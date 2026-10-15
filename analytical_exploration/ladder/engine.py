"""Run the UNCHANGED pipeline on a reduced Hamiltonian.

Every solve in tune-up, chevrons, length scans and scoring goes through
``ZhouCoupler.to_qutip_hamiltonian``. `install` replaces that one method, IN THIS
PROCESS ONLY, with a builder for the ladder level's effective Hamiltonian on the same
Fock space. Kets, subspace indices, the virtual-Z fit and the fidelity formula are then
the pipeline's own. No file in src/ is touched, and a running scan is unaffected.
Forked workers (ProcessPoolExecutor's default on Linux) inherit the patch.

The pulse is the coupler's own: ``eta(t) = cpl._eta(tone, t)``, which carries the
envelope, the amplitude, the Legendre chirp phase and phi_p. A carrier offset
``w_p - |w_b - w_a|`` (the calibration's `wp_offset_GHz`, or a chevron row's probe
offset) adds the phase ``e^{-i s 2pi dw t}`` to a term with s pump quanta, exactly as a
chirp does.
"""
from __future__ import annotations

import cmath
import os
from typing import Any, Dict, Optional

import numpy as np

from ladder.levels import DEFAULT_ORDER, DEFAULT_WINDOW_GHZ, LEVELS, cached_model

TWO_PI = 2.0 * np.pi
_STATE: Dict[str, Any] = {}


def model_for(cpl, order: Optional[int] = None, window_GHz: Optional[float] = None):
    if cpl.n_modes != 3 or cpl.coupler_index != 2:
        raise ValueError("the ladder engine handles the (a, b, coupler) system only")
    freqs = tuple(float(w) / TWO_PI for w in cpl.omega)
    lam = tuple(float(v) for v in cpl.participation)
    alpha = tuple(float(v) / TWO_PI for v in cpl.anharm)
    g = {n: float(v) / TWO_PI for n, v in cpl.g_n.items()}
    return cached_model(freqs, lam, alpha, g, tuple(cpl.dims),
                        order=order or _STATE.get("order", DEFAULT_ORDER),
                        window_GHz=window_GHz or _STATE.get("window", DEFAULT_WINDOW_GHZ),
                        q_final=_STATE.get("q_final"), method=_STATE.get("method", "matrix"),
                        cache_dir=_STATE.get("cache_dir"))


def reduced_qobjevo(cpl, level: str, order: Optional[int] = None,
                    window_GHz: Optional[float] = None):
    """The level's H_eff(t) as a QuTiP list-format QobjEvo on ``cpl``'s Fock space (rad/ns)."""
    import qutip as qt
    if len(cpl._pump_tones) != 1:
        raise ValueError("the ladder engine handles a single pump tone")
    model = model_for(cpl, order, window_GHz)
    tone = cpl._pump_tones[0]
    dw = float(tone.w_p_GHz) - model.ref_wp_GHz
    dims = [list(cpl.dims), list(cpl.dims)]
    cache: Dict[str, Any] = {"t": None, "eta": 0j}

    def eta_at(t: float) -> complex:
        if cache["t"] != t:
            cache["t"], cache["eta"] = t, cpl._eta(tone, t)
        return cache["eta"]

    def make_coeff(s: int, power: int, carrier: float):
        om = TWO_PI * (carrier + s * dw)
        n = abs(s)

        def coeff(t: float, **kw) -> complex:
            e = eta_at(t)
            ph = (e if s > 0 else e.conjugate()) ** n if n else 1.0
            return ph * (abs(e) ** power) * cmath.exp(-1j * om * t) * TWO_PI
        return coeff

    H = []
    static = None
    for e in model.select(level):
        q = qt.Qobj(e.matrix, dims=dims)
        if e.s == 0 and e.power == 0 and abs(e.carrier_GHz) < 1e-12:
            static = q * TWO_PI if static is None else static + q * TWO_PI
        else:
            H.append([q, make_coeff(e.s, e.power, e.carrier_GHz)])
    if static is not None:
        H.append(static)
    return qt.QobjEvo(H)


def install(level: str, *, order: int = DEFAULT_ORDER, window_GHz: float = DEFAULT_WINDOW_GHZ,
            q_final: Optional[int] = None, method: str = "matrix",
            cache_dir: Optional[str] = None) -> None:
    """Make every ZhouCoupler in this process (and its forks) solve the `level` model."""
    from snail_solver import zhou_coupler as Z
    if level not in LEVELS:
        raise ValueError(f"level {level!r}: expected one of {LEVELS}")
    if "orig" not in _STATE:
        _STATE["orig"] = Z.ZhouCoupler.to_qutip_hamiltonian
    _STATE.update(level=level, order=int(order), window=float(window_GHz),
                  q_final=q_final, method=method, cache_dir=cache_dir)
    os.environ["LADDER_LEVEL"] = level         # visible in logs of forked workers

    def to_qutip_hamiltonian(self, cutoff_GHz: float = np.inf, sparse: bool = True):
        return reduced_qobjevo(self, _STATE["level"])
    Z.ZhouCoupler.to_qutip_hamiltonian = to_qutip_hamiltonian


def uninstall() -> None:
    from snail_solver import zhou_coupler as Z
    if "orig" in _STATE:
        Z.ZhouCoupler.to_qutip_hamiltonian = _STATE.pop("orig")
