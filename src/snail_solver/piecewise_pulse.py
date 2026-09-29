r"""Ramp / plateau / ramp: evolve a flat-top gate segment by segment and read the plateau.

A :class:`~snail_solver.envelope.SinePowerRamp` with ``t_rise < t_g/2`` splits the gate
into three Hamiltonians played back to back::

    H_ramp   on [0, t_r]              |eta| rising
    H_flat   on [t_r, t_g - t_r]      |eta| = eta_flat, constant
    H_ramp~  on [t_g - t_r, t_g]      the rise mirrored, eps(t_g - t) = eps(t)

each handing its final state to the next. On the plateau the AC-Stark shift is
STATIC, so with the carrier parked on the Stark-shifted resonance at ``eta_flat``
(no chirp) ``H_flat`` is what a perfect chirp leaves behind: anything still pulling
population away there is a frequency collision no chirp can remove.

Only the ENVELOPE is mirrored, not time-reversed: every term carries its own carrier
``e^{-i Omega t}`` (:meth:`ZhouCoupler.expand_terms`), so a literal ``H(t_g - t)``
is not a playable pulse. The three segments are windows of ONE Hamiltonian in
absolute time, and :func:`evolve_piecewise` over them equals the single-shot solve
exactly (pinned by the tests). This module only imports the pipeline.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from snail_solver.envelope import SinePowerRamp
from snail_solver.find_stark_resonance import CHANNELS, population_channels

TWO_PI = 2.0 * np.pi

# the channels a collision can deposit population into, keyed by audit category
CATEGORY_CHANNEL = {"coupler": "P_coupler", "spectator": "P_spectator",
                    "leakage": "P_leak", "other": "P_leak", "target": "P10"}


# ===========================================================================
# Segments
# ===========================================================================
def segment_bounds(env) -> List[Tuple[float, float]]:
    """``[(0, t_r), (t_r, t_g - t_r), (t_g - t_r, t_g)]`` for a flat-top envelope.

    With no plateau (``t_rise = t_g/2``, or a non-ramp envelope) the middle window
    has zero length, so a Hann gate reads as "all transient".
    """
    t_g = float(env.t_g)
    t_r = float(getattr(env, "t_rise", t_g / 2.0))
    return [(0.0, t_r), (t_r, t_g - t_r), (t_g - t_r, t_g)]


SEGMENT_NAMES = ("ramp_up", "flat", "ramp_down")


def _tlist(t0: float, t1: float, dt_out: Optional[float]) -> np.ndarray:
    """Output grid for one window: ~5 ns chunks (the solver's nsteps cap, as in
    ``ZhouCoupler._sesolve_final``), or `dt_out` spacing when a trajectory is wanted."""
    span = max(float(t1) - float(t0), 0.0)
    n = int(np.clip(np.ceil(span / 5.0), 1, 4000)) + 1
    if dt_out is not None:
        n = max(n, int(np.ceil(span / float(dt_out))) + 1)
    return np.linspace(float(t0), float(t1), n)


def _as_qobj(cpl, state: np.ndarray):
    import qutip as qt
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        return qt.Qobj(state.reshape(-1, 1), dims=[cpl.dims, [1] * cpl.n_modes])
    return qt.Qobj(state, dims=[cpl.dims, cpl.dims])


def evolve_piecewise(cpl, state0: np.ndarray, bounds: Sequence[Tuple[float, float]],
                     *, dt_out: Optional[Dict[int, float]] = None,
                     c_ops: Optional[Sequence[Any]] = None,
                     atol: float = 1e-10, rtol: float = 1e-8,
                     nsteps: int = 500000) -> Dict[str, Any]:
    """Evolve `state0` through consecutive windows, handing each final state on.

    Parameters
    ----------
    cpl : ZhouCoupler
        Pumped coupler; every window is a slice of its ONE Hamiltonian.
    state0 : ndarray
        A ket, shape ``(dim,)``, or a density matrix, shape ``(dim, dim)``.
    bounds : sequence of (t0, t1)
        Consecutive windows, e.g. from :func:`segment_bounds`.
    dt_out : dict, optional
        ``{segment_index: spacing_ns}`` for windows whose trajectory is wanted densely
        (the plateau, for its spectrum). Others keep ~5 ns chunks.
    c_ops : sequence of Qobj, optional
        Collapse operators (``open_system.collapse_ops``). Given, every window runs
        ``mesolve`` and hands on a density matrix; otherwise ``sesolve`` and a ket.

    Returns
    -------
    dict
        ``times`` / ``states`` per window (numpy; kets ``(n, dim)`` or density
        matrices ``(n, dim, dim)``), and ``boundary``: the state at ``t0`` of the
        first window followed by the state at the end of each window.
    """
    import qutip as qt
    H = cpl.to_qutip_hamiltonian()
    options = cpl._qutip_options(atol, rtol, nsteps)
    dt_out = dt_out or {}
    open_system = bool(c_ops)
    state = _as_qobj(cpl, state0)
    if open_system and state.isket:
        state = qt.ket2dm(state)

    times, states, boundary = [], [], [state.full() if not state.isket
                                        else state.full().ravel()]
    for k, (t0, t1) in enumerate(bounds):
        tl = _tlist(t0, t1, dt_out.get(k))
        if t1 - t0 <= 0.0:                       # empty window: nothing happens
            times.append(np.array([float(t0)]))
            states.append(np.asarray([boundary[-1]]))
            boundary.append(boundary[-1])
            continue
        if open_system:
            res = qt.mesolve(H, state, tl, c_ops=list(c_ops), options=options)
            arr = np.stack([s.full() for s in res.states])
        else:
            res = qt.sesolve(H, state, tl, options=options)
            arr = np.stack([s.full().ravel() for s in res.states])
        state = res.states[-1]
        times.append(tl)
        states.append(arr)
        boundary.append(arr[-1])
    return {"bounds": [tuple(map(float, b)) for b in bounds], "times": times,
            "states": states, "boundary": boundary, "open_system": open_system}


# ===========================================================================
# Populations -- kets, density matrices or bare probabilities
# ===========================================================================
def probabilities(states: np.ndarray, is_rho: bool = False) -> np.ndarray:
    """Fock populations ``(n, dim)`` from kets ``(n, dim)`` or density matrices
    ``(n, dim, dim)``. A single ket ``(dim,)`` gains a leading axis; a single density
    matrix must be passed as ``rho[None]`` or with ``is_rho=True`` -- a square 2-D
    array is otherwise read as a trajectory of kets."""
    s = np.asarray(states)
    if s.ndim == 1:
        return np.abs(s[None, :]) ** 2
    if s.ndim == 2 and is_rho:
        return np.real(np.diag(s))[None, :]
    if s.ndim == 2:
        return np.abs(s) ** 2
    return np.real(np.einsum("nii->ni", s))


def channel_populations(cpl, states: np.ndarray, init: Sequence[int],
                        tgt: Sequence[int], is_rho: bool = False
                        ) -> Dict[str, np.ndarray]:
    """``find_stark_resonance.population_channels`` for kets OR density matrices.

    That function only reads ``|psi|^2``, so it is fed ``sqrt(p)`` (whose square is
    exactly diag(rho)). Returns ``{channel: array over time}``.
    """
    p = probabilities(states, is_rho=is_rho)
    arr = population_channels(cpl, np.sqrt(np.clip(p, 0.0, None)), init, tgt)
    return {name: arr[i] for i, name in enumerate(CHANNELS)}


def project_computational(cpl, state: np.ndarray, occupations: Sequence[Sequence[int]]
                          ) -> np.ndarray:
    """`state` restricted to the listed BARE Fock states and renormalized.

    Starting the plateau from this is a SUDDEN switch-on of ``H_flat``: it strips the
    ramp's adiabatic dressing, so the plateau rings at every collision's ``W_j`` --
    its intrinsic fingerprint. It is NOT a clean reference for ramp contamination (an
    adiabatic ramp hands over a dressed state that rings far less); use a slow ramp.
    """
    idx = [cpl.fock_index(list(o)) for o in occupations]
    s = np.asarray(state, dtype=complex)
    out = np.zeros_like(s)
    if s.ndim == 1:
        out[idx] = s[idx]
        return out / np.linalg.norm(out)
    out[np.ix_(idx, idx)] = s[np.ix_(idx, idx)]
    return out / np.real(np.trace(out))


# ===========================================================================
# The plateau gate
# ===========================================================================
def plateau_t_g(config: Dict[str, Any], eta_flat: float, t_rise_ns: float) -> float:
    r"""Gate length that is a full iSWAP at plateau drive `eta_flat`.

    A sine-power ramp is point-symmetric (``R(t_r - s) = 1 - R(s)``), so the area is
    ``eta_flat (t_g - t_r)``; pinning it to the iSWAP area ``A`` gives
    ``t_g = A/eta_flat + t_r``. At fixed ramp time, plateau amplitude and length are
    ONE knob: ``t_flat = A/eta_flat - t_r``.
    """
    from snail_solver.tune_up import _area
    t_g = _area(config) / float(eta_flat) + float(t_rise_ns)
    if t_g < 2.0 * float(t_rise_ns) - 1e-9:
        raise ValueError(f"eta_flat={eta_flat:g} with t_rise={t_rise_ns:g} ns leaves no "
                         f"plateau (t_g={t_g:.2f} ns < 2 t_rise)")
    return float(t_g)


def plateau_carrier_GHz(wp_offset0_GHz: float, k2_MHz: float, k4_MHz: float,
                        eta: float) -> float:
    """Static carrier offset on the Stark-shifted resonance at constant drive `eta`:
    the calibrated static part plus the measured law ``k2 eta^2 + k4 eta^4``."""
    e = float(eta)
    return float(wp_offset0_GHz) + (float(k2_MHz) * e ** 2 + float(k4_MHz) * e ** 4) * 1e-3


def build_plateau_gate(config: Dict[str, Any], eta_flat: float, t_rise_ns: float,
                       carrier_offset_GHz: float, *, m: int = 3,
                       phase_mod: Optional["PlateauPhaseMod"] = None,
                       t_g_ns: Optional[float] = None):
    """A flat-top iSWAP at plateau drive `eta_flat`, carrier fixed, no chirp, no DRAG.

    Built through ``device_utils.build_coupler`` on a COPY of `config` whose envelope
    is a sine-power ramp with the requested rise. `t_g_ns` overrides the area-derived
    length while holding the amplitude at `eta_flat`, so the gate then mis-rotates --
    for probing, not scoring.

    Returns ``(cpl, t_g, cfg)``.
    """
    from snail_solver.device_utils import build_coupler
    from snail_solver.tune_up import fixed_eta_amp_scale

    t_g = plateau_t_g(config, eta_flat, t_rise_ns) if t_g_ns is None else float(t_g_ns)
    cfg = {**config, "envelope": "sine_power", "envelope_m": int(m),
           "envelope_rise_frac": float(t_rise_ns) / t_g, "chirp_coeffs_GHz": []}
    amp = fixed_eta_amp_scale(cfg, t_g, float(eta_flat))
    cpl, _w_p, peak = build_coupler(cfg, t_g, amp, float(carrier_offset_GHz), None,
                                    None, chirp_coeffs_GHz=[])
    if abs(peak - float(eta_flat)) > 1e-6 * max(1.0, float(eta_flat)):
        raise RuntimeError(f"plateau |eta| came out {peak:.6g}, wanted {eta_flat:.6g}")
    if phase_mod is not None:
        tone = cpl._pump_tones[0]
        cpl.set_pump(replace(tone, chirp=phase_mod))    # same (already-scaled) envelope
    return cpl, t_g, cfg


# ===========================================================================
# What the plateau SHOULD do: the static collision map of H_flat
# ===========================================================================
def predict_plateau_channels(cpl, t_g_ns: float, *, window_GHz: float = 1.0,
                             min_g_MHz: float = 0.05) -> List[Dict[str, Any]]:
    r"""Every collision of the post-Stark plateau Hamiltonian, with its two-level fate.

    ``spectator_audit.interaction_channels`` on `cpl` (whose peak IS the plateau
    drive) gives each process's coupling `g` and detuning ``det_j`` from the playing
    pump; since the pump sits on the dressed target, ``det_j`` already carries the
    target's Stark shift. The channel's OWN levels shift too (``level_stark_shifts``,
    second order)::

        Delta_j = det_j - [dE(f) - dE(i)]          (pump minus transition, MHz)

    That dressing is badly wrong next to a strong collision (``g/|Delta| ~ 0.3-1``),
    where the measured lines follow ``det_j`` far better. So ``W_MHz`` uses ``det_j``
    and ``W_dressed_MHz`` the dressed variant; :func:`match_lines` accepts either.

    A suddenly switched two-level process ``[[0, g], [g, Delta]]`` moves
    ``P = 4g^2/W^2 sin^2(pi W t)``, ``W = sqrt(Delta^2 + 4 g^2)``: `W` is the line the
    plateau should show. Adiabatic ramps leave only the static dressing
    ``~ (g/Delta)^2``, so a strong plateau line measures how non-adiabatic the ramp was.
    """
    from snail_solver.spectator_audit import interaction_channels, level_stark_shifts

    eta = float(cpl.peak_eta())
    rows = interaction_channels(cpl, window_GHz=window_GHz, t_g_ns=float(t_g_ns),
                                min_g_MHz=min_g_MHz)
    dE = level_stark_shifts(cpl, eta, window_GHz=window_GHz + 1.0) * 1e3   # MHz
    out = []
    for r in rows:
        i, f = int(r["i_index"]), int(r["f_index"])
        g = float(r["g_MHz"])
        det = float(r["detuning_MHz"])
        dressed = det - (dE[f] - dE[i])
        W = float(np.hypot(det, 2.0 * g))
        out.append({
            "name": r["name"], "category": r["category"], "n_pump": int(r["n_pump"]),
            "i_index": i, "f_index": f, "i_occ": cpl.decode_index(i),
            "f_occ": cpl.decode_index(f), "g_MHz": g, "detuning_MHz": det,
            "dressed_detuning_MHz": float(dressed), "W_MHz": W,
            "W_dressed_MHz": float(np.hypot(dressed, 2.0 * g)),
            "P_max": float(4.0 * g * g / W ** 2) if W > 0 else 1.0,
            "g_over_det": float(g / max(abs(dressed), 1e-9)),
            "channel": CATEGORY_CHANNEL.get(r["category"], "P_leak"),
        })
    return out


# ===========================================================================
# What the plateau DOES: its spectral fingerprint
# ===========================================================================
def matrix_pencil(t: np.ndarray, y: np.ndarray, *, order: Optional[int] = None,
                  rel_sv: float = 1e-3, pencil_frac: float = 0.4) -> List[Dict[str, float]]:
    """Frequencies, dampings and amplitudes of ``y(t) = sum_k a_k e^{s_k t}``.

    Hua & Sarkar's matrix pencil, not an FFT: on a ~50-100 ns plateau an FFT resolves
    only ``1/t_flat`` ~ 10-20 MHz, while the pencil's resolution is set by SNR and
    these trajectories are noise-free to solver tolerance.

    Returns the positive-frequency components, strongest first, as
    ``{f_MHz, damping_per_us, amplitude}`` (``amplitude`` is the peak-to-peak/2 of the
    real oscillation).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float) - float(np.mean(y))
    N = t.size
    if N < 8 or not np.any(y):
        return []
    dt = float(np.mean(np.diff(t)))
    L = max(2, int(pencil_frac * N))
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    _U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    M = int(order) if order is not None else int(np.sum(s > rel_sv * s[0]))
    M = max(1, min(M, L))
    V = Vh[:M].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    # a pole that grows (or dies) by more than e^20 over the record is a fit
    # artefact, and its powers would overflow the Vandermonde solve below
    z = z[np.isfinite(z) & (np.abs(np.log(np.abs(z) + 1e-300)) * N < 20.0)]
    if z.size == 0:
        return []
    Z = np.vander(z, N, increasing=True).T                      # (N, M)
    try:
        amps, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
    except np.linalg.LinAlgError:
        return []
    out = []
    for zk, ak in zip(z, amps):
        f = float(np.angle(zk)) / (TWO_PI * dt) * 1e3           # MHz
        if f <= 0.0:
            continue                                            # conjugate partner
        out.append({"f_MHz": f,
                    "damping_per_us": float(-np.log(max(abs(zk), 1e-300)) / dt * 1e3),
                    "amplitude": float(2.0 * abs(ak))})
    out.sort(key=lambda d: -d["amplitude"])
    return out


def match_lines(lines: Sequence[Dict[str, float]], predicted: Sequence[Dict[str, Any]],
                *, channel: str, tol_MHz: float = 8.0, min_amp: float = 1e-5
                ) -> List[Dict[str, Any]]:
    """Pair each measured line with the nearest predicted ``W_j``.

    Prefers channels whose category deposits into `channel`. An unmatched strong line
    is a process the audit does not list (higher order, outside the window, or a
    truncation artefact).
    """
    cands = [p for p in predicted if p["category"] != "target"]
    out = []
    for ln in lines:
        if ln["amplitude"] < min_amp:
            continue
        best, best_d = None, np.inf
        for p in cands:
            d = min(abs(p["W_MHz"] - ln["f_MHz"]),
                    abs(p.get("W_dressed_MHz", p["W_MHz"]) - ln["f_MHz"]))
            if p["channel"] != channel and channel != "P_leak":
                d += 0.5 * tol_MHz                              # soft category penalty
            if d < best_d:
                best, best_d = p, d
        rec = dict(ln, channel=channel)
        if best is not None and best_d <= tol_MHz:
            rec.update(match=best["name"], match_category=best["category"],
                       predicted_W_MHz=best["W_MHz"], predicted_P_max=best["P_max"],
                       mismatch_MHz=float(ln["f_MHz"] - best["W_MHz"]))
        else:
            rec.update(match=None)
        out.append(rec)
    return out


# ===========================================================================
# Pulse shaping ON the plateau: a windowed phase modulation
# ===========================================================================
class PlateauPhaseMod:
    r"""Phase modulation ``Phi(t) = A w(t) sin(w_m (t - t_r))`` confined to the plateau.

    Duck-types :class:`envelope.Chirp`, so ``ZhouCoupler._eta_at`` plays it as the
    tone's chirp: ``eta -> eta e^{-i Phi(t)}``. A term carrying `k` pump quanta sees
    ``e^{-i k Phi}``, i.e. sidebands at multiples of ``w_m`` weighted by ``J_n(k A)``;
    a channel with ``J_0(k A) = 0`` loses its carrier line. The one-pump target is
    suppressed by ``J_0(A)`` too, and :func:`build_plateau_gate` does not
    re-normalize for it.

    ``w(t)`` is 0 on the ramps and rises as a sine-power ramp (order `m`) over
    `edge_ns` inside each end of the plateau, so ``Phi`` is C^m and the ramps are
    exactly the un-modulated ones.
    """

    def __init__(self, A_rad: float, f_m_GHz: float, t_rise_ns: float, t_g_ns: float,
                 edge_ns: float = 5.0, m: int = 3) -> None:
        self.A = float(A_rad)
        self.f_m_GHz = float(f_m_GHz)
        self.t_r = float(t_rise_ns)
        self.t_g = float(t_g_ns)
        self.m = int(m)
        flat = self.t_g - 2.0 * self.t_r
        self.edge = float(min(edge_ns, 0.5 * flat)) if flat > 0 else 0.0
        self._window = (SinePowerRamp(1.0, flat, m=self.m, t_rise=self.edge)
                        if self.edge > 0 else None)

    def __repr__(self) -> str:  # noqa: D105
        return (f"PlateauPhaseMod(A={self.A:g} rad, f_m={self.f_m_GHz * 1e3:g} MHz, "
                f"t_r={self.t_r:g}, t_g={self.t_g:g}, edge={self.edge:g})")

    @property
    def is_trivial(self) -> bool:
        return self.A == 0.0 or self.edge <= 0.0

    @property
    def coeffs_GHz(self) -> np.ndarray:           # only read by error messages
        return np.array([self.A, self.f_m_GHz])

    def _window_jet(self, t, order: int, xp):
        """``w, w', ..., w^(order)``: a :class:`envelope.SinePowerRamp` over the plateau,
        so ``delta = Phi'`` is smooth through ``delta^(m-1)`` (a ``sin^2`` edge would
        put a step in ``delta'``)."""
        t = xp.asarray(t)
        return list(self._window.jet_at(t - self.t_r, order, xp))

    def _phase_jet(self, t, order: int, xp):
        """``Phi, Phi', ..., Phi^(order)`` by Leibniz on ``A w(t) sin(w_m (t - t_r))``."""
        from math import comb
        t = xp.asarray(t)
        wm = TWO_PI * self.f_m_GHz
        W = self._window_jet(t, order, xp)
        S = [wm ** n * xp.sin(wm * (t - self.t_r) + n * np.pi / 2)
             for n in range(order + 1)]
        return [self.A * sum(comb(n, j) * W[j] * S[n - j] for j in range(n + 1))
                for n in range(order + 1)]

    # -- the Chirp interface ------------------------------------------------
    def phase(self, t: Any, xp: Any = np) -> Any:
        """Phi(t), radians."""
        if self.is_trivial:
            return 0.0 * xp.asarray(t)
        return self._phase_jet(t, 0, xp)[0]

    def detuning(self, t: Any, xp: Any = np) -> Any:
        """delta(t) = dPhi/dt, rad/ns."""
        if self.is_trivial:
            return 0.0 * xp.asarray(t)
        return self._phase_jet(t, 1, xp)[1]

    def detuning_jet(self, t: Any, order: int, xp: Any = np) -> tuple:
        """``delta, delta', ..., delta^(order)``."""
        order = int(order)
        if self.is_trivial:
            z = 0.0 * xp.asarray(t)
            return tuple(z for _ in range(order + 1))
        return tuple(self._phase_jet(t, order + 1, xp)[1:])

    @property
    def n_params(self) -> int:
        return 2

    def get_params(self) -> np.ndarray:
        return np.array([self.A, self.f_m_GHz])

    def set_params(self, p) -> None:
        p = np.asarray(p, dtype=float).ravel()
        if p.size != 2:
            raise ValueError(f"expected (A_rad, f_m_GHz), got {p.size} values")
        self.A, self.f_m_GHz = float(p[0]), float(p[1])


# ===========================================================================
# One study point
# ===========================================================================
def segment_budget(cpl, run: Dict[str, Any], init: Sequence[int], tgt: Sequence[int]
                   ) -> Dict[str, Dict[str, float]]:
    """Per channel: population at each boundary, and what each segment ADDED."""
    B = channel_populations(cpl, np.stack(run["boundary"]), init, tgt)
    out: Dict[str, Dict[str, float]] = {}
    for name, v in B.items():
        rec = {"start": float(v[0])}
        for k, seg in enumerate(SEGMENT_NAMES[:len(v) - 1]):
            rec[f"end_{seg}"] = float(v[k + 1])
            rec[f"d_{seg}"] = float(v[k + 1] - v[k])
        out[name] = rec
    return out


def plateau_fingerprint(cpl, run: Dict[str, Any], predicted: Sequence[Dict[str, Any]],
                        init: Sequence[int], tgt: Sequence[int], *,
                        channels: Sequence[str] = ("P_coupler", "P_f_a", "P_f_b",
                                                   "P_leak"),
                        tol_MHz: float = 8.0) -> Dict[str, Any]:
    """Matrix-pencil lines of each leakage channel over the plateau, matched to the
    predicted collision map."""
    t = run["times"][1]
    P = channel_populations(cpl, run["states"][1], init, tgt)
    out: Dict[str, Any] = {}
    for ch in channels:
        lines = matrix_pencil(t, P[ch])
        out[ch] = {"mean": float(np.mean(P[ch])), "ptp": float(np.ptp(P[ch])),
                   "lines": match_lines(lines[:8], predicted, channel=ch,
                                        tol_MHz=tol_MHz)}
    return out
