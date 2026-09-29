"""Optimal-control comparison for the Zhou SNAIL iSWAP.

Optimizes the complex pump envelope eta(t) to maximize the leakage-aware iSWAP
fidelity and compares it against the DRAG-shaped raised-cosine gate the sweeps use.
The pump enters the Hamiltonian NONLINEARLY (eta, eta^2, eta^3 from g3 X^3), so
control-linear GRAPE does not apply. Backends:

* backend='qutip', alg='CRAB' (default): Chopped RAndom Basis -- a randomized
  Fourier basis over a fixed shape, minimized gradient-free (Nelder-Mead, as in
  qutip-qtrl). The black box is ``ZhouCoupler.propagator_columns`` (exact
  ``qt.sesolve``) on an ``envelope.IQFourierEnvelope``; needs only ``qutip``.
  ``crab_restarts>1`` gives DCRAB-style monotone super-iterations.
* backend='qutip', alg='JOPT': ``jax.grad`` of the reduced-model propagator over the
  analytic I/Q ansatz (``_iq_ansatz``), driven by scipy L-BFGS-B; needs only ``jax``.
  NOT qutip-qoc's JOPT: that scores the full-dimension propagator with no virtual-Z
  freedom, and on this gate the Z fit is worth +0.76 of a fidelity of 0.95, so a
  phase-rigid objective walks downhill (see ``_jax_pipeline_infidelity``). Slower per
  point than CRAB (~600-step expm chain), so opt-in. Everything the gradient traces
  must be built from ``jax.numpy`` with no ``float()``/``asarray(dtype=)`` on
  parameter-dependent values -- hence ``_iq_ansatz`` takes ``xp``.
* backend='reduced': scipy L-BFGS-B over the rotating-frame reduced model (scipy
  expm, no QuTiP); keeps the near-resonant band via ``cutoff_GHz``.

All score the 4x4 projected propagator with the SAME
``ZhouCoupler._iswap_fidelity_from_U`` the sweeps use, so F_baseline/F_grape compare
directly to the sweep's F_avg. Validate an optimized pulse in the full
``iswap_fidelity`` sim before use.

CLI
---
    python -m snail_solver.grape --device evan_device.json --t-g-ns 92.6 \
        --backend qutip --alg JOPT [--warmstart-drag-beat-GHz ...]
    python -m snail_solver.grape --device evan_device.json --t-g-ns 92.6 \
        --backend qutip --alg CRAB [--warmstart-drag-beat-GHz ...]
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

TWO_PI = 2.0 * np.pi


def _prepare(cpl, a: int, b: int, cutoff_GHz: float):
    """Precompute the rotating-frame terms, anharmonicity, subspace, max carrier.

    Returns
    -------
    terms : list of (Omega, n_pos, n_neg, O)
        Omega (rad/ns), pump exponents (eta^n_pos * conj(eta)^n_neg), operator.
    H_anh : ndarray
        Static transmon-anharmonicity operator (diagonal).
    idx : list of int
        The 4 computational-subspace fock indices (|00>,|01>,|10>,|11>).
    max_Omega : float
        Largest |Omega| kept (sets the fine-propagation step).
    """
    terms: List[Tuple[float, int, int, np.ndarray]] = []
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=cutoff_GHz):
        n_pos = sum(1 for (_ti, conj) in pump_sig if not conj)
        n_neg = sum(1 for (_ti, conj) in pump_sig if conj)
        terms.append((float(Omega), n_pos, n_neg, np.asarray(O, dtype=complex)))
    H_anh = np.asarray(cpl._anharm_op, dtype=complex)
    idx = list(cpl._subspace_indices(a, b))
    max_Omega = max((abs(t[0]) for t in terms), default=0.0)
    return terms, H_anh, idx, max_Omega


def _chirp_pad_rad(chirp, terms) -> float:
    """Extra carrier bandwidth (rad/ns) a chirp adds, for sizing the fine step.

    A term with k = n_pos - n_neg net pump quanta picks up ``e^{-i k Phi(t)}``, i.e. a
    carrier displaced by ``k delta(t)`` (cf. ``calibration_map.scan``'s pump-offset
    padding). Cheap insurance: it binds only for a pathologically large chirp.
    Returns 0.0 for no chirp, leaving every ``n_sub`` unchanged.
    """
    if chirp is None:
        return 0.0
    k_max = max((abs(n_pos - n_neg) for _Om, n_pos, n_neg, _O in terms), default=0)
    ts = np.linspace(0.0, chirp.t_g, 257)
    return float(k_max) * float(np.max(np.abs(chirp.detuning(ts, np))))


def _n_sub(max_Omega: float, chirp, terms, dt_ctrl: float, resolution: float = 0.3) -> int:
    """Fine sub-steps per control slice resolving the fastest (chirp-padded) carrier."""
    return max(1, int(np.ceil((max_Omega + _chirp_pad_rad(chirp, terms))
                              * dt_ctrl / resolution)))


# The three readers below take the pulse off the coupler's (single) tone rather than
# as arguments, so a caller cannot build a baseline that differs from what the solver
# actually plays.
def _tone_chirp(cpl):
    """The chirp on the coupler's pump tone, or None."""
    tones = getattr(cpl, "_pump_tones", None)
    return tones[0].chirp if tones else None


def _tone_n_pump(cpl) -> int:
    """Pump quanta of the process DRAG suppresses on this coupler's tone."""
    tones = getattr(cpl, "_pump_tones", None)
    return int(getattr(tones[0], "drag_n_pump", 1)) if tones else 1


def _tone_drag_channels(cpl) -> tuple:
    """Recursive-DRAG channels on this coupler's tone; ``()`` for legacy first-order
    DRAG, so `_raised_cosine_eta` keeps its original code path."""
    tones = getattr(cpl, "_pump_tones", None)
    if not tones or getattr(tones[0], "is_legacy_drag", False):
        return ()
    return tuple(tones[0].drag_channels_resolved())


def _H(t: float, eta: complex, terms, H_anh: np.ndarray,
       offset_rad: float = 0.0) -> np.ndarray:
    """Interaction-picture Hamiltonian at time t for pump amplitude eta.

    A pump-frequency offset shifts each carrier by (n_pos - n_neg) * offset, so one
    precomputed operator basis serves any offset.
    """
    H = H_anh.copy()
    for Omega, n_pos, n_neg, O in terms:
        Om = Omega + (n_pos - n_neg) * offset_rad
        f = (eta ** n_pos) * (np.conj(eta) ** n_neg)
        if Om != 0.0:
            f = f * np.exp(-1j * Om * t)
        H = H + f * O
    return H


def _propagate(eta_ctrl: np.ndarray, t_g: float, terms, H_anh, idx,
               n_sub: int, offset_rad: float = 0.0, *,
               chirp=None) -> np.ndarray:
    """Propagate the 4 computational states; return the 4x4 projected propagator.

    Parameters
    ----------
    eta_ctrl : ndarray (complex)
        N piecewise-constant control amplitudes -- the pulse SHAPE only; pass any chirp
        through `chirp`, not pre-multiplied in.
    t_g : float
        Gate duration (ns).
    terms, H_anh, idx : see _prepare
    n_sub : int
        Fine sub-steps per control slice; size with `_n_sub` (chirp-padded).
    offset_rad : float
        CONSTANT pump-frequency offset (rad/ns). Equivalent to a constant chirp c0
        (offset_rad = 2 pi c0) but free, which lets a calibration scan sweep it on one
        ``_prepare``. Composes additively with `chirp`.
    chirp : envelope.Chirp, optional
        Applied as ``eta -> eta e^{-i Phi(t)}`` like ``ZhouCoupler._eta_at``; since `_H`
        forms ``eta^n_pos conj(eta)^n_neg`` each term gets its k-quanta phase for free.
        Phi is evaluated at the FINE step: per control slice it can move ~1 rad (at
        t_g = 92.6 ns, n_ctrl = 32, c1 = 0.05 GHz), far too coarse to hold constant.
    """
    N = len(eta_ctrl)
    dt_ctrl = t_g / N
    dt_fine = dt_ctrl / n_sub
    dim = H_anh.shape[0]
    phase = None
    if chirp is not None:
        steps = np.arange(N * n_sub)
        t_all = (steps // n_sub) * dt_ctrl + ((steps % n_sub) + 0.5) * dt_fine
        phase = np.exp(-1j * np.asarray(chirp.phase(t_all, np)))
    Psi = np.zeros((dim, 4), dtype=complex)
    for col, s in enumerate(idx):
        Psi[s, col] = 1.0
    for j in range(N):
        eta = eta_ctrl[j]
        t0 = j * dt_ctrl
        for m in range(n_sub):
            t = t0 + (m + 0.5) * dt_fine
            eta_t = eta if phase is None else eta * phase[j * n_sub + m]
            Ustep = expm(-1j * _H(t, eta_t, terms, H_anh, offset_rad) * dt_fine)
            Psi = Ustep @ Psi
    return Psi[np.ix_(idx, range(4))]


def _score(U: np.ndarray, cpl) -> Tuple[float, float]:
    """Leakage-aware iSWAP (F, leakage) via the coupler's own scorer."""
    from snail_solver.zhou_coupler import ZhouCoupler
    return ZhouCoupler._iswap_fidelity_from_U(U, True)


def _dechirped_samples(cpl, tone, ts) -> np.ndarray:
    """Pump samples via ``cpl._eta`` (DRAG, prefactor, phase included) with the chirp
    divided back out, so `_propagate` can re-apply it on its fine grid. ``_eta``
    applies the chirp last, so this is exact."""
    eta = np.array([complex(cpl._eta(tone, float(t))) for t in ts])
    if tone.chirp is not None:
        eta = eta * np.exp(1j * np.asarray(tone.chirp.phase(ts, np)))
    return eta


def _raised_cosine_eta(t_g: float, peak: float, n_ctrl: int,
                       drag_beat_GHz: Optional[float], chirp=None,
                       drag_n_pump: int = 1, channels=None) -> np.ndarray:
    """DRAG-shaped raised-cosine envelope sampled at control-slice midpoints.

    This is the BASELINE every ``dF_grape`` is measured against, so its DRAG must match
    what the solver applies: on a chirped tone the beat is ``Delta_0 - k delta(t)``
    (``envelope.PumpTone.drag_detuning``), and with recursive-DRAG `channels` it is the
    composed pulse. With `channels` empty the first-order arithmetic runs unchanged.

    Returns AMPLITUDE only (no chirp phase) -- pass the same `chirp` to `_propagate`.
    Midpoints are strictly interior, so the envelope never vanishes at a sample.
    """
    ts = (np.arange(n_ctrl) + 0.5) * (t_g / n_ctrl)
    if channels:
        from snail_solver import drag as _drag
        from snail_solver.envelope import PumpTone, RaisedCosine
        env = RaisedCosine(peak, t_g)
        tone = PumpTone(w_p_GHz=0.0, envelope=env, chirp=chirp,
                        drag_channels=list(channels))
        order = _drag.required_order(channels)
        return np.asarray(_drag.apply_drag(
            env.jet_at(ts, order, np),
            [tone.channel_detuning_jet(c, ts, order, np) for c in channels],
            channels, np), dtype=complex)
    rc = 0.5 * (1.0 - np.cos(2.0 * np.pi * ts / t_g))
    eta = peak * rc.astype(complex)
    if drag_beat_GHz:                       # add the first-order DRAG quadrature
        drc = (np.pi / t_g) * np.sin(2.0 * np.pi * ts / t_g)
        detuning = 2.0 * np.pi * float(drag_beat_GHz)
        if chirp is not None and drag_n_pump:
            detuning = detuning - drag_n_pump * np.asarray(chirp.detuning(ts, np))
        eta = eta - 1j * peak * drc / detuning
    return eta


def _baseline_eta(cpl, t_g: float, peak: float, n_ctrl: int,
                  drag_beat_GHz: Optional[float], chirp) -> np.ndarray:
    """`_raised_cosine_eta` with the DRAG settings read off the coupler's tone."""
    return _raised_cosine_eta(t_g, peak, n_ctrl, drag_beat_GHz, chirp,
                              _tone_n_pump(cpl), _tone_drag_channels(cpl))


def _iq_ansatz(t_g: float, peak: float, n_basis: int, xp=np):
    """Analytic complex pump ansatz eta(t; p) for the gradient method.

    eta(t) = peak * env(t) * [ (1 + sum_k pI_k s_k(t)) + i * sum_k pQ_k s_k(t) ],
    env = raised cosine, s_k(t) = sin(k pi t/t_g) (zero-ended). p = [pI_1..K, pQ_1..K];
    p = 0 is the plain raised cosine.

    ``xp`` is ``numpy`` or ``jax.numpy``; under JAX the body is traced in ``p``, so it
    must never concretize p (no ``asarray(p, dtype=...)``, no ``float(...)``).
    """
    ks = xp.arange(1, n_basis + 1)

    def eta(t, p):
        p = xp.asarray(p)                     # no dtype= : must not concretize a tracer
        env = 0.5 * (1.0 - xp.cos(2.0 * xp.pi * t / t_g))
        s = xp.sin(ks * xp.pi * t / t_g)
        amp_I = 1.0 + xp.dot(p[:n_basis], s)
        amp_Q = xp.dot(p[n_basis:], s)
        return peak * env * (amp_I + 1j * amp_Q)

    return eta


def _drag_seed_params(t_g: float, peak: float, n_basis: int,
                      warmstart_beat_GHz: Optional[float]) -> np.ndarray:
    """Initial ansatz parameters: zeros (raised cosine), or a DRAG warm start loading
    the k=2 quadrature (~ the raised-cosine derivative) with the first-order weight.

    Uses the STATIC beat even on a chirped tone: a swept beat is not one Fourier
    coefficient, and seeds are scored before use, so an imperfect one is harmless.
    """
    p0 = np.zeros(2 * n_basis)
    if warmstart_beat_GHz:                       # DRAG: Q ~ -eta'(t) / (2 pi beat)
        # env'(t) = (pi/t_g) sin(2 pi t/t_g) = (pi/t_g) s_2(t); Q amplitude coeff on s_2
        p0[n_basis + 1] = -(np.pi / t_g) / (2.0 * np.pi * float(warmstart_beat_GHz))
    return p0


def _jax_pipeline_infidelity(terms, H_anh, idx, t_g: float, n_sub: int,
                             n_ctrl: int, eta_max: float, z_grid: int = 1024,
                             chirp=None):
    """Build a JAX-traceable ``eta_ctrl -> 1 - F`` on the SWEEP's own metric.

    qutip-qoc's objectives (TRACEDIFF / PSU / SU) score the full-dimension propagator
    rigidly in phase; the sweep scores the 4x4 projection with virtual-Z phases fitted
    out, worth +0.76 of a fidelity of 0.95 on the raised-cosine baseline. So the
    objective is rebuilt here, term for term the same as ``_propagate`` +
    ``ZhouCoupler._iswap_fidelity_from_U(U, True)``:

      * propagate the 4 computational columns through the same slices (differentiable
        expm), project onto the 4x4 block;
      * fit the virtual Z as ``_fit_virtual_z`` does: with c_k = (U U_ideal^dag)_kk the
        overlap is A(pa) + e^{i pb} B(pa), maximized over pb ANALYTICALLY as |A| + |B|,
        with pa swept over ``z_grid`` points (gradient flows through the winning point,
        as in max-pooling). The numpy version also ternary-searches the winning cell,
        so this is marginally pessimistic, by ~(2 pi / z_grid)^2;
      * Pedersen leakage-aware fidelity (overlap + Tr[U^dag U]) / 20.

    Optimizer and reporter thus evaluate the SAME function.

    Parameters
    ----------
    eta_max : float
        Largest |eta| the caller can present (the optimizer's amplitude ceiling, not
        the nominal peak); bounds ||H|| for the squaring count. A chirp (pure phase)
        cannot change it.
    chirp : envelope.Chirp, optional
        Fixed device chirp ``eta -> eta e^{-i Phi(t)}`` on the fine grid, baked in as a
        constant.

    Returns
    -------
    callable
        ``infid(eta_ctrl, chirp_coeffs=None)`` with ``eta_ctrl`` length ``n_ctrl``.
    """
    import jax
    import jax.numpy as jnp

    dt_ctrl = t_g / n_ctrl
    dt_fine = dt_ctrl / n_sub
    dim = H_anh.shape[0]

    # Scaling-and-squaring with the squaring count fixed at BUILD time:
    # jax.scipy.linalg.expm reads ||A|| at runtime (data-dependent control flow), which
    # cannot compile inside lax.scan. The count must come from a real ||H|| bound --
    # carrier_resolution alone ignores H_anh and the operator norms, and too few
    # squarings silently return a diverged series (observed: 0.798 vs a true 0.222).
    # ||H|| <= ||H_anh|| + sum_j ||O_j|| eta_max^(n_pos+n_neg); scale to <= 1/2, where
    # order 12 truncates at ~2e-14.
    _op_norm = float(np.linalg.norm(np.asarray(H_anh), 2))
    for _Om, _np_, _nn, _O in terms:
        _op_norm += float(np.linalg.norm(np.asarray(_O), 2)) * eta_max ** (_np_ + _nn)
    _SQ = int(max(0, np.ceil(np.log2(max(_op_norm * dt_fine, 1e-12) / 0.5))))
    _ORDER = 12
    from snail_solver.jax_engine import _expm_taylor

    # static (traced-constant) pieces, promoted to JAX arrays once
    H_anh_j = jnp.asarray(np.asarray(H_anh), dtype=jnp.complex128)
    Om_j = jnp.asarray([Om for Om, _, _, _ in terms], dtype=jnp.float64)
    npos = jnp.asarray([p for _, p, _, _ in terms], dtype=jnp.float64)
    nneg = jnp.asarray([n for _, _, n, _ in terms], dtype=jnp.float64)
    Ops = jnp.asarray(np.array([np.asarray(O) for _, _, _, O in terms]),
                      dtype=jnp.complex128)                      # (n_terms, dim, dim)

    Psi0 = jnp.asarray(np.eye(dim, dtype=complex)[:, list(idx)])  # (dim, 4)
    U_ideal = jnp.asarray(_ideal_iswap_np(), dtype=jnp.complex128)
    phases = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, z_grid, endpoint=False))

    def H_at(t, eta):
        f = (eta ** npos) * (jnp.conj(eta) ** nneg) * jnp.exp(-1j * Om_j * t)
        return H_anh_j + jnp.tensordot(f.astype(jnp.complex128), Ops, axes=(0, 0))

    # Flattened fine-step schedule (step k at t_all[k], control slice slice_of[k]),
    # driven by lax.scan: unrolling ~600 expm calls makes the reverse-mode compile
    # effectively never finish, whereas scan compiles ONE step.
    n_steps = n_ctrl * n_sub
    t_all = jnp.asarray(((np.arange(n_steps) % n_sub) + 0.5) * dt_fine
                        + (np.arange(n_steps) // n_sub) * dt_ctrl)
    slice_of = jnp.asarray(np.repeat(np.arange(n_ctrl), n_sub))
    chirp_phase = (None if chirp is None
                   else jnp.asarray(np.exp(-1j * np.asarray(
                       chirp.phase(np.asarray(t_all), np)))))

    def infid(eta_ctrl, chirp_coeffs=None):
        eta_steps = eta_ctrl[slice_of]                            # (n_steps,)
        if chirp_coeffs is not None:
            # traced coefficients SUPERSEDE the baked-in device chirp (never both)
            eta_steps = eta_steps * jnp.exp(
                -1j * _chirp_phase_jax(chirp_coeffs, t_all, t_g, jnp))
        elif chirp_phase is not None:
            eta_steps = eta_steps * chirp_phase

        def step(Psi, xs):
            t, eta = xs
            return _expm_taylor(-1j * H_at(t, eta) * dt_fine, _ORDER, _SQ, jnp) @ Psi, None

        Psi, _ = jax.lax.scan(step, Psi0, (t_all, eta_steps))
        U = Psi[jnp.asarray(list(idx)), :]                        # (4, 4)

        # virtual-Z fit: sum_k z_k c_k = (c0 + e^{i pa} c2) + e^{i pb} (c1 + e^{i pa} c3)
        c = jnp.diag(U @ jnp.conj(U_ideal).T)                     # c_k, length 4
        e = jnp.exp(1j * phases)
        merit = jnp.abs(c[0] + e * c[2]) + jnp.abs(c[1] + e * c[3])
        overlap = jnp.max(merit) ** 2                             # virtual-Z fitted
        trace_UU = jnp.real(jnp.trace(jnp.conj(U).T @ U))
        return 1.0 - (overlap + trace_UU) / 20.0                  # d*(d+1), d = 4

    return infid


def _chirp_phase_jax(coeffs, t, t_g: float, xp):
    """Accumulated chirp phase Phi(t) with TRACED coefficients (so ``jax.grad`` can
    differentiate through them); ``envelope.Chirp.phase`` only takes concrete ones.
    Shares ``jax_engine._chirp_phase``, which a test pins to ``Chirp.phase``."""
    from snail_solver.jax_engine import _chirp_phase
    return _chirp_phase({"chirp": [coeffs]}, t, 0, t_g, xp)


def _ideal_iswap_np() -> np.ndarray:
    """Local copy of the 4x4 target (avoids importing zhou_coupler at module import)."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = U[2, 2] = 0.0
    U[1, 2] = U[2, 1] = 1j
    return U


def _nan_or_float(x: Optional[float]) -> float:
    return np.nan if x is None else float(x)


def _float_list_or_none(xs: Optional[Sequence[float]]) -> Optional[List[float]]:
    return None if xs is None else [float(c) for c in xs]


def _scaled_chirp_tail(coeffs_GHz, n_chirp: int, chirp_bound_GHz: float) -> np.ndarray:
    """``y_k = c_k / chirp_bound_GHz`` for k = 1..n_chirp (c_0 skipped), zero-padded."""
    y = np.zeros(n_chirp)
    if coeffs_GHz is not None:
        tail = np.asarray(coeffs_GHz, dtype=float).ravel()[1:n_chirp + 1]
        y[:tail.size] = tail / chirp_bound_GHz
    return y


def _optimize_jax(cpl, a: int, b: int, t_g: float, *, n_basis: int, cutoff_GHz: float,
                  drag_beat_GHz: Optional[float], warmstart_beat_GHz: Optional[float],
                  maxiter: int, n_time: int, chirp_degree: int = 0,
                  chirp_seed_GHz: Optional[Sequence[float]] = None,
                  chirp_bound_GHz: float = 0.02,
                  verbose: bool = False) -> Dict[str, Any]:
    """JAX-autodiff gradient optimizer (alg='JOPT') over the pipeline's OWN metric.

    The ``_iq_ansatz`` parameters (plus an optional scaled chirp tail) are optimized
    against ``_jax_pipeline_infidelity`` by scipy L-BFGS-B with ``jac=True``. Since the
    optimizer and the reporter are the same function, ``F_grape`` is exact rather than
    re-scored. Requires ``jax`` only.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            f"alg='JOPT' needs jax ({exc}). Install it, or use alg='CRAB' -- the "
            f"gradient-free optimizer, which requires only qutip.") from exc
    jax.config.update("jax_enable_x64", True)     # 1e-4 fidelities need float64

    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    peak = float(cpl.peak_eta())
    chirp = _tone_chirp(cpl)
    n_sub = _n_sub(max_Omega, chirp, terms, t_g / max(int(n_time), 1))

    eta_fn = _iq_ansatz(t_g, peak, n_basis, xp=jnp)
    ts = (np.arange(n_time) + 0.5) * (t_g / n_time)
    ts_j = jnp.asarray(ts)
    # the L-BFGS-B box caps amp_I/amp_Q at 1 + n_basis*bound = 2, so |eta| <= 2*peak
    infid_eta = _jax_pipeline_infidelity(terms, H_anh, idx, t_g, n_sub, n_time,
                                         eta_max=2.0 * peak, chirp=chirp)

    # x = [pI(n_basis) | pQ(n_basis) | y_1 .. y_D], y_k = c_k / chirp_bound_GHz so every
    # parameter is O(1) for the line search
    n_chirp = max(int(chirp_degree), 0)

    def objective(x):
        p = x[:2 * n_basis]
        eta_ctrl = jnp.stack([eta_fn(t, p) for t in ts_j])
        if not n_chirp:
            return infid_eta(eta_ctrl)
        # c_0 pinned to zero INSIDE the trace: it is degenerate with wp_offset_GHz
        cc = jnp.concatenate([jnp.zeros(1),
                              x[2 * n_basis:] * chirp_bound_GHz])
        return infid_eta(eta_ctrl, cc)

    obj_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_obj(x):
        v, g = obj_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    p0 = _drag_seed_params(t_g, peak, n_basis, warmstart_beat_GHz)
    bound = 1.0 / float(n_basis)       # |eta| <= 2 peak: amp_I = 1 + sum p_k s_k, |s_k|<=1
    bounds = [(-bound, bound)] * (2 * n_basis)
    if n_chirp:
        y0 = np.clip(_scaled_chirp_tail(chirp_seed_GHz, n_chirp, chirp_bound_GHz),
                     -1.0, 1.0)
        p0 = np.concatenate([p0, y0])
        bounds = bounds + [(-1.0, 1.0)] * n_chirp
    res = minimize(scipy_obj, p0, method="L-BFGS-B", jac=True,
                   bounds=bounds,
                   options=dict(maxiter=maxiter, ftol=1e-12, disp=verbose))
    x_star = np.asarray(res.x, dtype=float)
    p_star = x_star[:2 * n_basis]
    chirp_opt = None
    if n_chirp:
        from snail_solver.envelope import Chirp
        chirp_opt = Chirp(np.concatenate([[0.0], x_star[2 * n_basis:] * chirp_bound_GHz]),
                          t_g)

    # reconstruct + score in numpy on the identical model (a check, not a re-score)
    eta_np = _iq_ansatz(t_g, peak, n_basis, xp=np)
    eta_opt = np.array([eta_np(t, p_star) for t in ts], dtype=complex)
    eta0 = _baseline_eta(cpl, t_g, peak, n_time, drag_beat_GHz, chirp)
    # the optimized chirp supersedes the device one; the BASELINE keeps the device chirp
    chirp_star = chirp_opt if n_chirp else chirp
    Fg, leakg = _score(_propagate(eta_opt, t_g, terms, H_anh, idx, n_sub,
                                  chirp=chirp_star), cpl)
    F0, leak0 = _score(_propagate(eta0, t_g, terms, H_anh, idx, n_sub,
                                  chirp=chirp), cpl)

    out = dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
               eta_opt=eta_opt, F_grape=Fg, leak_grape=leakg,
               n_ctrl=n_time, n_sub=n_sub, cutoff_GHz=cutoff_GHz,
               nfev=int(res.nfev),
               warmstart_beat_GHz=_nan_or_float(warmstart_beat_GHz),
               backend="qutip", alg="JOPT", n_basis=n_basis,
               qoc_fid_err=float(res.fun))
    if n_chirp:
        # same envelope, chirp switched off -- isolates what the chirp alone bought
        F_off, _ = _score(_propagate(eta_opt, t_g, terms, H_anh, idx, n_sub), cpl)
        out.update(chirp_coeffs_GHz=[float(c) for c in chirp_opt.coeffs_GHz],
                   chirp_degree=n_chirp, chirp_bound_GHz=float(chirp_bound_GHz),
                   chirp_seed_GHz=_float_list_or_none(chirp_seed_GHz),
                   F_chirp_off=float(F_off), dF_chirp=float(Fg - F_off))
    return out


def _crab_frequencies(n_basis: int, t_g: float, rng, jitter: float = 0.5) -> np.ndarray:
    """Randomized CRAB basis frequencies omega_k = 2 pi k (1 + r_k) / t_g, rad/ns.

    The offsets r_k ~ U(-jitter, jitter) remove the exact-harmonic blind spots (every
    sin(2 pi k t / t_g) vanishes at t_g/2) and let independent draws explore different
    subspaces.
    """
    k = np.arange(1, int(n_basis) + 1, dtype=float)
    return TWO_PI * k * (1.0 + rng.uniform(-jitter, jitter, size=k.size)) / t_g


def _drag_seed_crab(t_g: float, freqs: np.ndarray,
                    warmstart_beat_GHz: Optional[float]) -> np.ndarray:
    """Flat CRAB coefficients approximating the first-order DRAG pulse.

    eta -> eta - i eta'/(2 pi beta) with Hann S'(t) = (pi/t_g) sin(2 pi t/t_g) is a pure
    first-harmonic sin quadrature of weight -1/(2 beta t_g), loaded into sin_Q[0]
    (approximate, since that frequency is jittered).
    """
    n = int(np.asarray(freqs).size)
    p = np.zeros(4 * n)
    if warmstart_beat_GHz and n > 0:
        p[n] = -1.0 / (2.0 * float(warmstart_beat_GHz) * t_g)      # sin_Q[0]
    return p


def _optimize_crab(cpl, a: int, b: int, t_g: float, *, n_basis: int,
                   cutoff_GHz: float, drag_beat_GHz: Optional[float],
                   warmstart_beat_GHz: Optional[float], maxiter: int,
                   restarts: int, seed: Optional[int], score: str, method: str,
                   atol: float, rtol: float, nsteps: int,
                   chirp_degree: int = 0,
                   chirp_seed_GHz: Optional[Sequence[float]] = None,
                   chirp_bound_GHz: float = 0.02,
                   verbose: bool = False) -> Dict[str, Any]:
    """Optimize the pump envelope with CRAB (Chopped RAndom Basis).

    Gradient-free over a randomized Fourier basis on a fixed shape, so it needs no
    control-linear structure. ``qutip_qtrl``'s CRAB is not used because it builds the
    dynamics from a drift plus control operators, which would linearize away the
    eta^2/eta^3 terms (31-68% of the Hamiltonian here). Instead the objective is
    ``ZhouCoupler.propagator_columns`` (exact ``qt.sesolve``) on an
    :class:`envelope.IQFourierEnvelope`, scored like the sweeps. ``score='reduced'``
    swaps in the fast rotating-frame model for exploration (re-score with 'qutip').

    ``restarts > 1`` runs DCRAB super-iterations: each APPENDS a fresh random basis and
    warm-starts from the previous optimum (zeros on the new coefficients), so fidelity
    is monotone. The parameter count grows by ``4 * n_basis`` each time and
    Nelder-Mead degrades in high dimension -- prefer few harmonics and restarts.

    Chirp
    -----
    ``chirp_degree > 0`` appends a tail ``y_1 .. y_D`` carrying Legendre coefficients
    ``c_1 .. c_D`` of delta(t):

    * ``c_0`` is PINNED to zero: a constant chirp is a retuned carrier, degenerate with
      ``wp_offset_GHz``.
    * The tail is stored SCALED, ``y_k = c_k / chirp_bound_GHz``, so the Nelder-Mead
      simplex is O(1) in every direction (raw c_k ~ 5e-3 would stay frozen).

    The chirp is installed on the tone, so both score backends see it via ``cpl._eta``.

    Returns
    -------
    dict
        Same keys as :func:`optimize_pulse` plus ``crab_freqs`` / ``crab_params`` (the
        exact pulse) and, with a chirp, ``chirp_coeffs_GHz``, ``chirp_degree``,
        ``chirp_seed_GHz`` and ``F_chirp_off`` (winning envelope, chirp removed).
    """
    from snail_solver.zhou_coupler import ZhouCoupler
    from snail_solver.envelope import Chirp, IQFourierEnvelope

    rng = np.random.default_rng(seed)
    tone = cpl._pump_tones[0]
    env_saved, drag_saved, chirp_saved = tone.envelope, tone.drag, tone.chirp
    channels_saved = getattr(tone, "drag_channels", None)
    amp = float(env_saved.amp)
    n_chirp = max(int(chirp_degree), 0)          # free coefficients: c_1 .. c_D
    chirp_bound_GHz = float(chirp_bound_GHz)
    n_samples = max(int(4 * n_basis * max(restarts, 1)), 32)   # for stored eta_opt
    ts_samples = (np.arange(n_samples) + 0.5) * (t_g / n_samples)

    # reduced-model scaffolding (used by score='reduced', and cheap to build)
    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    n_sub_red = _n_sub(max_Omega, None, terms, t_g / n_samples)

    def score_current() -> Tuple[float, float]:
        """(F, leak) of whatever pulse is currently installed on the tone."""
        if score == "qutip":
            U = cpl.propagator_columns(a, b, t_g, atol=atol, rtol=rtol, nsteps=nsteps)
            return ZhouCoupler._iswap_fidelity_from_U(U, True)
        # sample through cpl._eta (not envelope.value, which would drop DRAG). Read
        # tone.chirp per call: the optimizer mutates it in place.
        eta = _dechirped_samples(cpl, tone, ts_samples)
        chirp = tone.chirp
        n_sub = max(n_sub_red, _n_sub(max_Omega, chirp, terms, t_g / n_samples))
        return _score(_propagate(eta, t_g, terms, H_anh, idx, n_sub, chirp=chirp), cpl)

    try:
        # (1) baseline: the gate actually applied at this point (DRAG or raised cosine)
        tone.envelope, tone.drag = env_saved, drag_saved
        if drag_beat_GHz is not None:
            tone.drag, tone.delta_drag_GHz = True, float(drag_beat_GHz)
        F0, leak0 = score_current()
        eta0 = np.array([complex(cpl._eta(tone, float(t))) for t in ts_samples])

        # (2) install the CRAB ansatz; zero coefficients reproduce the raised cosine
        freqs = _crab_frequencies(n_basis, t_g, rng)
        env = IQFourierEnvelope(amp, t_g, freqs=freqs)
        tone.envelope, tone.drag = env, False   # ansatz carries its own quadrature
        # An explicit channel list WINS over the legacy flags, so clearing `drag` alone
        # would keep the recursion firing on top of the ansatz.
        tone.drag_channels = None
        # the optimizer's own chirp, mutated in place by `_apply`
        opt_chirp = Chirp(np.zeros(n_chirp + 1), t_g) if n_chirp else None
        if n_chirp:
            tone.chirp = opt_chirp

        bound = 2.0                            # |IQ coefficient| bound (dimensionless)

        def _clip(p: np.ndarray) -> np.ndarray:
            """Box the IQ block and the (scaled) chirp tail separately."""
            q = np.asarray(p, dtype=float).copy()
            if n_chirp:
                q[:-n_chirp] = np.clip(q[:-n_chirp], -bound, bound)
                q[-n_chirp:] = np.clip(q[-n_chirp:], -1.0, 1.0)   # |c_k| <= the bound
            else:
                q = np.clip(q, -bound, bound)
            return q

        def _apply(p: np.ndarray) -> np.ndarray:
            """Install a full parameter vector (IQ block + chirp tail) on the tone."""
            q = _clip(p)
            if n_chirp:
                env.set_params(q[:-n_chirp])
                opt_chirp.set_params(np.concatenate(
                    [[0.0], q[-n_chirp:] * chirp_bound_GHz]))       # c_0 pinned
            else:
                env.set_params(q)
            return q

        def _pack(p_iq, coeffs=None) -> np.ndarray:
            """Full vector from an IQ block plus optional chirp coefficients (GHz)."""
            p_iq = np.asarray(p_iq, dtype=float)
            if not n_chirp:
                return p_iq
            return np.concatenate([p_iq, _scaled_chirp_tail(coeffs, n_chirp,
                                                            chirp_bound_GHz)])

        # Candidate starts, all scored first: zeros ARE the raised cosine and the DRAG
        # seed reproduces the DRAG baseline to first order, so starting from the best
        # keeps dF_grape from going negative because of a bad warm start.
        n_iq = 4 * int(n_basis)
        seeds = [("raised-cosine", _pack(np.zeros(n_iq)))]
        if drag_beat_GHz:
            seeds.append(("drag-baseline",
                          _pack(_drag_seed_crab(t_g, freqs, drag_beat_GHz))))
        if warmstart_beat_GHz:
            seeds.append(("warmstart",
                          _pack(_drag_seed_crab(t_g, freqs, warmstart_beat_GHz))))
        if n_chirp and chirp_seed_GHz is not None:
            seeds.append(("stark-chirp", _pack(np.zeros(n_iq), chirp_seed_GHz)))
        F_best, leak_best, p_best, seed_used = -1.0, 1.0, seeds[0][1], seeds[0][0]
        nfev_total = 0
        for name, p in seeds:
            _apply(p)
            F_s, leak_s = score_current()
            nfev_total += 1
            if F_s > F_best:
                F_best, leak_best, p_best, seed_used = F_s, leak_s, p, name
        if verbose:
            print(f"  CRAB start: {seed_used} (F = {F_best:.6f}) "
                  f"of {len(seeds)} candidate seed(s)"
                  + (f", chirp degree {n_chirp}" if n_chirp else ""))

        def infid(p: np.ndarray) -> float:
            _apply(p)
            F, _ = score_current()
            return 1.0 - F

        for sup in range(max(int(restarts), 1)):
            if sup > 0:
                # DCRAB: append a fresh random basis, previous optimum as the start.
                new = _crab_frequencies(n_basis, t_g, rng)
                freqs = np.concatenate([freqs, new])
                # the vector is [sI|sQ|cI|cQ|y]: split the chirp tail off first
                tail = p_best[-n_chirp:] if n_chirp else np.zeros(0)
                iq = p_best[:-n_chirp] if n_chirp else p_best
                k = iq.size // 4
                sI, sQ, cI, cQ = (iq[:k], iq[k:2 * k], iq[2 * k:3 * k], iq[3 * k:])
                z = np.zeros(new.size)
                p_best = np.concatenate([sI, z, sQ, z, cI, z, cQ, z, tail])
                env = IQFourierEnvelope(amp, t_g, freqs=freqs)
                tone.envelope = env
                _apply(p_best)
            res = minimize(infid, p_best, method=method,
                           options=dict(maxiter=int(maxiter), xatol=1e-6,
                                        fatol=1e-9, disp=verbose)
                           if method == "Nelder-Mead"
                           else dict(maxiter=int(maxiter), disp=verbose))
            nfev_total += int(res.nfev)
            p_try = _apply(res.x)
            F_try, leak_try = score_current()
            if F_try >= F_best:                # monotone: keep only improvements
                F_best, leak_best, p_best = F_try, leak_try, p_try
            else:
                _apply(p_best)
            if verbose:
                print(f"  CRAB super-iteration {sup + 1}/{restarts}: "
                      f"{freqs.size} harmonics, {p_best.size} params, F = {F_best:.6f}")

        _apply(p_best)
        eta_opt = env.samples(n_samples)
        out = dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                   eta_opt=eta_opt, F_grape=F_best, leak_grape=leak_best,
                   n_ctrl=n_samples, n_sub=n_sub_red, cutoff_GHz=cutoff_GHz,
                   nfev=nfev_total,
                   warmstart_beat_GHz=_nan_or_float(warmstart_beat_GHz),
                   backend="qutip", alg="CRAB", n_basis=int(n_basis),
                   crab_freqs=np.asarray(freqs, dtype=float),
                   crab_params=np.asarray(p_best, dtype=float),
                   crab_score=score, crab_restarts=int(restarts),
                   crab_seed_used=seed_used,
                   qoc_fid_err=float(1.0 - F_best))
        if n_chirp:
            coeffs = [float(c) for c in opt_chirp.coeffs_GHz]
            # F_grape - F_chirp_off is what the chirp alone bought on the same shape
            opt_chirp.set_params(np.zeros(n_chirp + 1))
            F_off, _leak_off = score_current()
            opt_chirp.set_params(np.asarray(coeffs, dtype=float))
            out.update(chirp_coeffs_GHz=coeffs, chirp_degree=n_chirp,
                       chirp_bound_GHz=chirp_bound_GHz,
                       chirp_seed_GHz=_float_list_or_none(chirp_seed_GHz),
                       F_chirp_off=float(F_off),
                       dF_chirp=float(F_best - F_off))
        return out
    finally:
        # never leave the caller's coupler holding the optimizer's pulse OR its chirp
        tone.envelope, tone.drag, tone.chirp = env_saved, drag_saved, chirp_saved
        tone.drag_channels = channels_saved


def optimize_pulse(cpl, a: int, b: int, t_g: float, *, n_ctrl: int = 24,
                   cutoff_GHz: float = 1.0, drag_beat_GHz: Optional[float] = None,
                   warmstart_beat_GHz: Optional[float] = None,
                   backend: str = "qutip", alg: str = "CRAB", n_basis: int = 6,
                   maxiter: int = 200, carrier_resolution: float = 0.3,
                   crab_restarts: int = 1, crab_seed: Optional[int] = None,
                   crab_score: str = "qutip", crab_method: str = "Nelder-Mead",
                   atol: float = 1e-10, rtol: float = 1e-8, nsteps: int = 500000,
                   chirp_degree: int = 0,
                   chirp_seed_GHz: Optional[Sequence[float]] = None,
                   chirp_bound_GHz: float = 0.02,
                   verbose: bool = False) -> Dict[str, Any]:
    """Optimize the pump envelope and compare to the DRAG raised-cosine baseline.

    Parameters
    ----------
    cpl : ZhouCoupler
        Coupler with a pump already set (its peak_eta sets the amplitude scale).
    a, b : int
        Target-qubit mode indices.
    t_g : float
        Gate duration (ns).
    backend : {'qutip', 'reduced'}
        'qutip': ``alg='CRAB'`` or ``alg='JOPT'`` (see the module docstring).
        'reduced': in-house scipy L-BFGS-B over the rotating-frame model (fast, no
        QuTiP).
    alg : {'JOPT', 'CRAB'}
        For ``backend='qutip'``. JOPT = JAX-autodiff gradient (needs ``jax``); CRAB =
        gradient-free over a randomized Fourier basis (needs only ``qutip``), the most
        robust choice, at the cost of many exact propagations.
    n_basis : int
        sin() functions per quadrature (JOPT) or randomized harmonics (CRAB, 4 real
        coefficients each). ``n_ctrl`` is the control count for 'reduced' and the
        sample resolution otherwise.
    crab_restarts : int
        CRAB DCRAB super-iterations (monotone).
    crab_seed : int, optional
        Seed for the random basis, for reproducible pulses.
    crab_score : {'qutip', 'reduced'}
        CRAB objective: exact QuTiP propagator (default) or the fast reduced model.
    crab_method : str
        Gradient-free scipy method for CRAB ('Nelder-Mead', or e.g. 'Powell').
    drag_beat_GHz : float or None
        Beat for the DRAG BASELINE quadrature (None -> plain raised cosine).
    warmstart_beat_GHz : float or None
        Seed the optimizer from a DRAG raised cosine at THIS beat; the baseline is
        unchanged. None -> start from the baseline.
    maxiter : int
        Optimizer iteration cap (per CRAB super-iteration).
    carrier_resolution : float
        Max Omega*dt_fine (rad) -- sets fine sub-steps per control slice.
    atol, rtol, nsteps
        QuTiP ODE controls for the exact objective (CRAB with crab_score='qutip').
    chirp_degree : int, default 0
        Also optimize the pump chirp over Legendre ``c_1 .. c_chirp_degree`` of
        delta(t); ``c_0`` is pinned (degenerate with ``wp_offset_GHz``). The tracked
        shape ``|eta(t)|^2 ~ cos^4(pi u / 2)`` is EVEN in gate time, so the useful terms
        are even: try 2 or 4. See ``stark_chirp``.
    chirp_seed_GHz : sequence of float, optional
        Extra scored start for the chirp (e.g. ``stark_chirp.seed_from_calibration_map``);
        a bad seed is ignored, never adopted.
    chirp_bound_GHz : float, default 0.02
        Box on each ``|c_k|``: ~5x a few-MHz Stark tracking amplitude, ~1% of the
        qubit-qubit detuning, well below collision/anharmonicity scales (~200-300 MHz).

    Returns
    -------
    dict
        eta_baseline, F_baseline, leak_baseline, eta_opt, F_grape, leak_grape,
        n_ctrl, n_sub, cutoff_GHz, nfev, warmstart_beat_GHz, backend/alg (+
        qoc_fid_err, and crab_freqs/crab_params for CRAB). Only CRAB with
        crab_score='qutip' reports FULL-QuTiP numbers; other paths score on the
        reduced model.
    """
    if backend == "qutip" and alg.upper() == "CRAB":
        return _optimize_crab(cpl, a, b, t_g, n_basis=n_basis, cutoff_GHz=cutoff_GHz,
                              drag_beat_GHz=drag_beat_GHz,
                              warmstart_beat_GHz=warmstart_beat_GHz,
                              maxiter=maxiter, restarts=crab_restarts,
                              seed=crab_seed, score=crab_score, method=crab_method,
                              atol=atol, rtol=rtol, nsteps=nsteps,
                              chirp_degree=chirp_degree,
                              chirp_seed_GHz=chirp_seed_GHz,
                              chirp_bound_GHz=chirp_bound_GHz, verbose=verbose)
    if backend == "qutip":
        return _optimize_jax(cpl, a, b, t_g, n_basis=n_basis, cutoff_GHz=cutoff_GHz,
                             drag_beat_GHz=drag_beat_GHz,
                             warmstart_beat_GHz=warmstart_beat_GHz,
                             maxiter=maxiter, n_time=n_ctrl,
                             chirp_degree=chirp_degree,
                             chirp_seed_GHz=chirp_seed_GHz,
                             chirp_bound_GHz=chirp_bound_GHz, verbose=verbose)
    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    chirp = _tone_chirp(cpl)
    n_sub = _n_sub(max_Omega, chirp, terms, t_g / n_ctrl, carrier_resolution)

    peak = float(cpl.peak_eta())
    eta0 = _baseline_eta(cpl, t_g, peak, n_ctrl, drag_beat_GHz, chirp)
    F0, leak0 = _score(_propagate(eta0, t_g, terms, H_anh, idx, n_sub, chirp=chirp), cpl)

    def infid(x: np.ndarray) -> float:
        eta = x[:n_ctrl] + 1j * x[n_ctrl:]
        U = _propagate(eta, t_g, terms, H_anh, idx, n_sub, chirp=chirp)
        F, _ = _score(U, cpl)
        return 1.0 - F

    # optimizer seed: the baseline pulse, or a DRAG raised cosine at a supplied beat
    eta_seed = (eta0 if warmstart_beat_GHz is None
                else _raised_cosine_eta(t_g, peak, n_ctrl, warmstart_beat_GHz, chirp,
                                        _tone_n_pump(cpl)))
    x0 = np.concatenate([eta_seed.real, eta_seed.imag])
    bound = 2.0 * (abs(peak) + 1e-6)       # keep amplitudes physical
    res = minimize(infid, x0, method="L-BFGS-B",
                   bounds=[(-bound, bound)] * (2 * n_ctrl),
                   options=dict(maxiter=maxiter, ftol=1e-9, disp=verbose))
    eta_opt = res.x[:n_ctrl] + 1j * res.x[n_ctrl:]
    Fg, leakg = _score(_propagate(eta_opt, t_g, terms, H_anh, idx, n_sub, chirp=chirp),
                       cpl)

    return dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                eta_opt=eta_opt, F_grape=Fg, leak_grape=leakg,
                n_ctrl=n_ctrl, n_sub=n_sub, cutoff_GHz=cutoff_GHz, nfev=res.nfev,
                warmstart_beat_GHz=_nan_or_float(warmstart_beat_GHz),
                backend="reduced")


def compare(config: Dict[str, Any], t_g: float, *, amp_scale: float = 1.0,
            wp_offset_GHz: float = 0.0, spec_abs_GHz: Optional[float] = None,
            **kw) -> Dict[str, Any]:
    """Build the coupler from a config and run optimize_pulse (a vs b = 0, 1)."""
    from snail_solver.device_utils import build_coupler
    cpl, w_p, eta_pk = build_coupler(config, t_g=t_g, amp_scale=amp_scale,
                                     wp_offset_GHz=wp_offset_GHz,
                                     spec_abs_GHz=spec_abs_GHz)
    out = optimize_pulse(cpl, 0, 1, t_g, **kw)
    out["w_p_GHz"] = w_p
    out["peak_eta"] = eta_pk
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.grape", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--t-g-ns", type=float, required=True)
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--wp-offset-GHz", type=float, default=0.0)
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--drag-beat-GHz", type=float, default=None)
    ap.add_argument("--warmstart-drag-beat-GHz", type=float, default=None,
                    help="seed the optimizer from a DRAG raised cosine at this beat "
                         "(baseline/dF unchanged); good start for a hard collision")
    ap.add_argument("--n-ctrl", type=int, default=24)
    ap.add_argument("--cutoff-GHz", type=float, default=1.0)
    ap.add_argument("--backend", choices=["qutip", "reduced"], default="qutip",
                    help="qutip = CRAB or JOPT optimal control (handles the control "
                         "nonlinearity); reduced = in-house scipy")
    ap.add_argument("--alg", choices=["JOPT", "CRAB"], default="CRAB",
                    help="qutip optimizer: JOPT is the JAX-autodiff gradient method "
                         "(needs jax); CRAB is gradient-free over a randomized basis "
                         "and needs only qutip")
    ap.add_argument("--crab-restarts", type=int, default=1,
                    help="[CRAB] DCRAB super-iterations; each appends a fresh random "
                         "basis and warm-starts from the previous optimum (monotone)")
    ap.add_argument("--crab-seed", type=int, default=None,
                    help="[CRAB] RNG seed for the random basis (reproducible pulses)")
    ap.add_argument("--crab-score", choices=["qutip", "reduced"], default="qutip",
                    help="[CRAB] objective: exact QuTiP propagator (default) or the "
                         "fast reduced model for exploration")
    ap.add_argument("--crab-method", default="Nelder-Mead",
                    help="[CRAB] gradient-free scipy method (Nelder-Mead, Powell, ...)")
    ap.add_argument("--n-basis", type=int, default=6,
                    help="sin() basis functions per quadrature (qutip backend); 0 "
                         "freezes the envelope to the plain raised cosine, which is "
                         "how to isolate the chirp as the ONLY free parameter")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--chirp-degree", type=int, default=0,
                    help="optimize the pump chirp too, over Legendre coefficients "
                         "c_1..c_D of delta(t) (c_0 is pinned -- it is degenerate "
                         "with --wp-offset-GHz). The Stark shape is EVEN in gate "
                         "time, so c_2 is the leading useful term: try 2 or 4")
    ap.add_argument("--chirp-bound-GHz", type=float, default=0.02,
                    help="box on each |c_k| (default 20 MHz)")
    ap.add_argument("--chirp-seed-GHz", default=None,
                    help="comma list seeding the chirp, e.g. from calibration_map's "
                         "'suggested chirp seed'; scored as one more candidate start")
    ap.add_argument("--chirp-seed-stark-MHz", type=float, default=None,
                    help="build the chirp seed analytically from a pulse-averaged "
                         "Stark shift (MHz) via stark_chirp.stark_chirp_seed")
    args = ap.parse_args()

    from snail_solver.paths import resolve_device
    from snail_solver.device_utils import load_device, parse_chirp_arg
    cfg = load_device(resolve_device(args.device))

    chirp_seed = parse_chirp_arg(args.chirp_seed_GHz)
    if args.chirp_seed_stark_MHz is not None:
        if chirp_seed:
            raise SystemExit("give either --chirp-seed-GHz or --chirp-seed-stark-MHz")
        from snail_solver.stark_chirp import describe_seed, stark_chirp_seed
        delta = float(args.chirp_seed_stark_MHz) * 1e-3
        chirp_seed = list(stark_chirp_seed(delta, degree=max(args.chirp_degree, 2)))
        print(describe_seed(chirp_seed, delta))

    out = compare(cfg, args.t_g_ns, amp_scale=args.amp_scale,
                  wp_offset_GHz=args.wp_offset_GHz, spec_abs_GHz=args.spec_abs_GHz,
                  drag_beat_GHz=args.drag_beat_GHz, n_ctrl=args.n_ctrl,
                  cutoff_GHz=args.cutoff_GHz, maxiter=args.maxiter,
                  warmstart_beat_GHz=args.warmstart_drag_beat_GHz,
                  backend=args.backend, alg=args.alg, n_basis=args.n_basis,
                  crab_restarts=args.crab_restarts, crab_seed=args.crab_seed,
                  crab_score=args.crab_score, crab_method=args.crab_method,
                  chirp_degree=args.chirp_degree, chirp_seed_GHz=chirp_seed,
                  chirp_bound_GHz=args.chirp_bound_GHz)
    if out.get("alg") == "CRAB":
        tag = (f"qutip/CRAB score={out.get('crab_score')} "
               f"{out.get('crab_freqs', np.zeros(0)).size} harmonics x "
               f"{out.get('crab_restarts')} super-iteration(s)")
    elif out.get("backend") == "qutip":
        tag = (f"qutip-qoc/{out.get('alg','JOPT')} "
               f"(fid_err={out.get('qoc_fid_err', float('nan')):.2e})")
    else:
        tag = f"reduced (cutoff={out['cutoff_GHz']} GHz)"
    print(f"{tag}, {out['n_ctrl']} pts, {out['nfev']} iters:")
    print(f"  DRAG raised-cosine : F = {out['F_baseline']:.5f}  leak = {out['leak_baseline']:.4f}")
    print(f"  GRAPE optimized    : F = {out['F_grape']:.5f}  leak = {out['leak_grape']:.4f}")
    print(f"  improvement dF = {out['F_grape'] - out['F_baseline']:+.5f}")
    if "chirp_coeffs_GHz" in out:
        nz = ", ".join(f"c{k}={c:+.5f}" for k, c in
                       enumerate(out["chirp_coeffs_GHz"]) if c)
        print(f"  chirp (degree {out['chirp_degree']}): [{nz or 'all zero'}] GHz")
        print(f"    same envelope, chirp OFF : F = {out['F_chirp_off']:.5f}")
        print(f"    attributable to the chirp: dF = {out['dF_chirp']:+.5f}")
        print(f"    --chirp-GHz \"{','.join(f'{c:g}' for c in out['chirp_coeffs_GHz'])}\"")
    print("  (validate the GRAPE pulse in the full QuTiP iswap_fidelity on the cluster)")


if __name__ == "__main__":
    main()
