"""Shared machinery for the Zhou SNAIL iSWAP sweeps.

Constants, DEFAULT_CONFIG, the Point grid record, grid IO, the analytic collision
search (_nearest_collision), the per-point Stark chevron (_stark_offset_GHz), result
collection, and CLI helpers. Used by sweep_spectator, sweep_target and run_sweep_zhou.
"""
from __future__ import annotations

import os
import glob
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


TWO_PI = 2.0 * np.pi

_DELTA_EPS_GHz = 5e-4

def _drag_skip_GHz(config: Dict[str, Any]) -> float:
    """|beat| (GHz) below which DRAG is skipped: the first-order quadrature ~ 1/beat
    diverges toward the collision (``drag_skip_below_MHz``, default 5 MHz)."""
    return max(float(config.get("drag_skip_below_MHz", 5.0)) / 1e3, _DELTA_EPS_GHz)


#: Pump quanta k per collision channel, in the convention ``beat = separation - k w_p``.
#: A chirp moves the DRAG beat as ``Delta(t) = beat - k delta(t)``
#: (``envelope.PumpTone.drag_detuning``); the pump-independent "static" channel has k=0.
_PUMP_QUANTA = {"onepump": 1, "static": 0, "subharm": 2, "none": 1}


def _pump_quanta_of(kind: str) -> int:
    """k for a collision kind as labelled by :func:`_nearest_collision`."""
    return _PUMP_QUANTA.get(str(kind), 1)


def _drag_ok_with_chirp(config: Dict[str, Any], beat_GHz: float, n_pump: int,
                        chirp_coeffs_GHz: Optional[Sequence[float]],
                        t_g: float) -> bool:
    """Is DRAG safe for this (beat, k, chirp) over the whole pulse?

    With a chirp the beat is swept during the gate and can cross zero mid-pulse, so
    ``|beat| >= skip`` must hold at every time. Sweeps DISABLE DRAG when this fails
    (rather than raising) so one bad point cannot kill a scan.
    """
    skip = _drag_skip_GHz(config)
    if abs(float(beat_GHz)) < skip:
        return False
    if not chirp_coeffs_GHz or not n_pump:
        return True
    from snail_solver.envelope import Chirp
    ts = np.linspace(0.0, float(t_g), 257)
    delta = np.asarray(Chirp(chirp_coeffs_GHz, float(t_g)).detuning(ts, np)) / TWO_PI
    return bool(np.min(np.abs(float(beat_GHz) - int(n_pump) * delta)) >= skip)


def _drag_channels_filtered(config: Dict[str, Any], channels: Sequence[Any],
                            chirp_coeffs_GHz: Optional[Sequence[float]],
                            t_g: float) -> tuple:
    """The subset of `channels` that is DRAG-safe for this (chirp, t_g); may be empty.

    For recursive DRAG only the offending channel is dropped, keeping the suppression
    the remaining channels provide.
    """
    return tuple(c for c in channels
                 if _drag_ok_with_chirp(config, float(c.beat_GHz), int(c.n_pump),
                                        chirp_coeffs_GHz, t_g))


def _chirp_of(config: Dict[str, Any]) -> Optional[Sequence[float]]:
    """The configured chirp, or None when unset/empty."""
    return config.get("chirp_coeffs_GHz") or None


def _drag_beat_if_ok(config: Dict[str, Any], want_drag: bool, beat_GHz: float,
                     n_pump: int) -> Optional[float]:
    """`beat_GHz` if DRAG is wanted and safe under the configured chirp, else None."""
    ok = want_drag and _drag_ok_with_chirp(config, beat_GHz, n_pump, _chirp_of(config),
                                           float(config["t_g_ns"]))
    return beat_GHz if ok else None


def _solver_opts(config: Dict[str, Any]) -> Dict[str, Any]:
    """QuTiP tolerances from the config."""
    return dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                nsteps=int(config.get("nsteps", 500000)))


def _nonlinearities(config: Dict[str, Any]) -> Dict[int, float]:
    """{3: g3} plus {4: g4} when g4 is nonzero."""
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])
    return nonlin


def _calibrate_point(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                     spec_abs_GHz: Optional[float], drag_beat_GHz: Optional[float],
                     drag_n_pump: int) -> Dict[str, Any]:
    """Per-point amplitude + Stark tune-up (calibrate_gate); returns its "final" record."""
    from snail_solver import calibrate_gate as CG
    sub = dict(config); sub["qubit_freqs_GHz"] = [wa_GHz, wb_GHz]
    return CG.run_calibration(
        sub, float(config["t_g_ns"]),
        iters=int(config.get("calibrate_iters", 1)),
        amp_bounds=(float(config.get("cal_amp_lo", 0.6)),
                    float(config.get("cal_amp_hi", 1.4))),
        amp_points=int(config.get("cal_amp_points", 9)),
        span_MHz=float(config.get("stark_span_MHz", 60.0)),
        chevron_points=int(config.get("stark_points", 21)),
        window_factor=float(config.get("stark_window_factor", 2.0)),
        time_points=int(config.get("stark_time_points", 120)),
        solver=_solver_opts(config), n_jobs=1,
        spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
        drag_n_pump=drag_n_pump)["final"]


#: Result-row fields filled only by the GRAPE add-on (blank otherwise).
_GRAPE_BLANKS = dict.fromkeys(("grape_baseline_F", "F_grape", "leak_grape", "dF_grape",
                               "grape_nfev", "grape_warmstart_GHz"), "")

DEFAULT_CONFIG = {
    # target qubits a, b
    "qubit_freqs_GHz": [5.00, 4.60],
    "qubit_levels":    3,            # >=3 so target-pair leakage is captured
    "lam_a":           0.20,         # participation lambda_as = g_as/Delta_as
    "lam_b":           0.20,         # participation lambda_bs = g_bs/Delta_bs
    # coupler S (the SNAIL)
    "coupler_freq_GHz": 7.00,
    "coupler_levels":   5,
    "g3_GHz":           0.10,        # measured cubic (engine of every process)
    "g4_GHz":           0.0,         # optional quartic (four-wave mixing)
    # spectator mode: a single 3-level anharmonic transmon; participation = lam_b
    "spec_levels":       3,          # spectator Hilbert dimension (captures |1>->|2> leakage)
    "anchor":            1,          # spectator freq measured below qubit b
    "anharm_qubit_GHz": -0.20,       # transmon anharmonicity of qubits a & b (0 = harmonic)
    "anharm_spec_GHz":  -0.20,       # spectator anharmonicity (0 = harmonic 3-level ladder)
    # target-allocation sweep (--sweep target): fixed w_a & w_s, scan (w_b, w_spec)
    "min_detuning_GHz":  0.05,       # drop placements with |w_b-w_a| below this
    "drag_compare":         False,   # target sweep: also run DRAG-on in the near-collision window
    "drag_compare_below_MHz": 100.0, # |nearest beat| window (MHz) for the DRAG comparison
    "drag_skip_below_MHz":    5.0,   # |beat| below this -> DRAG SKIPPED (quadrature ~ 1/beat
                                     #   blows up near the collision); ~ a few x g_iswap
    # GRAPE add-on (--grape): optimize the pump envelope per point and record the
    # headroom over the DRAG/raised-cosine baseline, scored with the sweep's own
    # leakage-aware metric (so grape_baseline_F tracks F_avg). Opt-in and costly.
    "grape":            False,       # run GRAPE at each integrated point
    "grape_backend":    "qutip",     # "qutip" (qutip-qoc optimal control) | "reduced"
    "grape_alg":        "CRAB",      # "CRAB": gradient-free, needs only qutip. "JOPT": JAX
                                     #   autodiff on the same metric, far slower per point
    "grape_crab_restarts": 1,        # [CRAB] DCRAB super-iterations (monotone)
    "grape_crab_seed":  None,        # [CRAB] RNG seed for the random basis
    "grape_crab_score": "qutip",     # [CRAB] objective: "qutip" (exact) | "reduced"
    "grape_crab_method": "Nelder-Mead",   # [CRAB] gradient-free scipy method
    "grape_nbasis":     6,           # sin() basis functions per quadrature (qutip backend)
    "grape_nctrl":      24,          # piecewise-constant control points (reduced) / tlist
    "grape_cutoff_GHz": 1.0,         # reduced-model carrier cutoff for the optimizer
    "grape_maxiter":    200,         # L-BFGS-B iteration cap
    "grape_warmstart_drag": False,   # DRAG-off points: seed GRAPE from DRAG at the nearest
                                     #   beat (baseline unchanged; skipped inside drag_skip)
    # pulse / solver
    "t_g_ns":   60.0,
    "envelope": "raised_cosine",
    "amp_scale":      1.0,           # calibrated pump-amplitude correction (see calibrate_gate.py)
    "wp_offset_GHz":  0.0,           # calibrated pump-frequency offset from w_b - w_a
    # propagation engine: "qutip" = exact per-point reference; "jax" = batched engine
    # (jax_engine.py), a rotating-wave REDUCTION at any finite engine_cutoff_GHz --
    # run `validate_engines.py --cutoff-scan` before trusting a number.
    "engine": "qutip",
    "engine_cutoff_GHz": float("inf"),  # [jax] inf (exact) ON PURPOSE: a finite cutoff is
                                     #   only safe at weak drive (3 GHz at |eta|~9 gave
                                     #   max|dU|~0.9); the speed comes from CF4 + sparse ops
                                     #   + batching, not pruning. Lower only with a cutoff-scan.
    "engine_carrier_resolution": 0.1,  # max |Omega|*dt; CF4 is 4th order (halve -> ~16x)
    "engine_batch": 64,              # grid points per vmapped call
    "engine_precision": "f64",       # f32 is MIXED (f64 phases, complex64 state)
    "chirp_coeffs_GHz": [],          # pump-frequency offset delta(t) as Legendre coefficients
                                     #   in u = 2t/t_g - 1 (GHz); [] = none, [c0] == adding c0
                                     #   to wp_offset_GHz. See envelope.Chirp.
    "stark_drive":    False,         # drive each point at its AC-Stark-shifted resonance (spectator-aware chevron)
    "stark_span_MHz": 60.0,          # per-point chevron scan width (see find_stark_resonance.py)
    "stark_points":   21,            # per-point chevron offset samples
    "stark_window_factor": 2.0,      # chevron time window = factor * t_g
    "stark_time_points":   120,      # chevron time samples
    "stark_jobs":     1,             # processes for the chevron's offset scan (>1 only in
                                     #   mode=point; mode=local forces 1)
    "stark_match_pulse": False,      # chevron uses the ACTUAL pulse (+ DRAG on DRAG-on points)
                                     #   instead of a constant probe
    "calibrate_points": False,       # per-point amplitude+Stark tune-up (calibrate_gate),
                                     #   spectator-present + DRAG-aware (hardware-style)
    "calibrate_iters":  1,           # amplitude/frequency rounds per point
    "cal_amp_lo":       0.6,         # per-point amplitude-scale search bounds
    "cal_amp_hi":       1.4,
    "cal_amp_points":   9,
    "drag_always":      False,       # force DRAG on for every allocation point
    "drag_subharmonic": True,        # DRAG may target subharmonics: the pump's 2nd harmonic
                                     #   (2 w_p, from the SNAIL) driving mode i, detuning
                                     #   w_i - 2 w_p (NOT w_i/2 - w_p). False reproduces a
                                     #   pre-default run (--no-drag-subharmonic).
    "subharmonic_modes": ["a", "b", "spec", "s"],  # subharmonic channels to include; an
                                     #   EMPTY list means none (see _collision_candidates)
    "no_spectator": False,           # [target] BARE a-b-coupler gate (lam_spec=0); vary w_b so
                                     #   2 w_p scans the SNAIL subharmonic w_c = 2 w_p
    "integrate": True,               # set False for the instant analytic map only
    "rtol": 1e-8, "atol": 1e-10,     # QuTiP ODE tolerances
    "nsteps": 500000,                # max internal solver steps between outputs
}

DEFAULT_SPECFREQS_GHz = [round(0.20 + 0.05 * k, 3) for k in range(15)]

DEFAULT_DRAGS = [False, True]

DEFAULT_TARGET_DRAGS = [False]


@dataclass
class Point:
    """One sweep point.

    index : grid position and output-filename suffix.
    spec_freq_GHz : spectator detuning Delta = w_b - w_spec (spectator sweep).
    drag : first-order DRAG requested.
    kind : "spectator" (move one spectator against a fixed pair) or "target"
        (fixed w_a & w_s; scan w_b and the spectator -- frequency allocation).
    wa_GHz, wb_GHz, spec_abs_GHz : pair and ABSOLUTE spectator frequency for
        kind="target"; None for the spectator sweep (pair read from the config).
    """

    index: int
    spec_freq_GHz: float = 0.0
    drag: bool = False
    kind: str = "spectator"
    wa_GHz: Optional[float] = None
    wb_GHz: Optional[float] = None
    spec_abs_GHz: Optional[float] = None

def write_grid(outdir: str, config: Dict[str, Any], points: List[Point]) -> str:
    """Write config + points to ``<outdir>/grid.json`` (creating ``points/``); return its path."""
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    path = os.path.join(outdir, "grid.json")
    with open(path, "w") as f:
        json.dump({"config": config, "points": [asdict(p) for p in points]},
                  f, indent=2)
    return path

def load_grid(outdir: str) -> Tuple[Dict[str, Any], List[Point]]:
    """Load (config, points) from ``<outdir>/grid.json``."""
    with open(os.path.join(outdir, "grid.json")) as f:
        blob = json.load(f)
    return blob["config"], [Point(**p) for p in blob["points"]]

def _stark_offset_GHz(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                      t_g: float, amp_scale: float, solver: Dict[str, Any],
                      spec_abs_GHz: Optional[float] = None,
                      drag_beat_GHz: Optional[float] = None,
                      drag_n_pump: int = 1) -> Dict[str, Any]:
    """Per-point AC-Stark-shifted iSWAP resonance (find_stark_resonance chevron).

    Runs the chevron at this point's (w_a, w_b) and amplitude; the located offset from
    |w_b - w_a| maximises swap contrast. With ``spec_abs_GHz`` the spectator is in the
    chevron, so the offset carries its dispersive pull; None -> bare pair. The offset
    scan uses ``config['stark_jobs']`` processes (1 inside a pool; cpus-per-task in
    mode=point). ``drag_beat_GHz`` matters only with ``config['stark_match_pulse']``:
    the chevron then uses the ACTUAL raised-cosine pulse with DRAG at this beat, i.e.
    it locates the DRAG-ON resonance.

    Returns the full find_stark_resonance.scan result; callers take
    ``resonance_offset_GHz`` and may persist the rest.
    """
    from snail_solver import find_stark_resonance as FS
    sub = dict(config)
    sub["qubit_freqs_GHz"] = [float(wa_GHz), float(wb_GHz)]
    span = float(config.get("stark_span_MHz", 60.0)) / 1000.0
    n_pts = int(config.get("stark_points", 21))
    offsets = np.linspace(-span / 2.0, span / 2.0, n_pts)
    window = float(config.get("stark_window_factor", 2.0)) * float(t_g)
    n_time = int(config.get("stark_time_points", 120))
    shaped = bool(config.get("stark_match_pulse", False))
    # Probe WITH the gate's chirp so the result is the residual offset on top of it;
    # otherwise the chirp's mean c0 is counted twice (once here, once by the gate). Only
    # the shaped probe can carry a chirp.
    chirp = _chirp_of(config)
    if chirp is not None and not shaped:
        print("WARNING: device carries a chirp but stark_match_pulse is off; the "
              "constant probe locates the UN-chirped resonance, so the chirp's mean "
              "component is double-counted. Set stark_match_pulse=true.")
    return FS.scan(sub, float(t_g), float(amp_scale), offsets, window, n_time,
                   solver, n_jobs=int(config.get("stark_jobs", 1)),
                   spec_abs_GHz=(None if spec_abs_GHz is None else float(spec_abs_GHz)),
                   shape=("raised_cosine" if shaped else "constant"),
                   drag_beat_GHz=(float(drag_beat_GHz) if (shaped and drag_beat_GHz is not None)
                                  else None),
                   chirp_coeffs_GHz=(chirp if shaped else None),
                   drag_n_pump=int(drag_n_pump))

def _nearest_collision(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                       ws_GHz: float, wspec_GHz: float, w_p_GHz: float):
    """Nearest spectator/mode collision to the pump (see :func:`_collision_candidates`).

    Returns ``(|beat|, signed beat, kind, target_label, target_idx)`` with kind in
    {"onepump", "static", "subharm"}, or ``(0.0, 0.0, "none", "-", -1)`` if there are
    no candidates. Ties keep the FIRST candidate in canonical order.
    """
    cands = _collision_candidates(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_GHz)
    return min(cands, key=lambda c: c[0]) if cands else (0.0, 0.0, "none", "-", -1)


def _collision_candidates(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                          ws_GHz: float, wspec_GHz: float, w_p_GHz: float) -> list:
    """Every spectator/mode collision channel, in canonical scan order.

    Per target qubit q in {a, b} vs the spectator: one-pump swap (resonant at
    ``|w_q - w_spec| = w_p``) and static exchange (``w_q = w_spec``), with
    ``beat = |w_q - w_spec| - n w_p`` (n = 1, 0). With ``drag_subharmonic`` (or
    ``no_spectator``), the subharmonic drives of mode i in ``subharmonic_modes`` by the
    pump's 2nd harmonic, ``beat = w_i - 2 w_p``. The beat is the DRAG detuning.
    Target-sweep mode indices: a=0, b=1, coupler=2, spectator=3.
    """
    a, b, coupler, spec = 0, 1, 2, 3
    out = []
    no_spec = bool(config.get("no_spectator", False))
    if not no_spec:
        for q_idx, q_freq, q_label in ((a, wa_GHz, "a"), (b, wb_GHz, "b")):
            sep = abs(q_freq - wspec_GHz)
            for kind, harm in (("onepump", w_p_GHz), ("static", 0.0)):
                beat = sep - harm
                out.append((abs(beat), float(beat), kind, q_label, q_idx))
    if no_spec or bool(config.get("drag_subharmonic", True)):
        # `is None`, not `or`: an EMPTY list deliberately means "no subharmonics"; only
        # a MISSING key falls back (to the SNAIL one alone when there is no spectator).
        sub_modes = config.get("subharmonic_modes")
        if sub_modes is None:
            sub_modes = ["s"] if no_spec else ["a", "b", "spec", "s"]
        for lab, idx, wi in (("a", a, wa_GHz), ("b", b, wb_GHz),
                             ("spec", spec, wspec_GHz), ("s", coupler, ws_GHz)):
            if lab not in sub_modes:
                continue
            if lab == "spec" and no_spec:               # no spectator mode present
                continue
            beat = wi - 2.0 * w_p_GHz                    # detuning of the 2-pump drive
            out.append((abs(beat), float(beat), "subharm", lab, idx))
    return out


def collision_drag_channels(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                            ws_GHz: float, wspec_GHz: float, w_p_GHz: float, *,
                            n: int = 3, chirp_coeffs_GHz=None, t_g: float = 1.0,
                            quotient_rule: bool = True) -> tuple:
    """The `n` nearest collisions as :class:`envelope.DragChannel` objects (recursive DRAG).

    One derivative correction per nearby process -- a single correction can only trade
    one process's error against another's (Li/Calarco/Motzoi). Unsafe channels are
    dropped (:func:`_drag_channels_filtered`). Distinct beats only: composing the same
    substitution twice would double-count rather than suppress a second process.
    """
    from snail_solver.envelope import DragChannel
    cands = sorted(_collision_candidates(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz,
                                         w_p_GHz), key=lambda c: c[0])
    chans, seen = [], set()
    for _absb, beat, kind, _lab, _idx in cands:
        key = round(float(beat), 9)
        if key in seen:
            continue
        seen.add(key)
        chans.append(DragChannel.from_collision(float(beat), kind,
                                                quotient_rule=quotient_rule))
        if len(chans) >= int(n):
            break
    return _drag_channels_filtered(config, chans, chirp_coeffs_GHz, t_g)

def _grape_augment(out: Dict[str, Any], cpl, a: int, b: int,
                   config: Dict[str, Any], drag_beat_GHz: Optional[float] = None,
                   nearest_beat_GHz: Optional[float] = None) -> None:
    """Run GRAPE on the point's coupler and record the result in `out` in place.

    The baseline is the gate actually applied (``drag_beat_GHz`` = applied beat, None
    for the plain gate), scored with the sweep's leakage-aware metric, so ``dF_grape``
    is the optimal-control headroom over it. With ``grape_warmstart_drag`` a DRAG-off
    point is seeded from DRAG at ``nearest_beat_GHz`` (outside the drag-skip window
    only); the baseline is unchanged.

    Writes F_grape, leak_grape, dF_grape, grape_baseline_F, grape_nfev,
    grape_warmstart_GHz, and stashes the optimized envelope in ``out["_grape"]`` for
    ``save_point``. Reduced-model result: validate in ``iswap_fidelity``.
    """
    from snail_solver import grape
    warmstart = None
    if (config.get("grape_warmstart_drag") and drag_beat_GHz is None
            and nearest_beat_GHz is not None
            and abs(float(nearest_beat_GHz)) >= _drag_skip_GHz(config)):
        warmstart = float(nearest_beat_GHz)
    res = grape.optimize_pulse(
        cpl, a, b, float(config["t_g_ns"]),
        n_ctrl=int(config.get("grape_nctrl", 24)),
        cutoff_GHz=float(config.get("grape_cutoff_GHz", 1.0)),
        drag_beat_GHz=drag_beat_GHz, warmstart_beat_GHz=warmstart,
        backend=str(config.get("grape_backend", "qutip")),
        alg=str(config.get("grape_alg", "CRAB")),
        n_basis=int(config.get("grape_nbasis", 6)),
        crab_restarts=int(config.get("grape_crab_restarts", 1)),
        crab_seed=(None if config.get("grape_crab_seed") is None
                   else int(config["grape_crab_seed"])),
        crab_score=str(config.get("grape_crab_score", "qutip")),
        crab_method=str(config.get("grape_crab_method", "Nelder-Mead")),
        maxiter=int(config.get("grape_maxiter", 200)))
    out["grape_baseline_F"] = round(float(res["F_baseline"]), 6)
    out["F_grape"] = round(float(res["F_grape"]), 6)
    out["leak_grape"] = round(float(res["leak_grape"]), 6)
    out["dF_grape"] = round(float(res["F_grape"] - res["F_baseline"]), 6)
    out["grape_nfev"] = int(res["nfev"])
    out["grape_warmstart_GHz"] = (np.nan if warmstart is None else round(warmstart, 6))
    eta = np.asarray(res["eta_opt"], dtype=complex)
    out["_grape"] = dict(eta_opt=eta, n_ctrl=int(res["n_ctrl"]),
                         cutoff_GHz=float(res["cutoff_GHz"]),
                         drag_beat_GHz=(np.nan if drag_beat_GHz is None
                                        else float(drag_beat_GHz)),
                         warmstart_beat_GHz=res["warmstart_beat_GHz"])
    if res.get("alg") == "CRAB":          # basis + coefficients reproduce the pulse
        out["_grape"]["crab_freqs"] = np.asarray(res["crab_freqs"], dtype=float)
        out["_grape"]["crab_params"] = np.asarray(res["crab_params"], dtype=float)


def save_point(result: Dict[str, Any], outdir: str) -> str:
    """Persist one `run_point` row to ``<outdir>/points/point_XXXXX.npz``; return the path.

    ``U_proj`` is stored as real/imag arrays, the GRAPE envelope / chevron as extra
    arrays, and the remaining scalars as JSON ``meta``.
    """
    path = os.path.join(outdir, "points", f"point_{result['index']:05d}.npz")
    U = result.pop("U_proj", None)
    if U is None:
        U = np.zeros((4, 4), dtype=complex)
    chev = result.pop("_chevron", None)        # popped before JSON so meta stays scalar
    grp = result.pop("_grape", None)           # GRAPE optimized envelope (array) -> extra
    extra: Dict[str, Any] = {}
    if grp is not None:
        eta = np.asarray(grp["eta_opt"], dtype=complex)
        extra.update(grape_eta_real=np.real(eta), grape_eta_imag=np.imag(eta),
                     grape_n_ctrl=np.asarray(grp["n_ctrl"], dtype=int),
                     grape_cutoff_GHz=np.asarray(grp["cutoff_GHz"], dtype=float),
                     grape_drag_beat_GHz=np.asarray(grp["drag_beat_GHz"], dtype=float),
                     grape_warmstart_beat_GHz=np.asarray(
                         grp.get("warmstart_beat_GHz", np.nan), dtype=float))
        if "crab_freqs" in grp:
            extra.update(grape_crab_freqs=np.asarray(grp["crab_freqs"], dtype=float),
                         grape_crab_params=np.asarray(grp["crab_params"], dtype=float))
    if chev is not None:
        # NB: replaces (does not merge with) any GRAPE arrays above.
        extra = {"chev_offsets_GHz": np.asarray(chev["offsets_GHz"], dtype=float),
                 "chev_times_ns": np.asarray(chev["times_ns"], dtype=float),
                 "chev_P10": np.asarray(chev["P10"], dtype=float),
                 "chev_max_transfer": np.asarray(chev["max_transfer"], dtype=float),
                 "chev_resonance_metric": np.asarray(
                     chev.get("resonance_metric", chev["max_transfer"]), dtype=float),
                 "chev_metric_label": str(chev.get("metric_label", "max-over-time P(|10>)")),
                 "chev_resonance_offset_GHz": float(chev["resonance_offset_GHz"]),
                 "chev_eta_op": float(chev["eta_op"]),
                 "chev_shape": str(chev.get("shape", "constant")),
                 "chev_drag_beat_GHz": float(chev.get("drag_beat_GHz", np.nan)),
                 "chev_spec_abs_GHz": float(chev.get("spec_abs_GHz", np.nan))}
    np.savez_compressed(path,
                        U_proj_real=np.real(U), U_proj_imag=np.imag(U),
                        meta=json.dumps(result), **extra)
    return path


_SPEC_COLS = ["index", "spec_freq_GHz", "lam_spec", "drag", "drag_applied",
              "beat_GHz", "nearest_kind", "nearest_target",
              "eta_peak", "g_iswap_eff_MHz", "g_spec_eff_MHz",
              "status", "F_avg", "leakage", "n_spec", "n_coupler", "p_transfer",
              "grape_baseline_F", "F_grape", "leak_grape", "dF_grape", "grape_nfev",
              "grape_warmstart_GHz",
              "w_p_GHz", "stark_offset_MHz", "amp_scale_used", "wp_offset_used_MHz",
              "w_spec_GHz", "t_g_ns", "wall_s"]
_TARGET_COLS = ["index", "kind", "wa_GHz", "wb_GHz", "w_snail_GHz",
                "spec_GHz", "detuning_GHz", "w_p_GHz", "stark_offset_MHz",
                "amp_scale_used", "wp_offset_used_MHz", "lam_spec", "drag",
                "drag_applied", "drag_compare_window", "eta_peak",
                "g_iswap_eff_MHz", "nearest_beat_GHz", "nearest_kind",
                "nearest_target", "g_collision_MHz", "status", "F_avg", "leakage",
                "F_avg_drag", "leakage_drag", "dF_drag",
                "grape_baseline_F", "F_grape", "leak_grape", "dF_grape", "grape_nfev",
                "grape_warmstart_GHz",
                "n_spec", "n_coupler",
                "p_transfer", "t_g_ns", "wall_s"]


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float))


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _print_target_summary(rows: List[Dict[str, Any]]) -> None:
    """Best allocation (allowing DRAG where compared) and the DRAG gain in the window."""
    def _best_F(r: Dict[str, Any]) -> float:
        vals = [v for v in (r.get("F_avg"), r.get("F_avg_drag")) if _is_num(v)]
        return max(vals) if vals else float("-inf")

    scored = [r for r in rows if _best_F(r) > float("-inf")]
    if scored:
        best = max(scored, key=_best_F)
        with_drag = (_is_num(best.get("F_avg_drag"))
                     and best["F_avg_drag"] >= best.get("F_avg", -1))
        _wb, _ws, _nb = (_as_float(best.get("wb_GHz")), _as_float(best.get("spec_GHz")),
                         _as_float(best.get("nearest_beat_GHz")))
        wb_str = f"{_wb:.4f}" if _wb is not None else str(best.get("wb_GHz"))
        ws_str = f"{_ws:.4f} GHz" if _ws is not None else "bare (no spectator)"
        nb_str = f"{_nb:+.3f}" if _nb is not None else str(best.get("nearest_beat_GHz"))
        print(f"Best allocation: w_b={wb_str} GHz, w_spec={ws_str} "
              f"({'with' if with_drag else 'no'} DRAG) -> F={_best_F(best):.5f}, "
              f"nearest_beat={nb_str} GHz")
    gains = [r["dF_drag"] for r in rows if _is_num(r.get("dF_drag"))]
    if gains:
        g = np.array(gains)
        print(f"DRAG effect over {g.size} points with |beat|<threshold: "
              f"mean dF={g.mean():+.4f}, best dF={g.max():+.4f}, "
              f"helped {int((g > 0).sum())}/{g.size}")


def collect(outdir: str) -> None:
    """Gather ``points/point_*.npz`` into ``summary.csv`` (sorted by index) and
    ``combined.npz`` (stacked 4x4 propagators); warn about unfinished grid points."""
    files = sorted(glob.glob(os.path.join(outdir, "points", "point_*.npz")))
    if not files:
        print("No point_*.npz found; nothing to collect.", file=sys.stderr)
        return
    rows, U_stack, idx_stack = [], [], []
    for fpath in files:
        with np.load(fpath, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            U = z["U_proj_real"] + 1j * z["U_proj_imag"]
        rows.append(meta); U_stack.append(U); idx_stack.append(meta["index"])

    rows.sort(key=lambda r: r["index"])
    is_target = bool(rows) and rows[0].get("kind") == "target"
    cols = _TARGET_COLS if is_target else _SPEC_COLS
    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    order = np.argsort(idx_stack)
    np.savez_compressed(os.path.join(outdir, "combined.npz"),
                        index=np.array(idx_stack)[order],
                        U_proj=np.array(U_stack)[order])
    print(f"Collected {len(rows)} points -> {csv_path} and combined.npz")

    try:  # flag gaps so a partial (timed-out) sweep is obvious here
        _cfg, _pts = load_grid(outdir)
        got = {r["index"] for r in rows}
        gaps = sorted(set(range(len(_pts))) - got)
        if gaps:
            print(f"  WARNING: {len(gaps)}/{len(_pts)} grid points have no file "
                  f"(unfinished). Run `missing` to list and resubmit them.")
    except Exception:
        pass  # no grid.json (e.g. collecting a hand-assembled points dir)

    if is_target:
        _print_target_summary(rows)

def plot_chevrons(outdir: str, indices: Optional[List[int]] = None) -> List[str]:
    """Render the per-point Stark chevrons saved by --stark runs (optionally only
    `indices`) to ``<outdir>/figs/chevrons/chevron_XXXXX.png``; return the PNG paths."""
    from snail_solver import find_stark_resonance as FS
    figs = os.path.join(outdir, "figs", "chevrons")
    os.makedirs(figs, exist_ok=True)
    written: List[str] = []
    for npz in sorted(glob.glob(os.path.join(outdir, "points", "point_*.npz"))):
        d = np.load(npz)
        if "chev_P10" not in d.files:
            continue
        meta = json.loads(str(d["meta"]))
        idx = int(meta.get("index", -1))
        if indices is not None and idx not in indices:
            continue
        chev = {"offsets_GHz": d["chev_offsets_GHz"], "times_ns": d["chev_times_ns"],
                "P10": d["chev_P10"], "max_transfer": d["chev_max_transfer"],
                "resonance_offset_GHz": float(d["chev_resonance_offset_GHz"]),
                "eta_op": float(d["chev_eta_op"]),
                "shape": str(d["chev_shape"]),
                "drag_beat_GHz": float(d["chev_drag_beat_GHz"])}
        if "chev_resonance_metric" in d.files:
            chev["resonance_metric"] = d["chev_resonance_metric"]
            chev["metric_label"] = (str(d["chev_metric_label"])
                                    if "chev_metric_label" in d.files else "")
        bits: List[str] = []
        if "spec_freq_GHz" in meta:
            bits.append(rf"$\Delta$={float(meta['spec_freq_GHz']):.3f} GHz")
        if _is_num(meta.get("beat_GHz")):
            bits.append(f"beat={float(meta['beat_GHz'])*1e3:+.0f} MHz")
        bits.append("DRAG on" if meta.get("drag") else "DRAG off")
        png = os.path.join(figs, f"chevron_{idx:05d}.png")
        FS.render_chevron(chev, png, title_suffix=f"pt {idx}  " + "  ".join(bits))
        written.append(png)
    return written

def _parse_list(s: Optional[str], cast: Callable[[str], Any]) -> Optional[List[Any]]:
    """Comma-separated CLI string -> list via `cast`; None when `s` is empty."""
    return [cast(x) for x in s.split(",")] if s else None

def _bool_list(s: Optional[str]) -> Optional[List[bool]]:
    """Comma-separated booleans ({1,true,t,yes,on} -> True); None when `s` is empty."""
    if not s:
        return None
    return [tok.strip().lower() in ("1", "true", "t", "yes", "on")
            for tok in s.split(",")]

def _log_line(res: Dict[str, Any]) -> str:
    """One-line human summary for a result row (both sweep kinds)."""
    f_str = res["F_avg"] if res.get("F_avg", "") != "" else "  --  "
    grape_tail = (f" F_grape={res['F_grape']:.4f}(dF={res['dF_grape']:+.4f})"
                  if _is_num(res.get("F_grape")) else "")
    if res.get("kind") == "target":
        tail = f" dF_drag={res['dF_drag']:+.4f}" if _is_num(res.get("dF_drag")) else ""
        tail += grape_tail
        _ws = res.get("spec_GHz", "")
        wspec_str = f"{_ws:.3f}" if _is_num(_ws) else "bare"
        return (f"wb={res['wb_GHz']:.3f} wspec={wspec_str} wp={res['w_p_GHz']:.3f} "
                f"nearest={res['nearest_beat_GHz']:+.3f}GHz({res['nearest_kind']}) "
                f"g_coll={res['g_collision_MHz']}MHz F={f_str}{tail}")
    return (f"beat={res['beat_GHz']:+.3f}GHz drag={res['drag_applied']} "
            f"eta={res['eta_peak']:.3f} g_spec={res['g_spec_eff_MHz']:.3f}MHz F={f_str}{grape_tail}")


def _chunking(m: int, max_array: int) -> Tuple[int, int]:
    """(points per task, number of tasks) so the array stays within `max_array`."""
    chunk = math.ceil(m / max_array)
    return chunk, math.ceil(m / chunk)


def _print_submit_hint(outdir: str, m: int, max_array: int = 1000) -> None:
    """Print the sbatch line, chunking the array (CHUNK contiguous points per task) when
    `m` exceeds a typical SLURM ``MaxArraySize`` (site-specific; conservative default)."""
    base = f"RUNNER=snail_solver.run_sweep_zhou OUTDIR={outdir}"
    if m <= max_array:
        print(f"Submit with:\n  {base} sbatch --array=0-{m - 1} slurm/snail_sweep.slurm")
        return
    chunk, ntasks = _chunking(m, max_array)
    print(f"Submit with (N={m} exceeds a typical MaxArraySize={max_array}, so CHUNK the array):")
    print(f"  {base} CHUNK={chunk} sbatch --array=0-{ntasks - 1} slurm/snail_sweep.slurm")
    print(f"  -> {ntasks} tasks x {chunk} points/task. Check your site limit with "
          f"`scontrol show config | grep MaxArraySize` and raise --array/lower CHUNK if it allows.")
    print(f"  (raise #SBATCH --time accordingly: each task now runs {chunk} points in series.)")

def _compress_ranges(indices: List[int]) -> str:
    """Sorted unique indices -> SLURM ``--array`` spec, e.g. [3, 7, 8, 9, 20] -> "3,7-9,20"."""
    if not indices:
        return ""
    parts: List[str] = []
    lo = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{lo}" if lo == prev else f"{lo}-{prev}")
        lo = prev = i
    parts.append(f"{lo}" if lo == prev else f"{lo}-{prev}")
    return ",".join(parts)

def find_missing(outdir: str, max_array: int = 1000) -> None:
    """Report grid points with no saved ``point_XXXXX.npz``, write ``missing.txt``, and
    print resubmission commands.

    A point file is written only after :func:`run_point` returns, so a missing file
    means the task died first. Finished-but-bad points (e.g. nan ``F_avg``) are NOT
    caught here; check ``summary.csv`` after ``collect``. `max_array` is the assumed
    SLURM ``MaxArraySize`` (``scontrol show config | grep MaxArraySize``).
    """
    _config, points = load_grid(outdir)
    n = len(points)
    done: set[int] = set()
    for p in glob.glob(os.path.join(outdir, "points", "point_*.npz")):
        stem = os.path.basename(p)[len("point_"):-len(".npz")]
        try:
            done.add(int(stem))
        except ValueError:
            pass
    missing = sorted(set(range(n)) - done)
    extra = sorted(i for i in done if i >= n)  # stray files from a stale/edited grid
    print(f"{n} grid points: {n - len(missing)} finished, {len(missing)} missing.")
    if extra:
        print(f"  note: {len(extra)} saved point file(s) have index >= {n} "
              f"(stale grid?): {_compress_ranges(extra)}")
    if not missing:
        print("nothing to resubmit.")
        return

    listpath = os.path.join(outdir, "missing.txt")
    with open(listpath, "w") as f:
        f.write("\n".join(str(i) for i in missing) + "\n")
    print(f"wrote {listpath}  ({len(missing)} indices)")
    print(f"  indices: {_compress_ranges(missing)}")

    m = len(missing)
    print("\nResubmit (robust for any indices -- array is 0..M-1, points read from the list):")
    if m <= max_array:
        print(f"  RESUME={listpath} OUTDIR={outdir} sbatch "
              f"--array=0-{m - 1} slurm/snail_sweep.slurm")
    else:
        chunk, ntasks = _chunking(m, max_array)
        print(f"  RESUME={listpath} OUTDIR={outdir} CHUNK={chunk} sbatch "
              f"--array=0-{ntasks - 1} slurm/snail_sweep.slurm")
        print(f"  ({ntasks} tasks x {chunk} points/task; raise #SBATCH --time to match.)")
    if missing[-1] < max_array:
        print("Or directly (only if the largest index is below MaxArraySize):")
        print(f"  OUTDIR={outdir} CHUNK=1 sbatch "
              f"--array={_compress_ranges(missing)} slurm/snail_sweep.slurm")
    print("Then re-run `collect` once the resubmitted tasks finish.")


__all__ = [
    'TWO_PI',
    '_DELTA_EPS_GHz',
    '_drag_skip_GHz',
    'DEFAULT_CONFIG',
    'DEFAULT_SPECFREQS_GHz',
    'DEFAULT_DRAGS',
    'DEFAULT_TARGET_DRAGS',
    'Point',
    'write_grid',
    'load_grid',
    '_stark_offset_GHz',
    '_nearest_collision',
    '_grape_augment',
    'save_point',
    'collect',
    'plot_chevrons',
    '_parse_list',
    '_bool_list',
    '_log_line',
    '_print_submit_hint',
    '_compress_ranges',
    'find_missing',
]
