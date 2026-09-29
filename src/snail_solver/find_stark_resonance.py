#!/usr/bin/env python3
"""
find_stark_resonance.py
=======================

Locate the AC-Stark-shifted iSWAP resonance, so the pump is driven at the frequency
that actually closes the swap rather than the bare w_p = |w_b - w_a|.

Through the SNAIL non-linearity the pump also Stark-shifts the qubits (and dresses
the coupler) by an amount ~ |eta|^2, moving the resonance to

    w_p^res = |w_b - w_a| + Delta_Stark(eta),   Delta_Stark = differential shift.

A pump at the bare frequency sits at a residual detuning delta, capping transfer at
g_eff^2 / (g_eff^2 + delta^2). This tool reproduces the hardware chevron: a
CONSTANT pump at the operating |eta|, swept in pump offset and time, recording the
|01> -> |10> exchange; the contrast-maximising offset is the resonance. Feed it back
as ``wp_offset_GHz``.

The constant pump is deliberate: on resonance the exchange reaches full contrast
whatever the amplitude calibration, so the vertex locates the frequency cleanly.
Anharmonicity and qutrit levels are included (they shift the resonance too).

Usage
-----
    python -m snail_solver.find_stark_resonance --device dev.json --target-eta 0.3 \
        --span-MHz 60 --points 41 --out stark.npz --plot stark.png
    python -m snail_solver.find_stark_resonance --device dev.json --t-g 200 --amp-scale 0.9 --jobs 16

`dev.json` is the run_sweep_zhou schema. The scan needs QuTiP; `locate_resonance`
is pure numpy.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from snail_solver.device_utils import check_drag_detuning, load_device, target_eta_area

TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Coupler at a fixed (constant) pump strength held over the whole window
# ---------------------------------------------------------------------------
def operating_eta(config: Dict[str, Any], t_g: float, amp_scale: float) -> float:
    """Constant-pump |eta| for a full iSWAP in t_g (ns), times amp_scale:

        eta = (pi/2) / (6 (2pi g3) la lb t_g) .

    This is half the raised-cosine peak for the same t_g, and a better proxy for
    that gate's pulse-averaged Stark shift (Hann <eta^2> = 0.375 eta_peak^2).
    """
    from snail_solver.device_utils import build_coupler
    sub = dict(config)
    sub["envelope"] = "constant"                 # Stark calibration is constant-pulse
    _cpl, _w_p, eta_peak = build_coupler(sub, t_g, amp_scale, 0.0)
    return float(eta_peak)


def build_chevron_coupler(config: Dict[str, Any], eta_op: float,
                          wp_offset_GHz: float, window_ns: float,
                          spec_abs_GHz: Optional[float] = None,
                          shape: str = "constant", t_g_ns: Optional[float] = None,
                          drag_beat_GHz: Optional[float] = None,
                          amp_scale: float = 1.0,
                          chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                          drag_n_pump: int = 1,
                          drag_channels=None):
    """(a, b, coupler[, spectator]) system driven by a probe pump at
    |w_b - w_a| + wp_offset_GHz. Returns ``(ZhouCoupler, w_p_GHz)``.

    * ``shape="constant"``: pump of |eta| = eta_op held over [0, window_ns].
      Amplitude-robust, but d eta/dt = 0 makes it blind to the DRAG-quadrature shift.
    * ``shape="gate"`` / ``"raised_cosine"``: the actual gate -- a full iSWAP over
      t_g_ns in ``config["envelope"]``, normalized on (a, b), scaled by amp_scale,
      with DRAG tuned to ``drag_beat_GHz`` (delta = Delta - w_p) if given, so the
      located resonance is the DRAG-on one.

    ``spec_abs_GHz`` adds a 4th spectator mode at that ABSOLUTE frequency
    (participation lam_b, ``spec_levels``, ``anharm_spec_GHz``).

    ``chirp_coeffs_GHz`` (gate shape only) must be the chirp the gate runs with, so
    the located offset is the RESIDUAL on top of it; omitting it double-counts c0
    (see :class:`envelope.Chirp`). A nonzero chirp on the constant probe raises
    ValueError: a chirp lives on u = 2t/t_g - 1, and there is no gate to normalize to.
    """
    from snail_solver.envelope import envelope_from_config
    from snail_solver.zhou_coupler import (ZhouCoupler, PumpTone, ConstantPulse,
                                           make_chirp)

    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    ws = float(config["coupler_freq_GHz"])
    w_p_GHz = abs(wb - wa) + wp_offset_GHz
    aq = float(config.get("anharm_qubit_GHz", 0.0))
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    freqs = [wa, wb, ws]
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    participations = {0: float(config["lam_a"]), 1: float(config["lam_b"])}
    anharm = {0: aq, 1: aq}
    if spec_abs_GHz is not None:                    # add the spectator as a 4th mode
        freqs.append(float(spec_abs_GHz))
        levels.append(int(config.get("spec_levels", 3)))
        participations[3] = float(config["lam_b"])           # spectator participation = lam_b
        anharm[3] = float(config.get("anharm_spec_GHz", 0.0))

    cpl = ZhouCoupler(mode_freqs_GHz=freqs, coupler_index=2,
                      participations=participations,
                      nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz=anharm)

    if shape in ("gate", "raised_cosine"):
        # Built as run_sweep_zhou.build_point does (normalize then scale). The
        # envelope comes from the config: on a sine_power device a Hann probe would
        # be a different pulse than the one being calibrated.
        t_g = float(t_g_ns if t_g_ns is not None else window_ns)
        env = envelope_from_config(config, t_g, amp=1.0)
        tone = PumpTone(w_p_GHz=w_p_GHz, envelope=env, is_eta=True,
                        drag=(drag_beat_GHz is not None),
                        delta_drag_GHz=drag_beat_GHz,
                        chirp=make_chirp(chirp_coeffs_GHz, t_g),
                        drag_n_pump=int(drag_n_pump),
                        drag_channels=(list(drag_channels) if drag_channels else None))
        if drag_beat_GHz is not None or drag_channels:
            check_drag_detuning(tone)      # chirp must not sweep the pump onto the beat
        cpl.set_pump(tone, normalize_iswap=(0, 1))
        cpl.scale_pump_amplitude(float(amp_scale))
    else:
        if chirp_coeffs_GHz is not None and np.any(np.asarray(chirp_coeffs_GHz,
                                                              dtype=float)):
            raise ValueError(
                "a chirp cannot be applied to the constant probe: Chirp is defined on "
                "the normalized gate time u = 2t/t_g - 1, and a constant pump held "
                "over `window_ns` has no gate to normalize against. Pass "
                "shape='raised_cosine' (with t_g_ns) to locate the resonance of a "
                "chirped gate.")
        # constant pump at fixed |eta| (is_eta=True, no normalization): peak_eta == eta_op
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=ConstantPulse(amp=eta_op, t_g=window_ns),
                              is_eta=True), normalize_iswap=None)
    return cpl, w_p_GHz


# ---------------------------------------------------------------------------
# Parallel chevron scan
# ---------------------------------------------------------------------------
def _resolve_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs and n_jobs > 0:
        return int(n_jobs)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


#: Population channels reported per time by :func:`population_channels`, in the
#: order they are stacked. Flat names (not a nested dict) so a scan result stays
#: np.savez-able -- see main().
CHANNELS: Tuple[str, ...] = ("P01", "P10", "P_leak", "P_f_a", "P_f_b",
                             "P_coupler", "P_double", "P_spectator", "norm_defect")


def population_channels(cpl, states: np.ndarray, init: Sequence[int],
                        tgt: Sequence[int]) -> np.ndarray:
    """Population per channel and output time, shape ``[len(CHANNELS), n_time]``.

    The chevron's two-level lineshape assumes population leaving `init` can only
    reach `tgt`; this reads what violates that from the existing states (no new solve).

    ``P_leak = norm - P01 - P10`` is the EXCLUSIVE complement. ``norm_defect =
    1 - norm`` is a truncation health check (~1e-9 when closed), not leakage.
    ``P_f_a``/``P_f_b`` (qutrit ``|2>``), ``P_coupler`` (any coupler excitation),
    ``P_double`` (``|11>``) and ``P_spectator`` (any 4th-mode excitation) are
    OVERLAPPING attributions, not a partition of ``P_leak``.
    """
    from snail_solver.spectroscopy import marginal_population

    probs = np.abs(np.asarray(states)) ** 2
    dims = cpl.dims
    n_time = probs.shape[0]

    norm = probs.sum(axis=1)
    P01 = probs[:, cpl.fock_index(list(init))]
    P10 = probs[:, cpl.fock_index(list(tgt))]
    P_leak = norm - P01 - P10

    def _level(mode: int, level: int) -> np.ndarray:
        if level >= dims[mode]:
            return np.zeros(n_time)
        return np.array([marginal_population(row, dims, mode, level) for row in probs])

    P_f_a = _level(0, 2)
    P_f_b = _level(1, 2)
    P_coupler = 1.0 - _level(cpl.coupler_index, 0)
    double_occ = [1, 1] + [0] * (cpl.n_modes - 2)
    P_double = probs[:, cpl.fock_index(double_occ)]
    P_spectator = 1.0 - _level(3, 0) if cpl.n_modes > 3 else np.zeros(n_time)
    norm_defect = 1.0 - norm

    return np.stack([P01, P10, P_leak, P_f_a, P_f_b, P_coupler, P_double,
                     P_spectator, norm_defect], axis=0)


def _chevron_worker(args: Tuple) -> np.ndarray:
    """One pump-offset column: the :func:`population_channels` stack over time,
    for any mode count (bare pair or + spectator) and probe shape (``build_kw``)."""
    config, eta_op, wp_offset, times, solver, spec_abs_GHz, build_kw = args
    cpl, _w_p = build_chevron_coupler(config, eta_op, wp_offset, float(times[-1]),
                                      spec_abs_GHz=spec_abs_GHz, **build_kw)
    init = [0] * cpl.n_modes; init[1] = 1          # |01...> : qubit b excited
    tgt = [0] * cpl.n_modes; tgt[0] = 1            # |10...> : qubit a excited
    states = cpl.evolve_trajectory(init, times, **solver)
    return population_channels(cpl, states, init, tgt)


def _parabolic_vertex(x: Sequence[float], y: Sequence[float]) -> float:
    """x of the maximum of sorted samples, refined by a 3-point parabola around the
    argmax (the grid point itself at the boundary)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    k = int(np.argmax(y))
    if k == 0 or k == len(x) - 1:
        return float(x[k])
    x0, x1, x2 = x[k - 1], x[k], x[k + 1]
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-15:
        return float(x1)
    # vertex of the parabola through the three points (uniform spacing not required)
    return float(x1 + 0.25 * (x2 - x0) * (y0 - y2) / denom)


def coupler_number_trace(cpl, states: np.ndarray) -> np.ndarray:
    """Coupler-mode photon number ``<n_s(t)>`` along a trajectory, shape (n_time,).

    ``eta`` here is a classical drive LABEL (``is_eta=True``), separate from the
    coupler's own Fock occupation; whether the two match McKinney et al.'s
    ``eta = sqrt(n_s)`` (arXiv:2409.18262, Eq. 9) is UNRESOLVED (see
    ``tune_up.verify_eta_matches_ns``). Read this as incidental coupler population,
    not a validated cross-check.
    """
    from snail_solver.spectroscopy import expected_number

    probs = np.abs(np.asarray(states)) ** 2
    return np.array([expected_number(row, cpl.dims, cpl.coupler_index) for row in probs])


def locate_resonance(offsets_GHz: np.ndarray, max_transfer: np.ndarray) -> float:
    """Resonance offset (GHz): the parabolically refined argmax of the chevron
    envelope `max_transfer` over `offsets_GHz`. Pure numpy."""
    return _parabolic_vertex(offsets_GHz, max_transfer)


def scan(config: Dict[str, Any], t_g: float, amp_scale: float,
         offsets_GHz: np.ndarray, window_ns: float, n_time: int,
         solver: Optional[Dict[str, Any]] = None,
         n_jobs: Optional[int] = None,
         spec_abs_GHz: Optional[float] = None,
         shape: str = "constant",
         drag_beat_GHz: Optional[float] = None,
         chirp_coeffs_GHz: Optional[Sequence[float]] = None,
         drag_n_pump: int = 1,
         drag_channels=None,
         eta_op: Optional[float] = None,
         keep_full_channels: bool = False) -> Dict[str, Any]:
    """Run the pump-frequency chevron and locate the Stark-shifted resonance.

    `offsets_GHz` are relative to |w_b - w_a|; `window_ns` ~2 t_g captures a full
    exchange. `n_jobs` 0/None -> SLURM_CPUS_PER_TASK or CPU count. `eta_op` drives
    the constant probe at that |eta| instead of :func:`operating_eta`'s (for drive
    sweeps; ignored by the gate shape). `spec_abs_GHz` / `chirp_coeffs_GHz` /
    `drag_beat_GHz`: see :func:`build_chevron_coupler`. `keep_full_channels` also
    returns the ``[n_off, n_time]`` ``P01``/``P_leak`` rasters (off so callers that
    keep whole dicts don't bloat).

    Returns offsets_GHz, times_ns, P10 [n_off, n_time], max_transfer [n_off],
    resonance_metric, metric_label, eta_op, w_p_bare_GHz, resonance_offset_GHz,
    resonance_w_p_GHz, shape, drag_beat_GHz, spec_abs_GHz, chirp_coeffs_GHz,
    metric_time_index [n_off], leak_at_metric / leak_max / leak_f_a / leak_f_b /
    leak_coupler / leak_double / leak_spectator [n_off], leak_on_resonance (at the
    offset nearest the resonance), norm_defect_max, and optionally P01 / P_leak.
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    eta_op = (float(eta_op) if eta_op is not None
              else operating_eta(config, t_g, amp_scale))
    times = np.linspace(0.0, window_ns, n_time)
    chirp_list = ([float(c) for c in chirp_coeffs_GHz]
                  if chirp_coeffs_GHz is not None else None)
    build_kw = {"shape": shape, "t_g_ns": float(t_g),
                "drag_beat_GHz": (float(drag_beat_GHz) if drag_beat_GHz is not None else None),
                "amp_scale": float(amp_scale),
                "chirp_coeffs_GHz": chirp_list,
                "drag_n_pump": int(drag_n_pump),
                # frozen dataclasses of scalars: picklable for the process pool
                "drag_channels": (list(drag_channels) if drag_channels else None)}
    args = [(config, eta_op, float(off), times, solver, spec_abs_GHz, build_kw)
            for off in offsets_GHz]

    jobs = _resolve_jobs(n_jobs)
    if jobs <= 1 or len(args) <= 1:
        cols = [_chevron_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            cols = list(pool.map(_chevron_worker, args))

    stack = np.array(cols)                      # [n_off, len(CHANNELS), n_time]
    P10 = stack[:, CHANNELS.index("P10"), :]    # [n_off, n_time]
    max_transfer = P10.max(axis=1)
    # Constant probe: max-over-time is the Rabi contrast. Shaped gate: use P(|10>)
    # AT t_g -- max-over-time would credit off-resonant offsets with mid-pulse
    # values, broadening the peak and pulling the resonance.
    if shape == "raised_cosine":
        k_tg = int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(t_g))))
        metric = P10[:, k_tg]
        metric_label = f"P(|10>) at t_g={float(t_g):.0f} ns"
        metric_time_index = np.full(P10.shape[0], k_tg, dtype=int)
    else:
        metric = max_transfer
        metric_label = "max-over-time P(|10>)"
        metric_time_index = np.argmax(P10, axis=1)
    offs = np.asarray(offsets_GHz, dtype=float)
    res_off = locate_resonance(offs, metric)
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    w_p_bare = abs(wb - wa)

    # Leakage summaries per offset (cheap: `stack` is already in hand).
    rows = np.arange(stack.shape[0])

    def _at_metric(channel: str) -> np.ndarray:
        return stack[rows, CHANNELS.index(channel), metric_time_index]

    leak_at_metric = _at_metric("P_leak")
    leak_max = stack[:, CHANNELS.index("P_leak"), :].max(axis=1)
    leak_f_a = _at_metric("P_f_a")
    leak_f_b = _at_metric("P_f_b")
    leak_coupler = _at_metric("P_coupler")
    leak_double = _at_metric("P_double")
    leak_spectator = _at_metric("P_spectator")
    norm_defect_max = float(stack[:, CHANNELS.index("norm_defect"), :].max())
    j_res = int(np.argmin(np.abs(offs - res_off)))
    leak_on_resonance = float(leak_at_metric[j_res])

    out = {"offsets_GHz": offs, "times_ns": times,
           "P10": P10, "max_transfer": max_transfer,
           "resonance_metric": metric, "metric_label": metric_label,
           "eta_op": float(eta_op),
           "w_p_bare_GHz": float(w_p_bare),
           "resonance_offset_GHz": float(res_off),
           "resonance_w_p_GHz": float(w_p_bare + res_off),
           "shape": shape,
           "drag_beat_GHz": (float(drag_beat_GHz) if drag_beat_GHz is not None else np.nan),
           "spec_abs_GHz": (float(spec_abs_GHz) if spec_abs_GHz is not None else np.nan),
           "chirp_coeffs_GHz": chirp_list,
           "metric_time_index": metric_time_index,
           "leak_at_metric": leak_at_metric, "leak_max": leak_max,
           "leak_f_a": leak_f_a, "leak_f_b": leak_f_b, "leak_coupler": leak_coupler,
           "leak_double": leak_double, "leak_spectator": leak_spectator,
           "leak_on_resonance": leak_on_resonance, "norm_defect_max": norm_defect_max}
    if keep_full_channels:
        out["P01"] = stack[:, CHANNELS.index("P01"), :]
        out["P_leak"] = stack[:, CHANNELS.index("P_leak"), :]
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def render_chevron(chev: Dict[str, Any], png_path: str, title_suffix: str = "") -> None:
    """Render one chevron (heatmap + resonance metric) to `png_path`.

    `chev` needs offsets_GHz, times_ns, P10 [n_off, n_time], max_transfer,
    resonance_offset_GHz, eta_op; shape / drag_beat_GHz / resonance_metric /
    metric_label are optional.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:
        pass

    off_MHz = np.asarray(chev["offsets_GHz"]) * 1e3
    res_MHz = float(chev["resonance_offset_GHz"]) * 1e3
    shape = str(chev.get("shape", "constant"))
    drag_beat = float(chev.get("drag_beat_GHz", np.nan))
    probe = ("raised-cosine" if shape == "raised_cosine" else "constant")
    if shape == "raised_cosine" and np.isfinite(drag_beat):
        probe += rf" + DRAG($\delta$={drag_beat*1e3:+.0f} MHz)"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    mesh = ax0.pcolormesh(off_MHz, np.asarray(chev["times_ns"]), np.asarray(chev["P10"]).T,
                          shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax0.axvline(res_MHz, color="r", ls="--", lw=1.6, label=f"resonance {res_MHz:+.1f} MHz")
    ax0.set_xlabel(r"pump offset from $|w_b-w_a|$ (MHz)"); ax0.set_ylabel("time (ns)")
    ax0.set_title(rf"chevron ({probe}): $P(|01\rangle\to|10\rangle)$")
    ax0.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(mesh, ax=ax0, label=r"$P(|10\rangle)$")

    ax1.plot(off_MHz, np.asarray(chev.get("resonance_metric", chev["max_transfer"])),
             "o-", ms=3)
    ax1.axvline(res_MHz, color="r", ls="--", lw=1.6)
    ax1.set_xlabel(r"pump offset from $|w_b-w_a|$ (MHz)")
    ax1.set_ylabel(str(chev.get("metric_label", "max-over-time $P(|10\\rangle)$")))
    ax1.set_title(rf"offset = {res_MHz:+.1f} MHz  ($|\eta|$={float(chev['eta_op']):.3f})  {title_suffix}")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chevron(npz_path: str, png_path: str) -> None:
    """Load a chevron .npz (from main()) and render it via render_chevron."""
    d = np.load(npz_path)
    chev = {"offsets_GHz": d["offsets_GHz"], "times_ns": d["times_ns"],
            "P10": d["P10"], "max_transfer": d["max_transfer"],
            "resonance_offset_GHz": float(d["resonance_offset_GHz"]),
            "eta_op": float(d["eta_op"]),
            "shape": str(d["shape"]) if "shape" in d.files else "constant",
            "drag_beat_GHz": float(d["drag_beat_GHz"]) if "drag_beat_GHz" in d.files else np.nan}
    if "resonance_metric" in d.files:
        chev["resonance_metric"] = d["resonance_metric"]
        chev["metric_label"] = str(d["metric_label"]) if "metric_label" in d.files else ""
    render_chevron(chev, png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Find the Stark-shifted iSWAP resonance and report the pump frequency to drive."""
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.find_stark_resonance", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON (run_sweep schema)")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="operating |eta| (sets t_g); overrides --t-g / device t_g_ns")
    ap.add_argument("--t-g", type=float, default=None, help="operating gate time (ns)")
    ap.add_argument("--amp-scale", type=float, default=1.0,
                    help="amplitude-scale correction from a prior amplitude calibration")
    ap.add_argument("--calibration", default=None,
                    help="calibration JSON to read amp_scale (and t_g) from")
    ap.add_argument("--span-MHz", type=float, default=60.0,
                    help="offset scan is +/- span/2 about |w_b-w_a|")
    ap.add_argument("--points", type=int, default=41, help="offset grid points")
    ap.add_argument("--window-factor", type=float, default=2.0,
                    help="time window = window_factor * t_g")
    ap.add_argument("--time-points", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = SLURM_CPUS_PER_TASK/CPU)")
    ap.add_argument("--gpu", action="store_true", help="run via qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--shape", choices=["constant", "raised_cosine"], default="constant",
                    help="probe pulse. 'constant' is amplitude-robust but has "
                         "d(eta)/dt = 0, so it is BLIND to DRAG and cannot carry a "
                         "chirp; use 'raised_cosine' (the actual gate pulse) to locate "
                         "the DRAG-on or chirped resonance")
    ap.add_argument("--spec-abs-GHz", type=float, default=None,
                    help="include a spectator at this ABSOLUTE frequency, so the "
                         "located resonance carries its dispersive pull")
    ap.add_argument("--drag-beat-GHz", type=float, default=None,
                    help="apply the DRAG quadrature at this beat (needs "
                         "--shape raised_cosine); the located offset is then the "
                         "DRAG-on resonance")
    ap.add_argument("--drag-n-pump", type=int, default=1,
                    help="pump quanta of the suppressed process; with a chirp the "
                         "beat moves as Delta(t) = beat - n*delta(t)")
    ap.add_argument("--chirp-GHz", default=None,
                    help="comma list of Legendre chirp coefficients (GHz) applied to "
                         "the probe, so the located offset is the RESIDUAL on top of "
                         "the chirp (needs --shape raised_cosine)")
    ap.add_argument("--out", default="stark.npz", help="output .npz")
    ap.add_argument("--plot", default=None, help="optional output PNG")
    ap.add_argument("--update-device", default=None,
                    help="write device JSON with wp_offset_GHz set to the resonance")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    args = ap.parse_args()

    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from snail_solver.paths import resolve_device, in_results
    args.out = in_results(args.out)
    if args.update_device:
        args.update_device = in_results(args.update_device)
    if args.plot:
        args.plot = in_results(args.plot)
    config = load_device(resolve_device(args.device))
    amp_scale = float(args.amp_scale)
    t_g = args.t_g
    if args.calibration:
        with open(args.calibration) as f:
            cal = json.load(f)
        amp_scale = float(cal.get("amp_scale", amp_scale))
        t_g = cal.get("t_g_ns", t_g)
    if args.target_eta is not None:
        # constant-pulse full iSWAP: area = eta * t_g = target_eta_area  ->  t_g = area/eta
        t_g = target_eta_area(float(config["g3_GHz"]), float(config["lam_a"]),
                              float(config["lam_b"])) / float(args.target_eta)
    if t_g is None:
        t_g = float(config.get("t_g_ns", 200.0))
    t_g = float(t_g)

    span = args.span_MHz / 1000.0
    offsets = np.linspace(-span / 2, span / 2, args.points)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    window = args.window_factor * t_g

    print(f"device={args.device}  t_g={t_g:.1f} ns  amp_scale={amp_scale}  "
          f"jobs={_resolve_jobs(args.jobs)}{' GPU' if args.gpu else ''}")
    from snail_solver.device_utils import parse_chirp_arg
    result = scan(config, t_g, amp_scale, offsets, window, args.time_points,
                  solver, n_jobs=args.jobs, shape=args.shape,
                  spec_abs_GHz=args.spec_abs_GHz,
                  drag_beat_GHz=args.drag_beat_GHz,
                  drag_n_pump=args.drag_n_pump,
                  chirp_coeffs_GHz=parse_chirp_arg(args.chirp_GHz))

    np.savez(args.out, t_g_ns=t_g, amp_scale=amp_scale, **result)
    print(f"operating |eta|        = {result['eta_op']:.4f}")
    print(f"bare w_p = |w_b-w_a|   = {result['w_p_bare_GHz']:.6f} GHz")
    print(f"Stark resonance offset = {result['resonance_offset_GHz']*1e3:+.2f} MHz")
    print(f"=> drive the pump at w_p = {result['resonance_w_p_GHz']:.6f} GHz "
          f"(set wp_offset_GHz = {result['resonance_offset_GHz']:.6f})")
    print(f"peak contrast at resonance ~ {result['max_transfer'].max():.4f}")
    print(f"written {args.out}")

    if args.update_device:
        dev = dict(config)
        dev["wp_offset_GHz"] = float(result["resonance_offset_GHz"])
        with open(args.update_device, "w") as f:
            json.dump(dev, f, indent=2)
        print(f"wrote {args.update_device} with wp_offset_GHz={dev['wp_offset_GHz']:.6f}")

    if args.plot:
        plot_chevron(args.out, args.plot)
        print(f"plotted {args.plot}")


if __name__ == "__main__":
    main()