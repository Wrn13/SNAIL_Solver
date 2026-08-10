"""2D calibration landscape: pump frequency offset vs pump strength.

Scans the (pump-frequency-offset, pump-amplitude) plane and scores each point by
the leakage-aware iSWAP fidelity or the |01>->|10> transfer probability (the swap
population), so the global optimum is read off directly instead of from
alternating 1D amplitude/frequency scans (which rail when the AC-Stark shift and
the Rabi amplitude feed back on each other).

Three propagation engines, identical grid and output shape:
  * engine='reduced' : the fast rotating-frame reduced model from ``grape.py``
    (scipy expm, no QuTiP); the pump offset shifts precomputed carriers so the
    operator basis is built once. Good for quick looks / weak-to-moderate drive.
  * engine='qutip'   : compiled ``qt.sesolve`` on the FULL Hamiltonian via
    ``ZhouCoupler.propagator_columns`` -- exact, the cluster path. A fresh coupler
    is built per grid point through the usual ``build_coupler`` plumbing
    (``build_system``), so the grid is embarrassingly parallel (``--jobs``) and
    free of pump-mutation hazards, and DRAG / spectator context come along for
    free exactly as in the reduced map.
  * engine='jax'     : the batched engine in ``jax_engine.py``. The whole grid is
    ONE vmapped program -- the device is fixed, so every point shares a single
    sparse operator stack while the amplitude rides in the pulse parameters and
    the pump offset is just the last entry of the frequency vector. All four
    computational columns propagate together as one (dim, 4) block, and it runs
    unchanged on GPU. It is a rotating-wave REDUCTION at any finite
    ``--engine-cutoff-GHz``: run ``validate_engines.py --cutoff-scan`` on the
    device first, which reports fidelity and ``max|U - U_qutip|`` per cutoff.

Results (offsets, amps, Z, leakage, optimum, operating point) are written to
``--save-npz``; the heatmap with the optimum marked goes to ``--out``. At strong
drive, raise the coupler truncation (``--coupler-levels``) and confirm the map is
converged before trusting values in the bright/leaky region.

CLI
---
    python -m snail_solver.calibration_map --device evan_device.json --t-g-ns 92.6 \
        --engine qutip --metric transfer --jobs 32 \
        --wp-span-MHz 40 --amp-lo 0.6 --amp-hi 1.6 \
        --save-npz figs/swap_map.npz --out figs/swap_map.png

    # batched engine (GPU-capable); validate the cutoff first
    python -m snail_solver.validate_engines --device evan_device.json --cutoff-scan
    python -m snail_solver.calibration_map --device evan_device.json --t-g-ns 92.6 \
        --engine jax --engine-cutoff-GHz 3.0 --engine-batch 128 \
        --out figs/swap_map.png
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from snail_solver import grape
from snail_solver.log_utils import setup_run_logger

TWO_PI = 2.0 * np.pi


def scan(cpl, a: int, b: int, t_g: float, *,
         wp_span_MHz: float = 40.0, wp_points: int = 41,
         amp_lo: float = 0.6, amp_hi: float = 1.4, amp_points: int = 41,
         cutoff_GHz: float = 1.0, n_ctrl: int = 32, metric: str = "fidelity",
         carrier_resolution: float = 0.3,
         logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Scan (pump offset, amplitude) and score each point with the REDUCED model.

    Parameters
    ----------
    cpl : ZhouCoupler
        Coupler with a pump set; ``peak_eta`` sets the amp=1 reference.
    a, b : int
        Target-qubit mode indices.
    t_g : float
        Gate duration (ns).
    wp_span_MHz : float
        Full width of the pump-offset axis (centered on 0), in MHz.
    wp_points, amp_points : int
        Grid resolution.
    amp_lo, amp_hi : float
        Amplitude-scale range (multiples of the nominal peak_eta).
    cutoff_GHz : float
        Rotating-frame carrier cutoff for the reduced model.
    n_ctrl : int
        Piecewise-constant slices approximating the raised cosine.
    metric : {'fidelity', 'transfer'}
        Score per point: leakage-aware iSWAP fidelity, or P(|01>->|10>).
    logger : logging.Logger, optional
        If given, one line per completed row is logged (in addition to the
        interactive ``tqdm`` bar, which isn't a good fit for a plain log file).

    Returns
    -------
    dict
        offsets_MHz, amps, Z (shape [amp_points, wp_points]), best (dict),
        metric, plus scan metadata.
    """
    from snail_solver.zhou_coupler import ZhouCoupler
    terms, H_anh, idx, max_Omega = grape._prepare(cpl, a, b, cutoff_GHz)
    dt_ctrl = t_g / n_ctrl
    # The chirp rides along fixed while (offset, amplitude) are scanned -- read it off
    # the coupler so it cannot be forgotten, and let `_propagate` apply it on the fine
    # grid. `base` below stays the UN-chirped raised cosine: the chirp is a carrier
    # rotation, not a feature of the pulse shape.
    chirp = grape._tone_chirp(cpl)
    n_sub = max(1, int(np.ceil((max_Omega + abs(wp_span_MHz) * 1e-3 * TWO_PI
                                + grape._chirp_pad_rad(chirp, terms))
                               * dt_ctrl / carrier_resolution)))

    peak = float(cpl.peak_eta())
    ts = (np.arange(n_ctrl) + 0.5) * dt_ctrl
    base = peak * 0.5 * (1.0 - np.cos(2.0 * np.pi * ts / t_g))    # raised cosine

    offsets = np.linspace(-wp_span_MHz / 2.0, wp_span_MHz / 2.0, wp_points)  # MHz
    amps = np.linspace(amp_lo, amp_hi, amp_points)
    Z = np.full((amp_points, wp_points), np.nan)

    # |01> is column 1 of the propagator basis (|00>,|01>,|10>,|11>); |10> is row 2
    for i, amp in enumerate(tqdm(amps, desc="calibration scan (reduced)", unit="row")):
        eta_ctrl = (amp * base).astype(complex)
        for j, off_MHz in enumerate(offsets):
            U = grape._propagate(eta_ctrl, t_g, terms, H_anh, idx, n_sub,
                                 offset_rad=off_MHz * 1e-3 * TWO_PI,
                                 chirp=chirp)
            if metric == "transfer":
                Z[i, j] = abs(U[2, 1]) ** 2                        # |01> -> |10>
            else:
                F, _ = ZhouCoupler._iswap_fidelity_from_U(U, True)
                Z[i, j] = F
        if logger:
            logger.info(f"row {i + 1}/{len(amps)} amp_scale={amp:.3f}")

    bi, bj = np.unravel_index(np.nanargmax(Z), Z.shape)
    best = dict(amp_scale=float(amps[bi]), wp_offset_MHz=float(offsets[bj]),
                score=float(Z[bi, bj]))
    return dict(offsets_MHz=offsets, amps=amps, Z=Z, best=best, metric=metric,
                n_sub=n_sub, cutoff_GHz=cutoff_GHz, peak_eta=peak, engine="reduced",
                chirp_coeffs_GHz=(None if chirp is None
                                  else list(map(float, chirp.coeffs_GHz))))


def scan_qutip(build_fn: Callable[[float, float], Tuple[Any, float, float]],
               t_g: float, *, wp_span_MHz: float = 40.0, wp_points: int = 41,
               amp_lo: float = 0.6, amp_hi: float = 1.4, amp_points: int = 41,
               metric: str = "transfer", atol: float = 1e-10, rtol: float = 1e-8,
               nsteps: int = 500000, jobs: int = 1,
               logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """QuTiP-exact analogue of ``scan``: identical grid and return shape, but every
    point is a compiled ``qt.sesolve`` on the FULL Hamiltonian via
    ``ZhouCoupler.propagator_columns`` -- no reduced model, no pruned terms.

    ``build_fn(grid_amp, grid_off_MHz) -> (cpl, w_p, eta_pk)`` returns a FRESH
    coupler with the grid point already folded into the pump (via the usual
    ``build_system`` / ``device_utils.build_coupler`` plumbing), so points are
    independent and safe to evaluate in parallel. Leakage is recorded for free
    from the same 4x4 U. ``metric`` selects the colour ('transfer' = swap
    population |U[2,1]|^2, or leakage-aware 'fidelity').
    """
    from snail_solver.zhou_coupler import ZhouCoupler
    offsets = np.linspace(-wp_span_MHz / 2.0, wp_span_MHz / 2.0, wp_points)  # MHz
    amps = np.linspace(amp_lo, amp_hi, amp_points)
    _, _, eta_pk = build_fn(1.0, 0.0)                      # nominal amp=1 reference

    def one(i: int, j: int, amp: float, off_MHz: float):
        cpl, _w_p, _eta = build_fn(amp, off_MHz)
        U = cpl.propagator_columns(0, 1, t_g, atol=atol, rtol=rtol, nsteps=nsteps)
        fid, leak = ZhouCoupler._iswap_fidelity_from_U(U, True)
        score = abs(U[2, 1]) ** 2 if metric == "transfer" else fid   # swap pop / F
        return i, j, float(score), float(leak)

    grid = [(i, j, a, o) for i, a in enumerate(amps) for j, o in enumerate(offsets)]
    t0 = time.time()
    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        # return_as="generator_unordered" streams results back as they finish, so
        # progress can be logged as it happens instead of only after the whole
        # grid returns (joblib's own verbose=5 prints straight to stdout and
        # can't be redirected into `logger`'s file).
        results = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(one)(i, j, a, o) for (i, j, a, o) in grid)
        out = []
        for k, r in enumerate(results, start=1):
            out.append(r)
            if logger and (k % wp_points == 0 or k == len(grid)):
                logger.info(f"  {k}/{len(grid)} points done "
                            f"({100 * k / len(grid):.0f}%), elapsed={time.time() - t0:.0f}s")
    else:
        out = []
        for k, (i, j, a, o) in enumerate(grid):
            out.append(one(i, j, a, o))
            if (k + 1) % wp_points == 0:
                msg = f"  row {k // wp_points + 1}/{amp_points} (amp_scale={a:.3f})"
                (logger.info if logger else print)(msg)

    Z = np.full((amp_points, wp_points), np.nan)
    L = np.full((amp_points, wp_points), np.nan)
    for i, j, s, l in out:
        Z[i, j], L[i, j] = s, l

    bi, bj = np.unravel_index(np.nanargmax(Z), Z.shape)
    best = dict(amp_scale=float(amps[bi]), wp_offset_MHz=float(offsets[bj]),
                score=float(Z[bi, bj]), leakage=float(L[bi, bj]))
    return dict(offsets_MHz=offsets, amps=amps, Z=Z, leakage=L, best=best,
                metric=metric, peak_eta=float(eta_pk), engine="qutip")


def plot_map(result: Dict[str, Any], out: str = "figs/calibration_map.png",
             title: Optional[str] = None) -> None:
    r"""Render the 2D calibration landscape with the optimum marked.

    The y axis is the PHYSICAL drive, peak :math:`|\eta|`, with ``amp_scale`` on a
    twin axis on the right. The two differ only by the constant
    :math:`\eta_{\rm nom} = 1/(12 g_3 \lambda_a \lambda_b t_g)`, but they answer
    different questions: :math:`|\eta|` says which drive REGIME the point is in (how
    large the eta^2 / eta^3 terms are), and is therefore comparable between maps at
    different ``t_g``, whereas ``amp_scale`` is what ``--save-point`` writes into the
    device and is only meaningful at one ``t_g``. The dotted line marks
    ``amp_scale = 1``, i.e. the amplitude at which the LEADING-ORDER pulse area is
    exactly pi/2 -- so the distance of the optimum from that line is a direct read
    on how far first-order theory is off at this operating point.
    """
    offsets, amps, Z = result["offsets_MHz"], result["amps"], result["Z"]
    metric = result["metric"]
    best = result["best"]
    eta_nom = float(result["peak_eta"])              # peak |eta| at amp_scale = 1
    eta_axis = amps * eta_nom
    eta_best = best["amp_scale"] * eta_nom
    zlabel = ("iSWAP fidelity $F$" if metric == "fidelity"
              else r"transfer $P(|01\rangle\!\to\!|10\rangle)$")

    fig, ax = plt.subplots(figsize=(7.6, 5.3), dpi=200)
    pcm = ax.pcolormesh(offsets, eta_axis, Z, shading="nearest", cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax, pad=0.12)
    cb.set_label(zlabel)
    # the pi/2 (leading-order) amplitude
    if eta_axis.min() <= eta_nom <= eta_axis.max():
        ax.axhline(eta_nom, color="w", ls=":", lw=1.1, alpha=0.85)
        ax.text(offsets[0], eta_nom, r"  $\pi/2$ normalisation (amp scale 1)",
                color="w", va="bottom", ha="left", fontsize=7.5, alpha=0.9)
    # optimum
    ax.plot(best["wp_offset_MHz"], eta_best, marker="*", ms=18,
            mfc="#C0392B", mec="white", mew=1.2, zorder=5)
    lead = zlabel.split("$")[0].strip() or metric
    ax.annotate(f"opt: {best['wp_offset_MHz']:+.1f} MHz\n"
                f"$|\\eta|$ = {eta_best:.3f}  (amp scale {best['amp_scale']:.3f})\n"
                f"{lead} = {best['score']:.4f}",
                xy=(best["wp_offset_MHz"], eta_best),
                xytext=(0.98, 0.02), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color="white",
                bbox=dict(boxstyle="round", fc="#00000099", ec="none"))
    ax.set_xlabel(r"pump frequency offset  $\delta\omega_p$  (MHz)")
    ax.set_ylabel(r"pump strength   peak $|\eta|$")
    # twin axis: the amp_scale that gets written to the device
    sec = ax.secondary_yaxis("right",
                             functions=(lambda e: e / eta_nom, lambda a: a * eta_nom))
    sec.set_ylabel(r"amp scale  ($\times\,\eta_{\rm nom}$, written to the device)")
    eng = result.get("engine", "reduced")
    t_g = result.get("t_g_ns")
    sub = f"({metric}, {eng}" + (f", $t_g$={t_g:.1f} ns, "
                                 f"$\\eta_{{\\rm nom}}$={eta_nom:.3f})" if t_g else ")")
    # only annotate a NON-trivial chirp, so un-chirped figures are unchanged
    chirp = (result.get("context") or {}).get("chirp_coeffs_GHz")
    if chirp is not None and np.any(np.asarray(chirp, dtype=float)):
        sub += "\nchirp $\\delta(t)/2\\pi$ = " + \
            "[" + ", ".join(f"{c:g}" for c in chirp) + "] GHz"
    ax.set_title(title or f"iSWAP calibration landscape  {sub}", fontsize=10.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out, "and", out.rsplit(".", 1)[0] + ".pdf")


def save_npz(result: Dict[str, Any], path: str) -> None:
    """Persist the full scan (arrays + optimum + context + operating point) so the
    map can be re-plotted or post-processed without re-solving."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = dict(
        offsets_MHz=result["offsets_MHz"], amps=result["amps"], Z=result["Z"],
        peak_eta_axis=result["amps"] * float(result["peak_eta"]),
        metric=str(result["metric"]), peak_eta=float(result["peak_eta"]),
        engine=str(result.get("engine", "reduced")))
    if "leakage" in result:
        payload["leakage"] = result["leakage"]
    # the Stark ridge and the chirp seed derived from it (see stark_chirp)
    for key in ("ridge_MHz", "stark_slope_MHz_per_amp2", "stark_fit_r2",
                "delta_stark_GHz", "chirp_seed_GHz"):
        if key in result:
            payload[key] = np.asarray(result[key], dtype=float)
    if "w_p_GHz" in result:
        payload["w_p_GHz"] = float(result["w_p_GHz"])
    if "t_g_ns" in result:
        payload["t_g_ns"] = float(result["t_g_ns"])
        payload["best_peak_eta"] = float(result["best"]["amp_scale"]
                                         * result["peak_eta"])
    for k, v in result["best"].items():
        payload[f"best_{k}"] = v
    for group in ("operating_point", "context"):
        if group in result:
            for k, v in result[group].items():
                if k == "chirp_coeffs_GHz":
                    # list-valued: store as a float array (empty when unset) rather
                    # than letting an empty/None value become a nan scalar, so a
                    # reader can always np.asarray it
                    payload[f"{group[:3]}_{k}"] = np.asarray(v or [], dtype=float)
                else:
                    payload[f"{group[:3]}_{k}"] = (np.nan if v is None else v)
    np.savez_compressed(path, **payload)
    print("saved", path)


def build_system(config: Dict[str, Any], t_g: float, *,
                 wa_GHz: Optional[float] = None, wb_GHz: Optional[float] = None,
                 delta_GHz: Optional[float] = None,
                 spec_abs_GHz: Optional[float] = None,
                 drag_beat_GHz: Optional[float] = None,
                 chirp_coeffs_GHz: Optional[Any] = None, drag_n_pump: int = 1,
                 amp_scale: float = 1.0, wp_offset_GHz: float = 0.0):
    """Build the gate system for ANY sweep context (bare / spectator / target).

    The three sweep families differ only in which frequencies move, so they all
    reduce to a choice of (w_a, w_b, spectator):

    * bare / no-spectator : leave ``spec_abs_GHz`` and ``delta_GHz`` unset.
    * target sweep        : pass the swept ``wb_GHz`` (and ``spec_abs_GHz`` for the
                            absolute spectator placement).
    * spectator sweep     : pass ``delta_GHz`` = w_b - w_spec; the absolute
                            spectator frequency is derived as w_b - delta.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    wa_GHz, wb_GHz : float, optional
        Override the device pair (target sweeps vary w_b).
    delta_GHz : float, optional
        Spectator-sweep detuning w_b - w_spec (GHz). Mutually exclusive with
        ``spec_abs_GHz``.
    spec_abs_GHz : float, optional
        Absolute spectator frequency (GHz).
    drag_beat_GHz : float, optional
        Apply a DRAG quadrature at this beat, so the map is of the DRAG-on gate.
    chirp_coeffs_GHz : sequence of float, optional
        Legendre coefficients of a pump chirp delta(t) (GHz) held FIXED across the
        grid, so the map is of the chirped gate. This is context, exactly like
        `drag_beat_GHz` -- the scan axes remain (pump offset, amplitude). Defaults to
        ``config["chirp_coeffs_GHz"]``. See :class:`envelope.Chirp`.
    amp_scale, wp_offset_GHz : float
        Nominal pump scaling/offset the scan grid is applied on top of.

    Returns
    -------
    (ZhouCoupler, float, float, dict)
        Coupler, pump frequency (GHz), peak |eta|, and the resolved context.
    """
    from snail_solver.device_utils import build_coupler

    pair = list(np.asarray(config["qubit_freqs_GHz"], dtype=float))
    if wa_GHz is not None:
        pair[0] = float(wa_GHz)
    if wb_GHz is not None:
        pair[1] = float(wb_GHz)
    if delta_GHz is not None:
        if spec_abs_GHz is not None:
            raise SystemExit("give either --delta-GHz or --spec-abs-GHz, not both")
        spec_abs_GHz = pair[1] - float(delta_GHz)      # spectator-sweep convention

    cfg = dict(config)
    cfg["qubit_freqs_GHz"] = pair
    cpl, w_p, eta_pk = build_coupler(cfg, t_g=t_g, amp_scale=amp_scale,
                                     wp_offset_GHz=wp_offset_GHz,
                                     spec_abs_GHz=spec_abs_GHz,
                                     drag_beat_GHz=drag_beat_GHz,
                                     chirp_coeffs_GHz=chirp_coeffs_GHz,
                                     drag_n_pump=drag_n_pump)
    # Resolve the chirp from the coupler rather than the argument, so the recorded
    # context is what was actually BUILT (build_coupler falls back to the config when
    # the argument is None). This one key then flows automatically into `run`'s
    # operating_point via **context, into save_npz, and into --save-point.
    tone_chirp = grape._tone_chirp(cpl)
    context = dict(wa_GHz=pair[0], wb_GHz=pair[1], t_g_ns=t_g,
                   spec_abs_GHz=(None if spec_abs_GHz is None else float(spec_abs_GHz)),
                   drag_beat_GHz=drag_beat_GHz,
                   chirp_coeffs_GHz=(None if tone_chirp is None
                                     else [float(c) for c in tone_chirp.coeffs_GHz]))
    return cpl, w_p, eta_pk, context


def run(config: Dict[str, Any], t_g: float, *, amp_scale: float = 1.0,
        wp_offset_GHz: float = 0.0, spec_abs_GHz: Optional[float] = None,
        wa_GHz: Optional[float] = None, wb_GHz: Optional[float] = None,
        delta_GHz: Optional[float] = None, drag_beat_GHz: Optional[float] = None,
        chirp_coeffs_GHz: Optional[Any] = None, drag_n_pump: int = 1,
        engine: str = "reduced", coupler_levels: Optional[int] = None,
        atol: float = 1e-10, rtol: float = 1e-8, nsteps: int = 500000,
        engine_cutoff_GHz: float = float("inf"), engine_carrier_resolution: float = 0.1,
        engine_batch: int = 64, engine_precision: str = "f64",
        jobs: int = 1, save_npz_path: Optional[str] = None,
        out: Optional[str] = "figs/calibration_map.png", title: Optional[str] = None,
        log_path: Optional[str] = None,
        **kw) -> Dict[str, Any]:
    """Build the system for the given sweep context, scan (reduced or QuTiP), plot,
    optionally persist, and return results.

    The returned dict includes ``context`` (where it was calibrated) and
    ``operating_point`` (a record ready for ``operating_points.save_point``).
    ``engine='qutip'`` runs the exact solver; both engines share the grid, the
    ``best``/``operating_point`` bookkeeping, and the plot.

    Progress is logged (timestamped) to stdout so a long ``engine='qutip'`` run
    can be tailed while it's still going -- under SLURM that lands in the job's
    own ``slurm-<jobid>.out``, so concurrent jobs (e.g. one per device) don't
    interleave into one shared file the way a colocated log file would. Pass
    ``log_path`` explicitly to also mirror progress into a file.
    """
    if coupler_levels is not None:                     # truncation override, reused
        config = {**config, "coupler_levels": int(coupler_levels)}

    logger = setup_run_logger(log_path, f"calibration_map:{log_path or 'stdout'}")
    logger.info(f"start: engine={engine} metric={kw.get('metric', 'fidelity')} "
                f"t_g={t_g}ns grid={kw.get('wp_points', 41)}x{kw.get('amp_points', 41)} "
                f"jobs={jobs}")
    t0 = time.time()

    cpl, w_p, eta_pk, context = build_system(
        config, t_g, wa_GHz=wa_GHz, wb_GHz=wb_GHz, delta_GHz=delta_GHz,
        spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
        chirp_coeffs_GHz=chirp_coeffs_GHz, drag_n_pump=drag_n_pump,
        amp_scale=amp_scale, wp_offset_GHz=wp_offset_GHz)
    logger.info(f"  chirp: {context['chirp_coeffs_GHz'] or 'none'} (held fixed "
                f"across the grid)")

    if engine == "qutip":
        # per-point rebuild through the SAME plumbing, grid folded onto the nominal
        def build_fn(grid_amp: float, grid_off_MHz: float):
            return build_system(
                config, t_g, wa_GHz=wa_GHz, wb_GHz=wb_GHz, delta_GHz=delta_GHz,
                spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
                chirp_coeffs_GHz=chirp_coeffs_GHz, drag_n_pump=drag_n_pump,
                amp_scale=amp_scale * grid_amp,
                wp_offset_GHz=wp_offset_GHz + grid_off_MHz * 1e-3)[:3]
        qkeys = ("wp_span_MHz", "wp_points", "amp_lo", "amp_hi", "amp_points", "metric")
        qkw = {k: v for k, v in kw.items() if k in qkeys}
        result = scan_qutip(build_fn, t_g, atol=atol, rtol=rtol, nsteps=nsteps,
                            jobs=jobs, logger=logger, **qkw)
    elif engine == "jax":
        # The whole grid is ONE batched program: the device is fixed, the amplitude
        # is a pulse parameter and the pump offset is just the last entry of the
        # frequency vector, so nothing has to be rebuilt per point.
        from snail_solver import jax_engine as JE
        span = float(kw.get("wp_span_MHz", 40.0))
        offsets = np.linspace(-span / 2.0, span / 2.0, int(kw.get("wp_points", 41)))
        amps = np.linspace(float(kw.get("amp_lo", 0.6)), float(kw.get("amp_hi", 1.4)),
                           int(kw.get("amp_points", 41)))
        logger.info(f"  jax engine: cutoff={engine_cutoff_GHz} GHz, "
                    f"carrier_resolution={engine_carrier_resolution}, batch={engine_batch}, "
                    f"precision={engine_precision}")
        result = JE.scan_amp_offset(cpl, t_g, amps, offsets,
                                    cutoff_GHz=engine_cutoff_GHz,
                                    carrier_resolution=engine_carrier_resolution,
                                    precision=engine_precision, batch=engine_batch,
                                    metric=str(kw.get("metric", "fidelity")))
        result["peak_eta"] = float(eta_pk)
        logger.info(f"  jax engine kept {result['n_terms']} terms "
                    f"(dropped {result['n_dropped']} above the cutoff)")
    else:
        result = scan(cpl, 0, 1, t_g, logger=logger, **kw)

    result["w_p_GHz"] = w_p
    result["t_g_ns"] = t_g
    result["context"] = context
    result["log_path"] = log_path

    # The per-row argmax over the offset axis IS the Stark shift vs drive -- the map
    # already computes it and used to keep only the single global optimum. Recovering
    # the ridge is pure post-processing on Z, and it is what a physically-motivated
    # chirp seed is built from (see stark_chirp).
    try:
        from snail_solver import stark_chirp as SC
        fit = SC.stark_slope_from_map(result)
        result["ridge_MHz"] = fit["ridge_MHz"]
        result["stark_slope_MHz_per_amp2"] = fit["slope_MHz_per_amp2"]
        result["stark_fit_r2"] = fit["r2"]
        result["delta_stark_GHz"] = fit["delta_stark_GHz"]
        result["chirp_seed_GHz"] = list(map(
            float, SC.stark_chirp_seed(fit["delta_stark_GHz"], degree=4)))
        logger.info(f"  stark ridge: {fit['slope_MHz_per_amp2']:+.3f} MHz/amp^2 "
                    f"(r2={fit['r2']:.4f}), delta_stark={fit['delta_stark_MHz']:+.3f} MHz "
                    f"at amp={fit['amp_scale']:.3f}")
        logger.info(f"  suggested chirp seed (degree 4): "
                    f"{[round(c, 6) for c in result['chirp_seed_GHz']]} GHz")
    except Exception as exc:                    # a ridge fit must never kill a map
        logger.info(f"  stark ridge fit skipped: {exc}")

    best = result["best"]
    # the scan grid is relative to the nominal (amp_scale, wp_offset) it was built on
    result["operating_point"] = dict(
        amp_scale=amp_scale * best["amp_scale"],
        wp_offset_GHz=wp_offset_GHz + best["wp_offset_MHz"] * 1e-3,
        metric=result["metric"], score=best["score"], **context)
    logger.info(f"done in {time.time() - t0:.0f}s: wp_offset={best['wp_offset_MHz']:+.2f}MHz "
                f"amp_scale={best['amp_scale']:.4f} score={best['score']:.5f}")
    if out:
        plot_map(result, out=out, title=title)
    if save_npz_path:
        save_npz(result, save_npz_path)
    return result


def scan_chirp_axis(config: Dict[str, Any], t_g: float, coeff_index: int,
                    values: np.ndarray, *, base_chirp: Optional[Any] = None,
                    out: Optional[str] = None, **run_kw) -> Dict[str, Any]:
    """Repeat the 2-D (offset, amplitude) map across one chirp coefficient.

    A landscape view of the third axis: for each value of ``c_[coeff_index]`` the full
    map is recomputed and its optimum recorded, so the (offset, amplitude) tune-up is
    re-done at every chirp rather than held fixed at a value calibrated without one.

    Implemented as a LOOP over ``run`` rather than a third batch axis. That keeps it
    engine-agnostic -- it works with reduced, qutip and jax identically -- and the
    honest cost is simply ``len(values)`` maps. Use the jax engine and a coarse grid
    for exploration; this is a diagnostic, not the recommended way to calibrate a
    chirp. For that, optimize it directly: ``grape --chirp-degree``, which searches
    the coefficients continuously instead of on a grid.

    Only EVEN coefficients are worth scanning: the Stark shape |eta(t)|^2 is even in
    the normalized gate time, so ``c_2`` is the leading useful term (``c_1`` is odd,
    and ``c_0`` is degenerate with the pump offset the map already scans). See
    ``stark_chirp``.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    coeff_index : int
        Which Legendre coefficient to scan (2 or 4 in practice).
    values : ndarray
        Values of that coefficient (GHz).
    base_chirp : sequence of float, optional
        Chirp the scanned coefficient is varied on top of.
    out : str, optional
        Path for the summary figure (score and optimum vs chirp).
    **run_kw
        Forwarded to :func:`run` (engine, grid, metric, ...). ``out``/``save_npz_path``
        of the inner maps are suppressed so the loop does not emit one figure per
        value.

    Returns
    -------
    dict
        ``values``, ``scores``, ``amp_scales``, ``wp_offsets_MHz`` (the per-value
        optimum), ``best`` (the overall winner) and ``results`` (every inner map).
    """
    values = np.asarray(values, dtype=float)
    base = list(base_chirp or [])
    n = max(len(base), coeff_index + 1)
    base = (base + [0.0] * n)[:n]

    run_kw = dict(run_kw)
    run_kw.pop("out", None)
    run_kw.pop("save_npz_path", None)

    results, scores, amps_, offs_ = [], [], [], []
    for v in values:
        coeffs = list(base)
        coeffs[coeff_index] = float(v)
        res = run(config, t_g, chirp_coeffs_GHz=coeffs, out=None,
                  save_npz_path=None, **run_kw)
        results.append(res)
        scores.append(res["best"]["score"])
        amps_.append(res["best"]["amp_scale"])
        offs_.append(res["best"]["wp_offset_MHz"])
        print(f"  c{coeff_index} = {v:+.5f} GHz -> {res['metric']} = "
              f"{res['best']['score']:.5f} at amp={res['best']['amp_scale']:.4f}, "
              f"offset={res['best']['wp_offset_MHz']:+.2f} MHz")

    scores = np.asarray(scores, dtype=float)
    k = int(np.nanargmax(scores))
    best_coeffs = list(base)
    best_coeffs[coeff_index] = float(values[k])
    best = dict(coeff_index=coeff_index, value=float(values[k]),
                score=float(scores[k]), amp_scale=float(amps_[k]),
                wp_offset_MHz=float(offs_[k]), chirp_coeffs_GHz=best_coeffs)
    outd = dict(values=values, scores=scores,
                amp_scales=np.asarray(amps_, dtype=float),
                wp_offsets_MHz=np.asarray(offs_, dtype=float),
                coeff_index=coeff_index, best=best, results=results,
                metric=results[0]["metric"], engine=results[0].get("engine"))
    if out:
        _plot_chirp_axis(outd, out)
    return outd


def _plot_chirp_axis(scan: Dict[str, Any], out: str) -> None:
    """Score and the (offset, amplitude) optimum vs the scanned chirp coefficient."""
    v, s = scan["values"], scan["scores"]
    k = int(np.nanargmax(s))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), dpi=200, sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]))
    ax.plot(v, s, "o-", color="#2C3E50", lw=1.4, ms=4)
    ax.plot(v[k], s[k], marker="*", ms=17, mfc="#C0392B", mec="white", mew=1.1,
            zorder=5, linestyle="none")
    ax.set_ylabel("best " + str(scan["metric"]))
    ax.grid(alpha=0.25)
    ax.set_title(f"chirp axis: $c_{{{scan['coeff_index']}}}$ "
                 f"({scan.get('engine', '')}); best {s[k]:.5f} at "
                 f"{v[k]:+.5f} GHz", fontsize=10)
    # the tune-up MOVES with the chirp -- that is the whole reason to re-optimize
    # (offset, amplitude) per chirp rather than hold a chirp-free calibration
    ax2.plot(v, scan["wp_offsets_MHz"], "s-", ms=3.5, lw=1.2,
             color="#2980B9", label="opt offset (MHz)")
    ax2b = ax2.twinx()
    ax2b.plot(v, scan["amp_scales"], "^-", ms=3.5, lw=1.2,
              color="#E67E22", label="opt amp scale")
    ax2.set_xlabel(f"chirp coefficient $c_{{{scan['coeff_index']}}}$  (GHz)")
    ax2.set_ylabel("offset (MHz)", color="#2980B9")
    ax2b.set_ylabel("amp scale", color="#E67E22")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.calibration_map", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--t-g-ns", type=float, required=True)
    ap.add_argument("--wa-GHz", type=float, default=None,
                    help="override w_a (default: device pair)")
    ap.add_argument("--wb-GHz", type=float, default=None,
                    help="override w_b -- the target sweep's swept frequency")
    ap.add_argument("--spec-abs-GHz", type=float, default=None,
                    help="absolute spectator frequency (target-sweep convention)")
    ap.add_argument("--delta-GHz", type=float, default=None,
                    help="spectator-sweep detuning Delta = w_b - w_spec (GHz)")
    ap.add_argument("--drag-beat-GHz", type=float, default=None,
                    help="map the DRAG-on gate, with the quadrature at this beat")
    ap.add_argument("--chirp-GHz", default=None,
                    help="comma list of Legendre chirp coefficients (GHz) held FIXED "
                         "across the grid, e.g. '0,0,-0.004'; the scan axes stay "
                         "(pump offset, amplitude). Omit to inherit the device's "
                         "chirp_coeffs_GHz; pass '' to force no chirp")
    ap.add_argument("--engine", choices=["reduced", "qutip", "jax"], default="reduced",
                    help="reduced rotating-frame model (fast, CPU), exact QuTiP "
                         "sesolve (--engine qutip for the trustworthy map), or the "
                         "batched jax engine (--engine jax: the whole grid in one "
                         "vmapped program, GPU-capable; validate the cutoff first "
                         "with validate_engines.py --cutoff-scan)")
    ap.add_argument("--engine-cutoff-GHz", type=float, default=float("inf"),
                    help="[jax] carrier cutoff. Defaults to inf (EXACT). A finite "
                         "cutoff is only safe at weak drive -- at |eta|~9 a 3 GHz "
                         "cutoff was off by max|dU|~0.9. The engine's speed comes "
                         "from the integrator and batching, not pruning, so inf is "
                         "cheap; lower it only with a --cutoff-scan to justify it")
    ap.add_argument("--engine-carrier-resolution", type=float, default=0.1,
                    help="[jax] max |Omega|*dt for the CF4 propagator (4th order, so "
                         "halving this cuts the error ~16x)")
    ap.add_argument("--engine-batch", type=int, default=64,
                    help="[jax] grid points per vmapped call")
    ap.add_argument("--engine-precision", choices=["f64", "f32"], default="f64",
                    help="[jax] f32 is MIXED (f64 phases, complex64 state); measure it "
                         "with validate_engines.py --precision-scan before trusting it")
    ap.add_argument("--coupler-levels", type=int, default=None,
                    help="override device coupler truncation; bump at strong drive")
    ap.add_argument("--jobs", type=int, default=1,
                    help="joblib workers over the grid (QuTiP engine)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--save-point", default=None,
                    help="save the optimum into the device JSON under this name")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --save-point to replace an existing point")
    ap.add_argument("--save-npz", default=None,
                    help="write the full scan (arrays + optimum + context) to this .npz")
    ap.add_argument("--wp-span-MHz", type=float, default=40.0)
    ap.add_argument("--wp-points", type=int, default=41)
    ap.add_argument("--amp-lo", type=float, default=0.6)
    ap.add_argument("--amp-hi", type=float, default=1.4)
    ap.add_argument("--amp-points", type=int, default=41)
    ap.add_argument("--cutoff-GHz", type=float, default=1.0,
                    help="reduced-engine carrier cutoff (ignored by --engine qutip)")
    ap.add_argument("--metric", choices=["fidelity", "transfer"], default="fidelity",
                    help="'transfer' colours by the swap population P(|01>->|10>)")
    ap.add_argument("--chirp-scan-coeff", type=int, default=None,
                    help="repeat the whole 2-D map across this Legendre chirp "
                         "coefficient (2 or 4; the Stark shape is EVEN, and c_0 is "
                         "degenerate with the offset axis). Re-optimizes (offset, "
                         "amplitude) at every chirp value. Cost is N maps -- for "
                         "actually calibrating a chirp prefer 'grape --chirp-degree'")
    ap.add_argument("--chirp-lo", type=float, default=-0.01)
    ap.add_argument("--chirp-hi", type=float, default=0.01)
    ap.add_argument("--chirp-points", type=int, default=7)
    ap.add_argument("--out", default="figs/calibration_map.png")
    ap.add_argument("--log", default=None,
                    help="optional progress log file; default is stdout only, so "
                         "concurrent jobs (e.g. one per device) land in each "
                         "job's own SLURM .out instead of one shared file")
    args = ap.parse_args()

    from snail_solver.paths import resolve_device
    from snail_solver.device_utils import load_device, parse_chirp_arg
    device_path = resolve_device(args.device)
    cfg = load_device(device_path)

    if args.chirp_scan_coeff is not None:
        values = np.linspace(args.chirp_lo, args.chirp_hi, args.chirp_points)
        print(f"chirp axis: c{args.chirp_scan_coeff} over {len(values)} values "
              f"in [{args.chirp_lo:+g}, {args.chirp_hi:+g}] GHz "
              f"({len(values)} full maps)")
        scan_res = scan_chirp_axis(
            cfg, args.t_g_ns, args.chirp_scan_coeff, values,
            base_chirp=parse_chirp_arg(args.chirp_GHz),
            out=args.out, wa_GHz=args.wa_GHz, wb_GHz=args.wb_GHz,
            spec_abs_GHz=args.spec_abs_GHz, delta_GHz=args.delta_GHz,
            drag_beat_GHz=args.drag_beat_GHz, engine=args.engine,
            coupler_levels=args.coupler_levels, atol=args.atol, rtol=args.rtol,
            nsteps=args.nsteps, jobs=args.jobs,
            engine_cutoff_GHz=args.engine_cutoff_GHz,
            engine_carrier_resolution=args.engine_carrier_resolution,
            engine_batch=args.engine_batch, engine_precision=args.engine_precision,
            wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
            amp_lo=args.amp_lo, amp_hi=args.amp_hi, amp_points=args.amp_points,
            cutoff_GHz=args.cutoff_GHz, metric=args.metric, log_path=args.log)
        bst = scan_res["best"]
        print(f"\nbest over the chirp axis: c{bst['coeff_index']} = {bst['value']:+.6f} "
              f"GHz, {args.metric} = {bst['score']:.5f} "
              f"(amp={bst['amp_scale']:.4f}, offset={bst['wp_offset_MHz']:+.2f} MHz)")
        print(f"  --chirp-GHz \"{','.join(f'{c:g}' for c in bst['chirp_coeffs_GHz'])}\"")
        if args.save_npz:
            np.savez_compressed(
                args.save_npz, values=scan_res["values"], scores=scan_res["scores"],
                amp_scales=scan_res["amp_scales"],
                wp_offsets_MHz=scan_res["wp_offsets_MHz"],
                coeff_index=scan_res["coeff_index"],
                best_chirp_coeffs_GHz=np.asarray(bst["chirp_coeffs_GHz"], dtype=float),
                best_value=bst["value"], best_score=bst["score"])
            print("saved", args.save_npz)
        return

    result = run(cfg, args.t_g_ns, wa_GHz=args.wa_GHz, wb_GHz=args.wb_GHz,
                 spec_abs_GHz=args.spec_abs_GHz, delta_GHz=args.delta_GHz,
                 drag_beat_GHz=args.drag_beat_GHz,
                 chirp_coeffs_GHz=parse_chirp_arg(args.chirp_GHz),
                 drag_n_pump=args.drag_n_pump, engine=args.engine,
                 coupler_levels=args.coupler_levels, atol=args.atol, rtol=args.rtol,
                 nsteps=args.nsteps, jobs=args.jobs, save_npz_path=args.save_npz,
                 engine_cutoff_GHz=args.engine_cutoff_GHz,
                 engine_carrier_resolution=args.engine_carrier_resolution,
                 engine_batch=args.engine_batch, engine_precision=args.engine_precision,
                 wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
                 amp_lo=args.amp_lo, amp_hi=args.amp_hi, amp_points=args.amp_points,
                 cutoff_GHz=args.cutoff_GHz, metric=args.metric, out=args.out,
                 log_path=args.log)
    print(f"log: {result['log_path'] or 'stdout (SLURM job output)'}")
    b, ctx = result["best"], result["context"]
    print(f"context: w_a={ctx['wa_GHz']} w_b={ctx['wb_GHz']} t_g={ctx['t_g_ns']} ns "
          f"spec={ctx['spec_abs_GHz']} drag_beat={ctx['drag_beat_GHz']} "
          f"chirp={ctx['chirp_coeffs_GHz'] or 'none'} engine={args.engine}")
    print(f"optimum: wp_offset = {b['wp_offset_MHz']:+.2f} MHz, "
          f"amp_scale = {b['amp_scale']:.4f}, {args.metric} = {b['score']:.5f}"
          + (f", leak = {b['leakage']:.4f}" if "leakage" in b else ""))
    if "chirp_seed_GHz" in result:
        print(f"stark ridge: {result['stark_slope_MHz_per_amp2']:+.3f} MHz/amp^2 "
              f"(r2 = {result['stark_fit_r2']:.4f}), "
              f"delta_stark = {result['delta_stark_GHz'] * 1e3:+.3f} MHz")
        seed = ",".join(f"{c:g}" for c in result["chirp_seed_GHz"])
        print(f"  suggested chirp seed: --chirp-GHz \"{seed}\"")
        if not (result["stark_fit_r2"] > 0.9):
            print("  (r2 is low -- the ridge is not Stark-dominated here, so treat "
                  "this seed with suspicion)")
    if args.save_point:
        from snail_solver.operating_points import save_point
        rec = save_point(device_path, args.save_point, result["operating_point"],
                         overwrite=args.overwrite)
        print(f"saved operating point {args.save_point!r} to {device_path}: "
              f"amp_scale={rec['amp_scale']:.4f}, wp_offset={rec['wp_offset_GHz']:+.6f} GHz"
              + (f", chirp={rec['chirp_coeffs_GHz']}"
                 if rec.get("chirp_coeffs_GHz") else ""))
    if args.engine == "reduced":
        print("  (reduced model -- validate this (offset, amp) with --engine qutip)")


if __name__ == "__main__":
    main()