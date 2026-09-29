#!/usr/bin/env python3
"""
tune_up_sweep.py
================

How far can the pump be driven before the chirped gate stops working?

``tune_up`` calibrates at ONE drive strength (``t_g0 = 2A/eta*``). This module fans
the tune-up out over a grid of ``target_eta``, re-calibrating from scratch at each
(new Rabi sweep, chirp, offset and length), then scores the REAL gate at every point.
Local counterpart of ``slurm/submit_tune_up.sh``.

Every point gets its own calibration because the AC-Stark shift grows as ``|eta|^2``:
scoring many drives against one chirp would measure the calibration going stale.

Three series per eta
--------------------
* ``chirped``    -- the calibrated chirp at the calibrated length (the real gate).
* ``flat``       -- chirp zeroed, all else held (``run_tune_up``'s ``chirp_ablation``).
  Isolates what tracking the shift THROUGH the pulse buys, since ``wp_offset`` already
  carries its mean -- but the length was fitted with the chirp on.
* ``flat_refit`` -- chirp zeroed AND length re-fitted (``--flat-refit-length``).

Metrics: ``F_avg``/``leakage`` from :meth:`ZhouCoupler.iswap_fidelity` (4-column
subspace, virtual-Z fitted out); ``transfer`` is ``P(|01> -> |10>)``, the phase- and
``|11>``-blind quantity the length scan maximises. They can diverge at high drive, and
that divergence is itself the leakage story.

Usage
-----
    python -m snail_solver.tune_up_sweep --device 1Gate4.2SNAIL.json \\
        --target-etas 1.2:2.0:9 --eta-lo 0.4 --eta-hi 1.0 \\
        --amp-points 9 --wp-points 61 --post-chirp-points 9 \\
        --flat-refit-length --jobs 61 --plot-each \\
        --outdir results/etasweep_1Gate4.2SNAIL \\
        --plot figs/etasweep_1Gate4.2SNAIL/gate_quality_vs_eta.png

Output layout (one HDF5 file, ``h5_io``)::

    eta_sweep.h5
      /sweep              <- summary: rows, settings, timings (+ /device, /figures)
      /runs/eta1p2        <- that eta's COMPLETE tune-up in tune_up's --out schema,
      /runs/eta1p8           also kept for a FAILED eta (+ /device, /figures)

Each run is addressable on its own (it carries its own device copy)::

    python -m snail_solver.tune_up --replot results/.../eta_sweep.h5:/runs/eta1p8 \\
        --plot-ridge figs/eta1p8_ridge.png
    python -m snail_solver.post_chirp \\
        --from-tuneup results/.../eta_sweep.h5:/runs/eta1p8 ...
    python -m snail_solver.h5_io eta_sweep.h5:/runs/eta1p8 --extract figs/eta1p8/

``--out something.json`` writes the summary as text, with the per-eta runs beside it
as ``tuneup_<tag>.h5``. Re-plot a finished run with no solves::

    python -m snail_solver.tune_up_sweep --replot \\
        results/etasweep_1Gate4.2SNAIL/eta_sweep.h5 --plot figs/final.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from snail_solver.h5_io import (attach_figures, has_group, is_hdf5, load_doc,
                                save_doc, save_tree, split_address)


def _plain(o: Any) -> Any:
    """json default: ndarrays and numpy scalars to plain Python."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def parse_etas(spec: str) -> List[float]:
    """``"1.2:2.0:9"`` (start:stop:points) or ``"1.2,1.5,1.8"`` -> a list of floats.

    Rejects a non-positive eta: ``t_g0 = 2A/eta*`` divides by it.
    """
    text = str(spec).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"--target-etas {spec!r}: expected start:stop:points")
        lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n < 1:
            raise ValueError(f"--target-etas {spec!r}: points must be >= 1")
        etas = [float(v) for v in np.linspace(lo, hi, n)]
    else:
        etas = [float(v) for v in text.split(",") if v.strip()]
    if not etas:
        raise ValueError(f"--target-etas {spec!r}: no values")
    if any(e <= 0.0 for e in etas):
        raise ValueError(f"--target-etas {spec!r}: every eta must be > 0")
    return etas


def sweep_holds_runs(sweep_path: Optional[str]) -> bool:
    """Whether this sweep output can hold the per-eta runs inside itself (HDF5 only)."""
    return bool(sweep_path
                and os.path.splitext(split_address(sweep_path)[0])[1].lower() != ".json")


def store_run(tag: str, doc: Dict[str, Any], *, sweep_path: Optional[str] = None,
              outdir: Optional[str] = None,
              attrs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Store one eta's complete tune-up document where this sweep keeps its runs.

    ``<sweep>.h5:/runs/<tag>`` when the sweep output is HDF5, else
    ``<outdir>/tuneup_<tag>.h5`` beside a JSON summary. The document is in
    ``tune_up``'s ``--out`` schema, so the returned address works with
    ``tune_up --replot`` and ``post_chirp --from-tuneup``. Called as each eta
    finishes, so a killed sweep keeps what it finished.

    Returns the address written (``file`` or ``file:/group``), or None when there is
    nowhere to put it.
    """
    if sweep_holds_runs(sweep_path):
        return save_doc(split_address(sweep_path)[0], doc, attrs=attrs,
                        group=f"runs/{tag}")
    if outdir:
        return save_doc(os.path.join(outdir, f"tuneup_{tag}.h5"), doc, attrs=attrs)
    return None


def embed_figures(run_address: Optional[str], figures: Dict[str, str]) -> int:
    """Embed this eta's figures in the document its data went to.

    The loose copies under ``<outdir>/figs/<tag>/`` stay; this one travels with the
    run. Silent no-op (returns 0) with no HDF5 run to put them in.
    """
    if not run_address or not figures:
        return 0
    if not is_hdf5(split_address(run_address)[0]):
        return 0
    return attach_figures(run_address, figures)


def eta_tag(eta: float) -> str:
    """``1.8 -> 'eta1p8'`` -- a filename fragment with no '.' in it."""
    return "eta" + f"{float(eta):g}".replace(".", "p").replace("-", "m")


# ===========================================================================
# Scoring one calibrated point
# ===========================================================================
def score_gate(config: Dict[str, Any], record: Dict[str, Any],
               chirp_coeffs_GHz: Optional[Sequence[float]], *,
               solver: Optional[Dict[str, Any]] = None,
               fit_virtual_z: bool = True, refit_length: bool = False,
               tg_points: int = 13, tg_lo: float = 0.7, tg_hi: float = 1.3,
               drag_channels=None,
               logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Score the REAL gate at a calibrated operating point.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    record : dict
        A ``run_tune_up`` operating point (``t_g_ns``, ``amp_scale``,
        ``wp_offset_GHz``, ``target_eta``, ``spec_abs_GHz``, ``drag_beat_GHz``).
    chirp_coeffs_GHz : sequence of float, or []
        The chirp to score. Pass ``[]`` for the flat carrier -- **never None**:
        ``build_coupler`` reads None as "inherit ``config['chirp_coeffs_GHz']``".
    drag_channels : sequence of DragChannel, optional
        The recursive-DRAG channels to PLAY. Multi-channel DRAG does not travel via
        ``record["drag_beat_GHz"]`` (the one-channel legacy field, None whenever
        channels were used), so omitting this scores an UN-DRAGGED pulse -- silently,
        since the chirp itself is DRAG-aware either way.
    refit_length : bool, default False
        Re-fit the length for THIS chirp first (a full length scan, ~`tg_points` + 19
        serial solves) -- the fair comparison for the flat carrier.

    Returns
    -------
    dict
        ``t_g_ns``, ``amp_scale``, ``wp_offset_GHz``, ``peak_eta``,
        ``chirp_coeffs_GHz``, ``F_avg``, ``leakage``, ``transfer``,
        ``refit_length``.
    """
    from snail_solver.device_utils import build_coupler, transfer_probability
    from snail_solver.tune_up import fixed_eta_amp_scale, length_rabi, nominal_t_g

    if chirp_coeffs_GHz is None:
        raise ValueError("score_gate: pass [] for no chirp, not None -- None means "
                         "'inherit config[\"chirp_coeffs_GHz\"]' in build_coupler")
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    chirp = [float(c) for c in chirp_coeffs_GHz]
    target_eta = float(record["target_eta"])
    wp = float(record["wp_offset_GHz"])
    spec_abs_GHz = record.get("spec_abs_GHz")
    drag_beat_GHz = record.get("drag_beat_GHz")
    drag_n_pump = int(record.get("drag_n_pump") or 1)

    t_g = float(record["t_g_ns"])
    if refit_length:
        t_g0 = nominal_t_g(config, target_eta)
        L = length_rabi(config, target_eta,
                        t_g0 * np.linspace(float(tg_lo), float(tg_hi),
                                           int(tg_points)),
                        wp_offset_GHz=wp, chirp_coeffs_GHz=chirp,
                        drag_beat_GHz=drag_beat_GHz, drag_n_pump=drag_n_pump,
                        spec_abs_GHz=spec_abs_GHz, solver=solver, logger=logger,
                        drag_channels=drag_channels)
        t_g = float(L["t_g_ns"])
    # amp_scale holds |eta| at the target for THIS length: recomputing it makes the
    # refit branch correct and is an exact no-op when the length is unchanged.
    amp = float(fixed_eta_amp_scale(config, t_g, target_eta))

    cpl, _w_p, peak_eta = build_coupler(
        config, t_g, amp, wp, spec_abs_GHz, drag_beat_GHz,
        chirp_coeffs_GHz=chirp, drag_n_pump=drag_n_pump,
        drag_channels=drag_channels)
    F, leak, _U = cpl.iswap_fidelity(0, 1, t_g, fit_virtual_z=fit_virtual_z, **solver)
    P = transfer_probability(config, t_g, amp, wp, solver,
                             spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
                             chirp_coeffs_GHz=chirp, drag_n_pump=drag_n_pump,
                             drag_channels=drag_channels)
    return {"t_g_ns": float(t_g), "amp_scale": float(amp), "wp_offset_GHz": float(wp),
            "peak_eta": float(peak_eta), "chirp_coeffs_GHz": chirp,
            "F_avg": float(F), "leakage": float(leak), "transfer": float(P),
            "refit_length": bool(refit_length),
            # What was PLAYED, so a score that dropped its DRAG is visible as such.
            "n_drag_channels": int(len(drag_channels) if drag_channels else 0),
            "drag_beat_GHz": (float(drag_beat_GHz)
                              if drag_beat_GHz is not None else None)}


# ===========================================================================
# The sweep
# ===========================================================================
def run_eta_sweep(config: Dict[str, Any], target_etas: Sequence[float], *,
                  device_path: Optional[str] = None,
                  eta_lo: float = 0.4, eta_hi: float = 1.0, amp_points: int = 9,
                  wp_points: int = 61, wp_span_MHz: Optional[float] = None,
                  adaptive_span: bool = False, span_linewidths: float = 4.0,
                  window_tg: float = 2.0, n_time: int = 161, tg_points: int = 13,
                  tg_lo: float = 0.7, tg_hi: float = 1.3,
                  chirp_degree: int = 8, quartic_warn: float = 0.25,
                  contrast_min: float = 0.35, do_time_rabi: bool = False,
                  post_chirp_points: int = 0,
                  flat_refit_length: bool = False, fit_virtual_z: bool = True,
                  jobs: int = 0, solver: Optional[Dict[str, Any]] = None,
                  outdir: Optional[str] = None, sweep_path: Optional[str] = None,
                  plot_each: bool = False,
                  save_point_prefix: Optional[str] = None, overwrite: bool = False,
                  stop_on_error: bool = False,
                  logger: Optional[logging.Logger] = None,
                  **map_kw) -> Dict[str, Any]:
    """Re-calibrate at every ``target_eta`` and score the gate at each.

    A failure at one eta is recorded and the sweep continues: a calibration that
    stops being measurable at high drive is a RESULT, not an outage. A
    ``RabiFitError``'s measured chevrons are still stored (and plotted).

    Parameters
    ----------
    target_etas : sequence of float
        The drive strengths to calibrate at. ``t_g0 = 2A/eta*`` for each.
    eta_lo, eta_hi : float
        The Rabi ROW window as a fraction of each target (not a target range);
        `eta_hi` > 1.0 makes the chirp an extrapolation.
    adaptive_span : bool, default False
        Per-row adaptive offset spans. The rows then share no offset axis, so
        `plot_chirp_ridge` cannot draw.
    flat_refit_length : bool, default False
        Also score a flat carrier with its OWN fitted length (a third series).
    sweep_path : str, optional
        The sweep's output file; when HDF5, each eta's tune-up is stored inside it
        under ``runs/<tag>`` (see :func:`store_run`), else in `outdir`.
    save_point_prefix : str, optional
        Save each result into the device JSON as ``<prefix>_eta1p8`` (needs
        `device_path`).
    **map_kw
        Forwarded via `run_tune_up` to `rabi_shift_table` (``r2_min``, ``leak_max``,
        ``nrmse_max``, ``secondary_max``).

    Returns
    -------
    dict
        The sweep document: ``device`` (a copy of the merged configuration every
        eta ran with), ``device_path``, ``settings``, ``rows`` (one per eta, each
        with ``ok`` and either the scores or an ``error``), and ``summary``.
    """
    from snail_solver import tune_up as TU

    log = logger or logging.getLogger("tune_up_sweep")
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    etas = [float(e) for e in target_etas]
    t0_all = time.perf_counter()

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i, eta_star in enumerate(etas):
        tag = eta_tag(eta_star)
        t_g0 = TU.nominal_t_g(config, eta_star)
        log.info(f"=== [{i + 1}/{len(etas)}] target_eta={eta_star:g} "
                 f"(t_g0={t_g0:.3f} ns) ===")
        t0 = time.perf_counter()

        span = wp_span_MHz
        if span is None and not adaptive_span:
            # One fixed span per eta so the rows share an offset axis (ridge plot).
            span, _want = TU.ridge_span_MHz(
                config, eta_star, eta_lo=eta_lo, eta_hi=eta_hi,
                span_linewidths=span_linewidths, wp_points=wp_points, logger=log)

        # The span ACTUALLY used: the default sizes it per eta.
        row: Dict[str, Any] = {"target_eta": eta_star, "tag": tag,
                               "t_g0_ns": float(t_g0),
                               "wp_span_MHz": (None if span is None else float(span))}
        try:
            out = TU.run_tune_up(
                config, eta_star, drag_beat_GHz=None, drag_channels=None,
                spec_abs_GHz=None, chirp_degree=chirp_degree,
                quartic_warn=quartic_warn, eta_lo=eta_lo, eta_hi=eta_hi,
                amp_points=amp_points, wp_span_MHz=span, wp_points=wp_points,
                tg_points=tg_points, tg_lo=tg_lo, tg_hi=tg_hi,
                window_tg=window_tg, n_time=n_time,
                span_linewidths=span_linewidths, contrast_min=contrast_min,
                do_time_rabi=do_time_rabi, jobs=jobs,
                post_chirp_points=post_chirp_points, solver=solver, logger=log,
                **map_kw)
        except TU.RabiFitError as exc:
            # Only the interpretation failed: keep the chevrons as the evidence.
            row.update({"ok": False, "seconds": time.perf_counter() - t0,
                        "error": {"type": "RabiFitError", "stage": "rabi",
                                  "message": str(exc)}})
            p = store_run(tag, {"stages": {"rabi": exc.table},
                                "device": dict(config)},
                          sweep_path=sweep_path, outdir=outdir,
                          attrs={"target_eta": float(eta_star),
                                 "status": "rabi_fit_failed",
                                 "error": str(exc)})
            if p:
                row["run"] = p
                log.info(f"  the chevrons that failed are in {p}")
            if plot_each and outdir:
                fig = os.path.join(outdir, "figs", tag, "rabi_chevrons_FAILED.png")
                try:
                    TU.plot_rabi_table(exc.table, fig)
                    row["figs"] = {"rabi": fig}
                    embed_figures(p, {"rabi": fig})
                except Exception as pexc:                # a figure is never fatal
                    log.info(f"  plot skipped: {type(pexc).__name__}: {pexc}")
            log.warning(f"  eta={eta_star:g} FAILED (RabiFitError): {exc}")
            rows.append(row)
            if stop_on_error:
                break
            continue
        except Exception as exc:
            row.update({"ok": False, "seconds": time.perf_counter() - t0,
                        "error": {"type": type(exc).__name__, "stage": "tune_up",
                                  "message": str(exc)}})
            log.warning(f"  eta={eta_star:g} FAILED "
                        f"({type(exc).__name__}): {exc}")
            rows.append(row)
            if stop_on_error:
                raise
            continue

        rec = out["operating_point"]
        stages = out["stages"]

        # tune_up's own --out schema, so tune_up --replot / post_chirp --from-tuneup
        # read it verbatim.
        p = store_run(tag, {"operating_point": rec, "t_g0_ns": out["t_g0_ns"],
                            "stages": stages, "device": dict(config)},
                      sweep_path=sweep_path, outdir=outdir,
                      attrs={"target_eta": float(eta_star), "status": "ok"})
        if p:
            row["run"] = p

        if plot_each and outdir:
            figs = {}
            fdir = os.path.join(outdir, "figs", tag)
            for name, fn in (
                    ("rabi", lambda: TU.plot_rabi_table(
                        stages["rabi"], os.path.join(fdir, "rabi_chevrons.png"))),
                    # named as tune_up names it (figure_names() reads the same)
                    ("chirp_ridge", lambda: TU.plot_chirp_ridge(
                        stages["rabi"], stages["chirp"], rec["wp_offset_GHz"],
                        rec["t_g_ns"], os.path.join(fdir, "chirp_ridge.png"))),
                    ("post_chirp", lambda: TU.plot_post_chirp_table(
                        stages["post_chirp"],
                        out=os.path.join(fdir, "post_chirp_chevrons.png"),
                        rabi_table=stages["rabi"]) if stages.get("post_chirp")
                        else None)):
                try:
                    got = fn()
                    if got:
                        figs[name] = got
                except Exception as pexc:                # a figure is never fatal
                    log.info(f"  {name} plot skipped: "
                             f"{type(pexc).__name__}: {pexc}")
            row["figs"] = figs
            embed_figures(p, figs)

        # --- score the real gate, chirped vs flat -----------------------------
        log.info(f"  scoring the gate at t_g={rec['t_g_ns']:.3f} ns")
        row["chirped"] = score_gate(config, rec, rec["chirp_coeffs_GHz"],
                                    solver=solver, fit_virtual_z=fit_virtual_z,
                                    logger=log)
        row["flat"] = score_gate(config, rec, [], solver=solver,
                                 fit_virtual_z=fit_virtual_z, logger=log)
        if flat_refit_length:
            row["flat_refit"] = score_gate(config, rec, [], solver=solver,
                                           fit_virtual_z=fit_virtual_z,
                                           refit_length=True, tg_points=tg_points,
                                           tg_lo=tg_lo, tg_hi=tg_hi,
                                           logger=log)

        fit = stages["rabi"]["fit"]
        chirp_st = stages["chirp"]
        length = stages["length"]
        row.update({
            "ok": True, "error": None, "seconds": time.perf_counter() - t0,
            "operating_point": rec,
            "rabi_fit": {k: fit.get(k) for k in
                         ("delta0", "k2", "k4", "r2", "n_used", "resid_MHz",
                          "stark_span_MHz")},
            "n_dropped": int(sum(1 for c in stages["rabi"]["chevrons"]
                                 if c.get("dropped"))),
            "chirp": {k: chirp_st.get(k) for k in
                      ("coeffs_GHz", "quartic_fraction", "rel_diff",
                       "extrapolation_ratio", "perturbative_ok",
                       "measured_eta_max")},
            "length": {"t_g_ns": length["t_g_ns"], "transfer": length["transfer"],
                       "railed": bool(length["railed"]), "nfev": int(length["nfev"])},
            "residual_MHz": float(stages.get("residual_GHz", 0.0)) * 1e3,
            "delta_F": float(row["chirped"]["F_avg"] - row["flat"]["F_avg"]),
            "delta_leak": float(row["chirped"]["leakage"] - row["flat"]["leakage"]),
        })

        if save_point_prefix and device_path:
            from snail_solver.operating_points import save_point
            name = f"{save_point_prefix}_{tag}"
            save_point(device_path, name, rec, overwrite=overwrite)
            row["point_name"] = name
            log.info(f"  saved operating point {name!r}")

        log.info(f"  eta={eta_star:g}: F_avg={row['chirped']['F_avg']:.5f} "
                 f"(flat {row['flat']['F_avg']:.5f}), "
                 f"leak={row['chirped']['leakage']:.3e} "
                 f"(flat {row['flat']['leakage']:.3e}), "
                 f"{row['seconds']:.0f} s")
        rows.append(row)

    n_ok = sum(1 for r in rows if r.get("ok"))
    return {
        "source": "tune_up_sweep",
        "device_path": device_path,
        "device": dict(config),      # a copy, so the sweep outlives device edits
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "area_ns": float(TU._area(config)),
        "target_etas": etas,
        "settings": {
            "eta_lo": eta_lo, "eta_hi": eta_hi, "amp_points": amp_points,
            "wp_points": wp_points, "wp_span_MHz": wp_span_MHz,
            "adaptive_span": adaptive_span, "span_linewidths": span_linewidths,
            "window_tg": window_tg, "n_time": n_time, "tg_points": tg_points,
            "tg_lo": tg_lo, "tg_hi": tg_hi,
            "chirp_degree": chirp_degree, "quartic_warn": quartic_warn,
            "contrast_min": contrast_min, "do_time_rabi": do_time_rabi,
            "post_chirp_points": post_chirp_points,
            "flat_refit_length": flat_refit_length, "fit_virtual_z": fit_virtual_z,
            "jobs": jobs, "solver": solver,
            "drag_beat_GHz": None, "spec_abs_GHz": None,
            "qubit_levels": config.get("qubit_levels"),
            "coupler_levels": config.get("coupler_levels"),
            "map_kw": dict(map_kw),
        },
        "rows": rows,
        "summary": {"n_ok": n_ok, "n_failed": len(rows) - n_ok,
                    "seconds": time.perf_counter() - t0_all},
    }


# ===========================================================================
# The figure
# ===========================================================================
_C_CHIRP = "#2a78d6"          # matches tune_up._C_LORENTZ
_C_FLAT = "#eb6834"           # matches tune_up._C_VERTEX
_C_INK = "#52514e"
_C_LEAK = "#1baf7a"


def plot_eta_sweep(doc: Dict[str, Any], out: str = "figs/gate_quality_vs_eta.png",
                   title: Optional[str] = None) -> str:
    """Gate quality versus drive strength: the speed/leakage trade-off curve.

    (a) ``1 - F_avg`` (log) -- chirped, flat, and flat-refit if run.
    (b) leakage, same series, with ``1 - transfer`` (one column, a different
        quantity) as a thin trace.
    (c) calibration health -- Rabi fit r2, chirp quartic fraction, dropped rows and
        railed lengths -- so a rise in (a) can be told apart from a calibration that
        stopped being measurable.

    Failed etas are drawn as dotted vertical rules, never dropped.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:                                    # style is never fatal
        pass

    rows = doc["rows"]
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    if not ok:
        raise ValueError("plot_eta_sweep: no successful etas to plot "
                         f"({len(bad)} failed)")

    eta = np.array([r["target_eta"] for r in ok], dtype=float)
    t_g = np.array([r["chirped"]["t_g_ns"] for r in ok], dtype=float)

    def series(key, field):
        return np.array([r[key][field] if key in r else np.nan for r in ok],
                        dtype=float)

    has_refit = any("flat_refit" in r for r in ok)

    fig, (ax, axl, axd) = plt.subplots(
        3, 1, figsize=(7.0, 8.6), sharex=True, layout="constrained")

    # -- (a) infidelity ----------------------------------------------------
    ax.semilogy(eta, 1.0 - series("chirped", "F_avg"), "-o", color=_C_CHIRP,
                lw=2.0, ms=5, label="chirped")
    ax.semilogy(eta, 1.0 - series("flat", "F_avg"), "--s", color=_C_FLAT,
                lw=1.8, ms=4.5, label="flat carrier (same $t_g$)")
    if has_refit:
        ax.semilogy(eta, 1.0 - series("flat_refit", "F_avg"), ":^", color=_C_FLAT,
                    lw=1.6, ms=4.5, alpha=0.75, label="flat carrier (own $t_g$)")
    ax.set_ylabel(r"$1 - F_{\mathrm{avg}}$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    ax.set_title(title or "Chirped-gate quality vs pump strength")

    # -- (b) leakage -------------------------------------------------------
    axl.semilogy(eta, series("chirped", "leakage"), "-o", color=_C_CHIRP,
                 lw=2.0, ms=5, label=r"chirped, $1-\mathrm{Tr}(U^\dagger U)/4$")
    axl.semilogy(eta, series("flat", "leakage"), "--s", color=_C_FLAT,
                 lw=1.8, ms=4.5, label="flat carrier")
    axl.semilogy(eta, 1.0 - series("chirped", "transfer"), "-", color=_C_CHIRP,
                 lw=1.0, alpha=0.5, label=r"chirped, $1-P(|10\rangle)$ (1 column)")
    axl.semilogy(eta, 1.0 - series("flat", "transfer"), "--", color=_C_FLAT,
                 lw=1.0, alpha=0.5, label="flat, same")
    axl.set_ylabel("leakage")
    axl.legend(loc="best", fontsize=7)
    axl.grid(True, which="both", alpha=0.25, lw=0.4)

    # -- (c) calibration health -------------------------------------------
    r2 = np.array([r["rabi_fit"]["r2"] for r in ok], dtype=float)
    qf = np.array([(r["chirp"]["quartic_fraction"] or np.nan) for r in ok],
                  dtype=float)
    axd.plot(eta, r2, "-o", color=_C_INK, lw=1.6, ms=4.5, label="Rabi fit $r^2$")
    axd.axhline(0.9, ls=":", lw=1.0, color=_C_INK, alpha=0.6)
    axd.set_ylabel("Rabi fit $r^2$")
    axd.set_ylim(0.0, 1.05)
    axd.grid(True, alpha=0.25, lw=0.4)

    axq = axd.twinx()
    axq.plot(eta, qf, "-^", color=_C_LEAK, lw=1.4, ms=4.5,
             label="chirp quartic fraction")
    axq.set_ylabel("quartic / quadratic", color=_C_LEAK)
    axq.tick_params(axis="y", labelcolor=_C_LEAK)

    for i, r in enumerate(ok):
        if r.get("n_dropped"):
            axd.annotate(f"{r['n_dropped']} row(s)\ndropped", (eta[i], r2[i]),
                         textcoords="offset points", xytext=(0, -22), ha="center",
                         fontsize=6, color=_C_INK, alpha=0.8)
        if r["length"].get("railed"):
            axd.plot([eta[i]], [r2[i]], "x", ms=9, mew=1.8, color="#c0392b",
                     label="length railed" if i == 0 else None)

    h1, l1 = axd.get_legend_handles_labels()
    h2, l2 = axq.get_legend_handles_labels()
    axd.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7)
    axd.set_xlabel(r"target peak $|\eta^*|$")

    # -- failed etas -------------------------------------------------------
    for r in bad:
        for a in (ax, axl, axd):
            a.axvline(r["target_eta"], ls=":", lw=1.2, color="#c0392b", alpha=0.7)
        ax.annotate(r["error"]["type"], (r["target_eta"], ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(0, -10), ha="center",
                    rotation=90, va="top", fontsize=6, color="#c0392b")

    # -- calibrated (not 2A/eta) gate length on a second axis -------------
    # LAST: a twiny only copies the limits it is given, and a failed eta's axvline
    # outside the successful ones widens the x axis, which would slide every t_g
    # label off its eta (test_renders_three_series_and_a_failed_eta).
    axt = ax.twiny()
    axt.set_xlim(ax.get_xlim())
    axt.set_xticks(eta)
    axt.set_xticklabels([f"{v:.0f}" for v in t_g], fontsize=7)
    axt.set_xlabel("calibrated $t_g$ (ns)", fontsize=8)

    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.tune_up_sweep", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="device JSON (bare name resolves under devices/); "
                         "required unless --replot")
    ap.add_argument("--target-etas", default="1.2:2.0:9",
                    help="start:stop:points, or a comma list. THE grid this "
                         "module sweeps [1.2:2.0:9]")
    ap.add_argument("--eta-lo", type=float, default=0.4,
                    help="Rabi ROW window low end, as a fraction of each "
                         "target_eta -- not a target range [0.4]")
    ap.add_argument("--eta-hi", type=float, default=1.0,
                    help="Rabi ROW window high end; above 1.0 the chirp "
                         "extrapolates past measured data [1.0]")
    ap.add_argument("--amp-points", type=int, default=9,
                    help="Rabi rows per eta; SERIAL, so this one costs [9]")
    ap.add_argument("--wp-points", type=int, default=61,
                    help="chevron offset points; fans out over the pool, so it "
                         "is ~free up to the core count [61]")
    ap.add_argument("--wp-span-MHz", type=float, default=None,
                    help="fixed offset span; default is sized per eta by "
                         "tune_up.ridge_span_MHz")
    ap.add_argument("--adaptive-span", action="store_true",
                    help="per-row spans instead of one fixed span: better weak-row "
                         "sampling, but no shared axis, so no ridge plot")
    ap.add_argument("--span-linewidths", type=float, default=4.0)
    ap.add_argument("--window-tg", type=float, default=2.0)
    ap.add_argument("--n-time", type=int, default=161)
    ap.add_argument("--tg-points", type=int, default=13)
    ap.add_argument("--tg-lo", type=float, default=0.7,
                    help="length-scan window low edge as a fraction of t_g0 [0.7]")
    ap.add_argument("--tg-hi", type=float, default=1.3,
                    help="length-scan window high edge as a fraction of t_g0 [1.3]")
    ap.add_argument("--chirp-degree", type=int, default=8)
    ap.add_argument("--quartic-warn", type=float, default=0.25)
    ap.add_argument("--contrast-min", type=float, default=0.35)
    ap.add_argument("--time-rabi", action="store_true",
                    help="run the constant-drive cross-check too (off by default "
                         "here: it is a per-eta cost the sweep does not need)")
    ap.add_argument("--post-chirp-points", type=int, default=0,
                    help="chirped-vs-flat shaped chevron rows per eta [0=off]")
    ap.add_argument("--flat-refit-length", action="store_true",
                    help="also score a flat carrier with its OWN fitted length -- "
                         "the fair comparison, at one more length scan per eta")
    ap.add_argument("--no-fit-virtual-z", action="store_true",
                    help="do not fit out virtual-Z phases when scoring")
    ap.add_argument("--leak-max", type=float, default=None,
                    help="chevron row leakage rejection threshold (tune_up "
                         "default 0.35); raise it to keep rows at high drive")
    ap.add_argument("--r2-min", type=float, default=None,
                    help="Rabi ridge fit r2 floor (tune_up default 0.9)")
    ap.add_argument("--coupler-levels", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--gpu", action="store_true",
                    help="run via qutip-jax/diffrax (forces --jobs 1); usually a "
                         "LOSS here, since the offset scan is what parallelises")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--outdir", default=None,
                    help="figures land here, and the per-eta runs too when --out "
                         "is JSON [results/etasweep_<device>]")
    ap.add_argument("--out", default=None,
                    help="the sweep file [<outdir>/eta_sweep.h5]: summary under "
                         "/sweep, each eta's tune-up under /runs/<tag>. A .json "
                         "target writes a text summary with the runs beside it as "
                         "tuneup_<tag>.h5")
    ap.add_argument("--plot", nargs="?", const="figs/gate_quality_vs_eta.png",
                    default=None, help="the summary figure")
    ap.add_argument("--plot-each", action="store_true",
                    help="also render each eta's rabi_chevrons / chirp_ridge "
                         "(/ post_chirp) into <outdir>/figs/<tag>/")
    ap.add_argument("--save-points", metavar="PREFIX", default=None,
                    help="save each result into the device JSON as PREFIX_eta1p8")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort on the first failing eta instead of recording it")
    ap.add_argument("--replot", metavar="FILE", default=None,
                    help="regenerate --plot from a previous --out (HDF5 or JSON) "
                         "-- no solves; only the summary is read")
    args = ap.parse_args()

    # --- the zero-solve path, before anything heavy is imported -----------
    if args.replot:
        # Read /sweep alone: the runs are the bulk of the file and are not drawn.
        target = (args.replot + ":/sweep" if has_group(args.replot, "sweep")
                  else args.replot)
        doc = load_doc(target)
        path = plot_eta_sweep(doc, out=args.plot or "figs/gate_quality_vs_eta.png")
        print(f"wrote {path}")
        embed_figures(target, {"gate_quality_vs_eta": path})
        return

    if not args.device:
        ap.error("--device is required unless --replot is given")
    if args.adaptive_span and args.plot_each:
        print("note: --adaptive-span means the rows share no offset axis, so the "
              "per-eta chirp_ridge figure will be skipped")

    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from snail_solver.paths import resolve_device, in_results
    from snail_solver.device_utils import load_device
    from snail_solver.log_utils import setup_run_logger

    device_path = resolve_device(args.device)
    config = load_device(device_path)
    if args.coupler_levels is not None:
        config = {**config, "coupler_levels": int(args.coupler_levels)}
    logger = setup_run_logger(None, "tune_up_sweep")

    etas = parse_etas(args.target_etas)
    stem = os.path.splitext(os.path.basename(args.device))[0]
    outdir = args.outdir or in_results(f"etasweep_{stem}")
    os.makedirs(outdir, exist_ok=True)
    out_path = args.out or os.path.join(outdir, "eta_sweep.h5")
    holds_runs = sweep_holds_runs(out_path)
    if holds_runs:
        # Truncate ONCE, up front, so a rerun cannot inherit a previous sweep's runs.
        save_tree(out_path, {}, attrs={
            "tool": "snail_solver.tune_up_sweep",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": "python -m snail_solver.tune_up_sweep "
                       + " ".join(shlex.quote(a) for a in sys.argv[1:]),
            "device_path": str(device_path), "host": platform.node()})

    map_kw: Dict[str, Any] = {}
    if args.leak_max is not None:
        map_kw["leak_max"] = float(args.leak_max)
    if args.r2_min is not None:
        map_kw["r2_min"] = float(args.r2_min)

    from snail_solver import find_stark_resonance as FSR
    print(f"device={args.device}  etas={[f'{e:g}' for e in etas]}  "
          f"jobs={FSR._resolve_jobs(args.jobs)}{' GPU' if args.gpu else ''}")
    print(f"outdir={outdir}")

    doc = run_eta_sweep(
        config, etas, device_path=device_path,
        eta_lo=args.eta_lo, eta_hi=args.eta_hi, amp_points=args.amp_points,
        wp_points=args.wp_points, wp_span_MHz=args.wp_span_MHz,
        adaptive_span=args.adaptive_span, span_linewidths=args.span_linewidths,
        window_tg=args.window_tg, n_time=args.n_time, tg_points=args.tg_points,
        tg_lo=args.tg_lo, tg_hi=args.tg_hi,
        chirp_degree=args.chirp_degree, quartic_warn=args.quartic_warn,
        contrast_min=args.contrast_min, do_time_rabi=args.time_rabi,
        post_chirp_points=args.post_chirp_points,
        flat_refit_length=args.flat_refit_length,
        fit_virtual_z=not args.no_fit_virtual_z,
        jobs=args.jobs,
        solver={"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps},
        outdir=outdir, sweep_path=out_path, plot_each=args.plot_each,
        save_point_prefix=args.save_points, overwrite=args.overwrite,
        stop_on_error=args.stop_on_error, logger=logger, **map_kw)

    if holds_runs:
        save_doc(out_path, doc, group="sweep")
    else:
        with open(out_path, "w") as fh:
            json.dump(doc, fh, indent=2, default=_plain)
    print(f"\n  written {out_path}")

    print("\n=== gate quality vs drive ===")
    print(f"  {'eta*':>6} {'t_g/ns':>8} {'F_chirp':>9} {'F_flat':>9} "
          f"{'leak_chirp':>11} {'leak_flat':>10} {'r2':>6}")
    for r in doc["rows"]:
        if not r.get("ok"):
            print(f"  {r['target_eta']:6.2f}   FAILED  {r['error']['type']}: "
                  f"{r['error']['message'][:60]}")
            continue
        print(f"  {r['target_eta']:6.2f} {r['chirped']['t_g_ns']:8.2f} "
              f"{r['chirped']['F_avg']:9.5f} {r['flat']['F_avg']:9.5f} "
              f"{r['chirped']['leakage']:11.3e} {r['flat']['leakage']:10.3e} "
              f"{r['rabi_fit']['r2']:6.3f}")
    s = doc["summary"]
    print(f"  {s['n_ok']} ok, {s['n_failed']} failed, {s['seconds'] / 60:.1f} min")

    if args.plot:
        # "Nothing to draw" (every eta failed) is a result, not a non-zero exit.
        try:
            fig = plot_eta_sweep(doc, out=args.plot)
            print(f"  wrote {fig}")
            if embed_figures(f"{out_path}:/sweep" if holds_runs else out_path,
                             {"gate_quality_vs_eta": fig}):
                print(f"  embedded it in {out_path} "
                      f"(python -m snail_solver.h5_io {out_path}:/sweep "
                      f"--extract DIR)")
        except ValueError as exc:
            print(f"  no figure: {exc}")


if __name__ == "__main__":
    main()
