#!/usr/bin/env python
r"""Flat-only chevrons: the plateau Hamiltonian by itself -- no ramps, no chirp, no DRAG.

The pump is switched on at ``t = 0`` at constant ``|eta| = eta_flat`` (a
``ConstantPulse``: the "ramps are small" limit) at a FIXED carrier scanned around the
Stark-shifted frequency. Per subharmonic-scan column and drive this maps

* a NARROW chevron (offset x time -> P(|10>)) around the law's carrier, fitted with
  ``tune_up.fit_chevron_center`` -- the measured plateau Stark shift and the law's error;
* a WIDE collision map: the time-AVERAGE of every leakage channel vs pump offset (the
  sudden switch-on leaves fast bounded dressing, e.g. the SNAIL's ~1 GHz
  counter-rotating terms, that saturates a time-max everywhere; a real collision
  accumulates), with each audited channel's predicted resonance overlaid;
* the leakage fingerprint AT the fitted resonance (matrix pencil vs the audit).

The probe is the calibration's own constant probe
(``find_stark_resonance.build_chevron_coupler(shape="constant")`` via
``_chevron_worker``). Stark laws are READ from the pass-A column caches; nothing in the
running pipeline is written.

    .venv/bin/python scripts/flat_chevron_study.py --deltas=0.015,0.1 --jobs 12
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as _dt
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from plateau_collision_study import (DEFAULT_LAWS, INK, INK2, SERIES,  # noqa: E402
                                     _style, column_config, eta_tag, load_column,
                                     parse_list)

LEAK_CHANNELS = ("P_leak", "P_coupler", "P_f_a", "P_f_b")


# ---------------------------------------------------------------------------
def _worker(args):
    from snail_solver.find_stark_resonance import _chevron_worker
    return _chevron_worker(args)


def chevron_stack(cfg, eta, offsets_GHz, times, solver, jobs) -> np.ndarray:
    """``[n_off, n_channel, n_time]`` population stack of the constant probe."""
    args = [(cfg, float(eta), float(o), times, solver, None, {"shape": "constant"})
            for o in offsets_GHz]
    if jobs <= 1:
        return np.stack([_worker(a) for a in args])
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return np.stack(list(ex.map(_worker, args)))


def constant_coupler(cfg, eta, offset_GHz, window_ns):
    from snail_solver.find_stark_resonance import build_chevron_coupler
    cpl, _w_p = build_chevron_coupler(cfg, float(eta), float(offset_GHz), float(window_ns),
                                      shape="constant")
    return cpl


def collision_offsets(cpl, offset0_GHz: float, t_ns: float, *, min_g_MHz: float = 0.2
                      ) -> List[Dict[str, Any]]:
    """Pump offset at which each audited parasitic process becomes resonant.

    An ``n``-pump process with audit detuning ``det = n f_p - (E_f - E_i)`` (pump minus
    transition) at offset ``o0`` is resonant at ``o0 - det/n``. The audit lists each
    process in both directions (``i -> f`` and its emission partner); only the
    upward one (bare ``E_f > E_i``) is kept, so each collision appears once. Static
    (0-pump) processes cannot be moved by the pump and are dropped.
    """
    from snail_solver.piecewise_pulse import predict_plateau_channels
    f = np.asarray(cpl.omega) / (2 * np.pi)
    E = lambda occ: float(np.dot(occ, f))            # noqa: E731  bare harmonic, GHz
    seen, out = set(), []
    for p in predict_plateau_channels(cpl, t_ns, min_g_MHz=min_g_MHz):
        if p["category"] == "target" or p["n_pump"] == 0:
            continue
        if E(p["f_occ"]) < E(p["i_occ"]):
            continue
        key = (min(p["i_index"], p["f_index"]), max(p["i_index"], p["f_index"]),
               p["n_pump"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(p, resonant_offset_MHz=1e3 * offset0_GHz
                        - p["detuning_MHz"] / p["n_pump"]))
    return out


def channel_index(name: str) -> int:
    from snail_solver.find_stark_resonance import CHANNELS
    return CHANNELS.index(name)


# ---------------------------------------------------------------------------
def study_drive(cfg, law, wp0, eta, args, solver) -> Dict[str, Any]:
    from snail_solver import piecewise_pulse as pp
    from snail_solver.tune_up import _area, chevron_quality, fit_chevron_center

    t_iswap = _area(cfg) / float(eta)
    T = float(args.window_iswaps) * t_iswap
    times = np.arange(0.0, T + 1e-9, float(args.dt_ns))
    c_law = pp.plateau_carrier_GHz(wp0, law["k2"], law["k4"], eta)
    i10 = channel_index("P10")

    def narrow_scan(centre_GHz):
        """(offsets, stack, P10 envelope, Lorentzian fit) of the narrow chevron."""
        grid = centre_GHz + np.linspace(-args.narrow_MHz, args.narrow_MHz,
                                        args.narrow_points) * 1e-3
        S = chevron_stack(cfg, eta, grid, times, solver, args.jobs)
        env = S[:, i10, :].max(axis=1)
        return grid, S, env, fit_chevron_center(grid, env)

    # narrow chevron around the law's Stark-shifted carrier
    t0 = time.perf_counter()
    narrow, S_n, env, cen = narrow_scan(c_law)
    it = int(np.argmin(np.abs(times - t_iswap)))
    c_meas = float(cen["center_GHz"])
    # The grid is centred on the LAW, which near the subharmonic can miss the plateau
    # resonance by ~15 MHz and cut the chevron off; re-centre on the measured peak.
    # Only a PASSED fit may move the window, and a rescan is kept only if its brightest
    # column is the same feature (within a half-width): a second resonance ~20 MHz
    # away can otherwise capture a failed fit's argmax and walk the window off.
    recentred = 0
    while (cen["ok"] and recentred < args.max_recentre
           and abs(c_meas - float(np.mean(narrow))) > 0.25e-3 * args.narrow_MHz):
        n2, S2, e2, c2 = narrow_scan(c_meas)
        vertex2 = float(n2[int(np.argmax(e2))])
        tol = max(float(cen["hwhm_GHz"]), 3e-3)
        if abs(vertex2 - c_meas) > tol:
            break                     # the brightest thing moved: a different feature
        if not c2["ok"]:
            # same feature, but its neighbour now shares the window and spoils the
            # single-Lorentzian fit: keep the centred DATA, keep the trusted centre
            c2 = dict(cen, vertex_GHz=vertex2, refit_failed=True)
        narrow, S_n, env, cen = n2, S2, e2, c2
        c_meas = float(cen["center_GHz"])
        recentred += 1
        if cen.get("refit_failed"):
            break
    j = int(np.argmin(np.abs(narrow - c_meas)))        # the on-resonance column
    leak_res = float(S_n[j, channel_index("P_leak"), :].max())
    qual = chevron_quality(cen, narrow, env, 2 * args.narrow_MHz, leak=leak_res)

    # wide collision map, centred on the measured resonance
    wide = c_meas + np.linspace(-args.wide_MHz, args.wide_MHz, args.wide_points) * 1e-3
    S_w = chevron_stack(cfg, eta, wide, times, solver, args.jobs)
    cpl0 = constant_coupler(cfg, eta, c_meas, T)
    coll = [c for c in collision_offsets(cpl0, c_meas, T)
            if abs(c["resonant_offset_MHz"] - 1e3 * c_meas) <= args.wide_MHz]

    # leakage peaks in the wide map, labelled by the nearest predicted collision
    from scipy.signal import find_peaks
    peaks = []
    for ch in LEAK_CHANNELS:
        y = S_w[:, channel_index(ch), :].mean(axis=1)
        floor = float(np.median(y))
        idx, props = find_peaks(y, prominence=args.peak_floor)
        for k, prom in zip(idx, props["prominences"]):
            o = 1e3 * wide[k]
            near = min(coll, key=lambda c: abs(c["resonant_offset_MHz"] - o),
                       default=None)
            d = None if near is None else abs(near["resonant_offset_MHz"] - o)
            peaks.append({"channel": ch, "offset_MHz": o,
                          "offset_rel_MHz": o - 1e3 * c_meas, "height": float(y[k]),
                          "above_median": float(y[k] - floor),
                          "prominence": float(prom),
                          "match": (near["name"] if near is not None
                                    and d <= args.match_tol_MHz else None),
                          "match_dist_MHz": d})

    # fingerprint at resonance
    cpl_r = constant_coupler(cfg, eta, float(narrow[j]), T)
    pred = pp.predict_plateau_channels(cpl_r, T)
    finger = {}
    for ch in LEAK_CHANNELS:
        y = S_n[j, channel_index(ch), :]
        finger[ch] = {"max": float(y.max()), "mean": float(y.mean()),
                      "lines": pp.match_lines(pp.matrix_pencil(times, y)[:8], pred,
                                              channel=ch, tol_MHz=args.match_tol_MHz)}

    return {
        "eta": float(eta), "t_iswap_ns": t_iswap, "window_ns": T,
        "carrier_law_MHz": 1e3 * c_law, "carrier_meas_MHz": 1e3 * c_meas,
        "law_error_MHz": 1e3 * (c_meas - c_law),
        "stark_shift_meas_MHz": 1e3 * (c_meas - wp0),
        "stark_shift_law_MHz": law["k2"] * eta ** 2 + law["k4"] * eta ** 4,
        "fit": cen, "quality": qual, "recentred": recentred,
        "P10_at_t_iswap_on_res": float(S_n[j, i10, it]),
        "P10_max_on_res": float(env[j]),
        "norm_defect_max": float(np.abs(S_n[:, channel_index("norm_defect"), :]).max()),
        "collisions": coll, "wide_peaks": peaks, "fingerprint": finger,
        "seconds": time.perf_counter() - t0,
        "_raster": {"times_ns": times, "narrow_offsets_GHz": narrow, "narrow": S_n,
                    "wide_offsets_GHz": wide, "wide": S_w},
    }


# ---------------------------------------------------------------------------
def plot_column(doc: Dict[str, Any], png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drives = doc["drives"]
    n = len(drives)
    fig, axs = plt.subplots(n, 3, figsize=(15, 3.9 * n), constrained_layout=True,
                            squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    i10 = channel_index("P10")
    for r, dr in enumerate(drives):
        R = dr["_raster"]
        t = R["times_ns"]
        c = dr["carrier_meas_MHz"]
        # chevron: pump offset (x) vs time (y), viridis like the pipeline's chevrons
        ax = axs[r, 0]
        off = 1e3 * np.asarray(R["narrow_offsets_GHz"])
        S10 = np.asarray(R["narrow"])[:, i10, :]
        im = ax.pcolormesh(off, t, S10.T, cmap="viridis", vmin=0, vmax=1,
                           shading="auto", rasterized=True)
        pk = float(off[int(np.argmax(S10.max(axis=1)))])
        ax.axvline(c, color="white", lw=1.4,
                   label=f"measured resonance (fit) {c:+.2f} MHz")
        ax.axvline(pk, color="white", lw=1.0, ls=":",
                   label=f"brightest column {pk:+.2f} MHz")
        ax.axvline(dr["carrier_law_MHz"], color=SERIES[1], lw=1.4, ls="--",
                   label=f"pass-A law prediction {dr['carrier_law_MHz']:+.2f} MHz "
                         f"({-dr['law_error_MHz']:+.2f} off)")
        ax.axhline(dr["t_iswap_ns"], color="white", lw=0.8, ls=":", alpha=0.7)
        fig.colorbar(im, ax=ax, label="P(|10⟩)")
        _style(ax, f"η_flat = {dr['eta']:.2f}: chevron (flat Stark shift "
                   f"{dr['stark_shift_meas_MHz']:+.2f} MHz)",
               "pump offset from |ω_b − ω_a| (MHz)", "time (ns)")
        ax.grid(False)
        leg = ax.legend(fontsize=6.5, frameon=True, loc="upper right")
        leg.get_frame().set_alpha(0.85)
        # wide collision map
        ax = axs[r, 1]
        offw = 1e3 * R["wide_offsets_GHz"] - c
        for k, ch in enumerate(LEAK_CHANNELS):
            y = R["wide"][:, channel_index(ch), :].mean(axis=1)
            ax.semilogy(offw, np.clip(y, 1e-6, None), lw=2 if k == 0 else 1.4,
                        color=SERIES[k], label=f"⟨{ch}⟩_t")
        ymax = ax.get_ylim()[1]
        for cl in sorted(dr["collisions"], key=lambda x: -x["g_MHz"])[:10]:
            x = cl["resonant_offset_MHz"] - c
            ax.axvline(x, color=INK2, lw=0.7, ls="--")
            ax.text(x, ymax, f" {cl['name']} ({cl['n_pump']}p, g={cl['g_MHz']:.1f})",
                    rotation=90, va="top", ha="right", fontsize=6, color=INK2)
        ax.axvline(0.0, color=INK, lw=0.9)
        _style(ax, "collision map (dashed: audit-predicted resonances)",
               "pump offset − measured resonance (MHz)", "time-averaged population")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        # trajectories on resonance
        ax = axs[r, 2]
        j = int(np.argmin(np.abs(R["narrow_offsets_GHz"] - 1e-3 * c)))
        ax.plot(t, R["narrow"][j, i10, :], color=SERIES[0], lw=2, label="P10")
        for k, ch in enumerate(LEAK_CHANNELS):
            ax.plot(t, R["narrow"][j, channel_index(ch), :], color=SERIES[k + 1], lw=1.4,
                    label=ch)
        ax.set_yscale("symlog", linthresh=1e-3)
        _style(ax, "on resonance, from |01⟩", "time (ns)", "population")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(doc["title"], fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def prior_calibration(prior_dir: str, delta: float, eta: float) -> Optional[float]:
    """The carrier the 2026-09-25 flat-top study calibrated (MHz), for a cross-check."""
    for f in glob.glob(os.path.join(prior_dir, f"d{delta * 1e3:+.0f}MHz_eta*.json")):
        with open(f) as fh:
            d = json.load(fh)
        for p in d.get("points", []):
            if (p.get("knob") == "A" and p.get("ok") and "carrier_cal" in p
                    and abs(p["eta_flat"] - eta) < 1e-3
                    and not p["carrier_cal"].get("railed")):
                return 1e3 * float(p["carrier_GHz"])
    return None


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=os.path.join(REPO, "devices", "6Gate4.7SNAIL.json"))
    ap.add_argument("--deltas", default="0.015,0.04,0.1,-0.045,-0.12")
    ap.add_argument("--eta", type=float, default=1.3, help="scan target_eta the law came from")
    ap.add_argument("--eta-flat", default="0.78,1.04,1.3", help="constant drives |eta|")
    ap.add_argument("--branch", default="above")
    ap.add_argument("--levels", type=int, default=9)
    ap.add_argument("--laws", default=DEFAULT_LAWS)
    ap.add_argument("--prior", default=os.path.join(REPO, "results", "plateau_study",
                                                    "2026-09-25"))
    ap.add_argument("--window-iswaps", type=float, default=2.0)
    ap.add_argument("--dt-ns", type=float, default=0.25)
    ap.add_argument("--narrow-MHz", type=float, default=20.0)
    ap.add_argument("--narrow-points", type=int, default=41)
    ap.add_argument("--max-recentre", type=int, default=2,
                    help="rescans allowed to put the chevron in the middle of the window")
    ap.add_argument("--wide-MHz", type=float, default=250.0)
    ap.add_argument("--wide-points", type=int, default=101)
    ap.add_argument("--peak-floor", type=float, default=0.01,
                    help="min prominence (time-averaged population) for a collision peak")
    ap.add_argument("--match-tol-MHz", type=float, default=8.0)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "plateau_study", f"flat_chevrons_{_dt.date.today().isoformat()}"))
    ap.add_argument("--replot", default="",
                    help="redraw every <tag>.png from the <tag>.json in this dir; no solves")
    args = ap.parse_args(argv)

    if args.replot:
        for f in sorted(glob.glob(os.path.join(args.replot, "d*MHz_eta*.json"))):
            with open(f) as fh:
                doc = json.load(fh)
            for dr in doc["drives"]:
                dr["_raster"] = {k: np.asarray(v) for k, v in dr["_raster"].items()}
            plot_column(doc, f[:-5] + ".png")
            print(f"redrew {f[:-5]}.png", flush=True)
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(args.device) as fh:
        device = json.load(fh)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": 500000}
    summary = []
    for delta in parse_list(args.deltas):
        tag = f"d{delta * 1e3:+.0f}MHz_eta{eta_tag(args.eta)}"
        try:
            col = load_column(args.laws, delta, args.eta)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{tag}] skipped: {exc}", flush=True)
            continue
        cfg = column_config(device, col, args.levels, args.branch)
        law = col["chirp"]
        wp0 = float(col["operating_point"]["wp_offset_GHz"])
        drives = []
        for eta in parse_list(args.eta_flat):
            dr = study_drive(cfg, law, wp0, eta, args, solver)
            dr["carrier_prior_cal_MHz"] = prior_calibration(args.prior, delta, eta)
            drives.append(dr)
            print(f"[{tag}] eta={eta:.2f}: resonance {dr['carrier_meas_MHz']:+.2f} MHz "
                  f"(law {dr['carrier_law_MHz']:+.2f}, err {dr['law_error_MHz']:+.2f}; "
                  f"flat-top cal {dr['carrier_prior_cal_MHz']}), fit ok={dr['fit']['ok']}, "
                  f"P10max={dr['P10_max_on_res']:.3f}, "
                  f"{sum(p['match'] is None for p in dr['wide_peaks'])} unexplained / "
                  f"{len(dr['wide_peaks'])} leak peaks, {dr['seconds']:.0f}s", flush=True)
            summary.append({"delta_MHz": 1e3 * delta, "eta": eta,
                            **{k: dr[k] for k in ("carrier_law_MHz", "carrier_meas_MHz",
                                                  "law_error_MHz", "stark_shift_meas_MHz",
                                                  "stark_shift_law_MHz",
                                                  "carrier_prior_cal_MHz",
                                                  "P10_max_on_res",
                                                  "P10_at_t_iswap_on_res")},
                            "fit_ok": dr["fit"]["ok"],
                            "quality_ok": dr["quality"].get("ok"),
                            "wide_peaks": dr["wide_peaks"]})
        doc = {"title": f"Flat-only chevrons (no ramps, no chirp), δ = {delta * 1e3:+.0f} MHz",
               "delta_GHz": delta, "w_p_GHz": col["w_p_GHz"], "law": law,
               "wp_offset0_GHz": wp0, "drives": drives,
               "settings": dict(vars(args))}
        plot_column(doc, os.path.join(args.out, f"{tag}.png"))
        with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
            json.dump(_jsonable(doc), fh)
        # merge into any existing summary, so re-running some columns keeps the rest
        spath = os.path.join(args.out, "summary.json")
        prev = []
        if os.path.exists(spath):
            with open(spath) as fh:
                prev = json.load(fh)
        mine = {(round(x["delta_MHz"], 3), round(x["eta"], 4)) for x in summary}
        merged = [x for x in prev
                  if (round(x["delta_MHz"], 3), round(x["eta"], 4)) not in mine] + summary
        merged.sort(key=lambda x: (x["delta_MHz"], x["eta"]))
        with open(spath, "w") as fh:
            json.dump(_jsonable(merged), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
