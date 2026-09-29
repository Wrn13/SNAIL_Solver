#!/usr/bin/env python
r"""Frequency collisions on the post-Stark-shifted plateau, and what plateau shaping does to them.

For each subharmonic-scan column the gate is rebuilt as a flat-top ``SinePowerRamp``
(ramp / plateau / mirrored ramp) with a STATIC carrier parked on the Stark-shifted
resonance at the plateau drive -- no chirp, no DRAG. The ramps are fixed transients.
Per point it reports:

* the predicted collision map of ``H_flat`` (``piecewise_pulse.predict_plateau_channels``);
* the per-segment population budget, from ``|01>`` and ``|11>``;
* the plateau's spectral fingerprint (matrix pencil), matched to that map;
* the plateau switched on SUDDENLY from bare ``{|01>, |10>}``: its intrinsic
  collision fingerprint, independent of the ramps;
* ramp contamination: the same plateau reached through a ``--slow-factor`` slower
  ramp (the near-adiabatic reference);
* the full-gate iSWAP fidelity and leakage.

Knobs (A, C, D at fixed iSWAP area; B deliberately releases it):

    A  plateau drive eta_flat   (at fixed t_rise this also sets t_flat = A/eta - t_r),
       optionally x several ramp times t_rise
    B  plateau length at fixed eta_flat (``--flat-scale``)
    C  static carrier offset    around the Stark-compensated value
    D  plateau phase modulation Phi = A sin(w_m t) (--phasemod), exploratory

Every knob-A point first CALIBRATES its carrier on the flat top (max transfer,
``--cal-span-MHz``): at strong drive the quartic-dominated pass-A law misplaces the
plateau resonance by several MHz, and mistuned gates would measure the law's error
(reported as ``law_error_MHz``), not the collisions. Knob B then scans the plateau
length at fixed drive (length calibration and collision-period sync are one scan), and
C/D are centred on the calibrated carrier and the best B length.

The Stark law (``k2``, ``k4``, static ``wp_offset``) is READ from the pass-A column
caches of the running 5 MHz scan; nothing there is written. Outputs go to ``--out``.

    .venv/bin/python scripts/plateau_collision_study.py --deltas=-0.06,0.015,0.04 \
        --eta 1.3 --jobs 8
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import argparse
import datetime as _dt
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

INIT, TGT, DOUBLE = [0, 1, 0], [1, 0, 0], [1, 1, 0]
DEFAULT_LAWS = os.path.join(REPO, "results", "drag_curve_5MHz_2026-09-22",
                            "passA_nodrag", "columns")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def eta_tag(eta: float) -> str:
    return f"{float(eta):g}".replace(".", "p")


def load_column(laws_dir: str, delta_GHz: float, eta: float) -> Dict[str, Any]:
    """The pass-A column cache: Stark law + the scored bare/chirp gate. Read-only."""
    from snail_solver.subharmonic_convergence import delta_tag
    path = os.path.join(laws_dir, f"col_{delta_tag(delta_GHz)}_eta{eta_tag(eta)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no solved column cache at {path} (column refused, "
                                f"failed, or not yet run)")
    with open(path) as fh:
        col = json.load(fh)
    if not col.get("ok") or not col.get("chirp"):
        raise ValueError(f"{path}: column has no Stark law ({col.get('error')})")
    return col


def column_config(device: Dict[str, Any], col: Dict[str, Any], levels: int,
                  branch: str) -> Dict[str, Any]:
    from snail_solver.subharmonic_convergence import config_at_wp
    return config_at_wp(device, float(col["w_p_GHz"]), branch=branch, levels=levels)


# ---------------------------------------------------------------------------
# one point
# ---------------------------------------------------------------------------
def transfer(cfg, eta, t_r, carrier_GHz, t_g_ns, solver) -> float:
    """``|<10|U|01>|^2`` of one flat-top gate: the single-solve calibration metric."""
    from snail_solver import piecewise_pulse as pp
    cpl, t_g, _ = pp.build_plateau_gate(cfg, eta, t_r, carrier_GHz, t_g_ns=t_g_ns)
    psi = cpl.evolve_state(INIT, t_g, **solver)
    return float(abs(psi[cpl.fock_index(TGT)]) ** 2)


def calibrate_carrier(cfg, eta, t_r, carrier0_GHz, t_g_ns, solver, span_MHz: float,
                      points: int) -> Dict[str, Any]:
    """Park the carrier on the PLATEAU's own resonance: max transfer over a grid
    around the law's value, refined by one parabolic step on the top three points.
    (The law fits shaped-pulse chevrons and need not put a flat top on resonance.)
    """
    grid = carrier0_GHz + np.linspace(-span_MHz, span_MHz, int(points)) * 1e-3
    vals = [transfer(cfg, eta, t_r, c, t_g_ns, solver) for c in grid]
    k = int(np.argmax(vals))
    best_c, best_v = float(grid[k]), float(vals[k])
    if 0 < k < len(grid) - 1:
        x, y = grid[k - 1:k + 2], np.asarray(vals[k - 1:k + 2])
        den = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / den
        b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0])
             + x[0] ** 2 * (y[1] - y[2])) / den
        if a < 0:
            c_v = float(-b / (2 * a))
            v_v = transfer(cfg, eta, t_r, c_v, t_g_ns, solver)
            if v_v > best_v:
                best_c, best_v = c_v, v_v
    return {"carrier_GHz": best_c, "transfer": best_v,
            "railed": bool(k in (0, len(grid) - 1)),
            "scan_MHz": [float(1e3 * (c - carrier0_GHz)) for c in grid],
            "scan_transfer": [float(v) for v in vals]}


def study_point(job: Dict[str, Any]) -> Dict[str, Any]:
    from snail_solver import piecewise_pulse as pp

    t0 = time.perf_counter()
    cfg, eta, t_r = job["config"], float(job["eta_flat"]), float(job["t_rise_ns"])
    solver = job["solver"]
    t_g_over = job.get("t_g_ns")
    rec: Dict[str, Any] = {k: v for k, v in job.items() if k != "config"}
    try:
        t_g_nom = pp.plateau_t_g(cfg, eta, t_r) if t_g_over is None else float(t_g_over)
        if t_g_nom < 2.0 * t_r:
            raise ValueError(f"t_g={t_g_nom:.2f} ns < 2 t_rise")
    except ValueError as exc:
        rec.update(ok=False, error=str(exc))
        return rec
    carrier = float(job["carrier_GHz"])
    if job.get("calibrate"):
        cal = calibrate_carrier(cfg, eta, t_r, carrier, t_g_over, solver,
                                job["cal_span_MHz"], job["cal_points"])
        rec["carrier_law_GHz"] = carrier
        rec["carrier_cal"] = cal
        carrier = cal["carrier_GHz"]
        rec["carrier_GHz"] = carrier
    pm = None
    if job.get("phasemod"):
        A_rad, f_m = job["phasemod"]
        pm = pp.PlateauPhaseMod(A_rad, f_m, t_r, t_g_nom, edge_ns=job["pm_edge_ns"])
    cpl, t_g, _cfg = pp.build_plateau_gate(cfg, eta, t_r, carrier, phase_mod=pm,
                                           t_g_ns=t_g_over)
    env = cpl._pump_tones[0].envelope
    bounds = pp.segment_bounds(env)
    t_flat = bounds[1][1] - bounds[1][0]
    rec.update(t_g_ns=t_g, t_flat_ns=t_flat)

    pred = pp.predict_plateau_channels(cpl, t_g)
    rec["predicted"] = pred

    def ket(occ):
        v = np.zeros(cpl.dim, complex)
        v[cpl.fock_index(occ)] = 1.0
        return v

    def fingerprint(trajectory):
        try:
            return pp.plateau_fingerprint(cpl, trajectory, pred, INIT, TGT,
                                          tol_MHz=job["match_tol_MHz"])
        except (np.linalg.LinAlgError, ValueError) as exc:
            return {"error": str(exc)}

    # 1. |01>: dense plateau trajectory
    run = pp.evolve_piecewise(cpl, ket(INIT), bounds, dt_out={1: job["dt_flat_ns"]},
                              **solver)
    rec["budget_01"] = pp.segment_budget(cpl, run, INIT, TGT)
    if t_flat > 4 * job["dt_flat_ns"]:
        rec["fingerprint"] = fingerprint(run)
    # 2a. the plateau alone, switched on suddenly from bare |01>/|10>: its intrinsic
    #     collision fingerprint (no dressing), independent of ramp shape
    start = pp.channel_populations(cpl, run["boundary"][1], INIT, TGT)
    raw_end = pp.channel_populations(cpl, run["boundary"][2], INIT, TGT)
    sudden0 = pp.project_computational(cpl, run["boundary"][1], [INIT, TGT])
    run_s = pp.evolve_piecewise(cpl, sudden0, [bounds[1]], dt_out={0: job["dt_flat_ns"]},
                                **solver)
    s_start = pp.channel_populations(cpl, sudden0, INIT, TGT)
    s_end = pp.channel_populations(cpl, run_s["boundary"][-1], INIT, TGT)
    rec["fingerprint_sudden"] = fingerprint(
        {"times": [None, run_s["times"][0]], "states": [None, run_s["states"][0]]})
    # 2b. ramp contamination: the same plateau (drive, carrier, length) reached by a
    #     `slow_factor`-times slower ramp -- the near-adiabatic reference
    t_r_slow = job["slow_factor"] * t_r
    cpl_s, _tg, _ = pp.build_plateau_gate(cfg, eta, t_r_slow, carrier,
                                          t_g_ns=2.0 * t_r_slow + t_flat)
    b_s = pp.segment_bounds(cpl_s._pump_tones[0].envelope)
    run_a = pp.evolve_piecewise(cpl_s, ket(INIT), b_s[:2], **solver)
    a_start = pp.channel_populations(cpl_s, run_a["boundary"][1], INIT, TGT)
    a_end = pp.channel_populations(cpl_s, run_a["boundary"][2], INIT, TGT)
    rec["contamination"] = {
        ch: {"at_t_r": float(start[ch][0]), "at_t_r_adiabatic": float(a_start[ch][0]),
             "d_flat_raw": float(raw_end[ch][0] - start[ch][0]),
             "d_flat_adiabatic": float(a_end[ch][0] - a_start[ch][0]),
             "d_flat_sudden": float(s_end[ch][0] - s_start[ch][0])}
        for ch in ("P_leak", "P_coupler", "P_f_a", "P_f_b")}
    rec["t_rise_slow_ns"] = float(t_r_slow)
    # 3. |11>: the leakage path (|11> -> |02>, |20>) the gate also has to survive
    run11 = pp.evolve_piecewise(cpl, ket(DOUBLE), bounds, **solver)
    rec["budget_11"] = pp.segment_budget(cpl, run11, DOUBLE, TGT)
    # 4. the whole gate
    F, leak, _U = cpl.iswap_fidelity(0, 1, t_g, fit_virtual_z=True, **solver)
    rec.update(F_avg=float(F), leakage=float(leak), ok=True,
               seconds=time.perf_counter() - t0)
    return rec


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def dominant(pred: List[Dict[str, Any]], n: int = 4) -> List[Dict[str, Any]]:
    """Strongest parasites by the sudden-switch population bound P_max."""
    return sorted([p for p in pred if p["category"] != "target"],
                  key=lambda p: -p["P_max"])[:n]


def sync_factor(p: Dict[str, Any], t_flat_ns: float) -> float:
    """``P_max sin^2(pi W t_flat)``: 0 when the plateau is a whole number of periods."""
    return float(p["P_max"] * np.sin(np.pi * p["W_MHz"] * 1e-3 * t_flat_ns) ** 2)


def summarise(r: Dict[str, Any]) -> Dict[str, Any]:
    if not r.get("ok"):
        return {k: r.get(k) for k in ("knob", "eta_flat", "t_rise_ns", "carrier_GHz",
                                      "phasemod", "ok", "error")}
    b = r["budget_01"]
    dom = dominant(r["predicted"])
    return {
        "knob": r["knob"], "eta_flat": r["eta_flat"], "t_rise_ns": r["t_rise_ns"],
        "carrier_MHz": 1e3 * r["carrier_GHz"], "carrier_shift_MHz": r.get("dc_MHz", 0.0),
        "phasemod": r.get("phasemod"), "t_g_ns": r["t_g_ns"], "t_flat_ns": r["t_flat_ns"],
        "law_error_MHz": (1e3 * (r["carrier_GHz"] - r["carrier_law_GHz"])
                          if "carrier_law_GHz" in r else None),
        "cal_railed": r.get("carrier_cal", {}).get("railed"),
        "F_avg": r["F_avg"], "infidelity": 1.0 - r["F_avg"], "leakage": r["leakage"],
        "dP_leak_up": b["P_leak"]["d_ramp_up"], "dP_leak_flat": b["P_leak"]["d_flat"],
        "dP_leak_down": b["P_leak"]["d_ramp_down"],
        "dP_leak_flat_adiabatic": r["contamination"]["P_leak"]["d_flat_adiabatic"],
        "dP_leak_flat_sudden": r["contamination"]["P_leak"]["d_flat_sudden"],
        "P_leak_at_t_r": r["contamination"]["P_leak"]["at_t_r"],
        "P_leak_at_t_r_adiabatic": r["contamination"]["P_leak"]["at_t_r_adiabatic"],
        "dP_leak11_flat": r["budget_11"]["P_leak"]["d_flat"],
        "dominant": [{"name": p["name"], "g_MHz": p["g_MHz"],
                      "detuning_MHz": p["detuning_MHz"],
                      "dressed_detuning_MHz": p["dressed_detuning_MHz"],
                      "W_MHz": p["W_MHz"], "P_max": p["P_max"],
                      "sync": sync_factor(p, r["t_flat_ns"])} for p in dom],
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot_column(doc: Dict[str, Any], png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in doc["summary"] if r.get("F_avg") is not None]
    A = [r for r in rows if r["knob"] == "A"]
    C = [r for r in rows if r["knob"] == "C"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    # (0,0) A: plateau leakage by segment vs eta_flat, one line per t_rise
    ax = axs[0, 0]
    for i, tr in enumerate(sorted({r["t_rise_ns"] for r in A})):
        rr = sorted([r for r in A if r["t_rise_ns"] == tr], key=lambda r: r["eta_flat"])
        x = [r["eta_flat"] for r in rr]
        ax.semilogy(x, [max(abs(r["dP_leak_flat"]), 1e-7) for r in rr], "-o", ms=4,
                    lw=2, color=SERIES[i], label=f"plateau, t_r={tr:g} ns")
        ax.semilogy(x, [max(abs(r["dP_leak_flat_adiabatic"]), 1e-7) for r in rr], "--",
                    lw=1.5, color=SERIES[i], label="plateau, slow-ramp reference")
    _style(ax, "A: plateau leakage from |01>", "plateau drive η_flat", "|ΔP_leak| on plateau")
    ax.legend(fontsize=7, frameon=False)

    # (0,1) A: dressed detuning of dominant channels vs eta_flat
    ax = axs[0, 1]
    tr0 = doc["settings"]["t_rise_ns"][0]
    rr = sorted([r for r in A if r["t_rise_ns"] == tr0], key=lambda r: r["eta_flat"])
    names: List[str] = []
    for r in rr:
        for d in r["dominant"]:
            if d["name"] not in names:
                names.append(d["name"])
    for i, nm in enumerate(names[:len(SERIES)]):
        pts = [(r["eta_flat"], d["detuning_MHz"]) for r in rr
               for d in r["dominant"] if d["name"] == nm]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, "-o", ms=4, lw=2, color=SERIES[i], label=nm)
    ax.axhline(0.0, color=INK2, lw=0.8)
    _style(ax, f"A: where the collisions sit (t_r={tr0:g} ns)", "plateau drive η_flat",
           "detuning from the playing pump Δ_j (MHz)")
    ax.legend(fontsize=7, frameon=False)

    # (1,0) B: plateau length at fixed drive
    ax = axs[1, 0]
    Bs = sorted([r for r in rows if r["knob"] == "B"], key=lambda r: r["t_flat_ns"])
    if Bs:
        x = [r["t_flat_ns"] for r in Bs]
        ax.semilogy(x, [r["infidelity"] for r in Bs], "-o", ms=4, lw=2, color=SERIES[0],
                    label="gate infidelity 1-F")
        ax.semilogy(x, [max(abs(r["dP_leak_flat"]), 1e-7) for r in Bs], "-o", ms=4,
                    lw=2, color=SERIES[1], label="|ΔP_leak| on plateau")
        ax.semilogy(x, [max(sum(d["sync"] for d in r["dominant"]), 1e-7) for r in Bs],
                    "--", lw=1.5, color=SERIES[2],
                    label="Σ P_max sin²(πW t_flat), two-level model")
        ax.legend(fontsize=7, frameon=False)
    _style(ax, f"B: plateau length at η_flat={doc.get('best_eta_flat', float('nan')):.3g}",
           "t_flat (ns)", "error / population")

    # (1,1) C: carrier trade-off
    ax = axs[1, 1]
    if C:
        C = sorted(C, key=lambda r: r["carrier_shift_MHz"])
        x = [r["carrier_shift_MHz"] for r in C]
        ax.semilogy(x, [r["infidelity"] for r in C], "-o", ms=4, lw=2, color=SERIES[0],
                    label="gate infidelity 1-F")
        ax.semilogy(x, [max(r["leakage"], 1e-7) for r in C], "-o", ms=4, lw=2,
                    color=SERIES[1], label="gate leakage")
        ax.legend(fontsize=7, frameon=False)
    _style(ax, f"C: carrier offset at η_flat={doc.get('best_eta_flat', float('nan')):.3g}",
           "carrier − calibrated plateau resonance (MHz)", "error")
    fig.suptitle(doc["title"], fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
def parse_list(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi, n = part.split(":")
            out.extend(np.linspace(float(lo), float(hi), int(n)).tolist())
        else:
            out.append(float(part))
    return out


def run_pool(jobs: List[Dict[str, Any]], n_jobs: int) -> List[Dict[str, Any]]:
    if n_jobs <= 1:
        return [study_point(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(study_point, jobs))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=os.path.join(REPO, "devices", "6Gate4.7SNAIL.json"))
    ap.add_argument("--deltas", required=True, help="column offsets delta (GHz), comma list")
    ap.add_argument("--eta", type=float, default=1.3, help="scan target_eta the law came from")
    ap.add_argument("--branch", default="above")
    ap.add_argument("--levels", type=int, default=9, help="coupler levels")
    ap.add_argument("--laws", default=DEFAULT_LAWS, help="pass-A column cache dir (read-only)")
    ap.add_argument("--eta-flat", default="0.5:1.0:11",
                    help="knob A grid, as FRACTIONS of --eta (the law's fitted range "
                         "tops out at 1.0)")
    ap.add_argument("--t-rise", default="15", help="knob B: ramp times (ns), comma list")
    ap.add_argument("--cal-span-MHz", type=float, default=12.0,
                    help="carrier calibration: +- span around the law's value")
    ap.add_argument("--cal-points", type=int, default=9)
    ap.add_argument("--flat-scale", default="0.7:1.3:13",
                    help="knob B: plateau length as multiples of the area-derived one")
    ap.add_argument("--carrier-span-MHz", type=float, default=4.0)
    ap.add_argument("--carrier-points", type=int, default=13)
    ap.add_argument("--phasemod", default="",
                    help="knob D grid 'A_lo:A_hi:nA/fm_lo:fm_hi:nf' (rad / MHz); empty = off")
    ap.add_argument("--pm-edge-ns", type=float, default=5.0)
    ap.add_argument("--slow-factor", type=float, default=3.0,
                    help="ramp-contamination reference: ramp this many times slower")
    ap.add_argument("--dt-flat-ns", type=float, default=0.25)
    ap.add_argument("--match-tol-MHz", type=float, default=8.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "plateau_study", _dt.date.today().isoformat()))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    with open(args.device) as fh:
        device = json.load(fh)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": 500000}
    from snail_solver import piecewise_pulse as pp

    t_rises = parse_list(args.t_rise)
    eta_fracs = parse_list(args.eta_flat)
    gains = []
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
        carrier = lambda e: pp.plateau_carrier_GHz(wp0, law["k2"], law["k4"], e)  # noqa: E731
        base = dict(config=cfg, solver=solver, dt_flat_ns=args.dt_flat_ns,
                    cal_span_MHz=args.cal_span_MHz, cal_points=args.cal_points,
                    slow_factor=args.slow_factor,
                    match_tol_MHz=args.match_tol_MHz, pm_edge_ns=args.pm_edge_ns)
        print(f"[{tag}] law k2={law['k2']:.3f} k4={law['k4']:.3f} MHz, "
              f"wp0={wp0 * 1e3:.3f} MHz", flush=True)

        # --- A: plateau drive (x ramp time), carrier calibrated on each flat top
        jobs = [dict(base, knob="A", eta_flat=f * args.eta, t_rise_ns=tr,
                     carrier_GHz=carrier(f * args.eta), calibrate=True)
                for tr in t_rises for f in eta_fracs]
        t0 = time.perf_counter()
        res = run_pool(jobs, args.jobs)
        okA = [r for r in res if r.get("ok")]
        if not okA:
            print(f"[{tag}] knob A produced no gate", flush=True)
            continue
        best = min(okA, key=lambda r: 1.0 - r["F_avg"])
        print(f"[{tag}] A: {len(okA)}/{len(res)} ok in {time.perf_counter() - t0:.0f}s; "
              f"best eta_flat={best['eta_flat']:.3f} t_r={best['t_rise_ns']:g} "
              f"F={best['F_avg']:.5f} (law off by "
              f"{1e3 * (best['carrier_GHz'] - best['carrier_law_GHz']):+.2f} MHz)",
              flush=True)
        at_best = dict(base, eta_flat=best["eta_flat"], t_rise_ns=best["t_rise_ns"])
        c_best = float(best["carrier_GHz"])

        # --- B: plateau length at fixed drive (length calibration + collision sync).
        #     Amplitude stays at eta_flat, so the rotation angle moves with t_flat.
        t_fl0 = float(best["t_flat_ns"])
        jobs = [dict(at_best, knob="B", carrier_GHz=c_best,
                     t_g_ns=2.0 * best["t_rise_ns"] + t_fl0 * float(x))
                for x in parse_list(args.flat_scale)]
        res += run_pool(jobs, args.jobs)
        okB = [r for r in res if r.get("ok") and r["knob"] in ("A", "B")
               and r["eta_flat"] == best["eta_flat"]
               and r["t_rise_ns"] == best["t_rise_ns"]]
        bestB = min(okB, key=lambda r: 1.0 - r["F_avg"])
        t_g_B = None if bestB["knob"] == "A" else float(bestB["t_g_ns"])

        # --- C: carrier offset around the CALIBRATED resonance
        dcs = np.linspace(-args.carrier_span_MHz, args.carrier_span_MHz,
                          args.carrier_points)
        jobs = [dict(at_best, knob="C", dc_MHz=float(dc), t_g_ns=t_g_B,
                     carrier_GHz=c_best + dc * 1e-3) for dc in dcs]
        res += run_pool(jobs, args.jobs)

        # --- D: plateau phase modulation (exploratory), carrier re-calibrated because
        #     the modulation's sidebands pull the target resonance too
        if args.phasemod:
            a_spec, f_spec = args.phasemod.split("/")
            jobs = [dict(at_best, knob="D", carrier_GHz=c_best, t_g_ns=t_g_B,
                         calibrate=True, phasemod=[float(a), float(fm) * 1e-3])
                    for a in parse_list(a_spec) for fm in parse_list(f_spec)]
            res += run_pool(jobs, args.jobs)

        summary = [summarise(r) for r in res]
        good = [s for s in summary if s.get("F_avg") is not None]
        best_all = min(good, key=lambda s: s["infidelity"])
        doc = {"title": f"Plateau collisions, δ = {delta * 1e3:+.0f} MHz, "
                        f"law from η* = {args.eta:g}",
               "delta_GHz": delta, "w_p_GHz": col["w_p_GHz"], "law": law,
               "wp_offset0_GHz": wp0, "best_eta_flat": best["eta_flat"],
               "settings": {"t_rise_ns": t_rises, "eta_fracs": eta_fracs,
                            "levels": args.levels, "branch": args.branch,
                            "solver": solver, "laws": args.laws},
               "best": best_all, "summary": summary, "points": res}
        with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
            json.dump(doc, fh, indent=1, default=float)
        plot_column(doc, os.path.join(args.out, f"{tag}.png"))

        gains.append({"delta_MHz": delta * 1e3, "eta": args.eta,
                      "bare_F": col.get("flat", {}).get("F_avg"),
                      "chirp_F": col.get("fidelity", {}).get("F_avg"),
                      "chirp_t_g_ns": col.get("fidelity", {}).get("t_g_ns"),
                      "plateau_best": {k: best_all[k] for k in
                                       ("knob", "eta_flat", "t_rise_ns", "carrier_shift_MHz",
                                        "phasemod", "t_g_ns", "F_avg", "leakage")}})
        g = gains[-1]
        print(f"[{tag}] bare F={g['bare_F']}, chirp F={g['chirp_F']}, "
              f"best plateau F={best_all['F_avg']:.5f} ({best_all['knob']})", flush=True)

    with open(os.path.join(args.out, "gain_vs_existing.json"), "w") as fh:
        json.dump(gains, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
