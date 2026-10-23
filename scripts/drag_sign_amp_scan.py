"""Gate fidelity vs peak drive |eta| at -185..-160 MHz, with the DRAG sign as played
and fixed.

Per (column, peak |eta|, variant) the pulse is rebuilt from STORED measurements only
-- the column's passA_ridge Rabi table (no chevron is re-solved):

    chirp    the ridge law (``ridge_chirp.law_from_table``) projected at this |eta|
             (``tune_up.chirp_from_measured_shift``); carrier = the law's static
             part + pulse mean (step 3's shaped-chevron residual is replaced by
             the carrier refinement below)
    bare     no chirp, no DRAG, the same carrier as `chirp`
    drag     the chirp with the stored subharm-leak channels at their PLAYED beats,
             chirp<->DRAG coupled at t_g0(|eta|) as the tune-up does
    fixed    the same with the fixed-audit beats

and every variant then gets its OWN length fit (``score_gate(refit_length=True)``,
the length scan bare traces get), with ``|eta|`` held at the target, and then its
carrier refined for F_avg within +/- 1/(2 t_g) at that length (`maximize_1d`; the
law-carrier score is kept as ``*_law_carrier``). Chirp shape follows the measured
ridge; length and carrier are the only things tuned, the same for every variant.

Results append to OUTDIR/amp_scan.jsonl as they finish (re-running skips done jobs).

Usage:  drag_sign_amp_scan.py OUTDIR [WORKERS] [ETAS] [DELTAS_MHz]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VARIANTS = ("bare", "chirp", "drag", "fixed")
SOLVER = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}     # score_gate's default


def _job_key(delta, eta, variant):
    return f"{delta:+.0f}|{eta:.3f}|{variant}"


def pulse_for(delta, eta, variant):
    """(cfg, record, chirp, channels) for one job."""
    import numpy as np  # noqa: F401
    import drag_sign_common as D
    from snail_solver.ridge_chirp import law_from_table
    from snail_solver.ridge_refit import load_npz_rabi
    from snail_solver.tune_up import (chirp_from_measured_shift, nominal_t_g,
                                      shape_config, DragFixedPointDiverged)

    c = D.column(delta)
    st = c["settings"]
    cfg = c["variants"]["chirp"]["cfg"]
    table = load_npz_rabi(os.path.join(D.RES, "passA_ridge", "columns",
                                       f"col_{D.col_tag(delta)}_rabi.npz"))
    t_g0 = nominal_t_g(cfg, eta)
    law = law_from_table(table, cfg, eta, probe_shape=st["probe_shape"],
                         moment_weighting=st["moment_weighting"], t_g_ns=t_g0,
                         contrast_min=float(st["contrast_min"]))
    chans = {"bare": None, "chirp": None,
             "drag": c["variants"]["drag"]["chans"],
             "fixed": c["variants"]["fixed"]["chans"]}[variant]
    shape, shape_kw = shape_config(cfg)
    kw = dict(degree=int(st["chirp_degree"]), t_g=t_g0, shape=shape, shape_kw=shape_kw,
              max_iters=int(st["chirp_max_passes"]), ridge_law=law,
              drag_channels=chans)
    decoupled = False
    try:
        proj = chirp_from_measured_shift(table, eta, couple_drag=True, **kw)
    except DragFixedPointDiverged:
        proj = chirp_from_measured_shift(table, eta, couple_drag=False, **kw)
        decoupled = True
    chirp = [] if variant == "bare" else [float(x) for x in proj["coeffs_GHz"]]
    rec = {"target_eta": float(eta), "t_g_ns": float(t_g0),
           "wp_offset_GHz": float(proj["mean_shift_GHz"]), "spec_abs_GHz": None,
           "drag_beat_GHz": None, "drag_n_pump": 1}
    info = {"railed": bool(law["railed"]), "extrapolation": float(eta) > float(
                law["measured_eta_max"]) + 1e-9,
            "wp_offset_GHz": rec["wp_offset_GHz"], "chirp_GHz": chirp,
            "drag_decoupled": decoupled,
            "beats_GHz": None if chans is None else [float(x.beat_GHz) for x in chans]}
    return cfg, rec, chirp, chans, st, info


def run_job(job):
    import warnings
    warnings.filterwarnings("ignore")
    from snail_solver.device_utils import build_coupler, maximize_1d
    from snail_solver.tune_up_sweep import score_gate
    delta, eta, variant = job
    t0 = time.perf_counter()
    try:
        cfg, rec, chirp, chans, st, info = pulse_for(delta, eta, variant)
        s = score_gate(cfg, rec, chirp, drag_channels=chans, refit_length=True,
                       tg_points=int(st["tg_points"]), tg_lo=float(st["tg_lo"]),
                       tg_hi=float(st["tg_hi"]))
        # stand-in for step 3: refine the carrier at the fitted length, +/- one
        # linewidth 1/(2 t_g) about the law's carrier
        t_g, amp, wp0 = float(s["t_g_ns"]), float(s["amp_scale"]), rec["wp_offset_GHz"]

        def fid(wp):
            cpl, _w, _p = build_coupler(cfg, t_g, amp, wp, None, None,
                                        chirp_coeffs_GHz=chirp, drag_channels=chans)
            return cpl.iswap_fidelity(0, 1, t_g, **SOLVER)[0]

        half = 1.0 / (2.0 * t_g)
        wp_best, F_best, n_eval = maximize_1d(fid, wp0 - half, wp0 + half,
                                              n_points=7, n_refine=2)
        cpl, _w, _p = build_coupler(cfg, t_g, amp, wp_best, None, None,
                                    chirp_coeffs_GHz=chirp, drag_channels=chans)
        F, leak, _U = cpl.iswap_fidelity(0, 1, t_g, **SOLVER)
        out = {"F_avg": float(F), "leakage": float(leak), "t_g_ns": t_g,
               "amp_scale": amp, "peak_eta": s["peak_eta"],
               "wp_refined_GHz": float(wp_best), "n_carrier_evals": int(n_eval),
               "F_avg_law_carrier": s["F_avg"], "leakage_law_carrier": s["leakage"],
               "transfer_law_carrier": s["transfer"],
               "n_drag_played": s["n_drag_channels"], **info, "error": None}
    except Exception as exc:                       # record, never drop, a failure
        out = {"error": f"{type(exc).__name__}: {exc}"}
    return {"key": _job_key(delta, eta, variant), "delta_MHz": delta, "eta": eta,
            "variant": variant, "seconds": time.perf_counter() - t0, **out}


def main():
    OUTDIR = sys.argv[1]
    WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    ETAS = ([float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3
            else [0.6, 0.8, 1.0, 1.1, 1.2, 1.3])
    DELTAS = ([float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4
              else [-185, -180, -175, -170, -165, -160])
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "amp_scan.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path) if l.strip()}
    # the longest (DRAG, strong drive) first, so the pool drains evenly
    jobs = [(d, e, v) for e in sorted(ETAS, reverse=True) for v in VARIANTS[::-1]
            for d in DELTAS if _job_key(d, e, v) not in done]
    print(f"{len(jobs)} jobs ({len(done)} already done) on {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex, open(path, "a") as f:
        futs = [ex.submit(run_job, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            msg = (f"1-F={1 - r['F_avg']:.3e} leak={r['leakage']:.2e} "
                   f"t_g={r['t_g_ns']:.1f}" if r.get("error") is None else r["error"])
            print(f"[{i}/{len(jobs)}] {r['key']:22s} {msg} ({r['seconds']:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
