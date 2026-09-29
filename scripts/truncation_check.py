"""Coupler-level convergence at the columns a paper would quote.

`predicted_levels` is a scaling argument, not a bound: on this device it called for
1.9 levels where the observable still moved 3x between 7 and 9. So the quoted points
get an explicit table: each eta's best column replayed at several cutoffs (same t_g,
amplitude, chirp and channels), so only the Hilbert space changes.

Usage:  truncation_check.py RUN.h5[,RUN2.h5...] [LEVELS] [WORKERS]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

SRC = sys.argv[1].split(",")
LEVELS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                           else ["7", "9", "11", "13"])]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def _one(job):
    import warnings
    warnings.filterwarnings("ignore")
    from snail_solver.envelope import DragChannel
    from snail_solver.subharmonic_convergence import config_at_wp
    from snail_solver.tune_up_sweep import score_gate
    r, dev, lev = job["row"], job["device"], job["levels"]
    chirp = [float(x) for x in r["chirp"]["coeffs_GHz"]]
    chans = [DragChannel(beat_GHz=float(c["beat_GHz"]), n_pump=int(c["n_pump"]),
                         n_photon=int(c.get("n_photon", c["n_pump"])),
                         quotient_rule=True)
             for c in (r.get("drag_channels") or [])]
    cfg = config_at_wp(dev, r["w_p_GHz"], branch=r["branch"], levels=lev,
                       chirp_coeffs_GHz=chirp)
    g = score_gate(cfg, r["operating_point"], chirp, drag_channels=chans)
    b = score_gate(cfg, r["operating_point"], [], drag_channels=None)
    return {"delta_GHz": r["delta_GHz"], "target_eta": r["target_eta"],
            "levels": lev, "inf_drag": 1.0 - g["F_avg"],
            "inf_bare": 1.0 - b["F_avg"], "leakage": g["leakage"]}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from snail_solver.h5_io import load_doc

    jobs = []
    for s in SRC:
        doc = load_doc(s)
        scan = doc.get("scan") or doc
        dev = scan["device"]
        ok = [r for r in scan.get("rows", []) if r.get("ok")]
        if not ok:
            continue
        # The column a paper would quote: the best coherent infidelity at this eta.
        best = min(ok, key=lambda r: 1.0 - r["fidelity"]["F_avg"])
        jobs += [{"row": best, "device": dev, "levels": lev} for lev in LEVELS]
    print(f"{len(jobs)} solves ({len(jobs) // max(len(LEVELS), 1)} column(s) x "
          f"{len(LEVELS)} cutoffs) on {WORKERS} workers", flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(_one, jobs):
            out.append(res)
            print(f"  eta={res['target_eta']:.1f} delta={res['delta_GHz']*1e3:+.0f} "
                  f"levels={res['levels']:>3}  1-F={res['inf_drag']:.4e}", flush=True)

    print(f"\n{'eta':>5} {'delta':>7} " + " ".join(f"{l:>11}" for l in LEVELS)
          + f" {'converged?':>28}")
    by = {}
    for r in out:
        by.setdefault((r["target_eta"], r["delta_GHz"]), {})[r["levels"]] = r
    for (eta, dm), d in sorted(by.items()):
        vals = [d[l]["inf_drag"] if l in d else float("nan") for l in LEVELS]
        top = [v for l, v in zip(LEVELS, vals) if l >= LEVELS[-2]]
        if len(top) > 1:
            rel = abs(top[-1] - top[0]) / max(top[-1], 1e-30)
            verdict = f"last two differ by {rel:.1%}"
        else:
            verdict = "need >= 2 cutoffs"
        print(f"{eta:>5.1f} {dm*1e3:>+7.0f} "
              + " ".join(f"{v:>11.4e}" for v in vals) + f" {verdict:>28}")
    json.dump(out, open("truncation_check.json", "w"), indent=1)
    print("\nwrote truncation_check.json")
