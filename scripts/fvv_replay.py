"""Replay stored pulses in the full model and in FVV effective models, side by side.

    fvv_replay.py --delta -0.14 --peaks 0.4,0.7,1.0,1.3 --orders 1,2,3,4 [--kicks]
                  [--levels 12] [--bare] [--out map.json]

For each column and peak |eta| (default: the stored target), the stored chirped pulse
(or, with --bare, the flat one) is replayed in the full model and in FVV models of each
order, on the same Fock space. The FVV gate is eq. (3) in the charge frame,
U = R e^{-i Lambda(t_g)} W e^{+i Lambda(0)} R^dag, with F(|eta(t)|) integrated for W.
The report gives 1-F of each, and the distance of each FVV gate from the full model's.
That distance does not depend on calibration, so it is the convergence measure away
from the calibrated amplitude.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))


def _floats(s):
    return [float(v) for v in s.split(",")] if s else []


def run(job):
    import warnings
    warnings.filterwarnings("ignore")
    from floquet import model as M, propagator as P, target
    cpl, t_g = target.pulse(target.column(job["delta"]), bare=job["bare"], peak_eta=job["peak"],
                            levels=job["levels"])
    t = time.time()
    model = None if job["order"] is None else M.for_coupler(cpl, cache_dir=job["cache"], order=job["order"])
    try:
        r = P.replay(cpl, t_g, model, kicks=job["kick"])
    except Exception as e:                       # e.g. the integrator on a stiff F
        return dict(job, inf=float("nan"), leak=float("nan"), sec=time.time() - t,
                    U=[[float("nan"), 0.0]] * 16, error=f"{type(e).__name__}: {e}"[:200])
    return dict(job, inf=1 - r["F_avg"], leak=r["leakage"], sec=time.time() - t,
                U=[[z.real, z.imag] for z in r["U"].ravel()])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delta", default="-0.14", help="column(s), GHz, comma-separated")
    ap.add_argument("--peaks", default="", help="peak |eta| values (default: stored target)")
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--kicks", action="store_true", help="also replay with the end kicks")
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--levels", type=int, default=None, help="coupler truncation")
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "fvv_cache"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import numpy as np
    from floquet import model as M, propagator as P, target
    orders = [int(v) for v in a.orders.split(",") if v]
    jobs = []
    for d in _floats(a.delta):
        cpl, _ = target.pulse(target.column(d), levels=a.levels)
        for o in orders:                          # build (and cache) the models up front
            M.for_coupler(cpl, cache_dir=a.cache, order=o)
        for p in (_floats(a.peaks) or [None]):
            base = dict(delta=d, peak=p, bare=a.bare, levels=a.levels, cache=a.cache)
            if not a.no_full:
                jobs.append(dict(base, order=None, kick=False))
            for o in orders:
                jobs.append(dict(base, order=o, kick=False))
                if a.kicks and o > 1:
                    jobs.append(dict(base, order=o, kick=True))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(run, jobs))
    U = lambda r: np.array([complex(x, y) for x, y in r["U"]]).reshape(4, 4)
    print(f"{'delta':>7} {'peak':>6} {'model':>12} {'1-F':>10} {'distance to full':>17}")
    for r in res:
        full = next((q for q in res if q["order"] is None and q["delta"] == r["delta"]
                     and q["peak"] == r["peak"]), None)
        r["distance"] = P.gate_distance(U(r), U(full)) if full and r["order"] else None
        name = "full" if r["order"] is None else f"FVV {r['order']}" + (" +kick" if r["kick"] else "")
        dist = f"{r['distance']:.2e}" if r["distance"] is not None else ""
        peak = "stored" if r["peak"] is None else f"{r['peak']}"
        print(f"{r['delta']:>7} {peak:>6} {name:>12} {r['inf']:>10.4e} {dist:>17}"
              + (f"   [{r['error']}]" if r.get("error") else ""))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
