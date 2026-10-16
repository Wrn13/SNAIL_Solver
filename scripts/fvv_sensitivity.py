"""Sensitivity of the gate to each process in the FVV effective Hamiltonian F.

    fvv_sensitivity.py --delta -0.14 --order 4 --peaks 0.7,1.0,1.3 [--eps 0.05] [--out s.json]

F^(n) is split by process: the change Delta of the mode occupations it makes. In the
charge frame that fixes the pump number, so Delta names the process. Each (order,
process) group, together with its Hermitian conjugate, is scaled by (1 +/- eps), and
the stored chirped pulse is replayed. Two measures per group:

  chi              leading-order average infidelity between the two scaled gates, per
                   (d ln c)^2. It measures how much the gate depends on the term and does
                   not depend on calibration.
  d(1-F)/d ln c    how the gate error moves with the term. Meaningful only where the
                   pulse is calibrated (the stored target |eta|).

Linear sensitivities miss strongly nonlinear, near-resonant processes (3 w_p = w_s);
knock those out instead.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))


def run(job):
    import warnings
    warnings.filterwarnings("ignore")
    from floquet import model as M, propagator as P, target
    cpl, t_g = target.pulse(target.column(job["delta"]), peak_eta=job["peak"])
    model = M.for_coupler(cpl, cache_dir=job["cache"], order=job["order"])
    scale = {(o, tuple(d)): 1 + job["sign"] * job["eps"] for o, d in job["keys"]}
    r = P.replay(cpl, t_g, model, kicks=job["kicks"], scale=scale)
    return dict(peak=job["peak"], group=job["group"], sign=job["sign"], inf=1 - r["F_avg"],
                U=[[z.real, z.imag] for z in r["U"].ravel()])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delta", type=float, default=-0.14)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--peaks", default="0.7,1.0,1.3")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--kicks", action="store_true")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "fvv_cache"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import numpy as np
    from floquet import fourier as Fo, model as M, propagator as P, target
    cpl, _ = target.pulse(target.column(a.delta))
    model = M.for_coupler(cpl, cache_dir=a.cache, order=a.order)
    groups = {}
    for tm in model.terms:                         # a process and its h.c. form one group
        canon = max(tm.delta, tuple(-d for d in tm.delta))
        groups.setdefault((tm.order, canon), set()).update(
            {(tm.order, tm.delta), (tm.order, tuple(-d for d in tm.delta))})
    peaks = [float(v) for v in a.peaks.split(",")]
    jobs = [dict(delta=a.delta, peak=p, group=str(g), keys=sorted(k), sign=s, eps=a.eps,
                 kicks=a.kicks, cache=a.cache, order=a.order)
            for p in peaks for g, k in groups.items() for s in (+1, -1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(run, jobs))
    U = lambda r: np.array([complex(x, y) for x, y in r["U"]]).reshape(4, 4)
    by = {}
    for r in res:
        by.setdefault((r["group"], r["peak"]), {})[r["sign"]] = r
    idx = cpl._subspace_indices(0, 1)
    rows = []
    for g, keys in groups.items():
        sig = M.process(g[1])
        row = {"order": g[0], "process": M.describe(g[1]),
               "carrier_MHz": Fo.carrier(sig, model.freqs_GHz) * 1e3}
        for p in peaks:
            d = by[(str(g), p)]
            row[f"chi@{p}"] = P.susceptibility(U(d[1]), U(d[-1]), a.eps)
            row[f"dF@{p}"] = (d[1]["inf"] - d[-1]["inf"]) / (2 * a.eps)
            row[f"size_MHz@{p}"] = 1e3 / (2 * np.pi) * max(
                abs(t.matrix.toarray()[:, idx]).max() * p ** t.power
                for t in model.terms if (t.order, t.delta) in keys)
        rows.append(row)
    top = peaks[-1]
    rows.sort(key=lambda r: -r[f"chi@{top}"])
    print(f"{'ord':>3} {'process':18s} {'carrier':>8} " + " ".join(f"{'chi@' + str(p):>9}" for p in peaks)
          + f" {'dF@' + str(top):>10}")
    for r in rows[:30]:
        print(f"{r['order']:>3} {r['process']:18s} {r['carrier_MHz']:+8.0f} "
              + " ".join(f"{r['chi@' + str(p)]:9.2e}" for p in peaks) + f" {r['dF@' + str(top)]:+10.2e}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
