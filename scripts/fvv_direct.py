"""Eq. (3) of Bock, Ernst & Uhrig in closed form, against the full model, at constant drive.

    fvv_direct.py --delta -0.14 --etas 0.4,0.7,1.0,1.3 --T 130 --orders 2,3,4

With a constant pump the gate Hamiltonian in the charge frame is exactly periodic at w_p,
so U(t) = R(t) e^{-i Lambda(w t)} e^{-i F t} e^{+i Lambda(0)} needs matrix exponentials only.
For each |eta| and order the script reports the distance of its 4x4 computational block
from the full model's at every output time, with and without the kicks. It also reports
the full model's coupler population, since a sudden switch-on excites the coupler.
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
    import numpy as np
    from snail_solver.zhou_coupler import PumpTone
    from snail_solver.envelope import ConstantPulse
    from floquet import model as M, propagator as P, target
    cpl, _ = target.pulse(target.column(job["delta"]))
    w_p = float(cpl._pump_tones[0].w_p_GHz)
    eta, T = job["eta"], job["T"]
    cpl.set_pump(PumpTone(w_p_GHz=w_p, envelope=ConstantPulse(amp=eta, t_g=T + 10.0), is_eta=True))
    times = list(np.linspace(T / job["nt"], T, job["nt"]))
    idx = cpl._subspace_indices(0, 1)
    states = P.propagate_columns(cpl, cpl.to_qutip_hamiltonian(), times)
    full = [S[idx, :] for S in states]
    ns = np.unravel_index(np.arange(cpl.dim), tuple(cpl.dims))[2]
    pop = [float(np.mean(np.sum(np.abs(S[ns > 0, :]) ** 2, axis=0))) for S in states]
    dev = M.device_of(cpl)
    out = dict(delta=job["delta"], eta=eta, T=T, times=times, coupler_pop=pop, rows=[])
    for n in job["orders"]:
        row = {"order": n}
        for kick in (True, False):
            Us, _ = P.constant_drive(*dev, eta, w_p, times, n, kicks=kick)
            row["kick" if kick else "nokick"] = [P.gate_distance(U[np.ix_(idx, idx)], fb)
                                                 for U, fb in zip(Us, full)]
        out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delta", type=float, default=-0.14)
    ap.add_argument("--etas", default="0.4,0.7,1.0,1.3")
    ap.add_argument("--T", type=float, default=130.0, help="evolution time, ns")
    ap.add_argument("--nt", type=int, default=13, help="output times on (0, T]")
    ap.add_argument("--orders", default="2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    orders = [int(v) for v in a.orders.split(",")]
    jobs = [dict(delta=a.delta, eta=float(e), T=a.T, nt=a.nt, orders=orders) for e in a.etas.split(",")]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(run, jobs))
    for r in res:
        print(f"|eta| = {r['eta']}, delta = {r['delta'] * 1e3:+.0f} MHz: distance from the full model "
              f"at t = {r['T']:.0f} ns (max over t), coupler population max {max(r['coupler_pop']):.3f}")
        for row in r["rows"]:
            print(f"   order {row['order']}:  with kicks {row['kick'][-1]:.2e} ({max(row['kick']):.1e})"
                  f"   without {row['nokick'][-1]:.2e} ({max(row['nokick']):.1e})")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
