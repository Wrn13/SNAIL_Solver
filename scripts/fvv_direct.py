"""Test eq. (3) of Bock, Ernst & Uhrig directly: a constant-amplitude pump, where the FVV
propagator is closed form, against the full model.

    fvv_direct.py --delta -0.14 --etas 0.4,0.7,1.0,1.3 --T 130 --orders 1,2,3,4

At constant |eta| the gate Hamiltonian is quasi-periodic, and FVV gives F and Lambda. In
the frame rotating at nu_k = f_k - q_k w_p (pump charge q = (2, 3, 3)), the
charge-conserving part of F is static, and

    U(t) = e^{-i Lambda(t)} R(t) e^{-i 2pi F_rot t} e^{+i Lambda(0)}

is evaluated with matrix exponentials only. For each |eta| and order it reports the
distance of the 4x4 computational block from the full model's, which is integrated
exactly, at every output time:

  direct          closed form, kicks at both ends (the paper's full eq. 3)
  direct-nokick   closed form, no kicks
  numeric         the same truncated F integrated numerically, every secular term kept
                  (method C), kicks at both ends
  check           the charge-conserving F integrated numerically, no kicks. It must equal
                  direct-nokick to solver precision, which validates the rotating frame.

It also lists the secular terms the closed form drops (charge-violating).
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
    from scipy.linalg import expm
    from snail_solver.zhou_coupler import PumpTone
    from snail_solver.envelope import ConstantPulse
    from floquet import fourier as Fo, model as M, propagator as P, target
    from floquet.expansion import floquet_magnus
    rec = target.column(job["delta"])
    cpl, _ = target.pulse(rec)
    w_p = float(cpl._pump_tones[0].w_p_GHz)
    eta, T = job["eta"], job["T"]
    cpl.set_pump(PumpTone(w_p_GHz=w_p, envelope=ConstantPulse(amp=eta, t_g=T + 10.0), is_eta=True))
    times = list(np.linspace(T / job["nt"], T, job["nt"]))
    f, lam, al, g, dims = M.device_of(cpl)
    idx = cpl._subspace_indices(0, 1)
    full_states = P.propagate_columns(cpl, cpl.to_qutip_hamiltonian(), times)
    full = [S[idx, :] for S in full_states]
    ns = np.unravel_index(np.arange(cpl.dim), tuple(cpl.dims))[2]
    coupler_pop = [float(np.mean(np.sum(np.abs(S[ns > 0, :]) ** 2, axis=0))) for S in full_states]
    H = Fo.coefficients(f, lam, al, g, dims, eta)
    ex = floquet_magnus(H, Fo.carrier_of(f), max(job["orders"]), window_GHz=job["window"])
    dw = w_p - Fo.pump_frequency(f)
    dist = lambda blocks: [P.gate_distance(b, fb) for b, fb in zip(blocks, full)]
    out = dict(delta=job["delta"], eta=eta, T=T, times=times, rows=[], coupler_pop=coupler_pop,
               U_full_T=[[z.real, z.imag] for z in full[-1].ravel()])
    dim = int(np.prod(dims))
    for n in job["orders"]:
        row = {"order": n}
        for kick in (True, False):
            Us, dropped = P.direct_propagator(ex, f, dims, w_p, times, n, kicks=kick)
            row["direct" if kick else "direct-nokick"] = dist([U[np.ix_(idx, idx)] for U in Us])
            if kick:
                row["U_direct_T"] = [[z.real, z.imag] for z in Us[-1][np.ix_(idx, idx)].ravel()]
        row["dropped"] = sorted(((v[0] * 1e3, Fo.describe(s), v[1] * 1e3) for s, v in dropped.items()),
                                reverse=True)[:6]
        Fn = ex.F_upto(n)
        L = ex.Lambda_upto(n - 1) if n > 1 else {}
        I = np.zeros((dim, 4), complex)
        I[idx, range(4)] = 1.0
        K0 = expm(1j * P.lambda_at(L, f, dw, 0.0)) if L else np.eye(dim)
        S = P.propagate_columns(cpl, P.qobjevo_bins(Fn, f, w_p, dims), times, init=K0 @ I)
        blocks = []
        for t, St in zip(times, S):
            K1 = expm(-1j * P.lambda_at(L, f, dw, t)) if L else np.eye(dim)
            blocks.append((K1 @ St)[idx, :])
        row["numeric"] = dist(blocks)
        if n == max(job["orders"]):
            S = P.propagate_columns(cpl, P.qobjevo_bins(Fn, f, w_p, dims, keep=lambda s: P.charge(s) == 0), times)
            Us, _ = P.direct_propagator(ex, f, dims, w_p, times, n, kicks=False)
            row["check"] = max(float(np.abs(St[idx, :] - U[np.ix_(idx, idx)]).max()) for St, U in zip(S, Us))
        out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delta", type=float, default=-0.14)
    ap.add_argument("--etas", default="0.4,0.7,1.0,1.3")
    ap.add_argument("--T", type=float, default=130.0, help="evolution time, ns")
    ap.add_argument("--nt", type=int, default=13, help="output times on (0, T]")
    ap.add_argument("--orders", default="1,2,3,4")
    ap.add_argument("--window", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    orders = [int(v) for v in a.orders.split(",")]
    jobs = [dict(delta=a.delta, eta=float(e), T=a.T, nt=a.nt, orders=orders, window=a.window)
            for e in a.etas.split(",")]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(run, jobs))
    for r in res:
        print(f"\n|eta| = {r['eta']}, delta = {r['delta'] * 1e3:+.0f} MHz: distance of the 4x4 block "
              f"from the full model at t = {r['T']:.0f} ns  (max over t <= T)")
        print(f"   {'order':>5} {'direct':>20} {'direct-nokick':>20} {'numeric (C)':>20}")
        for row in r["rows"]:
            cell = lambda k: f"{row[k][-1]:.2e} ({max(row[k]):.1e})"
            print(f"   {row['order']:>5} {cell('direct'):>20} {cell('direct-nokick'):>20} {cell('numeric'):>20}")
        print(f"   full model: coupler population (mean over the 4 kets) max {max(r['coupler_pop']):.3f}, at T {r['coupler_pop'][-1]:.3f}")
        last = r["rows"][-1]
        print(f"   rotating-frame check (direct-nokick vs integrated, conserving F): {last['check']:.1e}")
        print("   largest dropped (charge-violating) secular terms at the top order: "
              + "; ".join(f"{d[1]} {d[0]:.3f} MHz @ {d[2]:+.0f} MHz" for d in last["dropped"][:4]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
