"""Coupler (and target) AC-Stark shift vs drive from Floquet quasienergies.

Constant pump of displaced amplitude |eta| on the column's device, LAB frame, no RWA:

    H(t) = sum_i w_i n_i + H_anh + sum_n g_n [ X_s + 2 |eta| cos(w_p t) ]^n ,
    X_s  = sum_i lambda_i (a_i + a_i^dag)

built from the same ``ZhouCoupler`` operators the gate solver uses (its interaction
picture ``hamiltonian_matrix`` is ``e^{i H0 t} (H - H0) e^{-i H0 t}`` of this). H is
T = 1/f_p periodic, so the one-period propagator's eigenphases are the quasienergies
``eps = -arg(lambda) / T`` (mod w_p), its eigenvectors the Floquet modes at t = 0.

Shifts, drive part only (the |eta| = 0 value subtracted):

    coupler 0->1   eps[001] - eps[000]     coupler 1->2   eps[002] - eps[001]
    target         H_eff[010,010] - H_eff[100,100]

The modes are followed in |eta| by maximum overlap with the previous step, and each
quasienergy unwrapped onto the branch nearest its last value. |100> and |010> are
degenerate by design (w_b - w_a = w_p) and mix into the iSWAP doublet, so the target
is read off the 2x2 effective Hamiltonian of the two modes that live in that pair:
``H_eff = P diag(eps) P^-1``, P the modes' Fock amplitudes. ``weight_*`` is the
tracked mode's population on its own Fock state: where it dips, the level has run
into an avoided crossing (e.g. w_s ~ 3 w_p couples |000> and |001>).

Usage:  floquet_coupler_shift.py OUTDIR [DELTA_MHz,...] [ETA_MAX] [N_ETA] [N_STEPS]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TWO_PI = 2.0 * np.pi


def operators(delta_MHz, levels=None):
    """(H0 + H_anh, X_s, {n: g_n}, w_p rad/ns, fock_index) for one column, pump on
    the nominal carrier |w_b - w_a| (the ridge is measured relative to it)."""
    import drag_sign_common as D
    from snail_solver.device_utils import build_coupler
    from snail_solver.subharmonic_convergence import config_at_wp
    c = D.column(delta_MHz)
    cfg = c["variants"]["chirp"]["cfg"]
    if levels is not None:
        cfg = config_at_wp(c["device"], c["w_p_GHz"], branch=c["settings"]["branch"],
                           levels=int(levels))
    cpl, w_p, _ = build_coupler(cfg, 100.0, 1.0, 0.0, None, None, chirp_coeffs_GHz=[])
    H0 = sum(cpl.omega[i] * (cpl.ad_ops[i] @ cpl.a_ops[i]) for i in range(cpl.n_modes))
    H0 = np.asarray(H0, dtype=complex) + np.asarray(cpl._anharm_op, dtype=complex)
    Xs = sum(amp * np.asarray(op, dtype=complex) for op, _w, amp, _k in cpl._mode_letters)
    return H0, Xs, dict(cpl.g_n), float(w_p) * TWO_PI, cpl.fock_index


def period_propagator(H0, Xs, g_n, w_p, eta, n_steps):
    """U(T) by exponential midpoint steps; H(t) = A_0 + sum_k c(t)^k A_k, c = 2 eta cos."""
    from math import comb
    dim = H0.shape[0]
    P = [np.eye(dim, dtype=complex)]
    for _ in range(max(g_n)):
        P.append(P[-1] @ Xs)
    A = [np.zeros((dim, dim), dtype=complex) for _ in range(max(g_n) + 1)]
    for n, g in g_n.items():
        for k in range(n + 1):
            A[k] = A[k] + g * comb(n, k) * P[n - k]
    A[0] = A[0] + H0
    T = TWO_PI / w_p
    dt = T / n_steps
    U = np.eye(dim, dtype=complex)
    for j in range(n_steps):
        c = 2.0 * eta * np.cos(w_p * (j + 0.5) * dt)
        H = A[0] + sum(c ** k * A[k] for k in range(1, len(A)))
        U = expm(-1j * H * dt) @ U
    return U, T


def floquet(U, T):
    lam, V = np.linalg.eig(U)
    return -np.angle(lam) / T, V                                # rad/ns, mod w_p


def _nearest(prev, eps, w_p):
    """eps shifted by a multiple of w_p onto the branch nearest prev."""
    return eps + w_p * np.round((prev - eps) / w_p)


def scan(job):
    delta, etas, n_steps, levels = job
    H0, Xs, g_n, w_p, fi = operators(delta, levels)
    F = {o: fi(list(o)) for o in ((0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (0, 1, 0))}
    track = {o: None for o in ((0, 0, 0), (0, 0, 1), (0, 0, 2))}
    eps_prev = {}
    pair_prev = None
    rows = []
    for eta in etas:
        U, T = period_propagator(H0, Xs, g_n, w_p, eta, n_steps)
        eps, V = floquet(U, T)
        row = {"eta": float(eta)}
        for o in track:                                         # single levels
            ov = np.abs(V.conj().T @ (np.eye(len(eps))[:, F[o]] if track[o] is None
                                      else track[o])) ** 2
            j = int(np.argmax(ov))
            track[o] = V[:, j]
            e = eps[j] if o not in eps_prev else _nearest(eps_prev[o], eps[j], w_p)
            eps_prev[o] = e
            row[f"eps_{''.join(map(str, o))}"] = float(e)
            row[f"weight_{''.join(map(str, o))}"] = float(abs(V[F[o], j]) ** 2)
            row[f"overlap_step_{''.join(map(str, o))}"] = float(ov[j])
        # target pair: the two modes with most weight in span{|100>, |010>}
        w = np.abs(V[F[(1, 0, 0)], :]) ** 2 + np.abs(V[F[(0, 1, 0)], :]) ** 2
        jj = np.argsort(w)[-2:]
        P = np.array([[V[F[(1, 0, 0)], j] for j in jj], [V[F[(0, 1, 0)], j] for j in jj]])
        e2 = eps[jj]
        e2 = _nearest(e2[0], e2, w_p)                           # same branch as each other
        if pair_prev is not None:
            e2 = _nearest(pair_prev, e2, w_p)
        pair_prev = float(np.mean(e2))
        Heff = P @ np.diag(e2) @ np.linalg.inv(P)
        row["target_diag"] = float(np.real(Heff[1, 1] - Heff[0, 0]))
        row["target_pair_weight"] = float(np.min(w[jj]))
        rows.append(row)
    base = rows[0]
    for r in rows:
        r["coupler01_MHz"] = ((r["eps_001"] - r["eps_000"]) - (base["eps_001"] - base["eps_000"])) / TWO_PI * 1e3
        r["coupler12_MHz"] = ((r["eps_002"] - r["eps_001"]) - (base["eps_002"] - base["eps_001"])) / TWO_PI * 1e3
        d = r["target_diag"] - base["target_diag"]
        r["target_MHz"] = float(_nearest(0.0, d, w_p) / TWO_PI * 1e3)
    return {"delta_MHz": delta, "n_steps": n_steps, "levels": levels, "w_p_GHz": w_p / TWO_PI,
            "rows": rows}


def main():
    outdir = sys.argv[1]
    deltas = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [-185, -180, -175, -170, -165, -160])
    eta_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    n_eta = int(sys.argv[4]) if len(sys.argv) > 4 else 151
    n_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    etas = np.linspace(0.0, eta_max, n_eta)
    os.makedirs(outdir, exist_ok=True)
    with ProcessPoolExecutor(len(deltas)) as ex:
        res = list(ex.map(scan, [(d, etas, n_steps, None) for d in deltas]))
    path = os.path.join(outdir, f"floquet_shift_N{n_steps}.json")
    json.dump(res, open(path, "w"), indent=1)
    for r in res:
        for row in r["rows"]:
            if any(abs(row["eta"] - e) < 1e-9 for e in (0.4, 0.7, 1.0, 1.3)):
                print(f"{r['delta_MHz']:+5.0f} eta={row['eta']:.2f} coupler01 "
                      f"{row['coupler01_MHz']:+7.2f} 12 {row['coupler12_MHz']:+7.2f} "
                      f"target {row['target_MHz']:+7.2f} MHz | weight 001 "
                      f"{row['weight_001']:.3f} 000 {row['weight_000']:.3f}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
