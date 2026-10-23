"""Full Floquet quasienergy spectrum vs |eta|, for plotting the branch tracking.

Same Hamiltonian and propagator as `floquet_coupler_shift.py`. Per |eta| it stores
EVERY quasienergy with each mode's weight on a few Fock states, plus the branches
`floquet_coupler_shift.scan` tracks (by the same max-overlap rule), so the plot can
show the tracked line on top of the spectrum it was picked from.

Usage:  floquet_spectrum.py OUTDIR [DELTA_MHz,...] [ETA_MAX] [N_ETA] [N_STEPS]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floquet_coupler_shift as F  # noqa: E402

WATCH = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (2, 0, 1), (1, 1, 1))


def run(job):
    delta, etas, n_steps = job
    H0, Xs, g_n, w_p, fi = F.operators(delta)
    idx = {o: fi(list(o)) for o in WATCH}
    track = {o: None for o in ((0, 0, 0), (0, 0, 1))}
    out = {"delta_MHz": delta, "w_p_GHz": w_p / F.TWO_PI, "eta": [], "eps_MHz": [],
           "weights": {"".join(map(str, o)): [] for o in WATCH},
           "tracked_index": {"000": [], "001": []}}
    for eta in etas:
        U, T = F.period_propagator(H0, Xs, g_n, w_p, eta, n_steps)
        eps, V = F.floquet(U, T)
        out["eta"].append(float(eta))
        out["eps_MHz"].append((eps / F.TWO_PI * 1e3).tolist())
        for o in WATCH:
            out["weights"]["".join(map(str, o))].append(
                (np.abs(V[idx[o], :]) ** 2).tolist())
        for o in track:
            ref = np.eye(len(eps))[:, idx[o]] if track[o] is None else track[o]
            j = int(np.argmax(np.abs(V.conj().T @ ref)))
            track[o] = V[:, j]
            out["tracked_index"]["".join(map(str, o))].append(j)
    return out


def main():
    outdir = sys.argv[1]
    deltas = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [-185, -180, -175, -170, -165, -160])
    eta_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    n_eta = int(sys.argv[4]) if len(sys.argv) > 4 else 151
    n_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    etas = np.linspace(0.0, eta_max, n_eta)
    with ProcessPoolExecutor(len(deltas)) as ex:
        res = list(ex.map(run, [(d, etas, n_steps) for d in deltas]))
    path = os.path.join(outdir, "floquet_spectrum.json")
    json.dump(res, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
