"""Floquet branch tracking over a detuning grid (the 1 MHz-wide scan's device).

Per detuning delta (w_p = w_a/2 + delta), `snail_solver.floquet.track` labels every
Floquet mode by its |eta| = 0 Fock state and follows all of them in constant-pump |eta|
from 0, and forms the windowed diabatic energy of each watched Fock state. Stored per
(levels, delta) as OUTDIR/levels<L>/d<delta>.npz (a re-run skips what is done):

    eta (n,), labels (dim,), w_p_GHz
    eps_GHz (n, dim)          labelled branches, unwrapped
    own_weight (n, dim)       each branch's population on its own Fock state
    weights (S, n, dim)       each branch's population on each watched state
    diabatic_GHz, spread_GHz, captured (S, n)   per watched state
    watch (S,)                the watched state keys

Usage:  floquet_branch_map.py OUTDIR [--deltas LO:HI:N | a,b,c] [--levels 9]
                              [--eta-max 1.5] [--n-eta 201] [--steps 1000] [--workers N]
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCAN = "results/drag_curve_1MHz_2026-10-23"
# name: (initial, final, k pump quanta, label); "doublet" is the |100>/|010> pair on the
# |100> zone, so a process from |010> carries one pump quantum more here
PROCESSES = {
    "snail3": ("000", "001", 3, r"$3\omega_p \to \omega_s$: $|000\rangle\to|001\rangle$"),
    "k5_201": ("doublet", "201", 5, r"$|100\rangle\to|201\rangle$, 5 pumps"),
    "a12": ("doublet", "200", 2, r"A $|1\rangle\to|2\rangle$: $|100\rangle\to|200\rangle$, 2 pumps"),
    "asub": ("000", "doublet", 2, r"A subharmonic: $|000\rangle\to|100\rangle$, 2 pumps"),
    "b_002": ("doublet", "002", 3, r"$|010\rangle\to|002\rangle$, 2 pumps"),
    "k4_201": ("doublet", "201", 4, r"$|100\rangle\to|201\rangle$, 4 pumps"),
}
WATCH = ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (1, 0, 0), (0, 1, 0),
         (2, 0, 0), (2, 0, 1), (0, 2, 1), (1, 1, 0), (1, 0, 2), (1, 1, 1))


def device():
    from snail_solver.h5_io import load_doc
    d = load_doc(sorted(glob.glob(f"{SCAN}/passA_fine/*.h5"))[0])
    s = d.get("scan") or d
    return s["device"], s["settings"]["branch"]


def path_for(outdir, levels, delta):
    return os.path.join(outdir, f"levels{levels}", f"d{delta:+.0f}.npz")


def run(job):
    from snail_solver import floquet as F
    delta, levels, etas, steps, out = job
    dev, branch = device()
    m = F.model_at_detuning(dev, delta * 1e-3, branch=branch, levels=levels)
    tr = F.track(m, etas, n_steps=steps, weights_on=WATCH, diabatic_of=WATCH)
    keys = [F.key(o) for o in WATCH]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp, delta_MHz=delta, levels=levels, n_steps=steps, w_p_GHz=tr["w_p_GHz"],
        eta=tr["eta"], labels=np.array(tr["labels"]), watch=np.array(keys),
        eps_GHz=tr["eps_GHz"], own_weight=tr["own_weight"].astype(np.float32),
        weights=np.stack([tr["weights"][k] for k in keys]).astype(np.float32),
        diabatic_GHz=np.stack([tr["diabatic_GHz"][k] for k in keys]),
        spread_GHz=np.stack([tr["spread_GHz"][k] for k in keys]),
        captured=np.stack([tr["captured"][k] for k in keys]))
    os.replace(tmp, out)
    return delta


def load(path):
    """A stored map file as a :func:`snail_solver.floquet.track`-shaped dict."""
    z = np.load(path)
    keys = [str(k) for k in z["watch"]]
    tr = {"eta": z["eta"], "labels": [str(x) for x in z["labels"]],
          "eps_GHz": z["eps_GHz"], "own_weight": z["own_weight"],
          "w_p_GHz": float(z["w_p_GHz"]), "delta_MHz": float(z["delta_MHz"])}
    for f in ("weights", "diabatic_GHz", "spread_GHz", "captured"):
        tr[f] = {k: z[f][i] for i, k in enumerate(keys)}
    return tr


def parse_deltas(spec):
    if ":" in spec:
        lo, hi, n = spec.split(":")
        return [float(round(x)) for x in np.linspace(float(lo), float(hi), int(n))]
    return [float(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--deltas", default="-250:250:251")
    ap.add_argument("--levels", type=int, default=9)
    ap.add_argument("--eta-max", type=float, default=1.5)
    ap.add_argument("--n-eta", type=int, default=201)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    etas = np.linspace(0.0, a.eta_max, a.n_eta)
    jobs = [(d, a.levels, etas, a.steps, path_for(a.outdir, a.levels, d))
            for d in parse_deltas(a.deltas)]
    todo = [j for j in jobs if not os.path.exists(j[-1])]
    print(f"{len(jobs)} detunings at {a.levels} levels, {len(todo)} to run "
          f"on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, d in enumerate(ex.map(run, todo), 1):
            print(f"[{i}/{len(todo)}] delta={d:+.0f} MHz", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
