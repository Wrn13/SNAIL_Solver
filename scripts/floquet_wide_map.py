"""Floquet branch map over the whole pump band, w_p = 10 MHz .. 2.5 GHz.

The 1 MHz-wide scan's device (branch "above": w_b = w_a + w_p), labelled and tracked
exactly as `floquet_branch_map.py` does, with three changes for the wide band:

- the device's 50 MHz pump floor is lowered (FLOOR_GHZ) so w_p reaches 10 MHz;
- steps per period double below LOW_WP_GHZ: 1000 is off by 4.8 MHz at w_p = 10 MHz,
  |eta| = 2.2, against 3 kHz at 2000 (1000 holds ~10 kHz down to 30 MHz);
- the work list is sharded (--shard i/n) so one grid spreads over several boxes.

Files land at OUTDIR/levels<L>/d<delta>.npz with delta = w_p - w_a/2 in MHz, the same
layout and fields as floquet_branch_map.py, so its `load` reads them.

Usage:  floquet_wide_map.py OUTDIR [--wp LO:HI:STEP] [--levels 9] [--eta-max 2.2]
                            [--n-eta 294] [--shard i/n] [--workers N]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floquet_branch_map as M  # noqa: E402

FLOOR_GHZ = 0.005
LOW_WP_GHZ = 0.2
DEVICE_JSON = "device.json"


def steps_for(w_p_GHz):
    return 2000 if w_p_GHz < LOW_WP_GHZ else 1000


def run(job):
    from snail_solver import floquet as F
    delta, levels, etas, steps, out, dev, branch = job
    m = F.model_at_detuning(dev, delta * 1e-3, branch=branch, levels=levels)
    tr = F.track(m, etas, n_steps=steps, weights_on=M.WATCH, diabatic_of=M.WATCH)
    keys = [F.key(o) for o in M.WATCH]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp, delta_MHz=delta, levels=levels, n_steps=steps, w_p_GHz=tr["w_p_GHz"],
        eta=tr["eta"], labels=np.array(tr["labels"]), watch=np.array(keys),
        eps_GHz=tr["eps_GHz"], own_weight=tr["own_weight"].astype(np.float32),
        step_overlap=tr["step_overlap"].astype(np.float32),
        weights=np.stack([tr["weights"][k] for k in keys]).astype(np.float32),
        diabatic_GHz=np.stack([tr["diabatic_GHz"][k] for k in keys]),
        spread_GHz=np.stack([tr["spread_GHz"][k] for k in keys]),
        captured=np.stack([tr["captured"][k] for k in keys]))
    os.replace(tmp, out)
    return delta


def device(outdir):
    """The scan's device and branch, frozen into OUTDIR/device.json on first use so
    every box reads the same one (the boxes hold different scan shards)."""
    path = os.path.join(outdir, DEVICE_JSON)
    if not os.path.exists(path):
        dev, branch = M.device()
        os.makedirs(outdir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"device": dev, "branch": branch}, f, indent=1)
    with open(path) as f:
        d = json.load(f)
    dev = dict(d["device"])
    dev["min_detuning_GHz"] = FLOOR_GHZ
    return dev, d["branch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--wp", default="0.010:2.500:0.002", help="LO:HI:STEP in GHz")
    ap.add_argument("--levels", type=int, default=9)
    ap.add_argument("--eta-max", type=float, default=2.2)
    ap.add_argument("--n-eta", type=int, default=294)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    dev, branch = device(a.outdir)
    wa = float(dev["qubit_freqs_GHz"][0])
    lo, hi, step = (float(x) for x in a.wp.split(":"))
    wps = lo + step * np.arange(int(round((hi - lo) / step)) + 1)
    etas = np.linspace(0.0, a.eta_max, a.n_eta)
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = []
    for wp in wps[i::n]:
        delta = float(round((wp - 0.5 * wa) * 1e3))
        jobs.append((delta, a.levels, etas, steps_for(wp),
                     M.path_for(a.outdir, a.levels, delta), dev, branch))
    # slowest (low w_p: more steps, stiffer expm) first so they don't straggle
    todo = sorted((j for j in jobs if not os.path.exists(j[4])), key=lambda j: j[0])
    print(f"shard {i}/{n}: {len(jobs)} pumps at {a.levels} levels, |eta| <= {a.eta_max} "
          f"({a.n_eta} pts), {len(todo)} to run on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for k, d in enumerate(ex.map(run, todo), 1):
            print(f"[{k}/{len(todo)}] w_p={(d * 1e-3 + 0.5 * wa):.3f} GHz "
                  f"(delta={d:+.0f} MHz)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
