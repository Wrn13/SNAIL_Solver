"""Replay stored gate pulses and record where the leaked population goes, and when.

For each detuning, pass A's calibrated operating point (ridge chirp, DRAG off; from
``passA_fine/columns``) is rebuilt and evolved on the exact Hamiltonian from |100>,
|010> and |110>. Stored per start state: the 8 most populated Fock states at the end,
their populations P(t) on 301 times, and the mean coupler occupation <n_s>(t); plus
the pulse's |eta(t)|. Nothing is re-calibrated.

Usage:  replay_leak_trajectories.py OUT.json DELTA_MHz[,...] [SCAN_DIR]
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

STARTS = ((1, 0, 0), (0, 1, 0), (1, 1, 0))


def run(job):
    from snail_solver.device_utils import build_coupler
    from snail_solver.h5_io import load_doc
    from snail_solver.subharmonic_convergence import config_at_wp
    from snail_solver.subharmonic_gate_scan import column_tag
    delta, scan = job
    d = load_doc(sorted(glob.glob(f"{scan}/passA_fine/*.h5"))[0])
    s = d.get("scan") or d
    device, st = s["device"], s["settings"]
    rec = json.load(open(f"{scan}/passA_fine/columns/col_{column_tag(delta * 1e-3, 1.3)}.json"))
    op = rec["operating_point"]
    cfg = config_at_wp(device, rec["w_p_GHz"], branch=st["branch"],
                       levels=int(rec.get("coupler_levels", st["coupler_levels"])),
                       chirp_coeffs_GHz=op["chirp_coeffs_GHz"])
    cpl, _w_p, peak = build_coupler(cfg, op["t_g_ns"], op["amp_scale"], op["wp_offset_GHz"],
                                    None, None, chirp_coeffs_GHz=op["chirp_coeffs_GHz"])
    t = np.linspace(0.0, op["t_g_ns"], 301)
    eta = np.abs(np.asarray(cpl._eta_at(cpl._pump_tones[0], t, np)))
    out = {"delta_MHz": delta, "t_ns": t.tolist(), "eta": eta.tolist(),
           "t_g_ns": op["t_g_ns"], "peak_eta": float(peak)}
    for start in STARTS:
        P = np.abs(cpl.evolve_trajectory(start, t)) ** 2
        top = np.argsort(P[-1])[::-1][:8]
        labels = ["".join(map(str, cpl.decode_index(int(i)))) for i in top]
        out["".join(map(str, start))] = {
            "final": dict(zip(labels, (float(P[-1, i]) for i in top))),
            "traj": {lab: P[:, i].tolist() for lab, i in zip(labels, top)},
            "coupler_n": [float(cpl.mean_occupation(p, 2)) for p in P]}
    return out


def main():
    out = sys.argv[1]
    deltas = [float(x) for x in sys.argv[2].split(",")]
    scan = sys.argv[3] if len(sys.argv) > 3 else "results/drag_curve_1MHz_2026-10-23"
    with ProcessPoolExecutor(len(deltas)) as ex:
        res = list(ex.map(run, [(d, scan) for d in deltas]))
    json.dump(res, open(out, "w"))
    for r in res:
        print(f"delta={r['delta_MHz']:+.0f} MHz  t_g={r['t_g_ns']:.1f} ns  "
              f"peak|eta|={r['peak_eta']:.3f}")
        for k in ("100", "010", "110"):
            print(f"  from |{k}>  " + "  ".join(f"|{lab}> {p:.4f}"
                                              for lab, p in r[k]["final"].items()))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
