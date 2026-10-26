"""Leak processes from the Floquet branch map: hybridisation, resonance drive, LZS period.

Reads `floquet_branch_map.py`'s per-detuning files and, for each process i -> f with k
pump quanta (`floquet_branch_map.PROCESSES`), forms on the (delta, |eta|) grid

    hyb      the computational branch's population on |f>: the |100>/|010> branches'
             mean weight on |f> for a process from the doublet, the |000> branch's for
             one from |000>
    beat     the diabatic dressed beat E_f - E_i - k w_p (MHz), masked where either
             diabatic energy holds less than half its state's weight in its window

and per detuning the resonance drive

    eta_meet the first |eta| where a branch carrying |f> (weight >= W_MIN) comes within
             MEET_MHZ of the computational branch: the two have met, crossed or
             avoided. By |eta| ~ 1 a coupler level is shared over several branches, so
             neither a single label nor the diabatic energy follows it all the way; the
             meeting of SOME |f>-carrying branch with the computational one does

For the positive-side |010> -> |002> process a pulse whose peak exceeds eta_meet passes
the resonance on its rise and again on its fall. The two passages interfere with a
phase whose explicit delta-dependence is 3 (t2 - t1) per MHz (the beat 650 - 3 delta
carries 3 delta), so the infidelity ripples in delta with period 1 / (3 T),
T = t2 - t1 the time the column's own calibrated envelope spends above eta_meet. No
parameter is fitted. Writes FLOQUET_DIR/analysis_levels<L>.npz.

Usage:  analyze_floquet_map.py FLOQUET_DIR [SCAN_DIR] [LEVELS ...]
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floquet_branch_map as M  # noqa: E402

CAPTURED_MIN = 0.5
W_MIN = 0.05
MEET_MHZ = 15.0
LZS_SLOPE = 3            # |d beat / d delta| of |010> -> |002>


def comp_branches(tr, initial):
    L = tr["labels"]
    return ([L.index("100"), L.index("010")] if initial == "doublet" else [L.index(initial)])


def comp_weight(tr, initial, final):
    """(n,) population of the computational branch(es) of `initial` on |final>."""
    w = tr["weights"]
    on = w["100"] + w["010"] if final == "doublet" else w[final]
    return on[:, comp_branches(tr, initial)].mean(axis=1)


def diabatic_beat(tr, initial, final, k):
    def E(s):
        if s != "doublet":
            return tr["diabatic_GHz"][s], tr["captured"][s]
        e1, e2 = tr["diabatic_GHz"]["100"], tr["diabatic_GHz"]["010"]
        w_p = tr["w_p_GHz"]
        e = 0.5 * (e1 + e2 - w_p * np.round((e2 - e1) / w_p))
        return e, np.minimum(tr["captured"]["100"], tr["captured"]["010"])
    ei, ci = E(initial)
    ef, cf = E(final)
    b = (ef - ei - k * tr["w_p_GHz"]) * 1e3
    return np.where((ci >= CAPTURED_MIN) & (cf >= CAPTURED_MIN), b, np.nan)


def meet_eta(tr, initial, final):
    """First |eta| where an |f>-carrying branch reaches the computational branch."""
    if final == "doublet":                    # the A subharmonic: |000> -> doublet
        final = "100"
    E = tr["eps_GHz"] * 1e3
    z = tr["w_p_GHz"] * 1e3
    comp = comp_branches(tr, initial)
    if len(comp) == 2:
        e1, e2 = E[:, comp[0]], E[:, comp[1]]
        ref = 0.5 * (e1 + e2 - z * np.round((e2 - e1) / z))
    else:
        ref = E[:, comp[0]]
    rel = (E - ref[:, None] + z / 2) % z - z / 2
    W = np.array(tr["weights"][final], dtype=float)
    W[:, comp] = 0.0
    dist = np.where(W >= W_MIN, np.abs(rel), np.inf).min(axis=1)
    hit = np.nonzero(dist < MEET_MHZ)[0]
    return float(tr["eta"][hit[0]]) if hit.size else np.nan


def envelope(job):
    """(delta, t_ns, |eta(t)|) of pass A's calibrated pulse at one column."""
    from snail_solver.device_utils import build_coupler
    from snail_solver.h5_io import load_doc
    from snail_solver.subharmonic_convergence import config_at_wp
    from snail_solver.subharmonic_gate_scan import column_tag
    delta, scan = job
    path = f"{scan}/passA_fine/columns/col_{column_tag(delta * 1e-3, 1.3)}.json"
    if not os.path.exists(path):
        return delta, None, None
    d = load_doc(sorted(glob.glob(f"{scan}/passA_fine/*.h5"))[0])
    s = d.get("scan") or d
    rec = json.load(open(path))
    op = rec["operating_point"]
    cfg = config_at_wp(s["device"], rec["w_p_GHz"], branch=s["settings"]["branch"],
                       levels=int(s["settings"]["coupler_levels"]),
                       chirp_coeffs_GHz=op["chirp_coeffs_GHz"])
    cpl, _w, _pk = build_coupler(cfg, op["t_g_ns"], op["amp_scale"], op["wp_offset_GHz"],
                                 None, None, chirp_coeffs_GHz=op["chirp_coeffs_GHz"])
    t = np.linspace(0.0, op["t_g_ns"], 4001)
    return delta, t, np.abs(np.asarray(cpl._eta_at(cpl._pump_tones[0], t, np)))


def lzs_period(deltas, eta_meet, scan):
    """Per scan column: time above eta_meet (ns) and the predicted period 1/(3T) (MHz)."""
    cols = [float(d) for d in range(int(min(deltas)), int(max(deltas)) + 1)]
    em = np.interp(cols, deltas, eta_meet, left=np.nan, right=np.nan)
    with ProcessPoolExecutor(min(64, os.cpu_count())) as ex:
        envs = list(ex.map(envelope, [(d, scan) for d in cols]))
    T, peak = [], []
    for (d, t, e), m in zip(envs, em):
        if t is None or not np.isfinite(m):
            T.append(np.nan)
            peak.append(np.nan if t is None else float(e.max()))
            continue
        above = t[e >= m]
        T.append(float(above[-1] - above[0]) if above.size > 1 else np.nan)
        peak.append(float(e.max()))
    T = np.array(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        period = 1e3 / (LZS_SLOPE * T)
    return np.array(cols), T, period, np.array(peak)


def analyse(fdir, levels, scan):
    paths = sorted(glob.glob(os.path.join(fdir, f"levels{levels}", "d*.npz")),
                   key=lambda p: float(os.path.basename(p)[1:-4]))
    if not paths:
        return None
    trs = [M.load(p) for p in paths]
    eta = trs[0]["eta"]
    d = np.array([t["delta_MHz"] for t in trs])
    out = {"delta_MHz": d, "eta": eta}
    for name, (i, f, k, _lab) in M.PROCESSES.items():
        out[f"hyb_{name}"] = np.array([comp_weight(t, i, f) for t in trs])
        out[f"beat_{name}"] = np.array([diabatic_beat(t, i, f, k) for t in trs])
        out[f"eta_meet_{name}"] = np.array([meet_eta(t, i, f) for t in trs])
    if scan and levels == 9:
        m = (d >= 0) & (d <= 170)
        cols, T, period, peak = lzs_period(d[m], out["eta_meet_b_002"][m], scan)
        out.update(lzs_delta_MHz=cols, lzs_T_ns=T, lzs_period_MHz=period, lzs_peak_eta=peak)
    path = os.path.join(fdir, f"analysis_levels{levels}.npz")
    np.savez_compressed(path, **out)
    return path, out


def main():
    fdir = sys.argv[1]
    scan = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
    levels = [int(x) for x in sys.argv[2:] if x.isdigit()] or [9, 12]
    for L in levels:
        r = analyse(fdir, L, scan)
        if r is None:
            continue
        path, out = r
        d = out["delta_MHz"]
        print(f"levels {L}: {len(d)} detunings -> {path}")
        for name in M.PROCESSES:
            e = out[f"eta_meet_{name}"]
            m = np.isfinite(e) & (e <= 1.3)
            span = f"{d[m].min():+.0f}..{d[m].max():+.0f}" if m.any() else "none"
            print(f"  {name:7s} resonance below |eta| = 1.3 at {m.sum():3d} detunings, {span} MHz")
        if "lzs_period_MHz" in out:
            for x in (40, 60, 80, 100, 120, 140):
                j = int(np.argmin(np.abs(out["lzs_delta_MHz"] - x)))
                print(f"  LZS {x:+d} MHz: T = {out['lzs_T_ns'][j]:.0f} ns, "
                      f"period 1/(3T) = {out['lzs_period_MHz'][j]:.1f} MHz")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
