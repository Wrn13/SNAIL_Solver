"""Flag the columns whose chirp was never measured, from the laws already stored.

    backfill_chirp_exclusions.py RUNDIR [--max-frac 0.10] [--suffix _flagged]

WHY THIS EXISTS
---------------
A column can carry no chirp for two reasons that look identical in the output and
demand opposite treatment:

  * there was NO SHIFT TO CHIRP -- the law's excursion across the pulse is a
    negligible fraction of the resonance half-width. `chirp == bare` is then a
    RESULT, and a gain of 1.00x is the correct thing to report.

  * a chirp COULD NOT BE MEASURED -- the ridge fit failed while the law still
    swept a real fraction of a half-width, or the ridge railed and no law was
    fitted at all. `chirp == bare` is then an ARTEFACT of the calibration, and
    averaging it in at 1.00x drags the chirp's measured benefit down precisely at
    the detunings where the chirp does the most work.

On the 2026-09-22 grid the second kind is 14 of 61 columns at eta = 1.3, and every
one was reported at exactly 1.000x. Including them puts the chirp's gain at
geomean 1.610x; excluding them puts it at 1.855x, with the median moving 1.19x ->
1.85x because the 1.000x entries sat right at the middle of the distribution.

The discriminator is the excursion |k2 eta*^2 + k4 eta*^4| against the half-width
1/(2 t_g). Measured over those 31 chirp-free columns the two populations separate
cleanly on it: below 10% the fit residual is 1.0-16.7x the excursion (no signal),
above 20% it is 0.21-0.58x (a real law that merely missed r2_min).

NO SOLVES. Every quantity is read from the per-column JSONs the run already wrote:
the rejected law is stored even when it was not used. The curves files are NOT
edited in place -- flagged copies are written beside them -- so the originals stay
available for comparison.

Downstream, `chirp_excluded` is what paired_gain_table.py and plot_drag_curves.py
read; rows that are chirp-free for the legitimate reason are left untouched and
keep contributing their honest 1.00x.
"""
import argparse
import glob
import json
import os

LEGIT = "no_measurable_shift"
UNMEASURABLE = "unmeasurable_chirp"
RAILED = "railed_ridge"

#: Coupler occupation above which --coupler-levels 9 cannot represent the state at
#: all: the top of the ladder is populated and the leakage out of the model is
#: uncharged, so 1-F comes out too GOOD. The 7-level run was 3x optimistic at ~0.85
#: photons, so this is a factors-of-several problem. Flagged, not dropped -- the
#: column still locates a resonance, it just cannot be quoted as a fidelity.
N_COUPLER_SEVERE = 0.5


def excursion_frac(col):
    """|k2 eta^2 + k4 eta^4| / (1/(2 t_g)), or None when no law was stored.

    A railed ridge raises before `fit_shift_curve` runs, so it carries no k2/k4 at
    all. An unknown excursion is not a small one -- those columns can never qualify
    as legitimately chirp-free.
    """
    op = col.get("operating_point") or {}
    ch = col.get("chirp") or {}
    k2, k4, t_g = ch.get("k2"), ch.get("k4"), op.get("t_g_ns")
    eta = col.get("target_eta")
    if k2 is None or k4 is None or not t_g or eta is None:
        return None
    eta = float(eta)
    exc = abs(float(k2) * eta ** 2 + float(k4) * eta ** 4)
    return exc / (1e3 / (2.0 * float(t_g)))


def coupler_load(rundir):
    """{(eta, delta_mhz): n_coupler} from the per-column records."""
    out = {}
    for sub in ("passA_nodrag", "passB_drag"):
        for path in glob.glob(os.path.join(rundir, sub, "columns", "*.json")):
            try:
                col = json.load(open(path))
            except (OSError, ValueError):
                continue
            n = col.get("n_coupler")
            if n is None:
                continue
            key = (round(float(col["target_eta"]), 2),
                   round(float(col["delta_GHz"]) * 1e3))
            out[key] = max(float(n), out.get(key, 0.0))
    return out


def classify(rundir, max_frac):
    """{(eta, delta_mhz): (reason, frac)} for every CHIRP-FREE column in the run."""
    out = {}
    for sub in ("passA_nodrag", "passB_drag"):
        for path in glob.glob(os.path.join(rundir, sub, "columns", "*.json")):
            try:
                col = json.load(open(path))
            except (OSError, ValueError):
                continue
            if not ((col.get("operating_point") or {}).get("chirp_free")):
                continue
            key = (round(float(col["target_eta"]), 2),
                   round(float(col["delta_GHz"]) * 1e3))
            frac = excursion_frac(col)
            if frac is None:
                reason = RAILED
            elif frac <= max_frac:
                reason = LEGIT
            else:
                reason = UNMEASURABLE
            # passA and passB both hold the column; agree by taking the stricter
            # reading, since either pass failing to measure a chirp is enough.
            prev = out.get(key)
            if prev is None or (prev[0] == LEGIT and reason != LEGIT):
                out[key] = (reason, frac)
    return out


def apply_to(curves_path, cls, load, out_path, n_severe):
    rows = json.load(open(curves_path))
    n = {LEGIT: 0, UNMEASURABLE: 0, RAILED: 0}
    n_trunc = 0
    for r in rows:
        key = (round(float(r["target_eta"]), 2), round(float(r["delta_GHz"]) * 1e3))
        nc = load.get(key)
        if nc is not None:
            r["n_coupler"] = nc
            r["coupler_truncated"] = bool(nc > n_severe)
            n_trunc += bool(nc > n_severe)
        hit = cls.get(key)
        if hit is None:
            continue
        reason, frac = hit
        r["chirp_free_reason"] = reason
        r["chirp_excursion_frac_linewidth"] = frac
        r["chirp_excluded"] = reason != LEGIT
        n[reason] += 1
    json.dump(rows, open(out_path, "w"), indent=1)
    return n, len(rows), n_trunc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir")
    ap.add_argument("--max-frac", type=float, default=0.10,
                    help="largest excursion, as a fraction of the half-width "
                         "1/(2 t_g), that counts as genuinely chirp-free [0.10]")
    ap.add_argument("--suffix", default="_flagged",
                    help="written beside each curves file [_flagged]")
    ap.add_argument("--n-coupler-severe", type=float, default=N_COUPLER_SEVERE,
                    help=f"coupler occupation above which the column is flagged "
                         f"`coupler_truncated`: 9 levels cannot represent it and "
                         f"1-F reads too good [{N_COUPLER_SEVERE:g}]")
    args = ap.parse_args()

    cls = classify(args.rundir, args.max_frac)
    load = coupler_load(args.rundir)
    if not cls:
        raise SystemExit(f"no chirp-free columns found under {args.rundir}")

    print(f"chirp-free columns found: {len(cls)}  (threshold {args.max_frac:.0%} "
          f"of a half-linewidth)")
    for reason, label in ((LEGIT, "genuinely chirp-free -> keep at 1.00x"),
                          (UNMEASURABLE, "chirp existed, not measured -> EXCLUDE"),
                          (RAILED, "ridge railed, no law at all -> EXCLUDE")):
        hits = sorted((k[0], k[1], v[1]) for k, v in cls.items() if v[0] == reason)
        if not hits:
            continue
        print(f"\n  {label}  ({len(hits)})")
        for eta, dm, frac in hits:
            f = "   --  " if frac is None else f"{frac:6.1%}"
            print(f"    eta={eta:<4} delta={dm:+5d} MHz   excursion {f}")

    for name in ("curves_nodrag", "curves_drag"):
        src = os.path.join(args.rundir, f"{name}.json")
        if not os.path.exists(src):
            print(f"\n{src}: missing -- skipped")
            continue
        dst = os.path.join(args.rundir, f"{name}{args.suffix}.json")
        n, total, n_trunc = apply_to(src, cls, load, dst, args.n_coupler_severe)
        print(f"\n{os.path.basename(dst)}: {total} rows, flagged "
              f"{n[LEGIT]} legit + {n[UNMEASURABLE]} unmeasurable + {n[RAILED]} railed"
              f"; {n_trunc} coupler-truncated (n_coupler > "
              f"{args.n_coupler_severe:g})")


if __name__ == "__main__":
    main()
