#!/usr/bin/env python
"""Paired bare / chirp / chirp+DRAG comparison, per eta.

    paired_gain_table.py CURVES_NODRAG.json CURVES_DRAG.json [--csv OUT.csv]

Every ratio is computed over the columns where BOTH series in that ratio exist.
That is the whole point of this script: a median of `bare/chirp` over the columns
chirp solved, set beside a median of `bare/chirp+DRAG` over the columns DRAG solved,
is not a comparison -- the two medians land on different columns, and DRAG failing
on the columns where chirp helped most makes DRAG look worse than it is. Medians of
ratios also do not compose, so the third ratio is measured, never inferred from the
first two.

Series, and which file each comes from:

    bare        nodrag file, "bare" trace         no chirp, no DRAG, own length
    chirp       nodrag file, "chirp+DRAG" trace   chirp only (the pass played no
                                                  channels, so the flag is a no-op)
    chirp+DRAG  drag file,   "chirp+DRAG" trace   the real thing

Infidelity is the COHERENT one throughout: the open-system numbers exist to show
what slow gates cost, not to rank calibrations.
"""
import json
import sys
from statistics import median, geometric_mean

USAGE = __doc__.strip().splitlines()[2].strip()


def load(path, variant):
    """{(eta, delta_mhz): infidelity_coherent} for one variant of one curves file."""
    try:
        rows = json.load(open(path))
    except (OSError, ValueError) as exc:
        sys.exit(f"cannot read {path}: {exc}")
    out = {}
    for r in rows:
        tr = (r.get("traces") or {}).get(variant) or {}
        v = tr.get("infidelity_coherent")
        if v is not None and v > 0:
            out[(round(float(r["target_eta"]), 2),
                 round(float(r["delta_GHz"]) * 1e3))] = float(v)
    return out


def errors_by_stage(path):
    """{eta: {stage_or_type: [delta_mhz, ...]}} for the columns that did not solve."""
    out = {}
    for r in json.load(open(path)):
        if r.get("ok"):
            continue
        err = r.get("error") or {}
        kind = err.get("type") or "unknown"
        eta = round(float(r["target_eta"]), 2)
        out.setdefault(eta, {}).setdefault(kind, []).append(
            round(float(r["delta_GHz"]) * 1e3))
    return out


def ratios(num, den, keys):
    """num/den over `keys` -- den is the series being improved ON."""
    return [den[k] / num[k] for k in keys]


def block(name, vals):
    if not vals:
        return f"  {name:<24} --"
    helps = sum(1 for v in vals if v > 1.0)
    return (f"  {name:<24} n={len(vals):<3d} median {median(vals):5.2f}x  "
            f"geomean {geometric_mean(vals):5.3f}x  "
            f"range {min(vals):4.2f}-{max(vals):5.2f}x  helps {helps}/{len(vals)}")


def main(argv):
    if len(argv) < 3:
        sys.exit(f"usage: {USAGE}")
    nodrag_path, drag_path = argv[1], argv[2]
    bare = load(nodrag_path, "bare")
    chirp = load(nodrag_path, "chirp+DRAG")      # no channels were played
    cd = load(drag_path, "chirp+DRAG")

    etas = sorted({e for e, _ in set(bare) | set(chirp) | set(cd)})
    if not etas:
        sys.exit("no usable traces in either file")

    nd_err, d_err = errors_by_stage(nodrag_path), errors_by_stage(drag_path)
    rows_csv = [("eta", "delta_MHz", "bare", "chirp", "chirp_DRAG")]

    for eta in etas:
        at = lambda d: {k[1]: v for k, v in d.items() if k[0] == eta}
        b, c, x = at(bare), at(chirp), at(cd)
        triple = sorted(set(b) & set(c) & set(x))

        print(f"\n{'=' * 78}\neta = {eta}\n{'=' * 78}")
        print(f"  columns: bare {len(b)}, chirp {len(c)}, chirp+DRAG {len(x)}, "
              f"all three {len(triple)}")

        if triple:
            print("\n  Paired over the SAME columns (all three solved):")
            print(block("chirp over bare", ratios(c, b, triple)))
            print(block("chirp+DRAG over bare", ratios(x, b, triple)))
            print(block("DRAG on top of chirp", ratios(x, c, triple)))

        # Each pair over its own widest common set, so nothing is hidden by the
        # three-way intersection -- but labelled as a different set.
        wider = [("chirp over bare", c, b), ("chirp+DRAG over bare", x, b),
                 ("DRAG on top of chirp", x, c)]
        extra = [(n, sorted(set(num) & set(den))) for n, num, den in wider]
        if any(len(ks) != len(triple) for _, ks in extra):
            print("\n  Each pair over its own widest common set:")
            for (name, num, den), (_, ks) in zip(wider, extra):
                print(block(name, ratios(num, den, ks)))

        for label, series in (("bare", b), ("chirp", c), ("chirp+DRAG", x)):
            if series:
                d = min(series, key=series.get)
                print(f"\n  best {label:<11} delta={d:+5d} MHz  1-F={series[d]:.3e}",
                      end="")
        print()

        # A chirp built on a law that did not converge is not evidence either
        # way. Report those columns rather than averaging them into the gain.
        for path, label in ((nodrag_path, "chirp"), (drag_path, "chirp+DRAG")):
            bad = [(round(float(r["delta_GHz"]) * 1e3), r.get("quartic_fraction"))
                   for r in json.load(open(path))
                   if round(float(r["target_eta"]), 2) == eta and r.get("ok")
                   and r.get("perturbative_ok") is False]
            if bad:
                print(f"\n  {label}: shift law NOT converged (quartic fraction "
                      f"above the warn threshold) at {len(bad)} solved column(s):")
                for d, q in sorted(bad):
                    inpair = " [in the paired set]" if d in triple else ""
                    print(f"    delta={d:+5d}  quartic fraction {q:7.2f}{inpair}")

        print()
        for label, errs in (("chirp pass", nd_err), ("DRAG pass", d_err)):
            for kind, ds in sorted((errs.get(eta) or {}).items()):
                print(f"  {label} failed {kind:<24} {len(ds):>2d}: "
                      f"{', '.join(f'{v:+d}' for v in sorted(ds))}")

        for d in sorted(set(b) | set(c) | set(x)):
            rows_csv.append((eta, d, b.get(d, ""), c.get(d, ""), x.get(d, "")))

    if "--csv" in argv:
        path = argv[argv.index("--csv") + 1]
        with open(path, "w") as fh:
            fh.write("\n".join(",".join(str(v) for v in r) for r in rows_csv) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
