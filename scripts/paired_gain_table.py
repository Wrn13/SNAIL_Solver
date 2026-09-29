#!/usr/bin/env python
"""Paired bare / chirp / chirp+DRAG comparison, per eta.

    paired_gain_table.py CURVES_NODRAG.json CURVES_DRAG.json [--csv OUT.csv]

Every ratio is computed over the columns where BOTH its series exist: medians over
different column sets are not a comparison (DRAG failing where chirp helped most would
make DRAG look worse than it is). Medians of ratios do not compose either, so the
third ratio is measured, never inferred from the first two.

Series, and which file each comes from:

    bare        nodrag file, "bare" trace         no chirp, no DRAG, own length
    chirp       nodrag file, "chirp+DRAG" trace   chirp only (the pass played no
                                                  channels, so the flag is a no-op)
    chirp+DRAG  drag file,   "chirp+DRAG" trace   the real thing

Infidelity is the COHERENT one throughout (open-system numbers show what slow gates
cost; they do not rank calibrations).
"""
import json
import sys
from statistics import median, geometric_mean

USAGE = __doc__.strip().splitlines()[2].strip()


def _eta(r):
    return round(float(r["target_eta"]), 2)


def _dmhz(r):
    return round(float(r["delta_GHz"]) * 1e3)


def load(path, variant, drop_excluded=False):
    """{(eta, delta_mhz): infidelity_coherent} for one variant of one curves file.

    `drop_excluded` removes the columns where a chirp could not be MEASURED: their
    "chirp" trace is the bare gate by construction, and a 1.00x there is the absence
    of a measurement, biasing the chirp's benefit down where the device is hardest
    (14 of 31 chirp-free columns on the 2026-09-22 grid). Columns chirp-free because
    there was NO SHIFT TO CHIRP are kept: there the 1.00x is the result.
    """
    try:
        rows = json.load(open(path))
    except (OSError, ValueError) as exc:
        sys.exit(f"cannot read {path}: {exc}")
    out = {}
    for r in rows:
        if drop_excluded and r.get("chirp_excluded"):
            continue
        tr = (r.get("traces") or {}).get(variant) or {}
        v = tr.get("infidelity_coherent")
        if v is not None and v > 0:
            out[(_eta(r), _dmhz(r))] = float(v)
    return out


#: reason code -> how to say it: different failures to measure a chirp, and a reader
#: needs to know which one hit their detuning.
_WHY = {
    "stark_crossing": "the ridge changed transition inside the drive sweep",
    "unmeasurable_chirp": "the fit failed while the law still swept a real shift",
    "railed_ridge": "the ridge railed; no shift law was fitted at all",
    # run_tune_up's own name for what the back-fill classifier
    # (backfill_chirp_exclusions.py) calls the two above; both must resolve, or a
    # re-analysed grid and a fresh one describe the same column differently.
    "chirp_not_converged": "delta0 + k2|eta|^2 + k4|eta|^4 does not describe the "
                           "measured ridge, though a real shift is there",
}


def excluded_columns(path):
    """{eta: [(delta_mhz, reason, detail)]} for columns whose chirp is unmeasurable."""
    out = {}
    for r in json.load(open(path)):
        if not r.get("chirp_excluded"):
            continue
        ce = r.get("stark_crossing_eta")
        frac = r.get("chirp_excursion_frac_linewidth")
        detail = (f"crossing at |eta| = {float(ce):.3f}" if ce is not None else
                  f"excursion {float(frac):.1%} of a half-linewidth"
                  if frac is not None else "")
        out.setdefault(_eta(r), []).append(
            (_dmhz(r), r.get("chirp_free_reason") or "unknown", detail))
    return out


def best_gate(path, variant):
    """{eta: (delta_mhz, infidelity, chirp_excluded)} over EVERY scored column.

    Deliberately not restricted to the ratio set: a chirp-excluded column still played
    a real pulse with a real fidelity (at eta = 1.3 the best gate, delta = -100 at
    2.382e-03, is such a DRAG-only gate).
    """
    out = {}
    for r in json.load(open(path)):
        tr = (r.get("traces") or {}).get(variant) or {}
        v = tr.get("infidelity_coherent")
        if v is None or v <= 0:
            continue
        eta = _eta(r)
        d = _dmhz(r)
        cur = out.get(eta)
        if cur is None or float(v) < cur[1]:
            out[eta] = (d, float(v), bool(r.get("chirp_excluded")))
    return out


def errors_by_stage(path):
    """{eta: {stage_or_type: [delta_mhz, ...]}} for the columns that did not solve."""
    out = {}
    for r in json.load(open(path)):
        if r.get("ok"):
            continue
        kind = (r.get("error") or {}).get("type") or "unknown"
        out.setdefault(_eta(r), {}).setdefault(kind, []).append(_dmhz(r))
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
    # The chirp series drops the columns where no chirp could be measured; `bare`
    # does not, because the bare gate at those columns is perfectly well measured.
    chirp = load(nodrag_path, "chirp+DRAG", drop_excluded=True)
    cd = load(drag_path, "chirp+DRAG", drop_excluded=True)
    excl = {k: v for d in (excluded_columns(nodrag_path),
                           excluded_columns(drag_path))
            for k, v in d.items()}
    # Best gate over every scored column, not over the ratio set -- see best_gate.
    best = {"bare": best_gate(nodrag_path, "bare"),
            "chirp": best_gate(nodrag_path, "chirp+DRAG"),
            "chirp+DRAG": best_gate(drag_path, "chirp+DRAG")}

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

        print("\n  Best gate found, over EVERY scored column (a column dropped from "
              "the\n  ratios above still played a real pulse):")
        for label in ("bare", "chirp", "chirp+DRAG"):
            hit = best[label].get(eta)
            if hit is None:
                continue
            d, v, was_excl = hit
            note = "   [chirp unmeasurable here -- DRAG-only gate]" if was_excl else ""
            print(f"    {label:<11} delta={d:+5d} MHz  1-F={v:.3e}{note}")

        # A chirp from a non-converged law is not evidence either way: list them.
        for path, label in ((nodrag_path, "chirp"), (drag_path, "chirp+DRAG")):
            bad = [(_dmhz(r), r.get("quartic_fraction"))
                   for r in json.load(open(path))
                   if _eta(r) == eta and r.get("ok")
                   and r.get("perturbative_ok") is False]
            if bad:
                print(f"\n  {label}: shift law NOT converged (quartic fraction "
                      f"above the warn threshold) at {len(bad)} solved column(s):")
                for d, q in sorted(bad):
                    inpair = " [in the paired set]" if d in triple else ""
                    print(f"    delta={d:+5d}  quartic fraction {q:7.2f}{inpair}")

        print()
        if excl.get(eta):
            rows_e = sorted(excl[eta])
            print(f"\n  chirp EXCLUDED at {len(rows_e)} column(s) -- no chirp could "
                  f"be measured, so `chirp == bare` there is an artefact of the\n"
                  f"  calibration rather than a measurement, and is NOT counted as "
                  f"1.00x above:")
            for d, reason, detail in rows_e:
                why = _WHY.get(reason, reason)
                print(f"    delta={d:+5d}  {why}"
                      + (f" ({detail})" if detail else ""))
            print("    Their bare and DRAG numbers ARE measured and are kept, which "
                  "is why the\n    bare column count above exceeds the chirp one.")

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
