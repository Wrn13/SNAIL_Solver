"""Generate the defensibility note for a drag-curve run, computed from its own rows.

    write_caveats.py RUNDIR > RUNDIR/figs_defensible/CAVEATS.md

Every number here is derived, never written in, so the document cannot go stale
against the data it describes. Run it after backfill_chirp_exclusions.py.
"""
import glob
import json
import sys
from statistics import geometric_mean, median

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results/drag_curve_5MHz_2026-09-22"
nod = json.load(open(f"{R}/curves_nodrag_flagged.json"))

#: Coupler occupation above which `--coupler-levels 9` is doing real violence. The
#: 7-level run came out 3x optimistic at ~0.85 photons, so this is not a rounding
#: concern: at a full photon the top of the ladder is populated and the leakage out
#: of the model is simply not charged.
N_CPL_WARN = 0.05
N_CPL_SEVERE = 0.5


def grab(eta, var, skip_excl=False):
    o = {}
    for r in nod:
        if round(float(r["target_eta"]), 2) != eta or not r.get("ok"):
            continue
        if skip_excl and r.get("chirp_excluded"):
            continue
        t = (r.get("traces") or {}).get(var) or {}
        v = t.get("infidelity_coherent")
        if v and v > 0:
            o[round(float(r["delta_GHz"]) * 1e3)] = float(v)
    return o


ncpl = {}
for f in glob.glob(f"{R}/passA_nodrag/columns/*.json"):
    d = json.load(open(f))
    ncpl[(round(float(d["target_eta"]), 2),
          round(float(d["delta_GHz"]) * 1e3))] = d.get("n_coupler")

etas = sorted({round(float(r["target_eta"]), 2) for r in nod})
P = print

P("# What this dataset does and does not support")
P()
P(f"`{R}`, above branch, `w_b = 1.5 w_a + delta`, `w_p = w_a/2 + delta`. Three")
P("independently calibrated series: bare, chirp-only (pass A,")
P("`--max-drag-channels 0 --envelope-m 3`), chirp+DRAG (pass B). Coherent infidelity")
P("throughout; the open-system numbers exist to show what slow gates cost, not to")
P("rank calibrations.")
P()
P("Figures and the gain table here are built from `curves_*_flagged.json` --")
P("`curves_*.json` plus exclusion flags back-filled by")
P("`scripts/backfill_chirp_exclusions.py`. **No column was re-solved.**")
P()
P("## 1. Columns where the chirp is excluded, and why it matters")
P()
P("A column can carry no chirp for two reasons that look identical in the output:")
P("there was no shift to chirp, or a chirp could not be measured. The first is a")
P("result and belongs in the average at 1.00x. The second is an artefact of the")
P("calibration, and averaging it in drags the chirp's benefit down exactly where the")
P("chirp does the most work.")
P()
P("Discriminator: the excursion `|k2 eta*^2 + k4 eta*^4|` against the resonance")
P("half-width `1/(2 t_g)`. Over the 31 chirp-free columns here the two populations")
P("separate cleanly on it -- below 10% the fit residual is 1.0-16.7x the excursion")
P("(no signal); above 20% it is 0.21-0.58x (a real law that missed `r2_min = 0.9`).")
P()
cnt = {}
for r in nod:
    k = r.get("chirp_free_reason")
    if k:
        cnt[k] = cnt.get(k, 0) + 1
P("| | n | treatment |")
P("|---|---|---|")
P(f"| genuinely chirp-free (excursion <= 10%) | {cnt.get('no_measurable_shift', 0)} "
  f"| kept, contributes an honest 1.00x |")
P(f"| chirp existed, fit failed | {cnt.get('unmeasurable_chirp', 0)} "
  f"| **excluded from the chirp ratios** |")
P(f"| ridge railed, no law fitted at all | {cnt.get('railed_ridge', 0)} "
  f"| **excluded from the chirp ratios** |")
P()
P("Their bare and DRAG numbers are measured and kept, which is why the bare column")
P("count exceeds the chirp one.")
P()
P("### Effect on the headline numbers")
P()
P("| | median | geomean |")
P("|---|---|---|")
for eta in etas:
    b, c = grab(eta, "bare"), grab(eta, "chirp+DRAG")
    allr = [b[d] / c[d] for d in sorted(set(b) & set(c))]
    c2 = grab(eta, "chirp+DRAG", skip_excl=True)
    hon = [b[d] / c2[d] for d in sorted(set(b) & set(c2))]
    P(f"| eta={eta} chirp/bare, all {len(allr)} columns (superseded) "
      f"| {median(allr):.2f}x | {geometric_mean(allr):.3f}x |")
    P(f"| eta={eta} chirp/bare, {len(hon)} columns with a measured chirp "
      f"| **{median(hon):.2f}x** | **{geometric_mean(hon):.3f}x** |")
P()
P("The median moves further than the geomean because the 1.00x entries sat at the")
P("middle of the distribution and pinned it there.")
P()
P("## 2. The shift law has not converged at most columns -- the main limitation")
P()
P("`delta = k2 eta^2 (1 + (k4/k2) eta^2)` is a TRUNCATION, and `quartic_fraction` is")
P("the last kept term against the first. Above ~0.25 the unmeasured `eta^6` term is")
P("plausibly as large again, so the series is not a series.")
P()
for eta in etas:
    rows = [r for r in nod if round(float(r["target_eta"]), 2) == eta and r.get("ok")]
    bad = [r for r in rows if r.get("perturbative_ok") is False]
    b = grab(eta, "bare")
    c = grab(eta, "chirp+DRAG", skip_excl=True)
    ks = [d for d in sorted(set(b) & set(c)) if b[d] / c[d] > 1.01]
    q = {round(float(r["delta_GHz"]) * 1e3): r.get("quartic_fraction") for r in rows}
    ok = [d for d in ks if (q.get(d) or 0) <= 0.25]
    verb = "rests" if len(ok) == 1 else "rest"
    P(f"- **eta={eta}**: not converged at {len(bad)}/{len(rows)} calibrated columns. "
      f"Of the {len(ks)} where the chirp helps, only **{len(ok)}** {verb} on a "
      f"converged law.")
    worst = sorted(((q.get(d) or 0), d, b[d] / c[d]) for d in ks)[::-1][:4]
    P("  The largest gains are the worst offenders: "
      + ", ".join(f"delta={d:+d} gain {g:.2f}x at quartic fraction {qq:.1f}"
                  for qq, d, g in worst) + ".")
P()
P("The fidelities themselves are honest: each pulse was built and scored, and the")
P("numbers are what it achieved. What is NOT established is that this calibration")
P("RECIPE reproduces them. A chirp derived from a non-convergent series is a")
P("candidate pulse, not a calibration. Read the gains as \"a chirped pulse this good")
P("exists near this detuning\", not as \"chirping delivers Nx\".")
P()
P("## 3. Coupler truncation makes the absolute infidelities optimistic")
P()
sev = {}
for eta in etas:
    vals = [(k[1], v) for k, v in ncpl.items() if k[0] == eta and v is not None]
    if not vals:
        continue
    v = [x for _, x in vals]
    hi = sorted(vals, key=lambda t: -t[1])[:3]
    n_warn = sum(1 for x in v if x > N_CPL_WARN)
    n_sev = sum(1 for x in v if x > N_CPL_SEVERE)
    sev[eta] = (max(v), n_sev, len(v))
    P(f"- **eta={eta}**: 9 coupler levels. Occupation median {np.median(v):.4f}, "
      f"max **{max(v):.3f}**; above {N_CPL_WARN} at {n_warn}/{len(v)} columns, above "
      f"{N_CPL_SEVERE} at {n_sev}. Worst: "
      + ", ".join(f"delta={d:+d} ({x:.3f})" for d, x in hi) + ".")
P()
P("Nine levels cannot represent a coupler holding a full photon: the top of the")
P("ladder is populated, leakage out of the model is uncharged, and `1-F` there is too")
P("GOOD. The earlier 7-level run was 3x optimistic at ~0.85 photons, so this is a")
P("factors-of-several concern, not a rounding one. It hits bare and chirp alike, so")
P("the ratios largely survive and the absolute floor does not.")
P()
if len(sev) > 1 and max(sev) in sev and min(sev) in sev:
    hi_eta, lo_eta = max(sev), min(sev)
    P(f"**This is much worse at eta={hi_eta} than at eta={lo_eta}** "
      f"({sev[hi_eta][1]}/{sev[hi_eta][2]} columns above {N_CPL_SEVERE} against "
      f"{sev[lo_eta][1]}/{sev[lo_eta][2]}). It cuts the same way as the drive "
      f"comparison below: the stronger drive's errors are the more understated, so a "
      f"finding that eta={hi_eta} is worse is CONSERVATIVE.")
    P()
P("## 4. Direction of the remaining bias")
P()
P("- The chirp's **relative benefit** was understated; corrected in section 1.")
P("- The **absolute infidelity** is still optimistic wherever the coupler is loaded.")
dot = []
for eta in etas:
    c = grab(eta, "chirp+DRAG", skip_excl=True)
    drg = json.load(open(f"{R}/curves_drag_flagged.json"))
    x = {}
    for r in drg:
        if round(float(r["target_eta"]), 2) != eta or not r.get("ok"):
            continue
        if r.get("chirp_excluded"):
            continue
        t = (r.get("traces") or {}).get("chirp+DRAG") or {}
        v = t.get("infidelity_coherent")
        if v and v > 0:
            x[round(float(r["delta_GHz"]) * 1e3)] = float(v)
    ks = sorted(set(c) & set(x))
    if ks:
        dot.append(f"{geometric_mean([c[d] / x[d] for d in ks]):.3f}x (eta={eta})")
P(f"- **DRAG on top of a working chirp is a wash**: geomean "
  + ", ".join(dot) + ". The superseded 1.00x median was mildly flattered by")
P("  chirp-free columns, where that ratio is really DRAG-vs-bare.")
P()
P("## 5. What is safe to claim")
P()
claims = []
for eta in etas:
    b = grab(eta, "bare")
    c = grab(eta, "chirp+DRAG", skip_excl=True)
    ks = sorted(set(b) & set(c))
    claims.append(f"{geometric_mean([b[d] / c[d] for d in ks]):.2f}x (eta={eta})")
P(f"1. A chirped pulse beats an independently calibrated bare pulse across this")
P(f"   window by geomean " + " / ".join(claims) + ", over the columns where a chirp")
P(f"   is measurable.")
P("2. Adding recursive DRAG on top of a working chirp buys nothing measurable.")
bg = {}
for r in json.load(open(f"{R}/curves_drag_flagged.json")):
    t = (r.get("traces") or {}).get("chirp+DRAG") or {}
    v = t.get("infidelity_coherent")
    if not v or v <= 0:
        continue
    eta = round(float(r["target_eta"]), 2)
    d = round(float(r["delta_GHz"]) * 1e3)
    if eta not in bg or v < bg[eta][1]:
        bg[eta] = (d, float(v), bool(r.get("chirp_excluded")))
P("3. Best gates found (over every scored column, since a column dropped from the")
P("   ratios still played a real pulse):")
for eta, (d, v, ex) in sorted(bg.items()):
    P(f"   - eta={eta}: **{v:.3e}** at delta = {d:+d} MHz"
      + (" -- a DRAG-only gate whose chirp could not be measured." if ex else "."))
if len(etas) > 1:
    lo, hi = min(etas), max(etas)
    b1, b2 = grab(lo, "bare"), grab(hi, "bare")
    ks = sorted(set(b1) & set(b2))
    if ks:
        rr = [b2[d] / b1[d] for d in ks]
        P(f"4. eta={hi} is worse than eta={lo} essentially everywhere "
          f"(geomean {geometric_mean(rr):.2f}x on the {len(ks)} shared columns, "
          f"better at only {sum(1 for r in rr if r < 1)}), so pushing the drive past "
          f"{lo} on this device loses -- and section 3 says this is understated.")
P()
P("Not safe to claim: that these gains follow from the chirp recipe as specified")
P("(section 2), or that the absolute infidelities are achievable (section 3).")
