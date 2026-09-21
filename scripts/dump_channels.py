"""Plain-text listing of the DRAG channels targeted at every column.

The channel set lives nested inside each row of the scan HDF5, where a tree viewer
shows it as an opaque blob. This flattens it: what was PLAYED per column, what the
audit considered, and a tally of which physical processes the recursion actually
spent its slots on.

Usage:  dump_channels.py RUNDIR [OUT.txt]
"""
import glob
import os
import sys
from collections import Counter

from snail_solver.h5_io import load_doc

RUNDIR = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RUNDIR, "channels.txt")

L = []


def w(s=""):
    L.append(s)


tally = Counter()
played_tally = Counter()

for path in sorted(glob.glob(os.path.join(RUNDIR, "eta*.h5"))):
    doc = load_doc(path)
    scan = doc.get("scan") or doc
    rows = sorted(scan.get("rows", []), key=lambda r: r["delta_GHz"])
    if not rows:
        continue
    eta = rows[0].get("target_eta")
    w("=" * 100)
    w(f"eta* = {eta:g}    {os.path.basename(path)}    "
      f"{sum(1 for r in rows if r.get('ok'))}/{len(rows)} calibrated")
    w("=" * 100)
    for r in rows:
        dm = r["delta_GHz"] * 1e3
        if not r.get("ok"):
            st = (r.get("error") or {}).get("stage", "?")
            w(f"\ndelta = {dm:+7.0f} MHz   NOT CALIBRATED (failed at: {st}) "
              f"-- channels below were selected but never played")
        else:
            shed = r.get("drag_shed") or 0
            w(f"\ndelta = {dm:+7.0f} MHz   {r['n_drag_channels']} channel(s) played"
              + (f", {shed} shed by the chirp<->DRAG retry" if shed else "")
              + f"   t_g = {r['fidelity'].get('t_g_ns', float('nan')):.1f} ns")
        for c in (r.get("drag_channels") or []):
            lab = c.get("label") or "?"
            w(f"    {c['beat_GHz'] * 1e3:+9.1f} MHz  k={c['n_pump']}  "
              f"{str(c.get('category')):>8}  g={c.get('g_MHz') or float('nan'):7.2f} MHz"
              f"  g/|det|={c.get('ratio') or float('nan'):.3f}   {lab}")
            if r.get("ok"):
                played_tally[lab] += 1
            tally[lab] += 1
        # What the audit saw but did NOT correct, and why.
        for a in ((r.get("channel_audit") or {}).get("rows") or []):
            if a.get("selected") or a.get("category") == "target":
                continue
            if not a.get("reason"):
                continue
            if (a.get("g_MHz") or 0.0) < 1e-6:
                continue
            w(f"      (not corrected) {a['detuning_MHz']:+9.1f} MHz  k={a['n_pump']}  "
              f"{a['category']:>8}  g={a['g_MHz']:7.2f} MHz  -- {a['reason']}")

w("")
w("=" * 100)
w("TALLY -- how often each process was actually PLAYED (calibrated columns only)")
w("=" * 100)
for lab, n in played_tally.most_common():
    w(f"  {n:>4}x  {lab}")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}  ({len(L)} lines)")
print("\n".join(L[-12:]))
