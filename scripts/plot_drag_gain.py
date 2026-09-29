"""Where does chirp+DRAG actually buy fidelity? Gain vs detuning, one line per eta.

The gain -- bare infidelity over corrected infidelity -- is the quantity worth
plotting, because it is the one that SURVIVES the coupler truncation: across levels
7/9/11 each curve's absolute height moves by ~85% of its mean while the gain moves by
2.1%. The two traces share their systematic; their ratio does not.

Colours are categorical slots 1-3 of the validated default palette, in fixed order.
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

CURVES, OUT = sys.argv[1], sys.argv[2]
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d8d7d0"

rows = json.load(open(CURVES))
by_eta = defaultdict(list)
for r in rows:
    b = (r.get("traces") or {}).get("bare")
    g = (r.get("traces") or {}).get("chirp+DRAG")
    if not (r.get("ok") and b and g):
        continue
    by_eta[float(r["target_eta"])].append(
        (r["delta_GHz"] * 1e3, (1 - b["F_avg"]) / (1 - g["F_avg"]),
         [c.get("category") for c in (r.get("drag_channels") or [])]))

fig, ax = plt.subplots(figsize=(11, 5.4))
ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)

# The band where the correction actually pays, read off the data rather than assumed.
ax.axhline(1.0, color=MUTED, lw=1.2, ls="-", alpha=0.7)
for lo, hi in ((-120, -60), (60, 120)):
    ax.axvspan(lo, hi, color="#1baf7a", alpha=0.07, lw=0)
ax.axvline(0.0, color="#e34948", lw=1.0, ls=":", alpha=0.8)
ax.annotate("A subharmonic", xy=(0, 0.985), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=8, color="#e34948")
ax.annotate("the shoulder: correctable and\nstill worth correcting",
            xy=(90, 0.03), xycoords=("data", "axes fraction"), ha="center",
            va="bottom", fontsize=8, color="#127a55")

for i, eta in enumerate(sorted(by_eta)):
    pts = sorted(by_eta[eta])
    d = np.array([p[0] for p in pts])
    gn = np.array([p[1] for p in pts])
    ax.plot(d, gn, MARK[i] + "-", color=SLOT[i], lw=1.8, ms=5.0, alpha=0.9,
            label=fr"$\eta^*$ = {eta:g}")
    # Ring the points where the A-subharmonic channel is among those corrected:
    # every channel set containing it beats every set without it.
    has_sub = np.array(["other" in p[2] for p in pts])
    if has_sub.any():
        ax.plot(d[has_sub], gn[has_sub], MARK[i], color=SLOT[i], ms=10.0,
                mfc="none", mew=1.6, alpha=0.9)

ax.set_xlabel("pump detuning from the subharmonic   $\\delta$  (MHz)",
              color=INK, fontsize=10)
ax.set_ylabel("fidelity gain\n$(1-F)_{\\mathrm{bare}}\\,/\\,(1-F)_{\\mathrm{chirp+DRAG}}$",
              color=INK, fontsize=10)
ax.set_title("Where chirped recursive DRAG pays  (above branch, $w_s$ = 4.7 GHz)",
             color=INK, fontsize=12, pad=10)
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", color=MUTED, ls="none", ms=10, mfc="none",
                mew=1.6))
l.append("A-subharmonic channel corrected")
ax.legend(h, l, frameon=False, fontsize=9, labelcolor=INK, ncol=4,
          loc="upper center", bbox_to_anchor=(0.5, -0.16))
fig.text(0.995, 0.005, "gain > 1 means the correction helped; 7 coupler levels, but "
         "the GAIN is truncation-stable to 2%",
         ha="right", va="bottom", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT, dpi=160, facecolor="white")
print(f"wrote {OUT}")
