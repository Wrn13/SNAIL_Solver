"""Diagram: which interactions are DRAGged, and where in frequency they sit.

Built from a stored column's `channel_audit`, so every number is the one the code used.

Panel A -- the frequency landscape that PRODUCES the beats: a channel's beat is the
distance from the relevant pump harmonic (w_p or 2 w_p) to the transition it drives,
which is why moving `delta` moves the two-pump channels twice as fast.

Panel B -- the (detuning, coupling) plane, where DRAG lives. `drag_verdict` is about
g/|det|, so two lines bound the usable region:

    g = |det|        perturbativity: above it the recursion has no leading term
    g = 0.3 |det|    the selector's correctability threshold (max_ratio)

and |det| < 1/t_g (inside the pulse bandwidth, nothing to cancel) bounds it from the
other side. A channel is correctable only in the wedge between them.

Usage:  plot_channel_map.py COLUMN.json OUT.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(COL))
a = d["channel_audit"]
rows = [r for r in a["rows"] if r.get("g_MHz", 0) > 1e-9]      # drop the g=0 stubs
played = {(round(c["beat_GHz"] * 1e3), int(c["n_pump"]))
          for c in (d.get("drag_channels") or [])}

CAT = {"target": ("#5c5b55", "target (the gate)"),
       "coupler": ("#2a78d6", "coupler: SNAIL heating"),
       "leakage": ("#eb6834", r"leakage: $|2\rangle$ ladder"),
       "other": ("#1baf7a", "mode subharmonic"),
       "subharm": ("#1baf7a", "mode subharmonic")}
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d8d7d0"

w_p = float(d["w_p_GHz"])
dev = (d.get("run_doc") or {}).get("device") or {}
qf = dev.get("qubit_freqs_GHz") or [3.5, 5.13]
w_a, w_b = float(qf[0]), float(qf[1])
w_s = float(dev.get("coupler_freq_GHz", 4.7))
alpha = float(dev.get("anharm_qubit_GHz", -0.12))
bw = 1e3 / float(a["t_g_ns"])
delta_MHz = float(d["delta_GHz"]) * 1e3

fig, ax = plt.subplots(2, 1, figsize=(12, 9.4),
                       gridspec_kw={"height_ratios": [1.0, 1.45]})
for x in ax:
    x.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        x.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        x.spines[s].set_color(GRID)

# ---------------------------------------------------------------- panel A
A = ax[0]
A.set_xlim(1.0, 6.1)
A.set_ylim(-0.72, 1.45)
A.get_yaxis().set_visible(False)
A.spines["left"].set_visible(False)
A.axhline(0.0, color=GRID, lw=1.4)

levels = [(w_a + alpha, "$\\omega_a\\!+\\!\\alpha$\nA $|1\\rangle\\!\\to\\!|2\\rangle$",
           "#eb6834", 0.62, "right", -0.03),
          (w_a, "$\\omega_a$\nqubit A", "#1baf7a", 0.92, "left", 0.03),
          (w_s, "$\\omega_s$\nSNAIL", "#2a78d6", 0.62, "center", 0.0),
          (w_b, "$\\omega_b = \\omega_a\\!+\\!\\omega_p$\nqubit B",
           "#1baf7a", 0.92, "center", 0.0)]
for f, lab, col, ytop, ha, dx in levels:
    A.plot([f, f], [0, ytop], color=col, lw=2.4)
    A.plot([f], [ytop], "o", color=col, ms=7)
    A.annotate(lab, xy=(f + dx, ytop + 0.04), ha=ha, va="bottom", fontsize=8.5,
               color=col)

for f, lab, col, y in ((w_p, "$\\omega_p$ (pump)", "#4a3aa7", -0.24),
                       (2 * w_p, "$2\\omega_p$ (second harmonic)", "#4a3aa7", -0.50)):
    A.annotate("", xy=(f, 0.0), xytext=(f, y),
               arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
    A.annotate(lab, xy=(f, y - 0.04), ha="center", va="top", fontsize=8.5, color=col)

# The two-pump beats: each at its OWN height, labelled to the side of the gap so a
# 120 MHz gap on a 4 GHz axis is still readable.
for f, col, y in ((w_a + alpha, "#eb6834", 0.18), (w_a, "#1baf7a", 0.38)):
    A.annotate("", xy=(f, y), xytext=(2 * w_p, y),
               arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.4,
                               shrinkA=0, shrinkB=0))
    A.annotate(f"  {(f - 2 * w_p) * 1e3:+.0f} MHz", xy=(max(f, 2 * w_p), y),
               ha="left", va="center", fontsize=8.5, color=col)
# The coupler channel is a SUM process: a + pump -> SNAIL. On this branch
# w_a + w_p is exactly w_b, which is why the arrow starts under qubit B.
A.annotate("", xy=(w_s, 0.08), xytext=(w_a + w_p, 0.08),
           arrowprops=dict(arrowstyle="<|-|>", color="#2a78d6", lw=1.4,
                           shrinkA=0, shrinkB=0))
A.annotate(f"{(w_a + w_p - w_s) * 1e3:+.0f} MHz   $a\\!+\\!\\omega_p \\to s$  ",
           xy=(w_s, 0.08), ha="right", va="center", fontsize=8.5, color="#2a78d6")
A.set_xlabel("frequency  (GHz)", color=INK, fontsize=10)
A.set_title(f"What the pump reaches   ($\\delta$ = {delta_MHz:+.0f} MHz, "
            f"$\\eta^*$ = {d['target_eta']:g}, $\\omega_p$ = {w_p:.3f} GHz, "
            f"$2\\omega_p$ = {2*w_p:.3f} GHz)", color=INK, fontsize=12, pad=10)

# ---------------------------------------------------------------- panel B
B = ax[1]
det = np.array([abs(r["detuning_MHz"]) for r in rows])
g = np.array([r["g_MHz"] for r in rows])
lo, hi = 4.0, max(2200.0, det.max() * 1.4)
xs = np.array([lo, hi])
B.fill_between(xs, 0.3 * xs, xs, color="#eda100", alpha=0.10, lw=0)
B.fill_between(xs, xs, 1e4, color="#e34948", alpha=0.10, lw=0)
B.plot(xs, xs, color="#e34948", lw=1.6)
B.plot(xs, 0.3 * xs, color="#eda100", lw=1.6)
B.axvspan(lo, bw, color="#e34948", alpha=0.16, lw=0)
B.annotate(f"inside the pulse bandwidth\n$1/t_g$ = {bw:.1f} MHz\n(DRAG has nothing "
           f"to cancel)", xy=(bw * 1.15, 3.0), fontsize=8, color="#b22222",
           ha="left", va="bottom")
B.annotate("NOT perturbative   $g \\geq |\\Delta|$", xy=(hi * 0.42, hi * 0.52),
           fontsize=9, color="#b22222", ha="center")
B.annotate("too strong to correct   $g \\geq 0.3|\\Delta|$",
           xy=(hi * 0.42, hi * 0.155), fontsize=9, color="#9a6b00", ha="center")
B.annotate("DRAG effective", xy=(hi * 0.42, hi * 0.020), fontsize=10,
           color="#127a55", ha="center")

seen = set()
# Group by (|det|, g, n_pump): a conjugate pair (+/- the same beat) is one point in
# this plane, and drawing it twice just prints two labels on top of each other.
groups = {}
for r in rows:
    key = (round(abs(r["detuning_MHz"]), 3), round(r["g_MHz"], 6), int(r["n_pump"]))
    groups.setdefault(key, []).append(r)
for (adet, gval, k), grp in groups.items():
    r = grp[0]
    col, lab = CAT.get(r["category"], (MUTED, r["category"]))
    is_played = any((round(x["beat_GHz"] * 1e3), int(x["n_pump"])) in played
                    for x in grp)
    x = max(adet, lo * 1.02)
    B.plot([x], [gval], "o" if k == 1 else "^", color=col,
           ms=16 if is_played else 9, mfc=col if is_played else "none", mew=2.0,
           label=lab if lab not in seen else None, zorder=5)
    seen.add(lab)
    signs = sorted({("+" if x["detuning_MHz"] >= 0 else "-") for x in grp})
    tag = ("$\\pm$" if len(signs) == 2 else signs[0]) + f"{adet:.0f}"
    B.annotate(tag + ("  (x2)" if len(grp) > 1 and len(signs) == 1 else ""),
               xy=(x, gval * 1.28), ha="center", fontsize=8, color=col,
               fontweight="bold" if is_played else "normal")
B.set_xscale("log")
B.set_yscale("log")
B.set_xlim(lo, hi)
B.set_ylim(3.0, 500.0)
B.grid(True, which="both", color=GRID, lw=0.5, alpha=0.7)
B.set_axisbelow(True)
B.set_xlabel("channel detuning  $|\\Delta|$  (MHz)   -- how far the pump harmonic "
             "misses the transition", color=INK, fontsize=10)
B.set_ylabel("coupling  $g$  (MHz)", color=INK, fontsize=10)
B.set_title("Where each interaction sits, and whether DRAG can touch it   "
            "(filled = actually played;  circle = 1 pump, triangle = 2 pumps)",
            color=INK, fontsize=11, pad=10)
h, l = B.get_legend_handles_labels()
B.legend(h, l, frameon=False, fontsize=9, labelcolor=INK, ncol=4,
         loc="upper center", bbox_to_anchor=(0.5, -0.155))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT, dpi=160, facecolor="white")
print(f"wrote {OUT}")
