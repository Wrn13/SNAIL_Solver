"""Zoom on the +85 MHz infidelity spike of the 1 MHz-wide eta = 1.3 scan.

(a) coherent 1 - F of bare / chirp / chirp + DRAG over a detuning window;
(b) where the population goes: final populations of the leak states in exact replays
    of pass A's calibrated pulses (replay_leak_trajectories.py), from |100>, |010>
    (averaged) and from |110>;
(c) P(t) of those states during the pulse at one detuning, with |eta(t)| and the drive
    at which the |010> -> |002> branch meets the doublet (Floquet map, if present).

Usage:  plot_spike_zoom.py SCAN_DIR REPLAY_JSON OUT_STEM [LO HI [DELTA_T]]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

from plot_wide_curve import BAND, GRID, INK, INK2, SERIES, runs, series

CAT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
# (label, start states averaged, leak state)
LEAKS = ((r"$|002\rangle$", ("100", "010"), "002"),
         (r"$|004\rangle$", ("100", "010"), "004"),
         (r"$|003\rangle$", ("100", "010"), "003"),
         (r"$|102\rangle + |012\rangle$ (from $|110\rangle$)", ("110",), ("102", "012")),
         (r"$|104\rangle$ (from $|110\rangle$)", ("110",), "104"))


def pop(r, starts, states, final=True):
    states = (states,) if isinstance(states, str) else states
    acc = 0.0
    for s in starts:
        for lab in states:
            if final:
                acc = acc + r[s]["final"].get(lab, 0.0)
            elif lab in r[s]["traj"]:
                acc = acc + np.array(r[s]["traj"][lab])
    return acc / len(starts)


def main():
    scan, replay, stem = sys.argv[1:4]
    lo, hi = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (60, 110)
    dt = float(sys.argv[6]) if len(sys.argv) > 6 else 85
    fdir = os.path.join(scan, "floquet")
    fdir = fdir if os.path.exists(os.path.join(fdir, "analysis_levels9.npz")) else None
    d, ys, trunc, f_p0, eta_t = series(scan)
    R = sorted(json.load(open(replay)), key=lambda r: r["delta_MHz"])
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 1), hspace=0.55, wspace=0.32)
    a1 = fig.add_subplot(gs[0, :])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[1, 1])

    m = (d >= lo) & (d <= hi)
    for i, j in runs(trunc & m):
        a1.axvspan(d[i] - 0.5, d[j] + 0.5, color=BAND, lw=0, zorder=0)
    for key, label, col in SERIES:
        a1.plot(d[m], ys[key][m], color=col, lw=1.1, marker="o", ms=2, label=label)
    a1.axvline(dt, color=INK2, lw=0.5, ls=":")
    a1.set_yscale("log")
    a1.set_xlim(lo, hi)
    a1.set_ylabel(r"Coherent gate infidelity $1-F$")
    a1.set_xlabel(r"Pump detuning $\delta = (\omega_p - \omega_a/2)/2\pi$ (MHz)")
    top = a1.secondary_xaxis("top", functions=(lambda x: f_p0 + x * 1e-3,
                                               lambda f: (f - f_p0) * 1e3))
    top.set_xlabel(r"Pump frequency $\omega_p/2\pi$ (GHz)")
    a1.set_ylim(0.012, 0.25)
    a1.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2]))
    a1.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.legend(frameon=False, fontsize=7, loc="upper left", ncol=3,
              title=rf"(a)   $\eta^*$ = {eta_t:g}", title_fontsize=7.5,
              alignment="left")

    dr = np.array([r["delta_MHz"] for r in R])
    for (lab, starts, st), col in zip(LEAKS, CAT):
        a2.plot(dr, [pop(r, starts, st) for r in R], color=col, lw=1.1, marker="o", ms=2,
                label=lab)
    a2.axvline(dt, color=INK2, lw=0.5, ls=":")
    a2.set_xlabel(r"$\delta$ (MHz)")
    a2.set_ylabel("Final population")
    a2.set_title("(b) leak states at the end of the gate (chirp only)", fontsize=7.5, loc="left")
    a2.legend(frameon=False, fontsize=6, loc="upper left")

    r = next(x for x in R if round(x["delta_MHz"]) == round(dt))
    eta_meet = None
    if fdir:
        an = np.load(os.path.join(fdir, "analysis_levels9.npz"))
        eta_meet = float(np.interp(dt, an["delta_MHz"], an["eta_meet_b_002"]))
    t = np.array(r["t_ns"])
    for (lab, starts, st), col in zip(LEAKS, CAT):
        p = pop(r, starts, st, final=False)
        if np.ndim(p):
            a3.plot(t, p, color=col, lw=0.7, label=lab)
    a3.set_xlabel("t (ns)")
    a3.set_ylabel("Population")
    a3.set_xlim(0, t[-1])
    a3.set_title(rf"(c) during the pulse, $\delta = {dt:+.0f}$ MHz (colours as in b)", fontsize=7.5, loc="left")
    tw = a3.twinx()
    tw.plot(t, r["eta"], color=INK2, lw=0.8, ls="--")
    tw.set_ylabel(r"$|\eta(t)|$ (dashed)", color=INK2)
    tw.set_ylim(0, 1.45)
    if eta_meet is not None:
        tw.axhline(eta_meet, color=INK2, lw=0.5, ls=":")
        tw.text(1, eta_meet + 0.02,
                rf"$|010\rangle\to|002\rangle$ resonance" "\n" rf"$|\eta|$ = {eta_meet:.2f}",
                fontsize=6, color=INK2, va="bottom")
    for a in (a1, a2, a3):
        a.grid(True, color=GRID, lw=0.5)
        a.set_axisbelow(True)
    for a in (a2, a3):
        a.spines["top"].set_visible(False)
    a1.spines["right"].set_visible(False)
    a2.spines["right"].set_visible(False)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    print("wrote", stem + ".pdf/.png")


if __name__ == "__main__":
    main()
