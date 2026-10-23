"""Coupler 0->1 AC-Stark shift vs |eta| from `floquet_coupler_shift.py`.

Top: the shift per column, the 12-level re-run at -170 MHz, and the weak-drive
eta^2 law fitted for |eta| <= 0.3. Bottom: how much of each tracked Floquet mode is
still its Fock state -- where it is near zero the "coupler level" is no longer
identifiable and the shift above it is not meaningful.

Usage:  plot_floquet_coupler_shift.py OUTDIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOURS = ("#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948", "#eda100")


def main():
    outdir = sys.argv[1]
    res = json.load(open(os.path.join(outdir, "floquet_shift_N1000.json")))
    l12 = os.path.join(outdir, "floquet_shift_N1000_d-170_levels12.json")
    fig, (ax, axw) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    for col, r in zip(COLOURS, res):
        eta = np.array([x["eta"] for x in r["rows"]])
        s = np.array([x["coupler01_MHz"] for x in r["rows"]])
        ax.plot(eta, s, color=col, lw=1.5, label=rf"$\delta$ = {r['delta_MHz']:+.0f} MHz")
        axw.plot(eta, [x["weight_001"] for x in r["rows"]], color=col, lw=1.3)
    if os.path.exists(l12):
        r = json.load(open(l12))
        ax.plot([x["eta"] for x in r["rows"]], [x["coupler01_MHz"] for x in r["rows"]],
                color="k", lw=1.2, ls=(0, (4, 2)), label=r"$-170$ MHz, 12 coupler levels")
    r = res[0]
    eta = np.array([x["eta"] for x in r["rows"]])
    s = np.array([x["coupler01_MHz"] for x in r["rows"]])
    m = (eta > 0) & (eta <= 0.3)
    k = float(np.sum(s[m] * eta[m] ** 2) / np.sum(eta[m] ** 4))
    ax.plot(eta, k * eta ** 2, color="0.5", lw=1, ls=":",
            label=rf"weak-drive law {k:.0f} MHz $\times|\eta|^2$")
    ax.set_ylabel("coupler 0→1 shift (MHz)")
    ax.set_title("Coupler AC-Stark shift (Floquet, constant pump)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    axw.set_ylabel("|0,0,1⟩ weight of\ntracked mode")
    axw.set_xlabel(r"pump $|\eta|$")
    axw.set_ylim(0, 1.02)
    axw.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "floquet_coupler_shift.png"), dpi=150)
    print(f"weak-drive coefficient {k:.1f} MHz/|eta|^2 (delta {r['delta_MHz']:+.0f})")


if __name__ == "__main__":
    main()
