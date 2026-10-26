"""Publication figure: gate infidelity vs pump detuning for the 1 MHz-wide eta = 1.3 scan.

Coherent 1 - F of the bare pulse, the ridge chirp, and chirp + the fixed DRAG set, per
detuning delta = (w_p - w_a/2)/2pi (the series of figs_fine/table.txt), with the
absolute pump frequency on the top axis. Columns where no chirp could be measured leave
a gap in the Chirp series (chirp == bare there is a calibration artefact, not a
measurement), as do pass C's failed columns. Grey bands: the coupler holds more than its
9-level cutoff represents, so the values there read too good.

Usage:  plot_wide_curve.py SCAN_DIR OUT_STEM   -> OUT_STEM.pdf / OUT_STEM.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SERIES = (("bare", "Bare", "#2a78d6"),
          ("chirp", "Chirp", "#eb6834"),
          ("drag", "Chirp + DRAG", "#1baf7a"))
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"


def series(scan):
    A = {round(x["delta_GHz"] * 1e3): x for x in
         json.load(open(os.path.join(scan, "curves_nodrag_fine_flagged.json")))}
    C = {round(x["delta_GHz"] * 1e3): x for x in
         json.load(open(os.path.join(scan, "curves_subharmleak_fine_flagged.json")))}
    d = np.array(sorted(A))

    def get(D, name, mask=None):
        v = []
        for k in d:
            t = ((D[k].get("traces") or {}).get(name) or {})
            ok = t and (mask is None or not D[k].get(mask))
            v.append(t.get("infidelity_coherent", np.nan) if ok else np.nan)
        return np.array(v, float)

    trunc = np.array([bool(C[k].get("coupler_truncated") or A[k].get("coupler_truncated"))
                      for k in d])
    f_p0 = float(np.median([A[k]["w_p_GHz"] - A[k]["delta_GHz"] for k in d]))   # w_a/4pi
    return d, {"bare": get(A, "bare"), "chirp": get(A, "chirp+DRAG", "chirp_excluded"),
               "drag": get(C, "chirp+DRAG")}, trunc, f_p0


def runs(mask):
    """[(start, stop)] index runs where mask is True."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def main():
    scan, stem = sys.argv[1], sys.argv[2]
    d, ys, trunc, f_p0 = series(scan)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, j in runs(trunc):
        ax.axvspan(d[i] - 0.5, d[j] + 0.5, color=BAND, lw=0, zorder=0)
    for key, label, col in SERIES:
        ax.plot(d, ys[key], color=col, lw=1.0, label=label, zorder=2 + (key == "drag"),
                solid_joinstyle="round")
    ax.set_yscale("log")
    ax.set_xlim(-252, 252)
    ax.set_ylim(1.5e-3, 1.2)
    ax.set_xticks(np.arange(-250, 251, 50))
    ax.set_xlabel(r"Pump detuning $\delta = (\omega_p - \omega_a/2)/2\pi$ (MHz)")
    ax.set_ylabel(r"Coherent gate infidelity $1-F$")
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)

    # absolute pump frequency f_p = w_a/4pi + delta on the top axis
    top = ax.secondary_xaxis("top", functions=(lambda x: f_p0 + x * 1e-3,
                                               lambda f: (f - f_p0) * 1e3))
    top.set_xticks(np.arange(1.50, 2.001, 0.05))
    top.set_xlabel(r"Pump frequency $\omega_p/2\pi$ (GHz)")
    top.tick_params(colors=INK2, width=0.6)

    best = int(np.nanargmin(ys["drag"]))
    ax.annotate(f"best: {ys['drag'][best]:.1e}\nat {d[best]:+.0f} MHz",
                (d[best], ys["drag"][best]), (d[best] + 14, 2.2e-3), fontsize=7,
                color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(color=BAND, label="Coupler beyond 9-level cutoff"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4,
              frameon=False, handlelength=1.8, fontsize=7.5)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    print("wrote", stem + ".pdf/.png")


if __name__ == "__main__":
    main()
