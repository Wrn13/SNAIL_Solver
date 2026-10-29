"""Floquet branches at one detuning, only the states of the process, each labelled.

A stripped-down companion to floquet_branches_d<delta> (plot_floquet_tracking.py): the
quasienergy of each named branch, relative to the computational doublet's mean, vs the
constant pump |eta|. A branch is named by its zero-drive state (the tracking follows it
adiabatically), so past an avoided crossing the name is where it came from, not what
it holds.

Usage:  plot_floquet_branches_labeled.py FLOQUET_DIR DELTA_MHZ OUT_STEM [STATES]
        STATES: comma-separated, default 100,010,002
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floquet_branch_map as M  # noqa: E402
from plot_floquet_tracking import CAT, ETA_PEAK, INK2, clean, save, style  # noqa: E402


def main():
    fdir, delta, stem = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    states = sys.argv[4].split(",") if len(sys.argv) > 4 else ["100", "010", "002"]
    tr = M.load(M.path_for(fdir, 9, delta))
    L, eta = tr["labels"], tr["eta"]
    E = tr["eps_GHz"] * 1e3                                          # (n, dim) MHz
    zone = tr["w_p_GHz"] * 1e3
    i1, i2 = L.index("100"), L.index("010")
    ref = 0.5 * (E[:, i1] + E[:, i2] - zone * np.round((E[:, i2] - E[:, i1]) / zone))
    rel = (E - ref[:, None] + zone / 2) % zone - zone / 2

    style()
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ys = {}
    for s, col in zip(states, CAT):
        y = rel[:, L.index(s)]
        y = np.where(np.abs(np.diff(y, prepend=y[0])) > zone / 4, np.nan, y)   # fold jumps
        ys[s] = y
        ax.plot(eta, y, color=col, lw=1.4, label=rf"$|{s}\rangle$", zorder=3)
    lim = 50 * np.ceil(1.05 * np.nanmax(np.abs(np.concatenate(list(ys.values())))) / 50)
    ax.axvline(ETA_PEAK, color=INK2, lw=0.5, ls=":")
    ax.text(ETA_PEAK - 0.02, -0.96 * lim, "gate peak drive", fontsize=6.3, color=INK2,
            ha="right", va="bottom")
    ax.set_xlim(0, eta[-1])
    ax.set_ylim(-lim, lim)
    ax.set_yticks(np.arange(-lim, lim + 1, 50))
    ax.set_xlabel(r"Constant pump $|\eta|$")
    ax.set_ylabel("Quasienergy rel. to doublet (MHz)")
    ax.set_title(rf"$\omega_p/2\pi = {tr['w_p_GHz']:.3f}$ GHz", fontsize=7.2)
    ax.legend(loc="lower left", frameon=False, fontsize=7, handlelength=1.6,
              title="zero-drive state", title_fontsize=6.5)
    clean(ax)
    save(fig, stem)


if __name__ == "__main__":
    main()
