"""Floquet branch tracking: the quasienergy spectrum vs |eta| with the tracked branch.

Per column: every quasienergy relative to the tracked |0,0,0> branch, folded into
one zone (-w_p/2, w_p/2] (grey). Modes with weight on |0,0,1> are drawn in blue, on
|1,0,0> in green and on |2,0,1> in orange, sized by that weight; the black line is the branch
`floquet_coupler_shift.py` tracks as the coupler level (max overlap step to step).
Where the blue weight leaves the black line, the coupler has hit an avoided crossing.

Usage:  plot_floquet_branches.py OUTDIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fold(x, zone):
    return (x + zone / 2) % zone - zone / 2


def main():
    outdir = sys.argv[1]
    res = json.load(open(os.path.join(outdir, "floquet_spectrum.json")))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, r in zip(axes.ravel(), res):
        zone = r["w_p_GHz"] * 1e3
        eta = np.array(r["eta"])
        eps = np.array(r["eps_MHz"])                          # (n_eta, dim)
        ref = eps[np.arange(len(eta)), r["tracked_index"]["000"]]
        rel = fold(eps - ref[:, None], zone)
        X = np.repeat(eta[:, None], rel.shape[1], axis=1)
        ax.scatter(X.ravel(), rel.ravel(), s=0.4, color="0.75", rasterized=True)
        for key, col, lab in (("001", "#2a78d6", "weight on |0,0,1⟩"),
                              ("100", "#1baf7a", "weight on |1,0,0⟩"),
                              ("201", "#eb6834", "weight on |2,0,1⟩")):
            w = np.array(r["weights"][key])
            m = w > 0.03
            ax.scatter(X[m], rel[m], s=18 * w[m], color=col, alpha=0.8, lw=0,
                       rasterized=True, label=lab)
        tr = rel[np.arange(len(eta)), r["tracked_index"]["001"]]
        tr = np.where(np.abs(np.diff(tr, prepend=tr[0])) > zone / 2, np.nan, tr)
        ax.plot(eta, tr, color="k", lw=1.2, label="tracked coupler branch")
        ax.set_title(rf"$\delta$ = {r['delta_MHz']:+.0f} MHz", fontsize=10)
        ax.set_ylim(-zone / 2, zone / 2)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"pump $|\eta|$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"quasienergy $-\ \varepsilon_{000}$, folded (MHz)")
    axes[0, 0].legend(fontsize=8, loc="lower left", markerscale=1.5)
    fig.suptitle("Floquet quasienergies vs drive and the tracked coupler branch")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "floquet_branches.png"), dpi=150)


if __name__ == "__main__":
    main()
