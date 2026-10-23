"""Figure + table for `drag_sign_amp_scan.py`: 1 - F_avg vs peak |eta|, per column.

One panel per detuning, four variants (bare / ridge chirp / chirp + DRAG as played /
chirp + DRAG sign fixed). Hollow markers at eta = 1.3 are the STORED ridge-tracked
scores (figs_ridge_tracked/table.txt), which had the full step-3 calibration, for
reference.

Usage:  plot_drag_sign_amp_scan.py OUTDIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drag_sign_common as D  # noqa: E402

OUTDIR = sys.argv[1]
VARIANTS = (
    ("bare",  "#eb6834", (0, (1, 1.5)), "o", "bare"),
    ("chirp", "#1baf7a", (0, (5, 2)),   "s", "ridge chirp"),
    ("drag",  "#2a78d6", "-",           "^", "chirp + DRAG, as played"),
    ("fixed", "#4a3aa7", (0, (6, 1.5, 1.5, 1.5)), "D", "chirp + DRAG, sign fixed"),
)
STORED = {"bare": ("curves_nodrag_ridge_tracked.json", "bare"),
          "chirp": ("curves_nodrag_ridge_tracked.json", "chirp+DRAG"),
          "drag": ("curves_subharmleak_ridge_tracked.json", "chirp+DRAG")}
INK, MUTED, GRID, SURFACE = "#1a1a19", "#5c5b55", "#d8d7d0", "#fcfcfb"


def stored(delta, key):
    name, trace = STORED[key]
    rows = json.load(open(os.path.join(D.RES, name)))
    r = next(r for r in rows if round(r["delta_GHz"] * 1e3) == round(delta))
    return 1.0 - float(r["traces"][trace]["F_avg"])


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUTDIR, "amp_scan.jsonl"))
            if l.strip()]
    ok = [r for r in rows if r.get("error") is None]
    deltas = sorted({r["delta_MHz"] for r in rows})
    etas = sorted({r["eta"] for r in rows})

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for ax, d in zip(axes.ravel(), deltas):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6, which="both")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for key, col, ls, mk, lab in VARIANTS:
            pts = sorted((r["eta"], 1 - r["F_avg"]) for r in ok
                         if r["delta_MHz"] == d and r["variant"] == key)
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, color=col, ls=ls, lw=1.6, marker=mk, ms=6,
                        mec=SURFACE, mew=1.2, label=lab)
            if key in STORED:
                ax.plot([1.3], [stored(d, key)], marker=mk, ms=9, mfc="none",
                        mec=col, mew=1.4, ls="")
        ax.set_yscale("log")
        ax.set_title(rf"$\delta$ = {d:+.0f} MHz", fontsize=10, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"peak $|\eta|$", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$1 - F_{avg}$ (coherent)", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", mfc="none", mec=MUTED, ls=""))
    l.append(r"stored score at $\eta^*$ = 1.3 (full calibration)")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Gate error vs drive: ridge chirp from the stored Rabi tables, "
                 "length + carrier refit per point", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(os.path.join(OUTDIR, "fidelity_vs_peak_eta.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUTDIR, "fidelity_vs_peak_eta.txt"), "w") as f:
        f.write("1 - F_avg (coherent); length + carrier refit; '(x)' = failed\n")
        f.write(f"{'delta':>6} {'eta':>5} " + " ".join(f"{k:>10s}" for k, *_ in VARIANTS)
                + f" {'fixed/drag':>10s}\n")
        idx = {(r["delta_MHz"], r["eta"], r["variant"]): r for r in rows}
        for d in deltas:
            for e in etas:
                vals = []
                for k, *_ in VARIANTS:
                    r = idx.get((d, e, k))
                    vals.append(np.nan if r is None or r.get("error") else 1 - r["F_avg"])
                ratio = vals[3] / vals[2] if np.isfinite(vals[2]) and vals[2] else np.nan
                f.write(f"{d:+6.0f} {e:5.2f} " + " ".join(
                    f"{v:10.3e}" if np.isfinite(v) else f"{'(x)':>10s}" for v in vals)
                    + f" {ratio:10.2f}\n")
        errs = [r for r in rows if r.get("error")]
        for r in errs:
            f.write(f"FAILED {r['key']}: {r['error']}\n")
    print(open(os.path.join(OUTDIR, "fidelity_vs_peak_eta.txt")).read())


if __name__ == "__main__":
    main()
