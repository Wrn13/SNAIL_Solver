"""Gate infidelity across the A-SNAIL exchange (scripts/run_asnail.sh output).

delta = w_s - w_a - w_p (w_p = 0.7 GHz - delta on devices/4.0Gate4.7SNAIL_asnail.json).
Several eta*: one panel of coherent 1-F curves per series (bare / chirp / chirp+DRAG),
one line per eta*, plus a delta x eta* heat map of the chirp. One eta*: the three series
on one axis. Columns whose chirp could not be measured are blanked in the chirp series,
as in plot_wide_curve.py; coupler-truncated columns are hatched grey on the curves.
Prints a table (delta, eta*, 1-F per series, n_coupler) to stdout.

Usage:  plot_asnail_map.py SCAN_DIR OUT_STEM   -> OUT_STEM.png / OUT_STEM.pdf
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = (("bare", "Bare", "#2a78d6"),
          ("chirp", "Chirp", "#eb6834"),
          ("drag", "Chirp + DRAG (ranked)", "#1baf7a"))
INK2, BAND = "#52514e", "#efeeea"


def load(scan, name):
    for suffix in ("_flagged", ""):
        path = os.path.join(scan, f"{name}{suffix}.json")
        if os.path.exists(path):
            return {(round(r["target_eta"], 2), round(r["delta_GHz"] * 1e3)): r
                    for r in json.load(open(path))}
    return {}


def coherent(r, trace, mask=None):
    t = ((r or {}).get("traces") or {}).get(trace) or {}
    if not t or (mask and r.get(mask)):
        return np.nan
    return t.get("infidelity_coherent", np.nan)


def main():
    scan, stem = sys.argv[1], sys.argv[2]
    A, C = load(scan, "curves_nodrag_fine"), load(scan, "curves_subharmleak_fine")
    etas = sorted({k[0] for k in A})
    ds = np.array(sorted({k[1] for k in A}))
    Y = {s: np.full((len(etas), len(ds)), np.nan) for s, _, _ in SERIES}
    trunc = np.zeros((len(etas), len(ds)), bool)
    print(f"{'delta':>6} {'eta*':>5} {'bare':>10} {'chirp':>10} {'drag':>10} {'n_cpl':>6}")
    for i, e in enumerate(etas):
        for j, d in enumerate(ds):
            a, c = A.get((e, d)), C.get((e, d))
            Y["bare"][i, j] = coherent(a, "bare")
            Y["chirp"][i, j] = coherent(a, "chirp+DRAG", "chirp_excluded")
            Y["drag"][i, j] = coherent(c, "chirp+DRAG")
            trunc[i, j] = bool((a or {}).get("coupler_truncated")
                               or (c or {}).get("coupler_truncated"))
            nc = (a or {}).get("n_coupler")
            print(f"{d:+6d} {e:5.2f} " + " ".join(f"{Y[s][i, j]:10.3e}" for s, _, _ in SERIES)
                  + (f" {nc:6.2f}" if nc is not None else "     --")
                  + ("  TRUNC" if trunc[i, j] else ""))

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    present = [s for s in SERIES if np.isfinite(Y[s[0]]).any()]
    step = float(np.median(np.diff(ds))) if len(ds) > 1 else 1.0
    if len(etas) == 1:
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        for j in np.flatnonzero(trunc[0]):
            ax.axvspan(ds[j] - step / 2, ds[j] + step / 2, color=BAND, lw=0, zorder=0)
        for key, label, col in present:
            ax.plot(ds, Y[key][0], color=col, lw=1.0, label=label)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\delta = \omega_s - \omega_a - \omega_p$ (MHz)")
        ax.set_ylabel(r"coherent $1-F$")
        ax.set_title(rf"A-SNAIL exchange scan, $\eta^*={etas[0]:g}$", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3, lw=0.4)
    else:
        n = len(present) + 1
        fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.2 * n), sharex=True)
        cmap = plt.get_cmap("viridis")
        for ax, (key, label, _) in zip(axes, present):
            for i, e in enumerate(etas):
                y = Y[key][i]
                ax.plot(ds, y, color=cmap(i / max(1, len(etas) - 1)), lw=1.0,
                        marker="o", ms=2, label=rf"$\eta^*={e:g}$")
                ax.plot(ds[trunc[i]], y[trunc[i]], "x", color=INK2, ms=3)
            ax.set_yscale("log")
            ax.set_ylabel(r"coherent $1-F$")
            ax.set_title(label + "   (x = coupler-truncated)", fontsize=8, loc="left")
            ax.grid(alpha=0.3, lw=0.4)
        axes[0].legend(frameon=False, fontsize=6.5, ncol=len(etas))
        ax = axes[-1]
        z = np.log10(Y["chirp"])
        de = np.diff(etas).mean() if len(etas) > 1 else 0.1
        im = ax.imshow(z, aspect="auto", origin="lower", cmap="magma_r",
                       extent=(ds[0] - step / 2, ds[-1] + step / 2,
                               etas[0] - de / 2, etas[-1] + de / 2))
        fig.colorbar(im, ax=ax, label=r"$\log_{10}(1-F)$ chirp", pad=0.01)
        ax.set_ylabel(r"$\eta^*$")
        ax.set_xlabel(r"$\delta = \omega_s - \omega_a - \omega_p$ (MHz)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
