"""Figures of the Floquet branch tracking (see docs/floquet-analysis.md).

    floquet_hybridization_maps     per leak process, the computational branch's
                                   population on the leak state over (delta, |eta|):
                                   each resonance is a fan from its zero-drive point
    floquet_branches_d<delta>      the labelled Floquet branches vs |eta| at one
                                   detuning, relative to the process's computational
                                   branch, coloured by their weight on the leak state
    floquet_leak_mechanisms        the measured infidelity against the two mechanisms:
                                   g/|beat| of the qubit-level processes, and the drive
                                   |eta|* at which the coupler process |010> -> |002>
                                   comes into resonance (crossed where below the peak)
    floquet_stueckelberg           the positive-side ripple: measured infidelity, and its
                                   period against 1/(3T), T the time each column's pulse
                                   spends above |eta|*
    replay_leak_dynamics           exact replays of calibrated pulses: P(t) of the
                                   leak states and the coupler occupation

Usage:  plot_floquet_tracking.py FLOQUET_DIR SCAN_DIR OUTDIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floquet_branch_map as M  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
ORDER = ("snail3", "k5_201", "a12", "asub", "b_002", "k4_201")
ETA_PEAK = 1.3
# weight on the leak state: surface grey -> orange 600 (one hue, light -> dark)
WEIGHT_CMAP = LinearSegmentedColormap.from_list("w", ["#dcdad3", "#f3a581", "#eb6834", "#a8401a"])
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6",
                                                     "#184f95", "#0d366b"])
BRANCH_FIGS = {-240: "snail3", -170: "k5_201", -70: "a12", 40: "b_002", 120: "b_002",
               200: "k4_201"}


def style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "axes.titlesize": 8.5})


def clean(ax, grid=True):
    if grid:
        ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out):
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".png/.pdf")


def hybridization_maps(an, outdir):
    d, eta = an["delta_MHz"], an["eta"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    norm = LogNorm(1e-3, 1)
    for ax, p in zip(axes.ravel(), ORDER):
        H = np.clip(an[f"hyb_{p}"], 1e-3, 1)
        im = ax.pcolormesh(d, eta, H.T, cmap=SEQ_CMAP, norm=norm, shading="nearest",
                           rasterized=True)
        ax.axhline(ETA_PEAK, color=CAT[1], lw=0.8, ls=(0, (3, 2)))
        ax.set_title(M.PROCESSES[p][3], fontsize=7)
        ax.set_xlim(-250, 250)
        ax.set_ylim(0, 1.5)
        clean(ax, grid=False)
    for ax in axes[1]:
        ax.set_xlabel(r"Pump detuning $\delta$ (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Constant pump $|\eta|$")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Computational branch's weight on the leak state")
    cb.outline.set_visible(False)
    axes[0, 0].text(245, ETA_PEAK - 0.04, "gate peak drive", fontsize=6.3, color=CAT[1],
                    va="top", ha="right")
    save(fig, os.path.join(outdir, "floquet_hybridization_maps"))


def branches_figure(fdir, delta, proc, outdir, window=320.0):
    path = M.path_for(fdir, 9, delta)
    if not os.path.exists(path):
        print("no map file at", delta)
        return
    tr = M.load(path)
    ini, fin, k, label = M.PROCESSES[proc]
    L = tr["labels"]
    eta = tr["eta"]
    E = tr["eps_GHz"] * 1e3                                          # (n, dim) MHz
    zone = tr["w_p_GHz"] * 1e3
    if ini == "doublet":
        ref = 0.5 * (E[:, L.index("100")] + E[:, L.index("010")]
                     - zone * np.round((E[:, L.index("010")] - E[:, L.index("100")]) / zone))
        comp = [L.index("100"), L.index("010")]
    else:
        ref = E[:, L.index(ini)]
        comp = [L.index(ini)]
    W = tr["weights"][fin]                                           # (n, dim)
    rel = (E - ref[:, None] + zone / 2) % zone - zone / 2
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    segs, cols = [], []
    for m in range(rel.shape[1]):
        y = rel[:, m]
        if m in comp or np.all(np.abs(y) > window):
            continue
        pts = np.column_stack([eta, y])
        s = np.stack([pts[:-1], pts[1:]], axis=1)
        ok = np.abs(np.diff(y)) < zone / 4                           # no fold jumps
        segs.append(s[ok])
        cols.append(0.5 * (W[:-1, m] + W[1:, m])[ok])
    segs = np.concatenate(segs)
    cols = np.concatenate(cols)
    o = np.argsort(cols)                                              # heavy on top
    lc = LineCollection(segs[o], cmap=WEIGHT_CMAP, norm=plt.Normalize(0, 0.5),
                        linewidths=0.4 + 1.6 * np.clip(cols[o] / 0.5, 0, 1))
    lc.set_array(cols[o])
    ax.add_collection(lc)
    for m in comp:
        y = rel[:, m]
        y = np.where(np.abs(np.diff(y, prepend=y[0])) > zone / 4, np.nan, y)
        ax.plot(eta, y, color=INK, lw=1.3, zorder=5)
    ax.axvline(ETA_PEAK, color=INK2, lw=0.5, ls=":")
    ax.set_xlim(0, eta[-1])
    ax.set_ylim(-window, window)
    ax.set_xlabel(r"Constant pump $|\eta|$")
    ax.set_ylabel(r"Quasienergy $-$ computational branch (MHz)")
    ini_lab = r"$|100\rangle,|010\rangle$" if ini == "doublet" else rf"$|{ini}\rangle$"
    ax.set_title(rf"$\omega_p/2\pi = {tr['w_p_GHz']:.3f}$ GHz: {label}", fontsize=7.2)
    ax.text(0.02, 0.97, f"black: {ini_lab} branch" + ("es" if ini == "doublet" else ""),
            transform=ax.transAxes, fontsize=6.5, va="top", color=INK)
    cb = fig.colorbar(lc, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label(rf"branch weight on $|{fin}\rangle$")
    cb.outline.set_visible(False)
    clean(ax)
    save(fig, os.path.join(outdir, f"floquet_branches_d{delta:+.0f}"))


def infidelity(scan):
    A = json.load(open(os.path.join(scan, "curves_nodrag_fine_flagged.json")))
    C = json.load(open(os.path.join(scan, "curves_subharmleak_fine_flagged.json")))

    def get(rows, name):
        d = np.array([round(x["delta_GHz"] * 1e3) for x in rows])
        y = np.array([((x.get("traces") or {}).get(name) or {}).get("infidelity_coherent",
                                                                  np.nan) for x in rows])
        o = np.argsort(d)
        return d[o], y[o]
    d, bare = get(A, "bare")
    _, drag = get(C, "chirp+DRAG")
    return d, bare, drag


def leak_mechanisms(an, scan, outdir):
    d, bare, drag = infidelity(scan)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True,
                                     gridspec_kw={"height_ratios": [1.4, 1, 1], "hspace": 0.12})
    a1.semilogy(d, bare, color=CAT[0], lw=1.0, label="Bare")
    a1.semilogy(d, drag, color=CAT[2], lw=1.0, label="Chirp + DRAG")
    a1.set_ylabel(r"$1-F$")
    a1.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    clean(a1)
    # qubit-level processes: first-order g at the target drive over the bare beat
    x = np.linspace(-150, 170, 1281)
    for (g, beat, lab, col) in ((43.0, -120 - 2 * x, r"A/B $|1\rangle\to|2\rangle$ (beat $-120-2\delta$)", CAT[3]),
                                (30.4, -2 * x, r"A subharmonic (beat $-2\delta$)", CAT[4])):
        with np.errstate(divide="ignore"):
            a2.semilogy(x, g / np.abs(beat), color=col, lw=1.1, label=lab)
    a2.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a2.set_ylim(0.05, 20)
    a2.set_ylabel(r"$g\,/\,|\mathrm{beat}|$")
    a2.legend(fontsize=6.5, frameon=False, loc="upper right")
    clean(a2)
    dl = an["delta_MHz"]
    em = np.where(dl >= 30, an["eta_meet_b_002"], np.nan)           # the identified branch
    crossed = np.isfinite(em) & (em < ETA_PEAK)
    a3.axvspan(dl[crossed].min(), dl[crossed].max(), color="#fbe3d6", lw=0, zorder=0)
    a3.plot(dl, em, color=CAT[1], lw=1.3, marker="o", ms=2.2)
    a3.axhline(ETA_PEAK, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a3.text(-148, ETA_PEAK + 0.03, "gate peak drive", fontsize=6.5, color=INK2, va="bottom")
    a3.text(100, 0.12,
            r"pulse crosses $|010\rangle\to|002\rangle$ twice", fontsize=6.8,
            color=INK2, ha="center")
    a3.set_ylim(0, 1.5)
    a3.set_ylabel(r"$|\eta|^*$ of $|010\rangle\to|002\rangle$")
    a3.set_xlabel(r"Pump detuning $\delta$ (MHz)")
    a3.set_xlim(-150, 170)
    clean(a3)
    save(fig, os.path.join(outdir, "floquet_leak_mechanisms"))


def stueckelberg_figure(an, scan, outdir):
    from scipy.signal import find_peaks
    d, bare, drag = infidelity(scan)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.1})
    a1.semilogy(d, bare, color=CAT[0], lw=1.0, label="Bare")
    a1.semilogy(d, drag, color=CAT[2], lw=1.0, label="Chirp + DRAG")
    a1.set_ylabel(r"$1-F$")
    a1.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    clean(a1)
    a2.plot(an["lzs_delta_MHz"], an["lzs_period_MHz"], color=INK, lw=1.2,
            label=r"predicted $1/(3T)$, no fitted parameter")
    for y, col, mk, lab in ((drag, CAT[2], "o", "measured, Chirp + DRAG"),
                            (bare, CAT[0], "s", "measured, Bare")):
        m = (d >= 10) & (d <= 165) & np.isfinite(y)
        xs, ys = d[m], np.log10(y[m])
        pk, _ = find_peaks(ys, prominence=0.08)
        xp = xs[pk]
        a2.plot(0.5 * (xp[1:] + xp[:-1]), np.diff(xp), mk, ms=3.6, mfc=col, mec="white",
                mew=0.6, ls="none", label=f"{lab} (peak spacing)")
    a2.set_ylim(0, 25)
    a2.set_xlim(0, 170)
    a2.set_ylabel(r"Ripple period in $\delta$ (MHz)")
    a2.set_xlabel(r"Pump detuning $\delta$ (MHz)")
    a2.legend(fontsize=6.5, frameon=False, loc="upper right")
    clean(a2)
    save(fig, os.path.join(outdir, "floquet_stueckelberg"))


def replay_figure(replay_json, outdir):
    if not os.path.exists(replay_json):
        print("no", replay_json)
        return
    R = {round(r["delta_MHz"]): r for r in json.load(open(replay_json))}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    for col, d in zip(CAT, (36, 41, 48, 52)):
        if d in R:
            r = R[d]
            t = np.asarray(r["t_ns"]) / r["t_g_ns"]
            ax.plot(t, r["100"]["traj"].get("002", np.zeros_like(t)), color=col, lw=0.9,
                    label=f"{d:+d} MHz")
    ax.set_title(r"$|100\rangle$ start: population of $|002\rangle$", fontsize=7.5)
    ax.legend(fontsize=6.3, frameon=False)
    ax2 = axes[1]
    if 200 in R:
        r = R[200]
        t = np.asarray(r["t_ns"]) / r["t_g_ns"]
        for col, s in zip(CAT, ("010", "100", "201", "021")):
            for src in ("100",):
                if s in r[src]["traj"]:
                    ax2.plot(t, r[src]["traj"][s], color=col, lw=0.9, label=rf"$|{s}\rangle$")
        ax2.plot(t, np.asarray(r["eta"]) / max(r["eta"]), color=INK2, lw=0.7, ls=":",
                 label=r"$|\eta(t)|$/peak")
        ax2.set_title(r"$\delta=+200$ MHz, $|100\rangle$ start", fontsize=7.5)
        ax2.legend(fontsize=6.3, frameon=False, loc="upper right")
    ax3 = axes[2]
    for col, d in zip((CAT[1], CAT[0], CAT[2]), (-240, 200, -130)):
        if d in R:
            r = R[d]
            ax3.semilogy(np.asarray(r["t_ns"]) / r["t_g_ns"], r["100"]["coupler_n"],
                         color=col, lw=0.9, label=f"{d:+d} MHz")
    ax3.axhline(8, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax3.text(0.02, 8.6, "9-level cutoff", fontsize=6.3, color=INK2)
    ax3.set_title(r"coupler $\langle n\rangle$, $|100\rangle$ start", fontsize=7.5)
    ax3.legend(fontsize=6.3, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
    for a in axes:
        a.set_xlabel(r"$t\,/\,t_g$")
        clean(a)
    fig.tight_layout()
    save(fig, os.path.join(outdir, "replay_leak_dynamics"))


def main():
    fdir, scan, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    style()
    an = dict(np.load(os.path.join(fdir, "analysis_levels9.npz")))
    hybridization_maps(an, outdir)
    for delta, proc in BRANCH_FIGS.items():
        branches_figure(fdir, delta, proc, outdir)
    leak_mechanisms(an, scan, outdir)
    stueckelberg_figure(an, scan, outdir)
    replay_figure(os.path.join(scan, "analysis", "replay_features.json"), outdir)


if __name__ == "__main__":
    main()
