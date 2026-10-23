"""Frequency spectra of the ridge-chirped pulses, with the processes they must avoid.

For each column, four variants (see `drag_sign_common`): bare, chirp, chirp + DRAG
as played (pre-fix beat signs) and chirp + DRAG with the fixed signs. A process that
absorbs ``n`` pump quanta sees the spectrum of ``eta^n`` at its DRAG beat
``Delta = (E_f - E_i) - n w_p``, so the panels are

    n = 1   |S[eta]|    about   w_p      (target, |2> leakage, A-SNAIL exchange)
    n = 2   |S[eta^2]|  about 2 w_p      (A subharmonics |0>->|1>, |1>->|2>)
    n = 3   |S[eta^3]|  about 3 w_p      (w_s = 3 w_p: an effective, higher-order
                                          process -- no g3 term carries eta^3 s)

on the NOMINAL pump axis (the carrier offset folded in, so markers do not move
between variants), in dB against the bare pulse's peak. A DRAG channel at beat
``Delta`` multiplies ``S[eta^n]`` by ``(1 - nu/Delta)``: its notch sits at ``nu =
Delta`` -- on the process only when the sign is right.

Usage:  plot_pulse_spectra.py OUTDIR [DELTA_MHz,...]
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
DELTAS = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
          else [-185, -180, -175, -170, -165, -160])
SPAN_MHz = 520.0
NU = np.linspace(-SPAN_MHz, SPAN_MHz, 2081) * 1e-3          # GHz, 0.5 MHz grid
FLOOR_DB = -160.0

# (key, colour, linestyle, width, label); palette slots as in plot_pulse_time.py
VARIANTS = (
    ("bare",  "#eb6834", (0, (1, 1.5)), 1.4, "bare (no chirp, no DRAG)"),
    ("chirp", "#1baf7a", (0, (5, 2)),   1.4, "ridge chirp"),
    ("drag",  "#2a78d6", "-",           1.6, "chirp + DRAG, as played"),
    ("fixed", "#4a3aa7", (0, (6, 1.5, 1.5, 1.5)), 1.6, "chirp + DRAG, sign fixed"),
)
INK, MUTED, GRID, SURFACE = "#1a1a19", "#5c5b55", "#d8d7d0", "#fcfcfb"
PLAYED_C, FIXED_C = "#2a78d6", "#4a3aa7"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def column_spectra(delta):
    c = D.column(delta)
    out = {"delta_MHz": delta, "w_p_GHz": c["w_p_GHz"], "spectra": {}, "processes": [],
           "channels": {}}
    for key, *_ in VARIANTS:
        t, eta = D.eta_nominal(c["variants"][key])
        out["spectra"][key] = {n: np.abs(D.spectrum(t, eta ** n, NU)) for n in (1, 2, 3)}
    for key in ("drag", "fixed"):
        out["channels"][key] = [(float(ch.beat_GHz), int(ch.n_pump))
                                for ch in c["variants"][key]["chans"]]
    for r in c["audit"]["rows"]:
        n = abs(int(r["pump_quanta"]))
        if n and abs(r["beat_MHz"]) <= SPAN_MHz:
            out["processes"].append({"n": n, "beat_MHz": float(r["beat_MHz"]),
                                     "label": r["transition"].replace(" (", "\n("),
                                     "category": r["category"],
                                     "g_MHz": float(r["g_MHz"])})
    # w_s = 3 w_p: effective only (not in the g3 audit), marked so the n=3 panel reads
    nom = c["w_p_GHz"]
    ws = float(c["device"]["coupler_freq_GHz"])
    out["processes"].append({"n": 3, "beat_MHz": (ws - 3 * nom) * 1e3,
                             "label": "s0->1\n(3 pump, eff.)", "category": "coupler",
                             "g_MHz": float("nan")})
    return out


def draw_row(axes, sp, show_legend=False, title=True):
    for j, n in enumerate((1, 2, 3)):
        ax = axes[j]
        _style(ax)
        ref = float(np.max(sp["spectra"]["bare"][n]))
        for key, col, ls, lw, lab in VARIANTS:
            y = 20 * np.log10(np.maximum(sp["spectra"][key][n] / ref, 1e-12))
            ax.plot(NU * 1e3, y, color=col, ls=ls, lw=lw, label=lab)
        ax.axhspan(FLOOR_DB, -100, color=GRID, alpha=0.25, lw=0, zorder=0)
        # played / fixed DRAG beats for the channels of this order
        for key, colr, yy in (("drag", PLAYED_C, -3.0), ("fixed", FIXED_C, -9.0)):
            for b, k in sp["channels"][key]:
                if k == n and abs(b * 1e3) <= SPAN_MHz:
                    ax.plot([b * 1e3], [yy], marker="v", ms=8, color=colr,
                            mec=SURFACE, mew=1.5, zorder=5)
        # one marker per beat: the eta and eta* rows of one process share it
        marks = {}
        for p in sp["processes"]:
            if p["n"] == n:
                marks.setdefault(round(p["beat_MHz"], 3), []).append(p)
        for x, ps in marks.items():
            target = any(p["category"] == "target" for p in ps)
            ax.axvline(x, color=INK if target else MUTED, lw=0.8, ls=(0, (2, 2)),
                       zorder=1)
            ax.text(x, FLOOR_DB + 3, " / ".join(p["label"].split("\n")[0] for p in ps)
                    + "\n" + ps[0]["label"].split("\n")[-1], rotation=90,
                    fontsize=6.5, color=INK, ha="right", va="bottom")
        ax.set_ylim(FLOOR_DB, 6)
        ax.set_xlim(-SPAN_MHz, SPAN_MHz)
        if title:
            ax.set_title(rf"$|S[\eta^{n}]|$ about ${n if n > 1 else ''}\omega_p$"
                         rf"  ($\delta$ = {sp['delta_MHz']:+.0f} MHz)",
                         fontsize=9, color=INK)
        ax.set_xlabel(rf"offset from {n if n > 1 else ''}$\omega_p^{{nom}}$ (MHz)",
                      fontsize=8, color=MUTED)
        if j == 0:
            ax.set_ylabel("dB vs bare peak", fontsize=8, color=MUTED)
    if show_legend:
        h, l = axes[0].get_legend_handles_labels()
        h += [plt.Line2D([], [], marker="v", ls="", color=PLAYED_C, ms=8),
              plt.Line2D([], [], marker="v", ls="", color=FIXED_C, ms=8),
              plt.Line2D([], [], ls=(0, (2, 2)), color=MUTED)]
        l += ["DRAG beat as played", "DRAG beat, fixed sign", "process (fixed audit beat)"]
        return h, l


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for d in DELTAS:
        print(f"delta = {d:+.0f} MHz", flush=True)
        sp = column_spectra(d)
        rows.append(sp)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
        h, l = draw_row(axes, sp, show_legend=True)
        fig.legend(h, l, loc="lower center", ncol=7, fontsize=7.5, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(OUTDIR, f"spectrum_d{d:+.0f}MHz.png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.3 * len(rows)),
                             facecolor=SURFACE, squeeze=False)
    for i, sp in enumerate(rows):
        h, l = draw_row(axes[i], sp, show_legend=True)
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Ridge-tracked 5 MHz curve: pulse spectra, eta* = 1.3 "
                 "(stored operating points)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    fig.savefig(os.path.join(OUTDIR, "spectra_grid.png"), dpi=130)
    plt.close(fig)

    # the number DRAG exists to lower: |S[eta^n]| AT each process, per variant
    summary = []
    for sp in rows:
        for p in sp["processes"]:
            k = int(np.argmin(np.abs(NU * 1e3 - p["beat_MHz"])))
            ref = float(np.max(sp["spectra"]["bare"][p["n"]]))
            summary.append({"delta_MHz": sp["delta_MHz"], "process": p["label"].replace("\n", " "),
                            "n": p["n"], "beat_MHz": p["beat_MHz"], "g_MHz": p["g_MHz"],
                            **{f"dB_{key}": float(20 * np.log10(
                                max(sp["spectra"][key][p["n"]][k] / ref, 1e-12)))
                               for key, *_ in VARIANTS}})
    json.dump(summary, open(os.path.join(OUTDIR, "spectra_at_processes.json"), "w"), indent=1)
    with open(os.path.join(OUTDIR, "spectra_at_processes.txt"), "w") as f:
        f.write("|S[eta^n]| at each process, dB vs the bare pulse's peak\n")
        f.write(f"{'delta':>6} {'process':28s} {'n':>2} {'beat':>8} "
                + " ".join(f"{k:>7s}" for k, *_ in VARIANTS) + "\n")
        for s in summary:
            f.write(f"{s['delta_MHz']:+6.0f} {s['process']:28s} {s['n']:2d} "
                    f"{s['beat_MHz']:+8.1f} "
                    + " ".join(f"{s['dB_' + k]:7.1f}" for k, *_ in VARIANTS) + "\n")
    print(open(os.path.join(OUTDIR, "spectra_at_processes.txt")).read())


if __name__ == "__main__":
    main()
