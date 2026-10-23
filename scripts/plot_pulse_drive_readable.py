"""Readable companions to `plot_pulse_spectra.py`: what each process actually gets.

1. ``excitation_at_processes.png`` -- per column, the first-order excitation
   probability every audited process receives from each pulse,

       P ~ | 2 pi (g / eta_pk^n) S[eta^n](Delta) |^2 ,

   with g the audit's coupling at peak |eta| = eta_pk (so g / eta_pk^n is the
   coupling per unit eta^n), S in ns and Delta the DRAG beat. One dot per pulse;
   the gate's own error (~3e-3) is drawn for scale. Linear response at the UNDRIVEN
   process frequencies: Stark shifts and adiabatic dressing are not in it.
2. ``spectrum_readable_d{delta}MHz.png`` -- the spectra as amplitude relative to the
   bare peak on a powers-of-ten axis, ripple removed (running max over 10 MHz),
   processes named along the top.

Usage:  plot_pulse_drive_readable.py OUTDIR [DELTA_MHz,...]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drag_sign_common as D  # noqa: E402

SPAN_MHz = 520.0
NU = np.linspace(-SPAN_MHz, SPAN_MHz, 2081) * 1e-3          # GHz, 0.5 MHz grid
SMOOTH_PTS = 21                                              # 10 MHz running max
VARIANTS = (
    ("bare",  "#eb6834", (0, (1, 1.5)), "o", "bare"),
    ("chirp", "#1baf7a", (0, (5, 2)),   "s", "ridge chirp"),
    ("drag",  "#2a78d6", "-",           "^", "chirp + DRAG, as played"),
    ("fixed", "#4a3aa7", (0, (6, 1.5, 1.5, 1.5)), "D", "chirp + DRAG, sign fixed"),
)
INK, MUTED, GRID, SURFACE = "#1a1a19", "#5c5b55", "#d8d7d0", "#fcfcfb"
GATE_ERR = 3e-3
P_FLOOR = 1e-30
NAMES = {"a1->0 b0->1 (1 pump)": "target iSWAP",
         "a1->2 (2 pump)": "A |1>->|2> subharmonic",
         "a1->0 (2 pump)": "A |0>-|1> subharmonic",
         "a0->1 (2 pump)": "A |0>-|1> subharmonic",
         "a1->0 b1->2 (1 pump)": "|11>->|02> leakage",
         "a1->2 b1->0 (1 pump)": "|11>->|20> leakage",
         "a1->0 s0->1 (1 pump)": "A -> SNAIL exchange"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def compute(delta):
    c = D.column(delta)
    eta_pk = float(c["audit"]["eta_peak"])
    spec = {}
    for key, *_ in VARIANTS:
        t, eta = D.eta_nominal(c["variants"][key])
        spec[key] = {n: D.spectrum(t, eta ** n, NU) for n in (1, 2, 3)}
    procs = {}                     # one entry per (name, beat): eta / eta* rows merge
    for r in c["audit"]["rows"]:
        n = abs(int(r["pump_quanta"]))
        if not n or r["category"] == "target" or abs(r["beat_MHz"]) > SPAN_MHz:
            continue
        name = NAMES.get(r["transition"], r["transition"])
        key = (name, round(r["beat_MHz"], 3))
        if key in procs:
            continue
        k = int(np.argmin(np.abs(NU * 1e3 - r["beat_MHz"])))
        g_unit = float(r["g_MHz"]) * 1e-3 / eta_pk ** n          # GHz per unit eta^n
        procs[key] = {"name": name, "n": n, "beat_MHz": float(r["beat_MHz"]),
                      "g_MHz": float(r["g_MHz"]),
                      "P": {v: float(abs(2 * np.pi * g_unit * spec[v][n][k]) ** 2)
                            for v in spec}}
    ws = float(c["device"]["coupler_freq_GHz"])
    return {"delta_MHz": delta, "spec": spec, "procs": list(procs.values()),
            "s3_beat_MHz": (ws - 3 * c["w_p_GHz"]) * 1e3}


def excitation_figure(cols, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), facecolor=SURFACE, sharex=True)
    for ax, col in zip(axes.ravel(), cols):
        _style(ax)
        procs = sorted(col["procs"], key=lambda p: (p["n"], p["beat_MHz"]))
        y = np.arange(len(procs))
        for j, (key, colr, _ls, mk, lab) in enumerate(VARIANTS):
            xs = [max(p["P"][key], P_FLOOR) for p in procs]
            ax.plot(xs, y + (j - 1.5) * 0.14, ls="", marker=mk, ms=7, color=colr,
                    mec=SURFACE, mew=1.0, label=lab)
        ax.axvline(GATE_ERR, color=INK, lw=1.0, ls=(0, (4, 2)))
        ax.set_yticks(y)
        ax.set_yticklabels([f"{p['name']}\n{p['beat_MHz']:+.0f} MHz, g={p['g_MHz']:.0f} MHz"
                            for p in procs], fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(P_FLOOR, 1)
        ax.set_title(rf"$\delta$ = {col['delta_MHz']:+.0f} MHz", fontsize=10, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("first-order excitation probability (log)", color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK, ls=(0, (4, 2))))
    l.append(f"gate error scale ({GATE_ERR:g})")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("How hard each pulse drives each audited process "
                 "(everything left of the dashed line is negligible)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def spectrum_figure(col, out):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), facecolor=SURFACE, sharex=True)
    titles = {1: r"$\eta$ about $\omega_p$ (1-pump processes)",
              2: r"$\eta^2$ about $2\omega_p$ (2-pump processes)",
              3: r"$\eta^3$ about $3\omega_p$ ($\omega_s \approx 3\omega_p$, effective)"}
    for ax, n in zip(axes, (1, 2, 3)):
        _style(ax)
        ref = float(np.max(np.abs(col["spec"]["bare"][n])))
        for key, colr, ls, _mk, lab in VARIANTS:
            y = maximum_filter1d(np.abs(col["spec"][key][n]) / ref, SMOOTH_PTS)
            ax.plot(NU * 1e3, np.maximum(y, 1e-16), color=colr, ls=ls, lw=1.7,
                    label=lab)
        marks = [(p["beat_MHz"], p["name"]) for p in col["procs"] if p["n"] == n]
        if n == 3:
            marks = [(col["s3_beat_MHz"], "SNAIL 3-pump")]
        for x, name in marks:
            ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (2, 2)))
            ax.annotate(name, (x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points", rotation=25,
                        fontsize=7.5, color=INK, ha="left", va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 3)
        ax.set_yticks([1, 1e-4, 1e-8, 1e-12, 1e-16])
        ax.set_ylabel("drive amplitude\n(relative to carrier)", color=MUTED, fontsize=8)
        ax.text(0.01, 0.95, titles[n], transform=ax.transAxes, fontsize=9, color=INK,
                va="top", bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    axes[-1].set_xlabel("offset from n x nominal pump frequency (MHz)", color=MUTED)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(rf"Pulse spectra, $\delta$ = {col['delta_MHz']:+.0f} MHz "
                 "(10 MHz running max; probability scales as amplitude squared)",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    outdir = sys.argv[1]
    deltas = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
              else [-185, -180, -175, -170, -165, -160])
    os.makedirs(outdir, exist_ok=True)
    cols = []
    for d in deltas:
        print(f"delta = {d:+.0f} MHz", flush=True)
        col = compute(d)
        cols.append(col)
        spectrum_figure(col, os.path.join(outdir, f"spectrum_readable_d{d:+.0f}MHz.png"))
    excitation_figure(cols, os.path.join(outdir, "excitation_at_processes.png"))
    table = [{"delta_MHz": c["delta_MHz"], **{k: v for k, v in p.items()}}
             for c in cols for p in c["procs"]]
    json.dump(table, open(os.path.join(outdir, "excitation_at_processes.json"), "w"),
              indent=1)
    for r in table:
        print(f"{r['delta_MHz']:+5.0f} {r['name']:26s} {r['beat_MHz']:+7.0f} "
              + " ".join(f"{k}={r['P'][k]:.1e}" for k, *_ in VARIANTS))


if __name__ == "__main__":
    main()
