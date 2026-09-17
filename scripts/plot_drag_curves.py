"""One figure per target eta: the bare pulse against the calibrated chirp+DRAG gate.

Reads `curves.json` from `curve_drag_vs_bare.py`. Three stacked panels on a shared
delta axis, because the three quantities have different units and a dual y-axis is
never the answer:

    1  coherent infidelity 1-F_avg (log), both traces, with total infidelity as a
       thin dotted companion where coherence times were given
    2  gate length t_g for each trace -- each was fitted for ITS OWN pulse, so these
       genuinely differ and the difference is part of the comparison
    3  DRAG channels actually PLAYED (a step plot). At strong drive the shed-and-retry
       drops channels, and a "DRAG" trace carrying one channel is a different claim
       from one carrying three, so the count travels with the curve.

Columns the calibration refused are shaded rather than interpolated across.

Colours are categorical slots 1 and 2 of the validated default palette, in fixed
order, with line style and marker also differing so identity is never colour-alone.

Usage:  plot_drag_curves.py curves.json OUTDIR
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES, OUTDIR = sys.argv[1], sys.argv[2]

# name -> (colour, linestyle, marker, label)
STYLE = {
    "bare":       ("#eb6834", "--", "s", "no chirp, no DRAG"),
    "chirp+DRAG": ("#2a78d6", "-",  "o", "chirp + DRAG"),
}
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d8d7d0"


def _panel(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def figure_for(eta, rows, outdir):
    rows = sorted(rows, key=lambda r: r["delta_GHz"])
    d = np.array([r["delta_GHz"] * 1e3 for r in rows])          # MHz

    fig, ax = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True,
                           height_ratios=[2.4, 1.0, 0.7])
    for a in ax:
        _panel(a)

    # Failed columns: shade, never interpolate. A gap in the curve is the result --
    # and WHICH stage gave up is different physics, so the bands are coloured by it:
    # an audit refusal is a frequency-allocation verdict, a Rabi failure means the
    # shift law stopped being measurable (coupler occupation), and a chirp failure
    # means the chirp<->DRAG fixed point diverged.
    stage_colour = {"audit": "#e34948", "rabi": "#eda100", "chirp": "#4a3aa7"}
    seen_stages = {}
    for r in rows:
        if not r["ok"]:
            x = r["delta_GHz"] * 1e3
            st = ((r.get("error") or {}).get("stage") or "other")
            col = stage_colour.get(st, "#5c5b55")
            seen_stages[st] = col
            for a in ax:
                a.axvspan(x - 5, x + 5, color=col, alpha=0.13, lw=0)
    for a in ax:
        a.axvline(0.0, color="#e34948", lw=1.0, ls=":", alpha=0.8)
    ax[0].annotate("A subharmonic\n$2\\omega_p=\\omega_a$", xy=(0, 0.97),
                   xycoords=("data", "axes fraction"), ha="center", va="top",
                   fontsize=8, color="#e34948")

    for name, (col, ls, mk, lab) in STYLE.items():
        def series(field, sub=None):
            out = []
            for r in rows:
                t = (r.get("traces") or {}).get(name)
                v = None if t is None else (t.get(field) if sub is None
                                            else t.get(field))
                out.append(np.nan if v is None else float(v))
            return np.array(out)

        coh = series("infidelity_coherent")
        tot = series("infidelity_total")
        tg = series("t_g_ns")
        nch = series("n_drag_played")

        ax[0].plot(d, coh, ls=ls, marker=mk, ms=3.6, lw=2.0, color=col,
                   label=f"{lab}  ($1-F$)")
        if np.isfinite(tot).any():
            ax[0].plot(d, tot, ls=":", lw=1.2, color=col, alpha=0.85,
                       label=f"{lab}  (incl. decoherence)")
        ax[1].plot(d, tg, ls=ls, marker=mk, ms=3.2, lw=1.8, color=col, label=lab)
        ax[2].step(d, nch, where="mid", ls=ls, lw=1.8, color=col, label=lab)

    ax[0].set_yscale("log")
    ax[0].set_ylabel("infidelity  $1-F_{\\mathrm{avg}}$", color=INK, fontsize=10)
    # No \tfrac: matplotlib's mathtext does not implement it and the parse throws
    # from deep inside tight_layout, where the traceback names none of this.
    ax[0].set_title(f"Bare pulse vs chirped recursive DRAG   "
                    f"$\\eta^*={eta:g}$   (above branch, "
                    f"$\\omega_b = 1.5\\,\\omega_a + \\delta$)",
                    color=INK, fontsize=12, pad=12)
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].set_ylabel("$t_g$  (ns)", color=INK, fontsize=10)
    ax[2].set_ylabel("channels\nplayed", color=INK, fontsize=10)
    ax[2].set_yticks([0, 1, 2, 3])
    ax[2].set_xlabel("pump detuning from the subharmonic   $\\delta$  (MHz)",
                     color=INK, fontsize=10)

    n_ok = sum(1 for r in rows if r["ok"])
    from matplotlib.patches import Patch
    order = [s for s in ("audit", "rabi", "chirp", "other") if s in seen_stages]
    band_lbl = {"audit": "refused: channel audit",
                "rabi": "failed: shift law unmeasurable",
                "chirp": "failed: chirp/DRAG fixed point",
                "other": "failed: other"}
    handles = handles + [Patch(facecolor=seen_stages[s], alpha=0.13,
                               label=band_lbl[s]) for s in order]
    labels = labels + [band_lbl[s] for s in order]
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.012))
    fig.text(0.995, 0.001, f"{n_ok}/{len(rows)} columns calibrated",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    out = f"{outdir}/curve_eta{('%g' % eta).replace('.', 'p')}.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    return out, n_ok, len(rows)


def table_for(rows):
    rows = sorted(rows, key=lambda r: (r["target_eta"], r["delta_GHz"]))
    hdr = (f"{'delta':>7} {'eta':>5} {'nch':>4} {'t_g bare':>9} {'t_g drag':>9} "
           f"{'1-F bare':>11} {'1-F drag':>11} {'gain':>7}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        b = (r.get("traces") or {}).get("bare")
        g = (r.get("traces") or {}).get("chirp+DRAG")
        if not r["ok"] or b is None or g is None:
            stage = ((r.get("error") or {}).get("stage") or "refused")
            out.append(f"{r['delta_GHz'] * 1e3:>7.0f} {r['target_eta']:>5.2f} "
                       f"{'--':>4} {('FAILED ' + stage):>9}")
            continue
        gain = (1 - b["F_avg"]) / (1 - g["F_avg"]) if g["F_avg"] < 1 else float("nan")
        out.append(f"{r['delta_GHz'] * 1e3:>7.0f} {r['target_eta']:>5.2f} "
                   f"{g['n_drag_played']:>4d} {b['t_g_ns']:>9.1f} {g['t_g_ns']:>9.1f} "
                   f"{1 - b['F_avg']:>11.4e} {1 - g['F_avg']:>11.4e} "
                   f"{gain:>6.2f}x")
    return "\n".join(out)


if __name__ == "__main__":
    rows = json.load(open(CURVES))
    by_eta = defaultdict(list)
    for r in rows:
        by_eta[float(r["target_eta"])].append(r)
    for eta in sorted(by_eta):
        out, n_ok, n = figure_for(eta, by_eta[eta], OUTDIR)
        print(f"eta={eta:g}: {n_ok}/{n} calibrated -> {out}")
    txt = table_for(rows)
    open(f"{OUTDIR}/table.txt", "w").write(txt + "\n")
    print(f"\n{txt}")
