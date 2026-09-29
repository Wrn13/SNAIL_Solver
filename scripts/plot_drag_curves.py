"""One figure per target eta: the bare pulse against the calibrated chirp+DRAG gate.

One panel, the two traces side by side; colour and style both carry the variant, and
in-window resonances are dashed lines. Gate lengths (fitted per trace) are listed per
column in `table.txt` instead.

Usage:  plot_drag_curves.py [--bars-only] curves.json OUTDIR [DEVICE.json] [BASELINE.json]
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --bars-only strips every overlay (trend, paired band, improvement wedge). The
# column-to-column structure is physics, not calibration noise -- a replayed fixed
# pulse loses to the locally calibrated one by 1.4-6x everywhere, and one resonance
# MOVES with drive -- so a 20 MHz smoothing kernel averages across real features.
_argv = [a for a in sys.argv if a != "--bars-only"]
BARS_ONLY = len(_argv) != len(sys.argv)
sys.argv = _argv

CURVES, OUTDIR = sys.argv[1], sys.argv[2]
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "devices/6Gate4.7SNAIL.json"
# Optional 4th arg: curves.json from a --max-drag-channels 0 run, the HONEST no-DRAG
# baseline. A DRAG run's own `bare` is not one: its length and carrier come from a
# chirp<->DRAG fixed point that assumed the correction would be played, and partly
# compensate for the missing chirp (at delta = -130: 4.80e-3 against 1.155e-2 for an
# honestly calibrated bare pulse, 2.4x too good). With a baseline, its bare replaces
# this run's.
BASELINE = sys.argv[4] if len(sys.argv) > 4 else None

# name -> (colour, linestyle, marker, filled, label). Categorical slots 1 and 2 of the
# validated default palette, in fixed order; style and fill duplicate the encoding so
# identity never rests on colour alone.
STYLE = {
    "bare":       ("#eb6834", "--", "s", False, "no chirp, no DRAG"),
    "chirp+DRAG": ("#2a78d6", "-",  "o", True,  "chirp + DRAG"),
}
BASE_STYLE = {
    "bare": ("#eb6834", "--", "s", False,
             "no chirp, no DRAG (independently calibrated)"),
}
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d8d7d0"


def resonances(device_path):
    """In-window channel resonances (MHz), derived from the device (a 5 MHz scan of
    `interaction_channels` over +-400 MHz finds exactly these two zero crossings):

      delta = 0        2 w_p = w_a          the qubit-A subharmonic the scan straddles
      delta = alpha/2  2 w_p = w_a + alpha  the A |1>->|2> ladder, from
                                            E(n) = n w_a + alpha n(n-1)/2

    Two more sit outside the window and are annotated rather than drawn.
    """
    A_SUB = "A subharmonic   $2\\omega_p=\\omega_a$"
    A_12 = "A $|1\\rangle\\!\\to\\!|2\\rangle$   $2\\omega_p=\\omega_a\\!+\\!\\alpha$"
    try:
        dev = json.load(open(device_path))
    except Exception:
        return [(0.0, A_SUB), (-60.0, A_12)], []
    w_a = float(dev["qubit_freqs_GHz"][0])
    w_s = float(dev["coupler_freq_GHz"])
    alpha = float(dev["anharm_qubit_GHz"])
    return ([(0.0, A_SUB), (alpha * 1e3 / 2.0, A_12)],
            [((w_s - w_a) * 1e3 / 2.0, "SNAIL subharmonic"),
             ((w_s - 1.5 * w_a) * 1e3, "A-SNAIL spectator")])


RES_IN, RES_OUT = resonances(DEVICE)


def _trend(d, y, sigma=20.0, gap=60.0):
    """Gaussian-kernel local mean and scatter of `y` over the detuning axis.

    Smoothed in DELTA, not column index: the grid is 10 MHz inside +-200 MHz and
    50-100 MHz outside, so an index kernel would mix very different spacings. The
    band is the local weighted sd.

    Returns (mean, sd, segments) where `segments` splits the axis at gaps wider than
    `gap` so the trend never bridges a region the calibration refused.
    """
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.empty_like(y)
    s = np.empty_like(y)
    for i, x0 in enumerate(d):
        w = np.exp(-0.5 * ((d - x0) / sigma) ** 2)
        m[i] = np.sum(w * y) / np.sum(w)
        s[i] = np.sqrt(max(np.sum(w * (y - m[i]) ** 2) / np.sum(w), 0.0))
    cut = np.nonzero(np.diff(d) > gap)[0]
    segments = np.split(np.arange(d.size), cut + 1)
    return m, s, segments


def _bkey(r):
    return (round(r["delta_GHz"] * 1e3), round(float(r["target_eta"]), 3))


def load_baseline(path):
    """{(delta_rounded, eta): row} from an independently calibrated no-DRAG run."""
    if not path:
        return {}
    try:
        return {_bkey(r): r for r in json.load(open(path))}
    except Exception as exc:
        print(f"(ignoring baseline {path}: {exc})")
        return {}


BASE_ROWS = load_baseline(BASELINE)


def figure_for(eta, rows, outdir):
    """One eta, one panel: grouped bars, bare against chirp+DRAG.

    A categorical column axis with log y, because the quantity is a per-column
    comparison, not a continuous function. Every scanned column gets a slot whether or
    not it calibrated, so a gap reads as a gap.
    """
    rows = sorted(rows, key=lambda r: r["delta_GHz"])
    x = np.arange(len(rows), dtype=float)
    d = np.array([r["delta_GHz"] * 1e3 for r in rows])

    fig, ax = plt.subplots(figsize=(max(11.0, 0.26 * len(rows) + 4.0), 6.2))
    ax.grid(axis="y", which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(axis="y", which="minor", color=GRID, lw=0.4, alpha=0.45)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)

    # Failed columns: a slot shaded by the stage that gave up (audit refusal,
    # unmeasurable shift law, diverged chirp fixed point are different physics).
    stage_colour = {"audit": "#e34948", "rabi": "#eda100", "chirp": "#4a3aa7"}
    seen_stages = {}
    for i, r in enumerate(rows):
        if not r["ok"]:
            st = ((r.get("error") or {}).get("stage") or "other")
            col = stage_colour.get(st, MUTED)
            seen_stages[st] = col
            ax.axvspan(i - 0.5, i + 0.5, color=col, alpha=0.13, lw=0, zorder=0)

    # Hatched: calibrated, but the CHIRP is not a measurement (the bare number stands).
    n_excluded = 0
    for i, r in enumerate(rows):
        if r.get("chirp_excluded"):
            n_excluded += 1
            ax.axvspan(i - 0.5, i + 0.5, facecolor="none", edgecolor=MUTED,
                       hatch="///", lw=0.0, alpha=0.5, zorder=0)

    # Triangles: coupler loaded past what 9 levels represent; both bars read too GOOD
    # (leakage off the ladder is uncharged). The comparison is fine, the absolute not.
    xt = [i for i, r in enumerate(rows) if r.get("coupler_truncated")]
    n_trunc = len(xt)
    if n_trunc:
        ax.plot(xt, [ax.get_ylim()[1] * 0.72] * len(xt), marker="v", ls="none",
                ms=6.0, color="#e34948", zorder=5, clip_on=False)

    # Series to draw: the run's own two, plus the independent baseline if given.
    if BASE_ROWS:
        # baseline bare + this run's chirp+DRAG; this run's own bare is superseded.
        series = ([(n, s, "baseline") for n, s in BASE_STYLE.items()]
                  + [(n, s, None) for n, s in STYLE.items() if n != "bare"])
    else:
        series = [(n, s, None) for n, s in STYLE.items()]
    nser = len(series)
    w = 0.86 / nser
    trend = {}
    for k, (name, (col, _ls, _mk, filled, lab), src_tag) in enumerate(series):
        if src_tag == "baseline":
            y = np.array([
                (1.0 - t["F_avg"])
                if (br := BASE_ROWS.get(_bkey(r)))
                and (t := (br.get("traces") or {}).get(name)) else np.nan
                for r in rows])
        else:
            y = np.array([(1.0 - t["F_avg"])
                          if (t := (r.get("traces") or {}).get(name)) else np.nan
                          for r in rows])
            if name == "chirp+DRAG":
                # No chirp MEASURED: this trace is the bare gate replayed, and two
                # equal bars would read as "chirping does nothing" where the law still
                # swept 20-64% of a half-linewidth. The slot is hatched instead.
                excl = np.array([bool(r.get("chirp_excluded")) for r in rows])
                y = np.where(excl, np.nan, y)
        ax.bar(x + (k - (nser - 1) / 2.0) * w, y, w, label=lab, color=col,
               edgecolor=col, linewidth=0.8,
               alpha=1.0 if filled else 0.45, zorder=3)
        ok = np.isfinite(y)
        if (not BARS_ONLY) and ok.sum() >= 4:
            lm, _sd, segs = _trend(d[ok], np.log10(y[ok]))
            trend[(name, src_tag)] = (x[ok], lm, segs, col, lab)

    # The wedge between the trends IS the improvement over the best pulse without
    # DRAG (the independent baseline if given, else this run's weaker bare trace).
    # Its band is the PAIRED scatter: both traces share one operating point, so their
    # column-to-column scatter is common mode (correlation 0.999) and the paired sd is
    # 11-27x narrower than either trace's own.
    ref_key = (("bare", "baseline") if ("bare", "baseline") in trend
               else ("bare", None))
    ref_lab = ("improvement over an independently calibrated bare pulse"
               if ref_key[1] == "baseline" else "improvement")
    if ref_key in trend and ("chirp+DRAG", None) in trend:
        xr, lr, segs, _cr, _lr = trend[ref_key]
        xg, lg, _s2, gcol, _l2 = trend[("chirp+DRAG", None)]
        if xr.size == xg.size and np.allclose(xr, xg):
            pair_ok = np.array([
                bool((BASE_ROWS.get(_bkey(r)) or r)
                     .get("traces", {}).get("bare")
                     and (r.get("traces") or {}).get("chirp+DRAG")) for r in rows])
            if pair_ok.sum() == xr.size:
                ratio = 10.0 ** (lr - lg)
                _pm, psd, segs_p = _trend(d[pair_ok], np.log10(ratio))
                for seg in segs_p:
                    if seg.size < 2:
                        continue
                    ax.fill_between(xr[seg], 10.0 ** lg[seg], 10.0 ** lr[seg],
                                    color="#1baf7a", alpha=0.22, lw=0, zorder=2,
                                    label=(ref_lab if seg is segs_p[0] else None))
                    ax.fill_between(xr[seg], 10.0 ** (lg[seg] - psd[seg]),
                                    10.0 ** (lg[seg] + psd[seg]), color=gcol,
                                    alpha=0.35, lw=0, zorder=4)

    for key in trend:
        xi, lm, segs, col, lab = trend[key]
        for seg in segs:
            if seg.size < 2:
                continue
            ax.plot(xi[seg], 10.0 ** lm[seg], "-", color=col, lw=2.4, alpha=0.95,
                    zorder=5, label=(f"{lab} — trend" if seg is segs[0] else None))

    # Resonances on a categorical axis: interpolate the slot position of each.
    for xr, lab in RES_IN:
        if not (d.min() - 1e-9 <= xr <= d.max() + 1e-9):
            continue
        xi = float(np.interp(xr, d, x))
        ax.axvline(xi, color="#e34948", lw=1.5, ls="--", alpha=0.9, zorder=5)
        ax.annotate(" " + lab, xy=(xi, 0.965), xycoords=("data", "axes fraction"),
                    rotation=90, ha="right", va="top", fontsize=8.5,
                    color="#e34948", zorder=6)

    ax.set_yscale("log")
    ax.set_xlim(-0.8, len(rows) - 0.2)
    step = 1 if len(rows) <= 24 else 2
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{v:+.0f}" for v in d[::step]], fontsize=8, rotation=90)
    # Columns are equally spaced though the grid is not; say so. (Resonance lines are
    # interpolated onto this axis, so they are placed correctly.)
    ax.set_xlabel("pump detuning from the subharmonic   $\\delta$  (MHz)"
                  "        [columns equally spaced, not linear in $\\delta$]",
                  color=INK, fontsize=10)
    ax.set_ylabel("coherent infidelity   $1-F_{\\mathrm{avg}}$",
                  color=INK, fontsize=10)
    ax.set_title(f"Bare pulse vs chirped recursive DRAG   $\\eta^*$ = {eta:g}"
                 f"\nabove branch, $\\omega_b = 1.5\\,\\omega_a + \\delta$, "
                 f"$\\omega_s$ = 4.7 GHz, 9 coupler levels",
                 color=INK, fontsize=12, pad=12)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    band = {"audit": "refused: channel audit",
            "rabi": "failed: shift law unmeasurable",
            "chirp": "failed: chirp/DRAG fixed point",
            "other": "failed: other"}
    for st in [s for s in ("audit", "rabi", "chirp", "other") if s in seen_stages]:
        h.append(Patch(facecolor=seen_stages[st], alpha=0.13))
        l.append(band[st])
    if n_excluded:
        h.append(Patch(facecolor="none", edgecolor=MUTED, hatch="///", alpha=0.5))
        l.append("chirp not measurable (bare still valid)")
    if n_trunc:
        h.append(Line2D([], [], marker="v", ls="none", color="#e34948", ms=6.0))
        l.append("coupler past 9 levels: 1-F too good")
    ax.legend(h, l, frameon=False, fontsize=8.5, labelcolor=INK, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))

    n_ok = sum(1 for r in rows if r["ok"])
    # Out-of-window resonances go in the footer, not on top of the end bars.
    outside = ";  ".join(f"{lab} at {xr:+.0f} MHz" for xr, lab in RES_OUT)
    if not BARS_ONLY:
        fig.text(0.005, 0.021, "green wedge = the improvement;  narrow band = +-1 "
                 "PAIRED sd (both traces share one operating point, so their common "
                 "scatter cancels). NOTE the column-to-column structure is PHYSICS, "
                 "not calibration noise -- a replayed fixed pulse is 1.4-6x worse "
                 "than the locally calibrated one -- so the trend averages across "
                 "real features, including a resonance that moves with drive.",
                 ha="left", va="bottom", fontsize=7.0, color=MUTED)
    # The caveats that survive the exclusions, computed from the rows.
    n_nonconv = sum(1 for r in rows if r["ok"] and r.get("perturbative_ok") is False)
    caveats = []
    if n_nonconv:
        caveats.append(
            f"shift law not converged (|k4 eta^4/k2 eta^2| > 0.25) at {n_nonconv}"
            f"/{n_ok} calibrated columns: those chirps are candidates, not "
            f"calibrations")
    if n_trunc:
        caveats.append(
            f"coupler occupation exceeds what 9 levels can represent at {n_trunc} "
            f"column(s) (red triangles): both bars there read too good")
    if caveats:
        fig.text(0.005, 0.021, "CAVEAT -- " + ";  ".join(caveats),
                 ha="left", va="bottom", fontsize=7.0, color="#8a3a1a")
    fig.text(0.005, 0.005, f"outside the window -- {outside}",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.text(0.995, 0.005,
             f"{n_ok}/{len(rows)} columns calibrated"
             + (f"; chirp unmeasurable at {n_excluded} of them" if n_excluded else "")
             + "; gate lengths in table.txt",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = (f"{outdir}/curve_eta{('%g' % eta).replace('.', 'p')}"
           f"{'_bars' if BARS_ONLY else ''}.png")
    fig.savefig(out, dpi=170, facecolor="white")
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
                   f"{1 - b['F_avg']:>11.4e} {1 - g['F_avg']:>11.4e} {gain:>6.2f}x")
    return "\n".join(out)


if __name__ == "__main__":
    rows = json.load(open(CURVES))
    by_eta = defaultdict(list)
    for r in rows:
        by_eta[float(r["target_eta"])].append(r)
    print("resonances marked: "
          + ", ".join(f"{x:+.0f} MHz" for x, _ in RES_IN)
          + "   (outside the window: "
          + ", ".join(f"{x:+.0f}" for x, _ in RES_OUT) + ")")
    for eta in sorted(by_eta):
        out, n_ok, n = figure_for(eta, by_eta[eta], OUTDIR)
        print(f"eta={eta:g}: {n_ok}/{n} calibrated -> {out}")
    txt = table_for(rows)
    open(f"{OUTDIR}/table.txt", "w").write(txt + "\n")
