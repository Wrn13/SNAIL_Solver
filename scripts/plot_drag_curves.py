"""One figure per target eta: the bare pulse, chirp only, and the chirp+DRAG gate.

One panel, the traces side by side; colour and style both carry the variant, and
in-window resonances are dashed lines. Every 5 MHz tick of the scanned range gets a
slot (an unsolved one stays empty), and the top axis gives the absolute pump
frequency. Gate lengths (fitted per trace) are listed per column in `table.txt`.

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
    "bare":       ("#eb6834", "--", "s", False, "No chirp, no DRAG"),
    "chirp+DRAG": ("#2a78d6", "-",  "o", True,  "Chirp + DRAG"),
}
BASE_STYLE = {
    "bare": ("#eb6834", "--", "s", False,
             "No chirp, no DRAG (independently calibrated)"),
    # Categorical slot 3. The baseline run plays no DRAG, so its "chirp+DRAG" trace
    # IS the chirp alone.
    "chirp only": ("#1baf7a", "-.", "^", True, "Chirp only, no DRAG"),
}
#: display name -> the baseline trace that holds it
BASE_TRACE = {"bare": "bare", "chirp only": "chirp+DRAG"}
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d8d7d0"


def _w_a(dev):
    """Qubit A's frequency (GHz); the pump sits at w_p = w_a/2 + delta."""
    return float(dev["qubit_freqs_GHz"][0])


def w_a_of(device_path, default=3.5):
    try:
        return _w_a(json.load(open(device_path)))
    except Exception:
        return float(default)


def resonances(device_path):
    """In-window channel resonances (MHz), derived from the device. A 5 MHz scan of
    `interaction_channels` over +-400 MHz finds the first two zero crossings; the
    third is second order in g3, which that first-order audit cannot see:

      delta = 0          2 w_p = w_a          the qubit-A subharmonic the scan straddles
      delta = alpha/2    2 w_p = w_a + alpha  the A |1>->|2> ladder, from
                                              E(n) = n w_a + alpha n(n-1)/2
      delta = w_s/3-w_a/2  3 w_p = w_s        SNAIL three-photon drive, ~g3^2 eta^3 s+

    Two more sit outside the window and are annotated rather than drawn.
    """
    A_SUB = "A subharmonic   $2\\omega_p=\\omega_a$"
    A_12 = "A $|1\\rangle\\!\\to\\!|2\\rangle$   $2\\omega_p=\\omega_a\\!+\\!\\alpha$"
    # 3 w_p = w_s: second order in g3 ([xi^2 s+, xi s+s] ~ g3^2 eta^3 s+ / Delta), so
    # absent from the first-order audit; drawn at the UNDRESSED position (the strong
    # pump shifts the SNAIL, so the damage sits ~10-20 MHz above it at eta* = 1.3).
    S3 = "SNAIL 3-photon   $3\\omega_p=\\omega_s$ (undressed)"
    try:
        dev = json.load(open(device_path))
    except Exception:
        return [(0.0, A_SUB), (-60.0, A_12), (-183.3, S3)], []
    w_a = _w_a(dev)
    w_s = float(dev["coupler_freq_GHz"])
    alpha = float(dev["anharm_qubit_GHz"])
    return ([(0.0, A_SUB), (alpha * 1e3 / 2.0, A_12),
             ((w_s / 3.0 - w_a / 2.0) * 1e3, S3)],
            [((w_s - w_a) * 1e3 / 2.0, "SNAIL subharmonic"),
             ((w_s - 1.5 * w_a) * 1e3, "A-SNAIL spectator")])


RES_IN, RES_OUT = resonances(DEVICE)
W_A_GHZ = w_a_of(DEVICE)


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

#: eta -> the caveat lines figure_for computed for that figure (-> caveats.txt)
CAVEATS = {}


def full_grid(eta, rows, step_MHz=5.0):
    """`rows` on every `step_MHz` tick of their range, stubs where nothing was solved.

    The range is the union of this run's and the baseline's columns at `eta`, so a
    column only the baseline solved still gets its slot. A stub carries ``stub`` and
    no traces: its slot stays empty, unshaded, and counts as not solved.
    """
    have = {round(r["delta_GHz"] * 1e3): r for r in rows}
    base = [k[0] for k in BASE_ROWS if abs(k[1] - round(float(eta), 3)) < 1e-9]
    ticks = sorted(set(have) | set(base))
    if not ticks:
        return []
    lo, hi = ticks[0], ticks[-1]
    grid = np.round(np.arange(lo, hi + 0.5 * step_MHz, step_MHz)).astype(int)
    out = []
    for dm in list(grid) + [t for t in ticks if t not in set(grid)]:
        r = have.get(int(dm))
        if r is None:
            r = {"delta_GHz": dm / 1e3, "target_eta": float(eta), "ok": False,
                 "stub": True, "error": None, "traces": {},
                 "w_p_GHz": 0.5 * W_A_GHZ + dm / 1e3}
        out.append(r)
    return sorted(out, key=lambda r: r["delta_GHz"])


def figure_for(eta, rows, outdir):
    """One eta, one panel: grouped bars, bare against chirp+DRAG.

    A categorical column axis with log y, because the quantity is a per-column
    comparison, not a continuous function. Every scanned column gets a slot whether or
    not it calibrated, so a gap reads as a gap.
    """
    rows = full_grid(eta, rows)
    x = np.arange(len(rows), dtype=float)
    d = np.array([r["delta_GHz"] * 1e3 for r in rows])

    fig, ax = plt.subplots(figsize=(max(11.0, 0.26 * len(rows) + 4.0), 6.2))
    ax.grid(axis="y", which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(axis="y", which="minor", color=GRID, lw=0.4, alpha=0.45)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom", "top"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)

    # Failed columns: a slot shaded by the stage that gave up (audit refusal,
    # unmeasurable shift law, diverged chirp fixed point are different physics).
    stage_colour = {"audit": "#e34948", "rabi": "#eda100", "chirp": "#4a3aa7"}
    seen_stages = {}
    for i, r in enumerate(rows):
        if not r["ok"] and not r.get("stub"):
            st = ((r.get("error") or {}).get("stage") or "other")
            if st not in stage_colour:
                st = "other"            # e.g. "tune_up": legend it, don't drop it
            col = stage_colour.get(st, MUTED)
            seen_stages[st] = col
            ax.axvspan(i - 0.5, i + 0.5, color=col, alpha=0.13, lw=0, zorder=0)

    # Columns whose chirp was not measured are drawn like any other: their chirp bars
    # are what that pulse scored. Those that PLAYED no chirp are counted in the footer.
    n_excluded = sum(1 for r in rows if r.get("chirp_excluded") and r.get("chirp_free"))

    # Coupler loaded past what 9 levels represent: both bars read too GOOD (leakage
    # off the ladder is uncharged). Named in the caveat line, not marked on the bars.
    d_trunc = [r["delta_GHz"] * 1e3 for r in rows if r.get("coupler_truncated")]
    n_trunc = len(d_trunc)

    # Series to draw: the run's own two, plus the independent baseline if given.
    if BASE_ROWS:
        # baseline bare + chirp only + this run's chirp+DRAG; this run's own bare is
        # superseded.
        series = ([(n, s, "baseline") for n, s in BASE_STYLE.items()]
                  + [(n, s, None) for n, s in STYLE.items() if n != "bare"])
    else:
        series = [(n, s, None) for n, s in STYLE.items()]
    nser = len(series)
    w = 0.86 / nser
    trend = {}
    for k, (name, (col, _ls, _mk, filled, lab), src_tag) in enumerate(series):
        if src_tag == "baseline":
            trace = BASE_TRACE.get(name, name)
            y = np.array([
                (1.0 - t["F_avg"])
                if (br := BASE_ROWS.get(_bkey(r)))
                and (t := (br.get("traces") or {}).get(trace))
                else np.nan
                for r in rows])
        else:
            y = np.array([(1.0 - t["F_avg"])
                          if (t := (r.get("traces") or {}).get(name)) else np.nan
                          for r in rows])
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

    # Resonances on a categorical axis: interpolate the slot position of each. One
    # dash pattern per resonance, so the legend (not text on the bars) names them.
    dashes = [(6, 3), (2, 2), (8, 2, 2, 2), (4, 1, 1, 1)]
    for k, (xr, lab) in enumerate(RES_IN):
        if not (d.min() - 1e-9 <= xr <= d.max() + 1e-9):
            continue
        xi = float(np.interp(xr, d, x))
        ax.axvline(xi, color="#e34948", lw=1.5, dashes=dashes[k % len(dashes)],
                   alpha=0.9, zorder=5, label=f"{lab}  ($\\delta$ = {xr:+.0f} MHz)")

    ax.set_yscale("log")
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:+.0f}" for v in d], fontsize=7.5, rotation=90)
    # Absolute pump frequency on top, one label per slot: w_p = w_a/2 + delta.
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels([f"{r.get('w_p_GHz') or 0.5 * W_A_GHZ + r['delta_GHz']:.3f}"
                         for r in rows], fontsize=7.5, rotation=90)
    top.tick_params(colors=MUTED)
    top.spines["top"].set_color(GRID)
    top.set_xlabel("pump frequency   $\\omega_p/2\\pi$  (GHz)", color=INK,
                   fontsize=10)
    ax.set_xlabel("pump detuning from the subharmonic   $\\delta$  (MHz)",
                  color=INK, fontsize=10)
    ax.set_ylabel("coherent infidelity   $1-F_{\\mathrm{avg}}$",
                  color=INK, fontsize=10)
    ax.set_title(f"Bare pulse vs chirp only vs chirped recursive DRAG   "
                 f"$\\eta^*$ = {eta:g}"
                 f"\n$\\omega_b = 1.5\\,\\omega_a + \\delta$, "
                 f"$\\omega_s$ = 4.7 GHz, 9 coupler levels",
                 color=INK, fontsize=12, pad=12)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    band = {"audit": "Refused: channel audit",
            "rabi": "Failed: shift law unmeasurable",
            "chirp": "Failed: chirp/DRAG fixed point",
            "other": "Failed: tune-up (no chirp+DRAG gate)"}
    for st in [s for s in ("audit", "rabi", "chirp", "other") if s in seen_stages]:
        h.append(Patch(facecolor=seen_stages[st], alpha=0.13))
        l.append(band[st])
    n_stub = sum(1 for r in rows if r.get("stub"))
    if n_stub:
        h.append(Patch(facecolor="none", edgecolor=GRID))
        l.append(f"Empty slot: not solved ({n_stub})")
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
            f"coupler occupation exceeds what 9 levels can represent at "
            f"delta = {', '.join(f'{v:+.0f}' for v in d_trunc)} MHz: both bars there "
            f"read too good")
    # Written to caveats.txt beside the figure, not drawn on it.
    CAVEATS[float(eta)] = caveats
    fig.text(0.005, 0.005, f"outside the window -- {outside}",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    # The column count goes to caveats.txt too, first, rather than on the figure.
    CAVEATS.setdefault(float(eta), []).insert(
        0, f"{n_ok}/{len(rows)} columns calibrated"
           + (f"; {n_excluded} of them played no chirp (none measured)"
              if n_excluded else "")
           + "; gate lengths in table.txt")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = (f"{outdir}/curve_eta{('%g' % eta).replace('.', 'p')}"
           f"{'_bars' if BARS_ONLY else ''}.png")
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out, n_ok, len(rows)


def table_for(rows):
    rows = sorted(rows, key=lambda r: (r["target_eta"], r["delta_GHz"]))
    hdr = (f"{'delta':>7} {'eta':>5} {'nch':>4} {'t_g bare':>9} {'t_g drag':>9} "
           f"{'1-F bare':>11} {'1-F chirp':>11} {'1-F drag':>11} {'gain':>7}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        br = BASE_ROWS.get(_bkey(r)) or {}
        # The baseline's bare supersedes this run's, exactly as in the figure.
        b = ((br.get("traces") or {}).get("bare") if BASE_ROWS
             else (r.get("traces") or {}).get("bare"))
        c = (br.get("traces") or {}).get("chirp+DRAG")
        g = (r.get("traces") or {}).get("chirp+DRAG")
        if not r["ok"] or b is None or g is None:
            stage = ((r.get("error") or {}).get("stage") or "refused")
            out.append(f"{r['delta_GHz'] * 1e3:>7.0f} {r['target_eta']:>5.2f} "
                       f"{'--':>4} {('FAILED ' + stage):>9}")
            continue
        gain = (1 - b["F_avg"]) / (1 - g["F_avg"]) if g["F_avg"] < 1 else float("nan")
        out.append(f"{r['delta_GHz'] * 1e3:>7.0f} {r['target_eta']:>5.2f} "
                   f"{g['n_drag_played']:>4d} {b['t_g_ns']:>9.1f} {g['t_g_ns']:>9.1f} "
                   f"{1 - b['F_avg']:>11.4e} "
                   + (f"{1 - c['F_avg']:>11.4e} " if c else f"{'--':>11} ")
                   + f"{1 - g['F_avg']:>11.4e} {gain:>6.2f}x")
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
    with open(f"{OUTDIR}/caveats.txt", "w") as fh:
        for eta in sorted(CAVEATS):
            fh.write(f"eta* = {eta:g}  ({CURVES})\n")
            for c in CAVEATS[eta] or ["none"]:
                fh.write(f"  - {c}\n")
            fh.write("\n")
