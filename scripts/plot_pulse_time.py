"""The time-dependent pulse a scan column plays, in three variants, at chosen detunings.

The curve figures plot SCORES; this draws the pulse behind them: the complex pump
amplitude eta(t), rebuilt from the stored operating point through the same
`config_at_wp` -> `build_coupler` path `curve_drag_vs_bare.py` scores and sampled with
`ZhouCoupler._eta_at` (what the solver sees), so the figure cannot drift from the
simulation.

Three variants per detuning, all at the CALIBRATED operating point (same t_g,
amp_scale and carrier), so they differ only in the corrections:

    chirp + DRAG    the gate as calibrated and scored
    chirp only      the same chirp, DRAG not played
    neither         the bare base envelope

(The `bare` trace's own length refit normally returns the calibrated length; where
it does not, the refit length is annotated -- this figure is about shape.)

Panels: 1. |eta(t)| -- the chirp is a PURE PHASE, so "chirp only" and "neither"
coincide and only DRAG moves |eta|, at second order. 2./3. I = Re eta and Q = Im eta
at the FIXED carrier w_p (what an AWG plays): "neither" has no Q, the chirp rotates
the envelope into it, and DRAG adds the derivative term.

Usage:  plot_pulse_time.py SCAN.h5 OUTDIR DELTA_MHz,DELTA_MHz[,...] [CURVES.json]

Writes one figure per detuning plus a side-by-side comparison of all of them.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snail_solver.envelope import DragChannel
from snail_solver.h5_io import load_doc
from snail_solver.subharmonic_convergence import config_at_wp
from snail_solver.device_utils import build_coupler, drag_correction_ratio
from snail_solver.tune_up import fixed_eta_amp_scale

SCAN, OUTDIR = sys.argv[1], sys.argv[2]
DELTAS_MHz = [float(x) for x in sys.argv[3].split(",")]
CURVES = sys.argv[4] if len(sys.argv) > 4 else None    # for the scored 1-F annotations
N_TIME = 2001

# (key, colour, linestyle, width, label). Palette slots 1-3 in the same roles as
# `plot_drag_curves.py`; line style duplicates the encoding.
VARIANTS = (
    ("drag",  "#2a78d6", "-",       1.9, "chirp + DRAG"),
    ("chirp", "#1baf7a", (0, (5, 2)), 1.5, "chirp only"),
    ("bare",  "#eb6834", (0, (1, 2)), 1.5, "neither"),
)
CHIRP_C = "#4a3aa7"
CHANNEL = ("#e34948", "#eda100", "#008300")        # one per DRAG channel, in order
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d8d7d0"
SURFACE = "#fcfcfb"


def _scored(curves_path, delta_GHz, target_eta):
    """The `curve_drag_vs_bare.py` traces for this column, or {} if unavailable."""
    if not curves_path or not os.path.exists(curves_path):
        return {}
    for r in json.load(open(curves_path)):
        if (abs(float(r["delta_GHz"]) - delta_GHz) < 5e-4
                and abs(float(r["target_eta"]) - target_eta) < 1e-6):
            return {k: v for k, v in (r.get("traces") or {}).items() if v}
    return {}


def column(scan, delta_MHz, curves_path=None):
    """The row of `scan` at this detuning, rebuilt as three pulses.

    Returns a dict of time traces plus the scalars the panels annotate. Raises if the
    column is absent or was never calibrated -- a failed column has no pulse.
    """
    want = delta_MHz * 1e-3
    hits = [r for r in scan["rows"] if abs(float(r["delta_GHz"]) - want) < 5e-4]
    if not hits:
        have = ", ".join(f"{float(r['delta_GHz']) * 1e3:+.0f}" for r in scan["rows"])
        raise SystemExit(f"no column at delta = {delta_MHz:+g} MHz; have: {have}")
    r = hits[0]
    if not r.get("ok"):
        raise SystemExit(f"delta = {delta_MHz:+g} MHz did not calibrate "
                         f"({r.get('error')}) -- it has no pulse to draw")

    op = r["operating_point"]
    chirp = [float(c) for c in r["chirp"]["coeffs_GHz"]]
    channels = [DragChannel(beat_GHz=float(c["beat_GHz"]), n_pump=int(c["n_pump"]),
                            n_photon=int(c.get("n_photon", c["n_pump"])),
                            quotient_rule=True)
                for c in (r.get("drag_channels") or [])]
    cfg = config_at_wp(scan["device"], float(r["w_p_GHz"]), branch=r["branch"],
                       levels=int(r["coupler_levels"]), chirp_coeffs_GHz=chirp)
    t_g = float(op["t_g_ns"])
    # amp_scale holds |eta| at the target for this length (common to all variants);
    # recomputing it is what `score_gate` does, an exact no-op vs the stored value.
    amp = float(fixed_eta_amp_scale(cfg, t_g, float(op["target_eta"])))
    t = np.linspace(0.0, t_g, N_TIME)

    def build(with_chirp, with_drag):
        # `[]` for no chirp, never None: None means "inherit config['chirp_coeffs_GHz']"
        # in build_coupler, which would silently restore the chirp this variant drops.
        cpl, w_p, peak = build_coupler(
            cfg, t_g, amp, float(op["wp_offset_GHz"]), op.get("spec_abs_GHz"),
            op.get("drag_beat_GHz") if with_drag else None,
            chirp_coeffs_GHz=(chirp if with_chirp else []),
            drag_n_pump=int(op.get("drag_n_pump") or 1),
            drag_channels=((channels or None) if with_drag else None))
        tone = cpl._pump_tones[0]
        return {"eta": np.asarray(cpl._eta_at(tone, t, np), dtype=complex),
                "tone": tone, "w_p": w_p, "peak_eta": peak,
                "ratio": drag_correction_ratio(tone)}

    pulses = {"drag": build(True, True), "chirp": build(True, False),
              "bare": build(False, False)}
    tone = pulses["drag"]["tone"]
    return {
        "row": r, "t": t, "t_g": t_g, "pulses": pulses, "amp_scale": amp,
        "w_p": pulses["drag"]["w_p"], "peak_eta": pulses["drag"]["peak_eta"],
        "base": np.abs(pulses["bare"]["eta"]),
        "delta_MHz": float(r["delta_GHz"]) * 1e3,
        "phase": (np.zeros_like(t) if tone.chirp is None
                  else np.asarray(tone.chirp.phase(t, np), dtype=float)),
        "delta_t": (np.zeros_like(t) if tone.chirp is None
                    else np.asarray(tone.chirp.detuning(t, np)) / (2 * np.pi) * 1e3),
        "beats": [(c, np.asarray(tone.channel_detuning(c, t, np)) / (2 * np.pi) * 1e3)
                  for c in channels],
        "scored": _scored(curves_path, float(r["delta_GHz"]),
                          float(op["target_eta"])),
    }


def _headroom(ax, frac, side="top"):
    """Open `frac` of the data range at one end, so an annotation has clear space."""
    lo, hi = ax.get_ylim()
    pad = frac * (hi - lo)
    ax.set_ylim(lo - (pad if side == "bottom" else 0.0),
                hi + (pad if side == "top" else 0.0))


def _frame(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def draw(axes, c, with_ylabels=True):
    """The three panels of one detuning into a length-3 column of axes."""
    t, t_g = c["t"], c["t_g"]
    for ax in axes:
        _frame(ax)
        ax.set_xlim(0.0, t_g)

    # -- 1. amplitude --------------------------------------------------------
    ax = axes[0]
    for key, colour, ls, lw, label in VARIANTS:
        ax.plot(t, np.abs(c["pulses"][key]["eta"]), color=colour, ls=ls, lw=lw,
                label=label)
    ax.set_title(f"$\\delta = {c['delta_MHz']:+.0f}$ MHz     "
                 f"$t_g = {t_g:.1f}$ ns,  $\\omega_p/2\\pi = {c['w_p']:.4f}$ GHz,  "
                 f"{len(c['beats'])} DRAG channel(s)",
                 color=INK, fontsize=10, pad=8)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, ncol=3,
              loc="lower center")
    if with_ylabels:
        ax.set_ylabel(r"amplitude  $|\eta|$", color=INK, fontsize=9)
    # The variants barely separate here, so state in numbers what the panel cannot.
    note = (f"peak $|\\eta|$ = {c['peak_eta']:.2f}   DRAG moves $|\\eta|$ by "
            f"{100 * c['pulses']['drag']['ratio']:.2f}% of the base peak; the chirp, "
            f"being a pure phase, by nothing\n"
            f"chirp phase $\\Phi$ swings {np.ptp(c['phase']):.2f} rad over the gate")
    scored = c["scored"]
    if scored:
        bits = []
        for key, name in (("chirp+DRAG", "chirp+DRAG"), ("bare", "neither")):
            s_ = scored.get(key)
            if s_:
                tag = (f" @ its own {s_['t_g_ns']:.1f} ns"
                       if s_.get("refit_length")
                       and abs(float(s_["t_g_ns"]) - t_g) > 0.05 else "")
                bits.append(f"{name} {s_['infidelity_total']:.3e}{tag}")
        note += "\nscan $1-F$:  " + ",   ".join(bits)
    _headroom(ax, 0.34)
    ax.text(0.02, 0.96, note, transform=ax.transAxes, fontsize=7.5, color=MUTED,
            va="top")

    # -- 2/3. the baseband quadratures at the fixed carrier ------------------
    for ax, part, sym in ((axes[1], np.real, "I = \\mathrm{Re}\\,\\eta"),
                          (axes[2], np.imag, "Q = \\mathrm{Im}\\,\\eta")):
        ax.axhline(0.0, color=MUTED, lw=0.7)
        for key, colour, ls, lw, label in VARIANTS:
            ax.plot(t, part(c["pulses"][key]["eta"]), color=colour, ls=ls, lw=lw,
                    label=label)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, ncol=3,
                  loc="upper right")
        if with_ylabels:
            ax.set_ylabel(f"${sym}$\n" r"at fixed $\omega_p$", color=INK, fontsize=9)
    _headroom(axes[1], 0.14)
    _headroom(axes[2], 0.14)
    axes[2].text(0.02, 0.94, "'neither' has no quadrature at all",
                 transform=axes[2].transAxes, fontsize=7.5, color=MUTED, va="top")
    axes[2].set_xlabel("time (ns)", color=INK, fontsize=9)


def figure(cols, path, title):
    """One figure, one column of panels per detuning."""
    n = len(cols)
    fig, axes = plt.subplots(3, n, figsize=(6.8 * n, 8.2), squeeze=False,
                             sharex="col")
    fig.patch.set_facecolor(SURFACE)
    for j, c in enumerate(cols):
        draw(axes[:, j], c, with_ylabels=(j == 0))
    fig.suptitle(title, color=INK, fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    doc = load_doc(SCAN, group="scan")
    etas = doc["settings"].get("target_etas") or []
    tag = f"eta = {etas[0]:g}" if len(etas) == 1 else "the scan"
    cols = [column(doc, d, CURVES) for d in DELTAS_MHz]
    os.makedirs(OUTDIR, exist_ok=True)
    for c in cols:
        d = f"{c['delta_MHz']:+.0f}".replace("+", "p").replace("-", "m")
        figure([c], os.path.join(OUTDIR, f"pulse_d{d}.png"),
               f"Pump pulse with and without its corrections, {tag}, "
               f"$\\delta = {c['delta_MHz']:+.0f}$ MHz")
    if len(cols) > 1:
        figure(cols, os.path.join(OUTDIR, "pulse_compare.png"),
               f"Pump pulse with and without its corrections, {tag}: "
               + " vs ".join(f"$\\delta = {c['delta_MHz']:+.0f}$ MHz" for c in cols))
