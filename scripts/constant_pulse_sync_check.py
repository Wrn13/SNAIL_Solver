#!/usr/bin/env python
r"""Sanity check of spectral-zero nulling with CONSTANT pulses -- no ramps, no chirp.

For a square pulse of length T the envelope spectrum is ``F(Delta) = T sinc(Delta T/2)``,
so a spectator with line ``W_j`` is nulled exactly when ``W_j T`` is an integer -- the
same statement as the two-level sudden-switch law

    P_j(T) = (4 g_j^2 / W_j^2) sin^2(pi W_j T).

With square pulses the only shaping knob is the length, and at fixed iSWAP area the
length is tied to the drive (T ~ A/eta). "Pulse shaping" is then the choice of a
MAGIC DRIVE where the dominant spectator returns to zero at the gate time. This
script tests that premise before any ramped spectral-zero design is built on it.

Part A (no solves): the two-level model against the saved flat-chevron trajectories
    (``flat_chevron_study.py`` output) -- band-passed channel population vs model.
Part B (solves): square iSWAP gates over an eta grid, each with its carrier
    calibrated on its own chevron and its length at the first P10 maximum; measured
    leakage and 1-F vs the predicted sum of sin^2 terms of the top-two spectators.

Imports only; nothing in the running pipeline or the earlier study files changes.

    .venv/bin/python scripts/constant_pulse_sync_check.py --deltas=0.1 --jobs 3
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as _dt
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from plateau_collision_study import (DEFAULT_LAWS, INK, INK2, SERIES, _style,  # noqa: E402
                                     column_config, eta_tag, load_column, parse_list)
from flat_chevron_study import (LEAK_CHANNELS, _jsonable, channel_index,  # noqa: E402
                                chevron_stack, constant_coupler)

FAST_MHZ = 400.0          # lines above this are the SNAIL/coupler counter-rotating dressing
MERGE_MHZ = 8.0
FLAT_DIR = os.path.join(REPO, "results", "plateau_study", "flat_chevrons_2026-09-28")


# ===========================================================================
# the two-level model
# ===========================================================================
def invert_line(W_MHz: float, amp: float) -> Dict[str, float]:
    """(W, cosine amplitude) of a sudden-switch line -> (g, Delta).

    ``P = (4g^2/W^2) sin^2(pi W t) = (2g^2/W^2)(1 - cos 2 pi W t)``, so the matrix
    pencil's cosine amplitude is ``2 g^2/W^2``: ``g = W sqrt(amp/2)``,
    ``Delta = sqrt(W^2 - 4 g^2)``.
    """
    W = float(W_MHz)
    a = float(np.clip(amp, 0.0, 0.5))           # 4g^2/W^2 <= 1
    g = W * np.sqrt(a / 2.0)
    return {"W_MHz": W, "amp": float(amp), "g_MHz": float(g),
            "Delta_MHz": float(np.sqrt(max(W * W - 4 * g * g, 0.0))),
            "P_max": float(4 * g * g / (W * W)) if W > 0 else 0.0}


def model_population(sp: Dict[str, float], t: np.ndarray) -> np.ndarray:
    return sp["P_max"] * np.sin(np.pi * sp["W_MHz"] * 1e-3 * np.asarray(t)) ** 2


def spectators_from_lines(finger: Dict[str, Any], *, fast_MHz: float = FAST_MHZ,
                          merge_MHz: float = MERGE_MHZ, top: int = 2,
                          slow_MHz: float = 1.0) -> List[Dict[str, Any]]:
    """Merge a fingerprint's lines across channels and rank by ``P_max``.

    One spectator shows up in several overlapping P_* channels (P_leak contains
    P_coupler, ...). Each merged spectator keeps the channel where its line is
    strongest -- the one to band-pass -- and the audit match if any.

    Lines below `slow_MHz` are dropped: pass ``1.5e3 / t_gate`` so the gate's OWN
    exchange (P_f_a etc. follow the |01> -> |10> swap, a ~5-20 MHz "line" with
    correlation ~1) is not ranked as a spectator.
    """
    lines = []
    for ch, v in finger.items():
        if not isinstance(v, dict):
            continue
        for ln in v.get("lines", []):
            if slow_MHz < ln["f_MHz"] < fast_MHz:
                lines.append(dict(ln, channel=ch))
    lines.sort(key=lambda l: -l["amplitude"])
    merged: List[Dict[str, Any]] = []
    for ln in lines:
        if any(abs(ln["f_MHz"] - m["W_MHz"]) < merge_MHz for m in merged):
            continue
        sp = invert_line(ln["f_MHz"], ln["amplitude"])
        sp.update(channel=ln["channel"], match=ln.get("match"))
        merged.append(sp)
    merged.sort(key=lambda s: -s["P_max"])
    return merged[:top]


def bandpass(t: np.ndarray, y: np.ndarray, f_MHz: float, rel: float = 0.3) -> np.ndarray:
    """Keep only the Fourier components within ``f (1 +- rel)`` (plus the mean)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    dt = float(np.mean(np.diff(t)))
    Y = np.fft.rfft(y - y.mean())
    f = np.fft.rfftfreq(y.size, dt) * 1e3                  # MHz
    Y[(f < f_MHz * (1 - rel)) | (f > f_MHz * (1 + rel))] = 0.0
    return np.fft.irfft(Y, n=y.size)


# ===========================================================================
# Part A
# ===========================================================================
def part_a(tag: str, etas: Sequence[float]) -> Dict[str, Any]:
    path = os.path.join(FLAT_DIR, f"{tag}.json")
    with open(path) as fh:
        doc = json.load(fh)
    i10 = channel_index("P10")
    out = []
    for dr in doc["drives"]:
        if not any(abs(dr["eta"] - e) < 1e-6 for e in etas):
            continue
        R = dr["_raster"]
        t = np.asarray(R["times_ns"])
        off = np.asarray(R["narrow_offsets_GHz"])
        S = np.asarray(R["narrow"])
        j = int(np.argmin(np.abs(off - 1e-3 * dr["carrier_meas_MHz"])))
        sps = spectators_from_lines(dr["fingerprint"],
                                    slow_MHz=1.5e3 / float(dr["t_iswap_ns"]))
        rows = []
        for sp in sps:
            y = S[j, channel_index(sp["channel"]), :]
            bp = bandpass(t, y, sp["W_MHz"])
            mdl = model_population(sp, t)
            mdl_bp = mdl - mdl.mean()
            corr = float(np.corrcoef(bp, mdl_bp)[0, 1]) if np.std(bp) > 0 else float("nan")
            ratio = float(np.std(bp) / max(np.std(mdl_bp), 1e-12))
            period = 1e3 / sp["W_MHz"]
            k = np.arange(1, int(t[-1] / period) + 1)
            returns = (k * period).tolist()
            t_sw = float(dr["t_iswap_ns"])
            near = min(returns, key=lambda r: abs(r - t_sw)) if returns else None
            rows.append(dict(sp, corr=corr, amp_ratio=ratio, returns_ns=returns,
                             t_iswap_ns=t_sw,
                             t_iswap_to_return_ns=(None if near is None
                                                   else float(t_sw - near)),
                             _t=t, _bp=bp, _model=mdl_bp))
        out.append({"eta": dr["eta"], "spectators": rows,
                    "P10": S[j, i10, :], "t": t})
    return {"tag": tag, "drives": out}


# ===========================================================================
# Part B
# ===========================================================================
def _fid_column(args):
    cfg, eta, off, T, start, subspace, solver = args
    cpl = constant_coupler(cfg, eta, off, T)
    H = cpl.to_qutip_hamiltonian()
    opts = cpl._qutip_options(solver["atol"], solver["rtol"], solver["nsteps"])
    psi = cpl._sesolve_final(H, start, T, opts)
    return [complex(psi[i]) for i in subspace]


def square_fidelity(cfg, eta, off_GHz, T, solver, jobs):
    """Leakage-aware iSWAP fidelity of a square pulse, its four columns in parallel."""
    from snail_solver.zhou_coupler import ZhouCoupler
    cpl = constant_coupler(cfg, eta, off_GHz, T)
    sub = cpl._subspace_indices(0, 1)
    args = [(cfg, eta, off_GHz, T, s, sub, solver) for s in sub]
    if jobs <= 1:
        cols = [_fid_column(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, 4)) as ex:
            cols = list(ex.map(_fid_column, args))
    U = np.array(cols, dtype=complex).T
    F, leak = ZhouCoupler._iswap_fidelity_from_U(U, True)
    return float(F), float(leak)


def gate_time_index(t: np.ndarray, p: np.ndarray, t_area: float,
                    smooth_ns: float = 8.0, lo: float = 0.6, hi: float = 1.4) -> int:
    """Index of the full-transfer time: the global maximum of P10, smoothed over
    `smooth_ns` (wider than the fast dressing ripple, whose local bumps otherwise
    look like early maxima), searched within ``[lo, hi] x`` the area time A/eta."""
    k = max(1, int(round(smooth_ns / float(np.mean(np.diff(t))))))
    ps = np.convolve(p, np.ones(k) / k, mode="same")
    win = (t >= lo * t_area) & (t <= hi * t_area)
    if not np.any(win):
        return int(np.argmax(ps))
    idx = np.nonzero(win)[0]
    return int(idx[np.argmax(ps[idx])])


def calibrate_square(cfg, eta, c0_GHz, span_MHz, points, times, solver, jobs):
    """Chevron-envelope carrier calibration (``tune_up.fit_chevron_center``) with the
    guarded single re-centre from ``flat_chevron_study``."""
    from snail_solver.tune_up import fit_chevron_center
    i10 = channel_index("P10")

    def scan(c):
        offs = c + np.linspace(-span_MHz, span_MHz, points) * 1e-3
        S = chevron_stack(cfg, eta, offs, times, solver, jobs)
        env = S[:, i10, :].max(axis=1)
        return offs, env, fit_chevron_center(offs, env)

    offs, env, cen = scan(c0_GHz)
    c = float(cen["center_GHz"])
    recentred = False
    k = int(np.argmax(env))
    if k in (0, len(offs) - 1):
        o2, e2, c2 = scan(float(offs[k]))
        v2 = float(o2[int(np.argmax(e2))])
        if abs(v2 - float(offs[k])) <= max(float(cen.get("hwhm_GHz") or 0), 3e-3):
            offs, env, cen, c, recentred = o2, e2, c2, float(c2["center_GHz"]), True
    return {"carrier_GHz": c, "fit_ok": bool(cen["ok"]), "recentred": recentred,
            "railed": bool(int(np.argmax(env)) in (0, len(offs) - 1)),
            "scan_MHz": (1e3 * offs).tolist(), "envelope": env.tolist()}


def part_b_point(cfg, eta, c0_GHz, extrapolated, args, solver) -> Dict[str, Any]:
    from snail_solver.piecewise_pulse import matrix_pencil
    from snail_solver.tune_up import _area

    t0 = time.perf_counter()
    A = _area(cfg)
    times = np.arange(0.0, 1.5 * A / eta + 1e-9, args.dt_ns)
    span, pts = ((args.far_span_MHz, args.far_points) if extrapolated
                 else (args.near_span_MHz, args.near_points))
    cal = calibrate_square(cfg, eta, c0_GHz, span, pts, times, solver, args.jobs)
    c = cal["carrier_GHz"]
    S = chevron_stack(cfg, eta, [c], times, solver, 1)[0]
    p10 = S[channel_index("P10")]
    it = gate_time_index(times, p10, A / eta)
    T = float(times[it])
    # lines of the trajectory up to T -> this drive's spectators
    finger = {}
    for ch in LEAK_CHANNELS:
        finger[ch] = {"lines": matrix_pencil(times[:it + 1],
                                             S[channel_index(ch), :it + 1])[:8]}
    sps = spectators_from_lines(finger, top=args.top, slow_MHz=1.5e3 / T)
    for sp in sps:
        sp["sin2"] = float(np.sin(np.pi * sp["W_MHz"] * 1e-3 * T) ** 2)
        sp["pred_P_T"] = sp["P_max"] * sp["sin2"]
        sp["W_T"] = sp["W_MHz"] * 1e-3 * T
    F, leak = square_fidelity(cfg, eta, c, T, solver, args.jobs)
    at_T = {ch: float(S[channel_index(ch), it]) for ch in LEAK_CHANNELS + ("P10",)}
    # the slow (< FAST_MHZ) part of P_leak at T, which is what the model describes
    y = S[channel_index("P_leak"), :it + 1]
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(y.size, args.dt_ns) * 1e3
    Y[f > FAST_MHZ] = 0.0
    slow = float(np.fft.irfft(Y, n=y.size)[-1])
    return {"eta": float(eta), "extrapolated": bool(extrapolated), "cal": cal,
            "carrier_MHz": 1e3 * c, "T_ns": T, "T_area_ns": A / eta,
            "P10_at_T": float(p10[it]), "F_avg": F, "infidelity": 1 - F,
            "leakage": leak, "at_T": at_T, "P_leak_slow_at_T": slow,
            "spectators": sps,
            "pred_sum_P_T": float(sum(sp["pred_P_T"] for sp in sps)),
            "seconds": time.perf_counter() - t0}


def carrier_seed(flat_doc: Dict[str, Any], eta: float, prev: List[Dict[str, Any]]
                 ) -> (float, bool):
    """Quadratic interpolation of the measured flat-chevron resonances for
    eta <= max measured; beyond, extrapolate from the last two measured points."""
    etas = np.array([d["eta"] for d in flat_doc["drives"]])
    cs = np.array([d["carrier_meas_MHz"] for d in flat_doc["drives"]]) * 1e-3
    if eta <= etas.max() + 1e-9:
        p = np.polyfit(etas, cs, min(2, len(etas) - 1))
        return float(np.polyval(p, eta)), False
    done = [r for r in prev if r.get("carrier_MHz") is not None]
    if len(done) >= 2:
        a, b = done[-2], done[-1]
        slope = (b["carrier_MHz"] - a["carrier_MHz"]) / (b["eta"] - a["eta"])
        return 1e-3 * (b["carrier_MHz"] + slope * (eta - b["eta"])), True
    p = np.polyfit(etas, cs, min(2, len(etas) - 1))
    return float(np.polyval(p, eta)), True


def magic_drives(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """eta where a spectator's W T crosses an integer (sin^2 -> 0), by linear
    interpolation of W T between grid points, tracked by line frequency."""
    out = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        for sa in a["spectators"]:
            sb = min(b["spectators"], key=lambda s: abs(s["W_MHz"] - sa["W_MHz"]),
                     default=None)
            if sb is None or abs(sb["W_MHz"] - sa["W_MHz"]) > 0.3 * sa["W_MHz"]:
                continue
            ka, kb = sa["W_T"], sb["W_T"]
            for k in range(int(np.ceil(min(ka, kb))), int(np.floor(max(ka, kb))) + 1):
                if ka == kb:
                    continue
                x = (k - ka) / (kb - ka)
                out.append({"eta": a["eta"] + x * (b["eta"] - a["eta"]), "k": k,
                            "W_MHz": sa["W_MHz"] + x * (sb["W_MHz"] - sa["W_MHz"]),
                            "channel": sa["channel"]})
    return out


# ===========================================================================
# plots
# ===========================================================================
def plot_part_a(A: Dict[str, Any], png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    drives = [d for d in A["drives"] if d["spectators"]]
    if not drives:
        return
    ncol = max(len(d["spectators"]) for d in drives)
    fig, axs = plt.subplots(len(drives), ncol, figsize=(6.5 * ncol, 3.2 * len(drives)),
                            squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    for r, d in enumerate(drives):
        for c in range(ncol):
            ax = axs[r, c]
            if c >= len(d["spectators"]):
                ax.axis("off")
                continue
            sp = d["spectators"][c]
            ax.plot(sp["_t"], sp["_bp"], color=SERIES[0], lw=1.6,
                    label=f"{sp['channel']} band-passed at W")
            ax.plot(sp["_t"], sp["_model"], color=SERIES[1], lw=1.4, ls="--",
                    label="two-level model")
            ax.axvline(sp["t_iswap_ns"], color=INK, lw=0.9, label="t_iSWAP")
            for rt in sp["returns_ns"]:
                ax.axvline(rt, color=INK2, lw=0.5, ls=":")
            _style(ax, f"η={d['eta']:.2f}: W={sp['W_MHz']:.1f} MHz, g={sp['g_MHz']:.2f}, "
                       f"corr={sp['corr']:.2f}, amp ratio={sp['amp_ratio']:.2f}",
                   "time (ns)", "population − mean")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Part A: two-level model vs flat trajectories ({A['tag']}); "
                 f"dotted = predicted returns k/W", fontsize=11, color=INK, x=0.01,
                 ha="left")
    fig.savefig(png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


CH_MARKER = {"P_coupler": "o", "P_leak": "s", "P_f_a": "^", "P_f_b": "v"}


def plot_part_b(doc: Dict[str, Any], png: str, *, gate_floor: float = 0.5) -> None:
    """Error vs drive, and WHERE each spectator line sits relative to the square
    pulse's spectral nulls ``f_k = k/T(eta)``.

    Panel 2 is the null ladder: grey curves ``k/T(eta)`` with every measured line
    ``W_j(eta)`` on top, coloured by ``sin^2(pi W_j T)`` (0 = sitting on a null).
    Panel 3 is the same information as a distance, ``W_j - k/T`` to the nearest null,
    inside the ``+-1/(2T)`` band; the inner band is ``sin^2 < 0.1``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = doc["rows"]
    eta = np.array([r["eta"] for r in rows])
    T = np.array([r["T_ns"] for r in rows])
    d_eta = float(np.median(np.diff(eta))) if len(eta) > 1 else 0.025
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 11.5), constrained_layout=True,
                            sharex=True, gridspec_kw={"height_ratios": [1.0, 1.35, 1.0]})
    fig.patch.set_facecolor("#fcfcfb")

    def shade(ax, label=True):
        ex = [r["eta"] for r in rows if r["extrapolated"]]
        if ex:
            ax.axvspan(min(ex) - d_eta / 2, max(ex) + d_eta / 2, color="#e4e3df",
                       alpha=0.55, lw=0,
                       label="carrier extrapolated past measured data" if label else None)
        broken = [r["eta"] for r in rows if r["P10_at_T"] < gate_floor]
        if broken:
            ax.axvspan(min(broken) - d_eta / 2, max(eta) + d_eta / 2, fill=False,
                       hatch="///", edgecolor="#c9c8c2", lw=0,
                       label=(f"not a working gate (P10(T) < {gate_floor:g})"
                              if label else None))

    # ---- 1: error vs drive
    ax = axs[0]
    shade(ax)
    ax.semilogy(eta, [r["infidelity"] for r in rows], "-o", ms=3.5, lw=2,
                color=SERIES[0], label="square-gate infidelity 1−F")
    ax.semilogy(eta, [max(r["P_leak_slow_at_T"], 1e-6) for r in rows], "-o", ms=3.5,
                lw=2, color=SERIES[1], label="measured P_leak(T) from |01⟩, < 400 MHz")
    ax.semilogy(eta, [max(r["pred_sum_P_T"], 1e-6) for r in rows], "--", lw=1.6,
                color=SERIES[2], label="two-level Σ P_max sin²(πW T), top two")
    _style(ax, "error vs drive", "", "probability")
    ax.set_ylim(top=3.0)
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.92, loc="upper left", ncol=2)

    # collect every stored line, splitting off the gate's own slow exchange
    pts, slow = [], []
    for r in rows:
        for sp in r["spectators"]:
            rec = (r["eta"], sp["W_MHz"], r["T_ns"], sp.get("channel", "P_leak"),
                   sp["P_max"])
            (slow if sp["W_MHz"] <= 1.5e3 / r["T_ns"] else pts).append(rec)

    # ---- 2: the null ladder
    ax = axs[1]
    shade(ax, label=False)
    wmax = max([p[1] for p in pts] + [50.0]) * 1.08
    kmax = int(np.ceil(wmax * 1e-3 * T.max()))
    eta_f = np.linspace(eta.min(), eta.max(), 400)
    T_f = np.interp(eta_f, eta, T)
    for k in range(1, kmax + 1):
        ax.plot(eta_f, 1e3 * k / T_f, color="#b9b8b2", lw=0.6, zorder=1)
    sc = None
    for ch, mk in CH_MARKER.items():
        sel = [p for p in pts if p[3] == ch]
        if not sel:
            continue
        x = [p[0] for p in sel]
        y = [p[1] for p in sel]
        c = [np.sin(np.pi * p[1] * 1e-3 * p[2]) ** 2 for p in sel]
        sz = [18 + 260 * min(p[4], 0.5) for p in sel]
        sc = ax.scatter(x, y, c=c, cmap="viridis", vmin=0, vmax=1, s=sz, marker=mk,
                        edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
    if slow:
        ax.scatter([p[0] for p in slow], [p[1] for p in slow], s=14, marker="x",
                   color="#9a998f", zorder=2)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="sin²(π W_j T)   (0 = on a null)")
    handles = [Line2D([], [], color="#b9b8b2", lw=0.8, label="spectral nulls k / T(η)")]
    handles += [Line2D([], [], ls="", marker=mk, color=INK2, label=f"line seen in {ch}")
                for ch, mk in CH_MARKER.items() if any(p[3] == ch for p in pts)]
    if slow:
        handles.append(Line2D([], [], ls="", marker="x", color="#9a998f",
                              label="gate's own exchange (< 1.5/T, not a spectator)"))
    ax.legend(handles=handles, fontsize=7, frameon=True, framealpha=0.9,
              loc="upper left")
    ax.set_ylim(0, wmax)
    _style(ax, "spectator lines on the square pulse's null ladder "
               "(marker size ∝ sudden-switch P_max; ladder kinks = jitter in the "
               "calibrated T)", "", "frequency (MHz)")

    # ---- 3: distance to the nearest null
    ax = axs[2]
    shade(ax, label=False)
    half = 1e3 / (2 * T_f)
    tol = 1e3 * np.arcsin(np.sqrt(0.1)) / (np.pi * T_f)
    ax.fill_between(eta_f, -half, half, color="#eeeeea", lw=0, zorder=0,
                    label="±1/(2T): farthest from any null")
    ax.fill_between(eta_f, -tol, tol, color="#cfe3d8", lw=0, zorder=1,
                    label="sin² < 0.1 (effectively nulled)")
    ax.axhline(0.0, color=INK, lw=0.9, zorder=2)
    for ch, mk in CH_MARKER.items():
        sel = [p for p in pts if p[3] == ch]
        if not sel:
            continue
        x = [p[0] for p in sel]
        off = [p[1] - 1e3 * np.round(p[1] * 1e-3 * p[2]) / p[2] for p in sel]
        c = [np.sin(np.pi * p[1] * 1e-3 * p[2]) ** 2 for p in sel]
        sz = [18 + 260 * min(p[4], 0.5) for p in sel]
        ax.scatter(x, off, c=c, cmap="viridis", vmin=0, vmax=1, s=sz, marker=mk,
                   edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
    _style(ax, "offset of each spectator from its nearest null, W_j − k/T",
           "plateau drive η_flat", "W_j − k/T (MHz)")
    ax.legend(fontsize=7, frameon=True, framealpha=0.92, loc="upper left")
    fig.suptitle(doc["title"], fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=os.path.join(REPO, "devices", "6Gate4.7SNAIL.json"))
    ap.add_argument("--deltas", default="0.04,0.1,-0.12,0.015")
    ap.add_argument("--eta", type=float, default=1.3)
    ap.add_argument("--eta-flat", default="0.9:1.7:33")
    ap.add_argument("--branch", default="above")
    ap.add_argument("--levels", type=int, default=9)
    ap.add_argument("--laws", default=DEFAULT_LAWS)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--dt-ns", type=float, default=0.25)
    ap.add_argument("--near-span-MHz", type=float, default=3.0)
    ap.add_argument("--near-points", type=int, default=7)
    ap.add_argument("--far-span-MHz", type=float, default=12.0)
    ap.add_argument("--far-points", type=int, default=11)
    ap.add_argument("--skip-b", action="store_true", help="Part A only (no solves)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "plateau_study", f"constant_sync_{_dt.date.today().isoformat()}"))
    ap.add_argument("--replot", default="",
                    help="redraw every *_partB.png from the JSONs in this dir; no solves")
    args = ap.parse_args(argv)

    if args.replot:
        import glob
        for f in sorted(glob.glob(os.path.join(args.replot, "d*MHz_eta*.json"))):
            with open(f) as fh:
                doc = json.load(fh)
            if doc.get("rows"):
                plot_part_b(doc, f[:-5] + "_partB.png")
                print(f"redrew {f[:-5]}_partB.png", flush=True)
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(args.device) as fh:
        device = json.load(fh)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": 500000}
    grid = parse_list(args.eta_flat)

    for delta in parse_list(args.deltas):
        tag = f"d{delta * 1e3:+.0f}MHz_eta{eta_tag(args.eta)}"
        # ---- Part A
        A = part_a(tag, [e for e in (0.78, 1.04, 1.3)
                         if min(grid) - 1e-9 <= e <= max(grid) + 1e-9])
        plot_part_a(A, os.path.join(args.out, f"{tag}_partA.png"))
        for d in A["drives"]:
            for sp in d["spectators"]:
                print(f"[{tag}] A eta={d['eta']:.2f} W={sp['W_MHz']:.1f} MHz "
                      f"g={sp['g_MHz']:.2f} ({sp['channel']}, {sp['match']}): "
                      f"corr={sp['corr']:.2f} amp_ratio={sp['amp_ratio']:.2f} "
                      f"t_iSWAP-return="
                      f"{'n/a' if sp['t_iswap_to_return_ns'] is None else format(sp['t_iswap_to_return_ns'], '+.2f')}"
                      f" ns", flush=True)
        doc: Dict[str, Any] = {"tag": tag, "delta_GHz": delta, "partA": A,
                               "settings": vars(args)}
        if not args.skip_b:
            col = load_column(args.laws, delta, args.eta)
            cfg = column_config(device, col, args.levels, args.branch)
            with open(os.path.join(FLAT_DIR, f"{tag}.json")) as fh:
                flat = json.load(fh)
            rows: List[Dict[str, Any]] = []
            for eta in grid:
                c0, extrap = carrier_seed(flat, eta, rows)
                r = part_b_point(cfg, eta, c0, extrap, args, solver)
                rows.append(r)
                sp = "; ".join(f"W={s['W_MHz']:.1f} WT={s['W_T']:.2f}"
                               for s in r["spectators"])
                print(f"[{tag}] B eta={eta:.3f}{' (extrap)' if extrap else ''}: "
                      f"carrier {r['carrier_MHz']:+.2f} MHz, T={r['T_ns']:.1f} ns "
                      f"(area {r['T_area_ns']:.1f}), F={r['F_avg']:.4f}, "
                      f"leak_slow={r['P_leak_slow_at_T']:.4f}, "
                      f"pred={r['pred_sum_P_T']:.4f} | {sp} | {r['seconds']:.0f}s",
                      flush=True)
            doc.update(rows=rows, magic=magic_drives(rows),
                       title=f"Square-pulse sync check, δ = {delta * 1e3:+.0f} MHz "
                             f"(no ramps, no chirp)")
            plot_part_b(doc, os.path.join(args.out, f"{tag}_partB.png"))
        # strip the plotting arrays out of Part A before saving
        for d in A["drives"]:
            for sp in d["spectators"]:
                for k in ("_t", "_bp", "_model"):
                    sp.pop(k, None)
            d.pop("P10", None)
            d.pop("t", None)
        with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
            json.dump(_jsonable(doc), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
