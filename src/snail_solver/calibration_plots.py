#!/usr/bin/env python3
"""
calibration_plots.py
====================

Two-panel figure showing the AC-Stark shift as pump frequency vs pulse length:

  (a) Chevron -- constant-pump P(|01>->|10>) over (pump-on time, pump offset) at one
      operating point; the vertex sits at a NON-zero offset, the Stark shift.
  (b) Stark dispersion -- located resonance offset vs full-iSWAP pulse length L.
      Each L carries its constant-pump pi/2 drive, eta = (pi/2) / (6 (2pi g3) la lb L),
      so shorter pulses mean larger |eta| and (shift ~ |eta|^2) a larger retune.
      A |eta|^2 guide is overlaid.

A diagnostics tool, independent of calibrate_gate.py (``--amp-scale`` probes at a
calibrated amplitude).

Usage
-----
    python -m snail_solver.calibration_plots --device dev.json \
        --lengths 90,120,160,220,320,500 --span-MHz 80 --out stark_dispersion.png
    python -m snail_solver.calibration_plots --device dev.json --eta-min 0.3 --eta-max 1.5 --n-lengths 8

Requires QuTiP (run on a compute node).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence

import numpy as np

from snail_solver.device_utils import load_device
from snail_solver import find_stark_resonance as FS

TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# length <-> eta bookkeeping (constant-pulse full-iSWAP normalization)
# ---------------------------------------------------------------------------
def rate_per_eta(config: Dict[str, Any]) -> float:
    """Leading-order iSWAP rate per unit |eta|, 6 (2pi g3) la lb (rad/ns)."""
    return 6.0 * (TWO_PI * float(config["g3_GHz"])) * float(config["lam_a"]) * float(config["lam_b"])


def eta_of_length(config: Dict[str, Any], length_ns: float) -> float:
    """Constant-pump |eta| of a full iSWAP of duration length_ns (the Stark
    calibration is constant-pulse): eta = (pi/2) / (6 (2pi g3) la lb L)."""
    return float((np.pi / 2.0) / (rate_per_eta(config) * float(length_ns)))


def length_of_eta(config: Dict[str, Any], eta: float) -> float:
    """Constant-pump full-iSWAP duration (ns) for |eta|: L = (pi/2)/(rate eta)."""
    return float((np.pi / 2.0) / (rate_per_eta(config) * float(eta)))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep_lengths(config: Dict[str, Any], lengths_ns: Sequence[float], amp_scale: float,
                  span_MHz: float, points: int, window_factor: float, time_points: int,
                  solver: Optional[Dict[str, Any]] = None,
                  n_jobs: Optional[int] = None,
                  panel_a_index: Optional[int] = None) -> Dict[str, Any]:
    """Locate the Stark-shifted resonance at each full-iSWAP pulse length
    (chevron over +/- span/2, time window ``window_factor * L``), keeping the chevron
    at ``panel_a_index`` (default: the middle length) for panel (a).

    Returns lengths_ns, etas, resonance_offset_GHz (per length) and ``panel_a``
    (offsets_GHz, times_ns, P10, resonance_offset_GHz, eta_op, length_ns).
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    lengths = np.asarray(lengths_ns, dtype=float)
    if panel_a_index is None:
        panel_a_index = len(lengths) // 2
    span = span_MHz / 1000.0
    offsets = np.linspace(-span / 2.0, span / 2.0, points)

    res_off = np.zeros(len(lengths))
    etas = np.zeros(len(lengths))
    panel_a: Dict[str, Any] = {}
    for k, L in enumerate(lengths):
        out = FS.scan(config, float(L), amp_scale, offsets, window_factor * float(L),
                      time_points, solver, n_jobs=n_jobs)
        res_off[k] = out["resonance_offset_GHz"]
        etas[k] = out["eta_op"]
        if k == panel_a_index:
            panel_a = {"offsets_GHz": out["offsets_GHz"], "times_ns": out["times_ns"],
                       "P10": out["P10"], "resonance_offset_GHz": out["resonance_offset_GHz"],
                       "eta_op": out["eta_op"], "length_ns": float(L)}
        print(f"  L={L:7.1f} ns  |eta|={etas[k]:.3f}  "
              f"resonance={res_off[k]*1e3:+.2f} MHz")

    return {"lengths_ns": lengths, "etas": etas, "resonance_offset_GHz": res_off,
            "panel_a": panel_a}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(result: Dict[str, Any], png_path: str, config: Dict[str, Any]) -> None:
    """Render the two-panel Stark figure from `sweep_lengths` output to `png_path`
    (`config` supplies the |eta| axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:
        pass

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.7), layout="constrained")

    # (a) chevron: pump-on duration (x) vs pump-frequency offset (y)
    pa = result["panel_a"]
    off_MHz = pa["offsets_GHz"] * 1e3
    res_MHz = float(pa["resonance_offset_GHz"]) * 1e3
    mesh = ax0.pcolormesh(pa["times_ns"], off_MHz, pa["P10"], shading="auto",
                          cmap="viridis", vmin=0, vmax=1)
    ax0.axhline(res_MHz, color="r", ls="--", lw=1.5,
                label=f"resonance {res_MHz:+.1f} MHz")
    ax0.axhline(0.0, color="w", ls=":", lw=1.0, label=r"bare $|w_b-w_a|$")
    ax0.set_xlabel("pulse length / pump-on time (ns)")
    ax0.set_ylabel(r"pump offset from $|w_b-w_a|$ (MHz)")
    ax0.set_title(f"(a) chevron at $|\\eta|$={float(pa['eta_op']):.3f} "
                  f"($L$={float(pa['length_ns']):.0f} ns)")
    ax0.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(mesh, ax=ax0, label=r"$P(|01\rangle\to|10\rangle)$")

    # (b) Stark dispersion: resonance offset vs full-iSWAP pulse length
    L = result["lengths_ns"]; res = result["resonance_offset_GHz"] * 1e3
    etas = result["etas"]
    ax1.plot(L, res, "o-", ms=5, color="C3", label="located resonance")
    # eta^2 guide anchored at the longest (weakest-pump) point
    k0 = int(np.argmax(L))
    if abs(res[k0]) > 1e-9:
        guide = res[k0] * (etas / etas[k0]) ** 2
        ax1.plot(L, guide, "k--", lw=1.2, alpha=0.7, label=r"$\propto|\eta|^2$ guide")
    ax1.axhline(0.0, color="0.6", ls=":", lw=1.0)
    ax1.set_xlabel("full-iSWAP pulse length $L$ (ns)")
    ax1.set_ylabel("Stark resonance offset (MHz)")
    ax1.set_title("(b) resonance shift vs pulse length")
    # annotate |eta| on a secondary axis (constant-pulse: eta = (pi/2)/(rate*L))
    _half_pi = np.pi / 2.0
    ax1b = ax1.secondary_xaxis("top",
                               functions=(lambda x: _half_pi / (rate_per_eta(config) * np.clip(x, 1e-9, None)),
                                          lambda e: _half_pi / (rate_per_eta(config) * np.clip(e, 1e-9, None))))
    ax1b.set_xlabel(r"constant-pump $|\eta|$")
    ax1.legend(loc="best", framealpha=0.9)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_lengths(args, config: Dict[str, Any]) -> np.ndarray:
    """Resolve the pulse-length grid from --lengths or an --eta-min/max range."""
    if args.lengths:
        return np.array([float(x) for x in args.lengths.split(",")], dtype=float)
    etas = np.linspace(args.eta_max, args.eta_min, args.n_lengths)   # long pulse last
    return np.array([length_of_eta(config, e) for e in etas], dtype=float)


def main() -> None:
    """Sweep pulse length, locate the Stark resonance at each, and plot the shift."""
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.calibration_plots", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON (run_sweep schema)")
    ap.add_argument("--lengths", default=None,
                    help="comma list of full-iSWAP pulse lengths (ns); overrides --eta range")
    ap.add_argument("--eta-min", type=float, default=0.3)
    ap.add_argument("--eta-max", type=float, default=1.5)
    ap.add_argument("--n-lengths", type=int, default=7)
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--span-MHz", type=float, default=80.0)
    ap.add_argument("--points", type=int, default=31)
    ap.add_argument("--window-factor", type=float, default=2.0)
    ap.add_argument("--time-points", type=int, default=160)
    ap.add_argument("--panel-a-index", type=int, default=None,
                    help="which length index to show as the chevron (default: middle)")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--out", default="stark_dispersion.png")
    ap.add_argument("--save-npz", default=None, help="also save the swept arrays")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    args = ap.parse_args()

    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from snail_solver.paths import resolve_device, in_results
    args.out = in_results(args.out)
    if args.save_npz:
        args.save_npz = in_results(args.save_npz)
    config = load_device(resolve_device(args.device))
    lengths = _parse_lengths(args, config)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    print(f"device={args.device}  lengths(ns)={np.round(lengths,1).tolist()}"
          f"{' GPU' if args.gpu else ''}")

    result = sweep_lengths(config, lengths, args.amp_scale, args.span_MHz, args.points,
                           args.window_factor, args.time_points, solver,
                           n_jobs=args.jobs, panel_a_index=args.panel_a_index)
    if args.save_npz:
        np.savez_compressed(args.save_npz, lengths_ns=result["lengths_ns"],
                            etas=result["etas"],
                            resonance_offset_GHz=result["resonance_offset_GHz"],
                            **{f"panel_a_{k}": v for k, v in result["panel_a"].items()})
        print(f"saved {args.save_npz}")
    plot(result, args.out, config)
    print(f"plotted {args.out}")


if __name__ == "__main__":
    main()