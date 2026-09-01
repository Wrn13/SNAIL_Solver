"""
validate_recursive_drag.py
==========================

Does recursive DRAG actually beat single-derivative DRAG on THIS device?

Everything else added with the recursion checks that the pulse we build is the one
Li, Calarco & Motzoi define (npj QI 10, 66 (2024)) -- exact derivatives, the right
composition order, agreement across all four solve paths. None of that is evidence
that composing the corrections *helps here*. This module is that evidence, or its
absence.

The experiment (their Fig. 2b, transposed to this device)
----------------------------------------------------------
Their claim is specifically about **frequency crowding**: a single derivative
correction has one knob, so with more than one nearby off-resonant process it can only
trade one process's error against another's, while one correction per process
suppresses all of them at once. So the sweep axis has to be the thing that controls
how crowded the spectrum is -- here the SPECTATOR frequency, which sets the beats of
the collision channels (``sweep_common._collision_candidates``).

At each spectator placement we score the same gate under four schemes::

    none        no DRAG at all
    F1          first-order DRAG on the NEAREST collision   (what the repo did before)
    F1oF1       the two nearest, composed
    F1oF1oF2    the three nearest, multi-photon innermost   (the paper's Eq. 8)

Reading the result honestly
---------------------------
The number to look at is ``dF`` = infidelity, and the comparison that matters is
``F1`` vs the composed schemes AT THE SAME spectator placement. Two ways this can
come out, and both are informative:

* the composed schemes win, by a margin that GROWS as the beats close in -- the
  paper's claim, reproduced;
* they do not -- which on this device would most likely mean the second and third
  collisions are far enough away to be irrelevant at these parameters, i.e. the
  regime is not frequency-crowded. ``beats_MHz`` in the output says which.

A composed scheme can also be WORSE, and the cause is usually visible in
``corr_ratio`` (:func:`device_utils.drag_correction_ratio`): once the correction is
comparable to the pulse it corrects, the perturbative expansion has stopped meaning
anything and adding another order makes it worse, not better. That is a real result
about the operating point, not a bug.

Base shape
----------
Defaults to ``sine_power`` with ``m = 3``. This is not cosmetic: a Hann window has
only two vanishing end derivatives, and under ``F1oF1oF2`` the pulse literally
diverges as ``t^(-1/2)`` at both gate edges (see :class:`envelope.SinePowerRamp`, and
``test_hann_diverges_under_the_full_recursion``). Running this comparison on a Hann
would measure that divergence rather than the physics. ``--shape raised_cosine`` is
allowed so the contrast can be shown deliberately.

CLI
---
    python -m snail_solver.validate_recursive_drag --device devices/evan_device.json \\
        --t-g 40 --points 7
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TWO_PI = 2.0 * np.pi

#: The schemes compared at every spectator placement, as (label, n_channels).
SCHEMES = (("none", 0), ("F1", 1), ("F1oF1", 2), ("F1oF1oF2", 3))


def _score_point(config: Dict[str, Any], t_g: float, spec_abs_GHz: float,
                 n_channels: int, *, amp_scale: float = 1.0,
                 wp_offset_GHz: float = 0.0,
                 chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                 solver: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One (spectator, scheme) cell: build the gate, solve it, score it."""
    from snail_solver.device_utils import build_coupler, drag_correction_ratio
    from snail_solver.sweep_common import collision_drag_channels
    from snail_solver.zhou_coupler import ZhouCoupler

    solver = solver or {"atol": 1e-9, "rtol": 1e-7, "nsteps": 200000}
    wa, wb = (float(x) for x in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    w_p = abs(wb - wa) + float(wp_offset_GHz)

    # spec_abs_GHz=None builds the bare (a, b, coupler) trio -- the reference gate
    channels = ()
    if n_channels and spec_abs_GHz is not None:
        channels = collision_drag_channels(config, wa, wb, ws, float(spec_abs_GHz),
                                           w_p, n=int(n_channels),
                                           chirp_coeffs_GHz=chirp_coeffs_GHz, t_g=t_g)
    cpl, _w_p, peak = build_coupler(
        config, t_g, amp_scale, wp_offset_GHz,
        spec_abs_GHz=(None if spec_abs_GHz is None else float(spec_abs_GHz)),
        chirp_coeffs_GHz=chirp_coeffs_GHz,
        drag_channels=(list(channels) if channels else None))
    U = cpl.propagator_columns(0, 1, t_g, **solver)
    F, leak = ZhouCoupler._iswap_fidelity_from_U(U, True)
    return {"F": float(F), "leak": float(leak), "dF": float(1.0 - F),
            "n_channels_used": len(channels),
            "beats_MHz": [float(c.beat_GHz) * 1e3 for c in channels],
            "n_photon": [int(c.n_photon) for c in channels],
            "corr_ratio": float(drag_correction_ratio(cpl._pump_tones[0])),
            "peak_eta": float(peak)}


def calibrate_operating_point(config: Dict[str, Any], target_eta: float, *,
                              span_MHz: float = 120.0, points: int = 25,
                              solver: Optional[Dict[str, Any]] = None,
                              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """A working gate to run the comparison ON, before any spectator is added.

    This is not optional set-up -- it is what makes the sweep measure anything. At an
    UNCALIBRATED point the gate is dominated by rotation-angle and detuning error
    (infidelity ~0.2 in a first attempt at these parameters), and a leakage-suppression
    scheme cannot be seen underneath that: every scheme scores roughly the same large
    number, and which one wins is noise.

    Fixes the length from ``nominal_t_g(target_eta)`` (the analytic full iSWAP at that
    drive) with ``amp_scale`` holding the peak there, then calibrates the carrier with
    one DRAG-off, spectator-free chevron. That single offset is then held across the
    whole sweep, so every cell differs only in the spectator and the DRAG scheme.
    """
    from snail_solver import find_stark_resonance as FSR
    from snail_solver.tune_up import fixed_eta_amp_scale, nominal_t_g

    t_g = nominal_t_g(config, float(target_eta))
    amp_scale = fixed_eta_amp_scale(config, t_g, float(target_eta))
    offsets = np.linspace(-span_MHz / 2, span_MHz / 2, int(points)) * 1e-3
    chev = FSR.scan(config, t_g, amp_scale, offsets, 1.05 * t_g, 81,
                    shape="raised_cosine", solver=solver, n_jobs=0)
    wp_offset = float(chev["resonance_offset_GHz"])
    if logger:
        logger.info(f"  calibrated: t_g={t_g:.3f} ns  amp_scale={amp_scale:.4f}  "
                    f"wp_offset={wp_offset * 1e3:+.3f} MHz  "
                    f"(peak transfer {np.max(chev['resonance_metric']):.4f})")
    return {"t_g_ns": t_g, "amp_scale": amp_scale, "wp_offset_GHz": wp_offset,
            "target_eta": float(target_eta),
            "peak_transfer": float(np.max(chev["resonance_metric"]))}


def sweep(config: Dict[str, Any], t_g: float, *,
          spec_GHz: Optional[Sequence[float]] = None, points: int = 7,
          span_GHz: float = 0.45, shape: str = "sine_power", m: int = 3,
          amp_scale: float = 1.0, wp_offset_GHz: float = 0.0,
          chirp_coeffs_GHz: Optional[Sequence[float]] = None,
          schemes: Sequence = SCHEMES,
          solver: Optional[Dict[str, Any]] = None,
          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Sweep the spectator frequency and score every scheme at each placement.

    Parameters
    ----------
    config : dict
        Merged device configuration. The base shape is overridden by `shape`/`m`.
    t_g : float
        Gate length (ns), held fixed across the sweep so the comparison is
        like-for-like.
    spec_GHz : sequence of float, optional
        Explicit spectator placements. Default: `points` values spanning
        ``+/- span_GHz/2`` about qubit b, skipping placements that land on top of a
        qubit (where no DRAG scheme is meaningful -- an on-resonant collision needs
        frequency allocation, not a derivative correction).
    schemes : sequence of (label, n_channels)
        Defaults to :data:`SCHEMES`.

    Returns
    -------
    dict
        ``spec_GHz``, ``schemes``, ``rows`` (one per placement, keyed by scheme
        label), ``t_g_ns``, ``shape``, and ``best`` (the scheme with the lowest mean
        infidelity across the sweep).
    """
    cfg = dict(config)
    cfg["envelope"] = shape
    if shape == "sine_power":
        cfg["envelope_m"] = int(m)
        cfg.setdefault("envelope_rise_frac", 0.5)

    wa, wb = (float(x) for x in cfg["qubit_freqs_GHz"])
    if spec_GHz is None:
        grid = wb + np.linspace(-float(span_GHz) / 2, float(span_GHz) / 2, int(points))
        # a spectator sitting exactly on a qubit is a resonant collision, not an
        # off-resonant one: no derivative correction applies, so it is not a fair cell
        grid = np.array([f for f in grid
                         if min(abs(f - wa), abs(f - wb)) > 0.02])
    else:
        grid = np.asarray(spec_GHz, dtype=float)

    # The no-spectator, no-DRAG gate. If THIS is already bad the comparison below is
    # meaningless -- a leakage-suppression scheme cannot be seen underneath a gate
    # that is not rotating, and whichever scheme happens to score best is noise. This
    # is not hypothetical: at peak |eta| ~ 1.8 the bundled device configs put ~88% of
    # the population in the coupler before any spectator is added.
    reference = _score_point(cfg, t_g, None, 0, amp_scale=amp_scale,
                             wp_offset_GHz=wp_offset_GHz,
                             chirp_coeffs_GHz=chirp_coeffs_GHz, solver=solver)
    if logger:
        logger.info(f"  reference gate (no spectator, no DRAG): "
                    f"dF={reference['dF']:.3e}  leak={reference['leak']:.3e}")

    rows: List[Dict[str, Any]] = []
    for i, spec in enumerate(grid):
        row: Dict[str, Any] = {"spec_GHz": float(spec)}
        for label, n_ch in schemes:
            try:
                row[label] = _score_point(
                    cfg, t_g, float(spec), int(n_ch), amp_scale=amp_scale,
                    wp_offset_GHz=wp_offset_GHz,
                    chirp_coeffs_GHz=chirp_coeffs_GHz, solver=solver)
            except Exception as exc:                       # one cell must not kill it
                row[label] = {"F": float("nan"), "dF": float("nan"),
                              "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if logger:
            beats = row[schemes[-1][0]].get("beats_MHz", [])
            logger.info(
                f"  spec {spec:.4f} GHz  beats "
                + ("/".join(f"{b:+.0f}" for b in beats) or "-") + " MHz  "
                + "  ".join(f"{lab} dF={row[lab]['dF']:.2e}" for lab, _ in schemes))

    means = {lab: float(np.nanmean([r[lab]["dF"] for r in rows]))
             for lab, _ in schemes}
    best = min(means, key=lambda k: means[k]) if means else None
    return {"spec_GHz": grid, "rows": rows, "t_g_ns": float(t_g), "shape": shape,
            "m": int(m), "schemes": [lab for lab, _ in schemes],
            "mean_dF": means, "best": best, "reference": reference}


def summarize(result: Dict[str, Any]) -> str:
    """Human-readable verdict: does the recursion beat single-derivative DRAG?"""
    labels = result["schemes"]
    rows = result["rows"]
    out = [f"gate {result['t_g_ns']:.1f} ns, shape={result['shape']}"
           f"{'(m=%d)' % result['m'] if result['shape'] == 'sine_power' else ''}, "
           f"{len(rows)} spectator placements", ""]
    head = f"  {'spec (GHz)':>11}  {'beats (MHz)':>22}  " + "  ".join(
        f"{lab:>11}" for lab in labels)
    out += [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        beats = r[labels[-1]].get("beats_MHz", [])
        out.append(f"  {r['spec_GHz']:>11.4f}  "
                   f"{('/'.join(f'{b:+.0f}' for b in beats) or '-'):>22}  "
                   + "  ".join(f"{r[lab]['dF']:>11.3e}" for lab in labels))
    out += ["", "  mean infidelity:"]
    for lab in labels:
        out.append(f"    {lab:>10}: {result['mean_dF'][lab]:.4e}")

    ref = result.get("reference")
    out.append("")
    if ref is not None:
        out.append(f"  reference gate (no spectator, no DRAG): dF={ref['dF']:.3e}  "
                   f"leak={ref['leak']:.3e}")
    # Refuse to draw a conclusion from a gate that does not work. A broken baseline
    # produces a confident-looking ranking that is pure noise, which is worse than
    # reporting nothing.
    if ref is not None and ref["dF"] > 0.1:
        out += ["",
                f"  NO VERDICT: the reference gate is already at dF={ref['dF']:.2f} "
                f"(leak={ref['leak']:.2f}) BEFORE any spectator or DRAG. Nothing here "
                f"measures collision suppression -- the gate is limited by something "
                f"else (at these drives, coupler leakage). Find an operating point "
                f"where the reference gate works, then re-run; the scheme ranking "
                f"above is not evidence either way."]
        return "\n".join(out)

    f1, best = result["mean_dF"].get("F1"), result["best"]
    if best is None:
        out.append("  no usable cells")
    elif best == "none":
        out.append("  VERDICT: no DRAG scheme helped at this operating point -- the "
                   "collisions are not what limits this gate.")
    elif best == "F1":
        out.append("  VERDICT: single-derivative DRAG was not improved on. Most "
                   "likely the spectrum is not crowded here (check the beats: if the "
                   "2nd/3rd are far away there is nothing for extra channels to do).")
    else:
        gain = f1 / result["mean_dF"][best] if result["mean_dF"][best] else float("inf")
        out.append(f"  VERDICT: {best} beat single-derivative DRAG, {gain:.2f}x lower "
                   f"mean infidelity. This is the paper's claim reproduced on this "
                   f"device.")
    ratios = [r[lab].get("corr_ratio", 0.0) for r in rows for lab in labels]
    worst = max((x for x in ratios if np.isfinite(x)), default=0.0)
    if worst > 0.3:
        out.append(f"  NOTE: max correction/pulse ratio {worst:.2f} -- at that size "
                   f"the perturbative expansion is no longer small, so treat the "
                   f"deepest scheme's numbers with suspicion.")
    return "\n".join(out)


def main() -> None:                                        # pragma: no cover
    ap = argparse.ArgumentParser("python -m snail_solver.validate_recursive_drag")
    ap.add_argument("--device", default=None, help="device JSON (merged over defaults)")
    ap.add_argument("--target-eta", type=float, default=1.0,
                    help="peak |eta| to calibrate at; sets t_g = 2A/eta* and the "
                         "amplitude. The comparison is meaningless on an uncalibrated "
                         "gate -- see calibrate_operating_point")
    ap.add_argument("--t-g", type=float, default=None,
                    help="override the calibrated gate length (ns)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the calibration and use --t-g with amp_scale=1 and no "
                         "carrier offset. Expect rotation error to swamp everything")
    ap.add_argument("--points", type=int, default=7)
    ap.add_argument("--span-GHz", type=float, default=0.45,
                    help="spectator window about qubit b")
    ap.add_argument("--shape", default="sine_power",
                    choices=["sine_power", "raised_cosine"],
                    help="base envelope. raised_cosine DIVERGES under the 3-channel "
                         "recursion (see the module docstring) -- opt in knowingly")
    ap.add_argument("--m", type=int, default=3, help="vanishing end derivatives")
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--wp-offset-GHz", type=float, default=0.0)
    ap.add_argument("--chirp", default=None,
                    help="comma-separated Legendre chirp coefficients (GHz)")
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("validate_recursive_drag")

    from snail_solver.sweep_common import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    if args.device:
        config.update(json.load(open(args.device)))
    config.setdefault("envelope", "raised_cosine")

    chirp = ([float(x) for x in args.chirp.split(",")] if args.chirp else None)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": 200000}

    t_g, amp_scale, wp_offset, cal = args.t_g, args.amp_scale, args.wp_offset_GHz, None
    # A device JSON that already carries a calibrated operating point is the best
    # starting gate there is -- re-deriving it costs a chevron sweep and can only be
    # worse. Explicit CLI flags still win.
    if not args.no_calibrate and "t_g_ns" in config and args.t_g is None:
        t_g = float(config["t_g_ns"])
        amp_scale = float(config.get("amp_scale", 1.0))
        wp_offset = float(config.get("wp_offset_GHz", 0.0))
        cal = {"source": "device config", "t_g_ns": t_g, "amp_scale": amp_scale,
               "wp_offset_GHz": wp_offset}
        log.info(f"using the device's calibrated point: t_g={t_g:.3f} ns  "
                 f"amp_scale={amp_scale:.4f}  wp_offset={wp_offset * 1e3:+.3f} MHz")
    elif not args.no_calibrate:
        log.info("calibrating the operating point (DRAG off, no spectator)")
        cal = calibrate_operating_point(config, args.target_eta, solver=solver,
                                        logger=log)
        t_g = args.t_g if args.t_g is not None else cal["t_g_ns"]
        amp_scale, wp_offset = cal["amp_scale"], cal["wp_offset_GHz"]
    elif t_g is None:
        t_g = 40.0

    result = sweep(config, t_g, points=args.points, span_GHz=args.span_GHz,
                   shape=args.shape, m=args.m, amp_scale=amp_scale,
                   wp_offset_GHz=wp_offset, chirp_coeffs_GHz=chirp,
                   solver=solver, logger=log)
    result["calibration"] = cal
    print()
    print(summarize(result))
    if args.out:
        def _plain(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            raise TypeError(type(o))
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=_plain)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":                                 # pragma: no cover
    main()
