"""
run_sweep_zhou.py
=================

Batch driver for ``zhou_coupler.ZhouCoupler``, the first-principles dressed-mode
model (Chao Zhou's thesis, Ch. 2): the Hamiltonian is built directly from measured
g3 (+ optional g4) and participations lambda_is = g_is/Delta_is, with the SNAIL as an
explicit mode.

Modes: 0 = qubit a, 1 = qubit b (spectator anchor), 2 = coupler S (SNAIL; carries
g3/g4), 3 = spectator. The pump sits on the coupler at w_p = w_b - w_a, normalized
to a full iSWAP at t_g.

Spectator sweep (default; spec_freq x drag)
-------------------------------------------
Only the spectator moves, w_spec = w_b - spec_freq, so the target gate is fixed.
* spec_freq : Delta = w_b - w_spec (GHz). For a single-tone pure-g3 coupler the only
  in-band collision is the one-pump exchange with the anchor at |Delta| = w_p
  (beat = |Delta| - w_p); the spectator may sit below or above w_b.
* drag      : first-order DRAG (Motzoi et al., PRL 103, 110501 (2009)),
  eta -> eta - i d(eta)/dt / (2 pi beat), tuned to the nearest collision beat;
  skipped (drag_applied=False) inside the drag-skip window, where allocation --
  not DRAG -- is the fix.
The spectator is one 3-level anharmonic transmon with participation ``lam_b``.

Frequency-allocation sweep (--sweep target)
-------------------------------------------
Fixes w_s and qubit a and scans a 2-D grid of partner w_b (``--wb-GHz``) x absolute
spectator w_spec (``--spec-GHz``). Moving w_b moves the pump, so each point sees a
different collision landscape; the analytic pass reports the NEAREST collision
(``onepump``: |w_q - w_spec| = w_p; ``static``: w_q = w_spec; ``subharm``:
w_i = 2 w_p) and ``collect`` prints the best-fidelity placement. Placements with
|w_b - w_a| < ``min_detuning_GHz`` are dropped. ``--drag-compare`` also runs DRAG-on
for points with |nearest beat| < ``--drag-compare-below-MHz`` (default 100), adding
F_avg_drag / leakage_drag / dF_drag.

Metrics per point
-----------------
* ANALYTIC (always, free): nearest collision, its beat, and the Eq.-62 rates
  g_spec_eff / g_iswap_eff -- the collision map.
* FULL (``integrate=true``, default): QuTiP ``sesolve`` of the exact Hamiltonian
  (no terms pruned) -> leakage-aware iSWAP fidelity plus spectator/coupler
  occupations. ``--no-integrate`` gives the instant analytic map (numpy/scipy only).

Workflow
--------
    python -m snail_solver.run_sweep_zhou prepare --outdir results_zhou/
    RUNNER=snail_solver.run_sweep_zhou OUTDIR=results_zhou sbatch --array=0-<M-1> slurm/snail_sweep.slurm
    python -m snail_solver.run_sweep_zhou local --outdir results_zhou/ --nproc 8
    python -m snail_solver.run_sweep_zhou collect --outdir results_zhou/

'prepare' and 'collect' do NOT import the solver; 'point'/'local' import it lazily.
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import time
from typing import Any, Dict

import numpy as np

# Shared machinery (constants, DEFAULT_*, Point, grid IO, collect, CLI helpers) is
# re-exported from here, alongside the two sweep implementations.
from snail_solver.sweep_common import *                     # noqa: F401,F403  (API re-exported)
from snail_solver.sweep_common import _parse_list, _bool_list, _log_line, _print_submit_hint
from snail_solver.sweep_spectator import build_grid, run_spectator_point
from snail_solver.sweep_target import build_target_grid, _run_target_point
from snail_solver.log_utils import setup_run_logger


def run_point(pt: "Point", config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one grid point, dispatching on ``pt.kind`` ("spectator" or "target")."""
    if getattr(pt, "kind", "spectator") == "target":
        return _run_target_point(pt, config)
    return run_spectator_point(pt, config)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.run_sweep_zhou", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prepare", "point", "local", "collect", "chevrons",
                                     "missing"])
    ap.add_argument("--outdir", default="results_zhou")
    ap.add_argument("--device", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--index", type=int, help="point index (mode=point); with --chunk N "
                    "this is the CHUNK index and the task runs points [index*N, (index+1)*N)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="points per array task (mode=point); use >1 when N exceeds "
                         "SLURM MaxArraySize (array size becomes ceil(N/chunk))")
    ap.add_argument("--grid", help="path to grid.json (default: <outdir>/grid.json)")
    ap.add_argument("--nproc", type=int, default=4, help="processes for mode=local")
    ap.add_argument("--stark-jobs", type=int, default=None,
                    help="processes for the per-point --stark chevron's offset scan "
                         "(mode=point: default SLURM_CPUS_PER_TASK; mode=local: forced to 1)")
    ap.add_argument("--sweep", choices=["spectator", "target"], default="spectator",
                    help="spectator sweep (default) or target-frequency allocation sweep")
    ap.add_argument("--specfreqs", help="comma list of spectator freqs Delta = w_b - w_spec (GHz)")
    ap.add_argument("--beats", help="[spectator] comma list of BEAT detunings delta = Delta - w_p "
                                    "(GHz); 0 = on the collision. Sets spec_freq = w_p + delta.")
    ap.add_argument("--beat-span-MHz", type=float, default=None,
                    help="[spectator] auto beat sweep: +/-span/2 about the collision (delta=0)")
    ap.add_argument("--beat-points", type=int, default=13,
                    help="[spectator] number of points for --beat-span-MHz (default 13)")
    ap.add_argument("--clip-band", action="store_true",
                    help="[spectator] drop points whose absolute w_spec = w_b - Delta falls "
                         "outside the physical band [w_a, w_b]")
    ap.add_argument("--drags", help="comma list of bools, e.g. false,true")
    ap.add_argument("--t-g-ns", type=float, default=None,
                    help="gate duration t_g (ns); overrides the device/default value")
    ap.add_argument("--resume-list", default=None,
                    help="[point] file of point indices (one per line, e.g. missing.txt, "
                         "written by `missing`); --index selects the chunk-th slice of it")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="set t_g via auto_t_g so the raised-cosine full-iSWAP pump has "
                         "peak |eta| = TARGET_ETA (t_g = 2*area/eta); takes precedence "
                         "over --t-g-ns")
    ap.add_argument("--chirp-GHz", default=None,
                    help="comma list of Legendre coefficients (GHz) of a pump-frequency "
                         "offset delta(t), e.g. '0,0.05' sweeps -50 -> +50 MHz across the "
                         "gate. A single value == adding it to wp_offset_GHz. Empty or "
                         "all-zero leaves the pump un-chirped.")
    ap.add_argument("--wb-GHz", help="[target] comma list of partner (w_b) freqs (GHz)")
    ap.add_argument("--spec-GHz", help="[target] comma list of spectator ABSOLUTE freqs (GHz)")
    ap.add_argument("--spec-min-GHz", type=float, default=None,
                    help="[target] lower edge of the auto spectator band (GHz); "
                         "default min(w_a, min w_b) - 0.40")
    ap.add_argument("--spec-max-GHz", type=float, default=None,
                    help="[target] upper edge of the auto spectator band (GHz); "
                         "default max(w_a, max w_b) + 0.40")
    ap.add_argument("--spec-step-GHz", type=float, default=None,
                    help="[target] spectator grid step for the auto band (GHz); default 0.10")
    ap.add_argument("--drag-compare", action="store_true",
                    help="[target] also run DRAG-on where |nearest beat| < --drag-compare-below-MHz")
    ap.add_argument("--drag-compare-below-MHz", type=float, default=None,
                    help="[target] near-collision window for the DRAG comparison (default 100)")
    ap.add_argument("--no-integrate", action="store_true",
                    help="analytic collision map only (no time integration)")
    ap.add_argument("--stark", action="store_true",
                    help="drive each point at its AC-Stark-shifted resonance from a per-point "
                         "chevron that INCLUDES the spectator; integrated run only, slower")
    ap.add_argument("--stark-match-pulse", action="store_true",
                    help="make the --stark chevron use the ACTUAL gate pulse (+ DRAG on "
                         "DRAG-on points) instead of a constant probe")
    ap.add_argument("--calibrate", action="store_true",
                    help="per-point amplitude+Stark tune-up (calibrate_gate) with the "
                         "spectator loaded and DRAG on if the point uses it; slowest")
    ap.add_argument("--calibrate-iters", type=int, default=None,
                    help="amplitude/frequency rounds per point for --calibrate (default 1)")
    ap.add_argument("--grape", action="store_true",
                    help="per-point GRAPE optimal control (grape.optimize_pulse) on the "
                         "integrated gate; records grape_baseline_F / F_grape / dF_grape / "
                         "leak_grape and the optimized envelope. Opt-in and costly")
    ap.add_argument("--grape-backend", choices=["qutip", "reduced"], default=None,
                    help="GRAPE engine: 'qutip' (qutip-qoc, default) or 'reduced' "
                         "(in-house scipy over the reduced model)")
    ap.add_argument("--grape-alg", choices=["JOPT", "CRAB"], default=None,
                    help="qutip GRAPE optimizer: JOPT (JAX gradients; needs qutip-qoc + "
                         "qutip-jax) or CRAB (gradient-free, needs only qutip)")
    ap.add_argument("--grape-crab-restarts", type=int, default=None,
                    help="[CRAB] DCRAB super-iterations (monotone)")
    ap.add_argument("--grape-crab-seed", type=int, default=None,
                    help="[CRAB] RNG seed for the random basis (reproducible pulses)")
    ap.add_argument("--grape-crab-score", choices=["qutip", "reduced"], default=None,
                    help="[CRAB] objective: exact QuTiP propagator (default) or the "
                         "fast reduced model")
    ap.add_argument("--grape-crab-method", default=None,
                    help="[CRAB] gradient-free scipy method (Nelder-Mead, Powell, ...)")
    ap.add_argument("--grape-nbasis", type=int, default=None,
                    help="sin() basis functions per quadrature (qutip GRAPE backend)")
    ap.add_argument("--grape-nctrl", type=int, default=None,
                    help="GRAPE piecewise-constant control points (default 24)")
    ap.add_argument("--grape-cutoff-GHz", type=float, default=None,
                    help="GRAPE reduced-model carrier cutoff (default 1.0 GHz)")
    ap.add_argument("--grape-maxiter", type=int, default=None,
                    help="GRAPE L-BFGS-B iteration cap (default 200)")
    ap.add_argument("--grape-warmstart-drag", action="store_true",
                    help="on DRAG-off points, seed GRAPE from DRAG at the nearest beat "
                         "(baseline/dF unchanged; skipped inside the drag-skip window)")
    ap.add_argument("--drag", action="store_true",
                    help="force DRAG on for every allocation point (tuned to the nearest beat)")
    ap.add_argument("--operating-point", default=None,
                    help="apply a calibrated operating point from the device JSON "
                         "(amp_scale + wp_offset, and t_g unless --t-g-ns/--target-eta); "
                         "create with calibration_map.py --save-point, list with "
                         "operating_points.py --device <dev>")
    ap.add_argument("--operating-point-strict", action="store_true",
                    help="fail instead of warning when the operating point was "
                         "calibrated in a different context (w_a/w_b/t_g/spectator)")
    sub = ap.add_mutually_exclusive_group()
    sub.add_argument("--drag-subharmonic", action="store_true",
                     help="let DRAG target the nearest SUBHARMONIC collision (2 w_p driving "
                          "mode i at w_i = 2 w_p, i in --subharmonic-modes). ON BY DEFAULT; "
                          "this flag only pins it explicitly")
    sub.add_argument("--no-drag-subharmonic", action="store_true",
                     help="turn the default-on subharmonic DRAG channels OFF (also "
                          "overrides a value baked into an existing grid.json)")
    ap.add_argument("--subharmonic-modes", default=None,
                    help="comma list, subset of {a,b,spec,s}, of subharmonic channels to "
                         "include (default all); an empty string selects NONE")
    ap.add_argument("--no-spectator", action="store_true",
                    help="[target] sweep the BARE a-b-coupler gate (lam_spec=0); vary "
                         "--wb-GHz so 2 w_p scans the SNAIL subharmonic w_c = 2 w_p "
                         "(needs w_p = w_c/2 in band -- e.g. coupler <= 4.4 for w_a=3.5)")
    ap.add_argument("--stark-span-MHz", type=float, default=None,
                    help="per-point Stark chevron width (default 60)")
    ap.add_argument("--stark-points", type=int, default=None,
                    help="per-point Stark chevron offset samples (default 21)")
    ap.add_argument("--gpu", action="store_true",
                    help="run the solver on GPU via qutip-jax/diffrax (see zhou_coupler.use_gpu)")
    return ap


def _subharmonic_modes(s: str) -> list:
    """Parse --subharmonic-modes; "" -> [] (none), honoured literally downstream."""
    return [m.strip() for m in s.split(",") if m.strip()]


# (arg dest, config key, cast) for the GRAPE tunables, applied whenever passed.
_GRAPE_TUNABLES = (("grape_nctrl", "grape_nctrl", int),
                   ("grape_backend", "grape_backend", None),
                   ("grape_alg", "grape_alg", None),
                   ("grape_nbasis", "grape_nbasis", int),
                   ("grape_crab_restarts", "grape_crab_restarts", int),
                   ("grape_crab_seed", "grape_crab_seed", int),
                   ("grape_crab_score", "grape_crab_score", None),
                   ("grape_crab_method", "grape_crab_method", None),
                   ("grape_cutoff_GHz", "grape_cutoff_GHz", float),
                   ("grape_maxiter", "grape_maxiter", int))


def _apply_grape_tunables(config: Dict[str, Any], args: argparse.Namespace) -> None:
    for dest, key, cast in _GRAPE_TUNABLES:
        val = getattr(args, dest)
        if val is not None:
            config[key] = val if cast is None else cast(val)


def _apply_subharmonic_flags(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.drag_subharmonic:
        config["drag_subharmonic"] = True
    if args.no_drag_subharmonic:
        config["drag_subharmonic"] = False
    if args.subharmonic_modes is not None:
        config["subharmonic_modes"] = _subharmonic_modes(args.subharmonic_modes)


def _resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """DEFAULT_CONFIG + device JSON + CLI overrides (what `prepare` bakes into the grid)."""
    from snail_solver.paths import resolve_device
    config = dict(DEFAULT_CONFIG)
    if args.device:
        with open(resolve_device(args.device)) as f:      # bare name -> devices/
            config.update(json.load(f))
    # gate duration: --t-g-ns, or --target-eta -> auto_t_g (target_eta wins)
    if args.t_g_ns is not None:
        config["t_g_ns"] = float(args.t_g_ns)
    if args.target_eta is not None:
        from snail_solver.device_utils import auto_t_g
        config["t_g_ns"] = float(auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                                          float(config["lam_b"]), float(args.target_eta)))
        _env = config.get("envelope", "raised_cosine")
        print(f"target_eta={args.target_eta} -> t_g = {config['t_g_ns']:.3f} ns "
              f"(auto_t_g, raised-cosine peak |eta|)")
        if _env != "raised_cosine":
            print(f"  warning: auto_t_g assumes a raised-cosine (Hann) pump; envelope is "
                  f"'{_env}', so the actual peak |eta| will differ (constant pulse: "
                  f"eta = area/t_g, not the Hann 2*area/t_g).")
    # a stored operating point supplies (amp_scale, wp_offset) -- and t_g unless given
    if getattr(args, "operating_point", None):
        from snail_solver.operating_points import resolve as _resolve_op
        _explicit_tg = (args.t_g_ns is not None) or (args.target_eta is not None)
        config, _pt = _resolve_op(config, args.operating_point,
                                  t_g=float(config["t_g_ns"]),
                                  strict=bool(getattr(args, "operating_point_strict", False)),
                                  set_t_g=not _explicit_tg)
        print(f"operating point '{args.operating_point}': amp_scale={config['amp_scale']}, "
              f"wp_offset={config['wp_offset_GHz']} GHz, t_g={config['t_g_ns']:.3f} ns"
              + ("  (t_g from CLI)" if _explicit_tg else ""))
    # AFTER the operating point, so an explicit --chirp-GHz wins; always report it,
    # since `--chirp-GHz ""` legitimately cancels a saved point's chirp.
    if getattr(args, "chirp_GHz", None) is not None:
        from snail_solver.device_utils import describe_chirp, parse_chirp_arg
        previous = list(config.get("chirp_coeffs_GHz") or [])
        config["chirp_coeffs_GHz"] = parse_chirp_arg(args.chirp_GHz)
        print(f"chirp (--chirp-GHz): {describe_chirp(config['chirp_coeffs_GHz'])}"
              f"  (Legendre in u = 2t/t_g - 1)")
        if previous and not config["chirp_coeffs_GHz"]:
            print(f"  NOTE: this OVERRIDES a configured chirp {previous} with no chirp")
    if args.no_integrate:
        config["integrate"] = False
    if args.stark:
        config["stark_drive"] = True
    if args.stark_match_pulse:
        config["stark_match_pulse"] = True
        if not config.get("stark_drive"):
            print("note: --stark-match-pulse also needs --stark to do anything "
                  "(it shapes the per-point Stark chevron).")
    if args.calibrate:
        config["calibrate_points"] = True
        if not config.get("integrate", True):
            print("note: --calibrate needs the integrated run; it has no effect with "
                  "--no-integrate (analytic map only).")
    if args.calibrate_iters is not None:
        config["calibrate_iters"] = int(args.calibrate_iters)
    if args.drag:
        config["drag_always"] = True
    _apply_subharmonic_flags(config, args)
    if args.stark_span_MHz is not None:
        config["stark_span_MHz"] = float(args.stark_span_MHz)
    if args.stark_points is not None:
        config["stark_points"] = int(args.stark_points)
    if args.drag_compare:
        config["drag_compare"] = True
    if args.drag_compare_below_MHz is not None:
        config["drag_compare_below_MHz"] = float(args.drag_compare_below_MHz)
    if args.grape:
        config["grape"] = True
        if not config.get("integrate", True):
            print("note: --grape needs the integrated run; it has no effect with "
                  "--no-integrate (nothing to optimize against).")
    _apply_grape_tunables(config, args)
    if args.grape_warmstart_drag:
        config["grape_warmstart_drag"] = True
        if not args.grape:
            print("note: --grape-warmstart-drag has no effect without --grape.")
    if args.no_spectator:
        config["no_spectator"] = True
    return config


def _print_integrate(config: Dict[str, Any]) -> None:
    print(f"integrate = {config['integrate']}  "
          f"({'FULL sim per point' if config['integrate'] else 'analytic map only'})")


def _prepare_target(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    nominal = config["qubit_freqs_GHz"]
    wa_GHz = float(nominal[0])                                   # fixed
    wb_default = [round(float(nominal[1]) - 0.30 + 0.02 * k, 3) for k in range(31)]
    wb_list = _parse_list(args.wb_GHz, float) or wb_default
    # default spectator band: from below the lower qubit to ABOVE the highest w_b
    wb_lo = min(float(w) for w in wb_list)
    wb_hi = max(float(w) for w in wb_list)
    band_lo = min(wa_GHz, wb_lo) - 0.40
    band_hi = max(wa_GHz, wb_hi) + 0.40
    spec_lo = band_lo if args.spec_min_GHz is None else float(args.spec_min_GHz)
    spec_hi = band_hi if args.spec_max_GHz is None else float(args.spec_max_GHz)
    spec_step = 0.10 if args.spec_step_GHz is None else float(args.spec_step_GHz)
    if spec_step <= 0:
        sys.exit("--spec-step-GHz must be > 0")
    if spec_hi < spec_lo:
        sys.exit("--spec-max-GHz must be >= --spec-min-GHz")
    n_spec = int(round((spec_hi - spec_lo) / spec_step)) + 1
    spec_default = [round(spec_lo + spec_step * k, 6) for k in range(max(n_spec, 1))]
    spec_list = _parse_list(args.spec_GHz, float) or spec_default
    if config.get("no_spectator"):
        # bare gate: one decoupled dummy spectator (lam_spec=0 at run time), far out
        # of band so it never collides; only w_b sweeps.
        spec_list = [round(max(band_hi, 5.7) + 1.5, 3)]
        print(f"  no-spectator: bare a-b-coupler gate, sweeping w_b only "
              f"({len(wb_list)} pts); 2*w_p scans the SNAIL subharmonic w_c=2w_p")
    if config.get("drag_compare"):
        if args.drags:
            print("note: --drag-compare runs DRAG on/off per point; ignoring --drags.")
        drags = [False]
    else:
        drags = _bool_list(args.drags) or DEFAULT_TARGET_DRAGS
    points = build_target_grid(wa_GHz, wb_list, spec_list, drags,
                               float(config.get("min_detuning_GHz", 0.05)))
    path = write_grid(args.outdir, config, points)
    m = len(points)
    print(f"Wrote {m} allocation points -> {path}")
    print(f"  fixed: w_a={wa_GHz:.4f} GHz, w_snail={config['coupler_freq_GHz']:.4f} GHz; "
          f"spectator lam={config['lam_b']} (3-level anharmonic)")
    print(f"  scan: w_b ({len(wb_list)}) x w_spec ({len(spec_list)}) x drag ({len(drags)}) "
          f"[dropped |detuning|<{config.get('min_detuning_GHz', 0.05)} GHz]")
    _above = sum(1 for s in spec_list if s > wb_hi)
    print(f"  spectator band: {min(spec_list):.3f}..{max(spec_list):.3f} GHz "
          f"({_above} point(s) above the highest w_b = {wb_hi:.3f} GHz)")
    if config.get("drag_compare"):
        print(f"  DRAG comparison ON for |nearest beat| < "
              f"{config.get('drag_compare_below_MHz', 100.0):.0f} MHz")
    _print_integrate(config)
    _print_submit_hint(args.outdir, m)


def _prepare_spectator(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    # Detuning axis: BEAT delta = spec_freq - w_p (0 = on the collision), converted to
    # spec_freq = w_p + delta; else explicit --specfreqs or the default broad axis.
    wa_g, wb_g = (float(x) for x in config["qubit_freqs_GHz"])
    w_p = abs(wb_g - wa_g) + float(config.get("wp_offset_GHz", 0.0))
    if args.beat_span_MHz is not None:
        half = float(args.beat_span_MHz) / 2000.0            # MHz full-width -> GHz half
        beats = np.linspace(-half, half, int(args.beat_points))
        specfreqs = [round(w_p + float(b), 6) for b in beats]
        print(f"beat-centered sweep: collision at spec_freq = w_p = {w_p:.4f} GHz; "
              f"delta in +/-{args.beat_span_MHz/2:.0f} MHz, {args.beat_points} points")
    elif args.beats:
        beats = _parse_list(args.beats, float)
        specfreqs = [round(w_p + float(b), 6) for b in beats]
        print(f"beat-centered sweep: collision at spec_freq = w_p = {w_p:.4f} GHz")
    else:
        specfreqs = _parse_list(args.specfreqs, float) or DEFAULT_SPECFREQS_GHz
    drags = _bool_list(args.drags) or DEFAULT_DRAGS
    # Physical band: a spectator qubit sits BETWEEN the computational qubits.
    band_lo, band_hi = min(wa_g, wb_g), max(wa_g, wb_g)
    wspec = [round(wb_g - float(sf), 6) for sf in specfreqs]
    oob = [(sf, ws) for sf, ws in zip(specfreqs, wspec)
           if not (band_lo <= ws <= band_hi)]
    if oob:
        print(f"WARNING: {len(oob)}/{len(specfreqs)} spectator points are OUTSIDE the "
              f"physical band [{band_lo:.3f}, {band_hi:.3f}] GHz: absolute w_spec in "
              f"[{min(w for _, w in oob):.3f}, {max(w for _, w in oob):.3f}] GHz "
              f"(w_spec = w_b - Delta). Visualize with plot_allocation.py.")
        if args.clip_band:
            kept = [sf for sf, ws in zip(specfreqs, wspec) if band_lo <= ws <= band_hi]
            print(f"         --clip-band: keeping {len(kept)} in-band points, "
                  f"dropping {len(specfreqs) - len(kept)}.")
            specfreqs = kept
        else:
            print("         (pass --clip-band to drop them, or --allow it by ignoring "
                  "this warning if you are deliberately probing out-of-band.)")
    points = build_grid(specfreqs, drags)
    path = write_grid(args.outdir, config, points)
    m = len(points)
    print(f"Wrote {m} points -> {path}")
    _print_integrate(config)
    _print_submit_hint(args.outdir, m)


def _reapply_runtime_flags(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Re-apply run-time flags onto the grid's config (load_grid replaced ours).

    store_true flags only force ON, so a value baked in at prepare survives an absent
    flag; --no-drag-subharmonic is the one that can force OFF (subharmonic DRAG is the
    default, so a grid bakes in True). GRAPE tunables override whenever passed.
    """
    if args.stark:
        config["stark_drive"] = True
    if args.stark_match_pulse:
        config["stark_match_pulse"] = True
    _apply_subharmonic_flags(config, args)
    if config.get("stark_match_pulse") and not config.get("stark_drive"):
        print("note: stark_match_pulse has no effect without --stark (no per-point chevron).")

    if args.grape:
        config["grape"] = True
    if args.grape_warmstart_drag:
        config["grape_warmstart_drag"] = True
    _apply_grape_tunables(config, args)
    if (config.get("grape") and str(config.get("grape_alg", "")).upper() == "CRAB"
            and str(config.get("grape_crab_score", "qutip")) == "qutip"):
        print("note: CRAB with crab_score=qutip runs MANY exact propagations per point "
              "(gradient-free); use --grape-crab-score reduced to explore cheaply.")
    if config.get("grape") and not config.get("integrate", True):
        print("note: grape needs the integrated run; it has no effect with integrate=false.")
    if config.get("grape_warmstart_drag") and not config.get("grape"):
        print("note: grape_warmstart_drag has no effect without --grape.")

    # Chevron parallelism depends on the mode: use the task's cores in mode=point,
    # stay serial in mode=local (points are already pooled; no nested pools).
    if args.mode == "point":
        config["stark_jobs"] = int(args.stark_jobs if args.stark_jobs is not None
                                   else os.environ.get("SLURM_CPUS_PER_TASK", 1))
    elif args.mode == "local":
        if args.stark_jobs and int(args.stark_jobs) > 1:
            print("note: mode=local pools points; forcing stark_jobs=1 (no nested pools).")
        config["stark_jobs"] = 1


def _run_logger(args: argparse.Namespace):
    """Tailable log in <outdir>/logs/ (SLURM's own .out files are awkward to find)."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if args.mode == "point":
        tag = os.environ.get("SLURM_ARRAY_TASK_ID", "task")
        log_name = f"point_{tag}" + (f"_{job_id}" if job_id else "")
    else:
        log_name = "local" + (f"_{job_id}" if job_id else "")
    log_path = os.path.join(args.outdir, "logs", f"{log_name}.log")
    logger = setup_run_logger(log_path, f"run_sweep_zhou:{log_path}")
    logger.info(f"progress log: {log_path}")
    return logger


def _run_point_mode(args, config, points, logger) -> None:
    if args.index is None:
        sys.exit("mode=point requires --index")
    chunk = max(1, int(args.chunk))
    if args.resume_list:
        # --index is the CHUNK position into the resume list, so the array size is
        # ceil(len(list)/chunk) however scattered the missing indices are.
        with open(args.resume_list) as f:
            order = [int(x) for x in f.read().split()]
        lo, hi = args.index * chunk, min(args.index * chunk + chunk, len(order))
        if lo >= len(order):
            logger.info(f"[resume {args.index}] no entries in [{lo}, {hi}) of {len(order)}; "
                        f"nothing to do.")
            return
        todo = order[lo:hi]
    else:
        # array task K runs points [K*chunk, (K+1)*chunk)
        lo, hi = args.index * chunk, min(args.index * chunk + chunk, len(points))
        if lo >= len(points):
            logger.info(f"[chunk {args.index}] no points in [{lo}, {hi}) (N={len(points)}); "
                        f"nothing to do.")
            return
        todo = list(range(lo, hi))
    logger.info(f"starting {len(todo)} point(s): {todo}")
    for i in todo:
        if not (0 <= i < len(points)):
            logger.info(f"[point {i}] out of range 0..{len(points)-1}; skipping.")
            continue
        res = run_point(points[i], config)
        path = save_point(res, args.outdir)
        logger.info(f"[point {i}] {_log_line(res)} ({res['wall_s']:.1f}s) -> {path}")


def _run_local_mode(args, config, points, logger) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    os.makedirs(os.path.join(args.outdir, "points"), exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    logger.info(f"starting {len(points)} point(s) over {args.nproc} worker(s)")
    with ProcessPoolExecutor(max_workers=args.nproc, mp_context=ctx) as ex:
        futs = [ex.submit(run_point, pt, config) for pt in points]
        for done, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            save_point(res, args.outdir)
            logger.info(f"[{done}/{len(points)}] idx={res['index']:>4} {_log_line(res)}")
    logger.info(f"Local sweep done: {len(points)} points in {time.time()-t0:.1f}s")
    collect(args.outdir)


def main() -> None:
    """CLI: ``prepare`` (write grid.json), ``point`` (run one --index / chunk),
    ``local`` (process pool over the grid, then collect), ``collect``, ``chevrons``,
    ``missing``. See ``--help``."""
    args = _build_parser().parse_args()

    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)

    from snail_solver.paths import in_results
    args.outdir = in_results(args.outdir)                 # bare name -> results/
    config = _resolve_config(args)

    if args.mode == "prepare":
        if args.sweep == "target":
            _prepare_target(args, config)
        else:
            _prepare_spectator(args, config)
        return

    if args.mode == "collect":
        collect(args.outdir)
        return

    if args.mode == "missing":
        find_missing(args.outdir)
        return

    if args.mode == "chevrons":
        idxs = [args.index] if args.index is not None else None
        paths = plot_chevrons(args.outdir, idxs)
        dest = os.path.join(args.outdir, "figs", "chevrons")
        if paths:
            print(f"Rendered {len(paths)} chevron figure(s) -> {dest}")
        else:
            print(f"No saved chevrons found in {args.outdir}/points (run with --stark).")
        return

    grid_path = args.grid or os.path.join(args.outdir, "grid.json")
    if not os.path.exists(grid_path):
        sys.exit(f"grid.json not found at {grid_path}; run `prepare` first.")
    config, points = load_grid(args.outdir if not args.grid
                               else os.path.dirname(args.grid) or ".")
    _reapply_runtime_flags(config, args)

    if args.mode in ("point", "local"):
        logger = _run_logger(args)
    if args.mode == "point":
        _run_point_mode(args, config, points, logger)
    elif args.mode == "local":
        _run_local_mode(args, config, points, logger)


if __name__ == "__main__":
    main()
