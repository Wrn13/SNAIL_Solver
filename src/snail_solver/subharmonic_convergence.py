#!/usr/bin/env python3
r"""
subharmonic_convergence.py
==========================

How far must the gate be detuned from the SNAIL subharmonic before a simulation
truncated at a FIXED ``coupler_levels`` can be trusted?

The map this builds is ``coupler_levels`` (y) against the subharmonic detuning
``Delta_sub = w_s - 2 w_p`` (x), coloured by the gate fidelity, with the cells
that agree with the largest truncation outlined. Read a row and you have the
answer to the experimental question: *at N coupler levels, how far do I have to
put the gate from the subharmonic before the number I compute is the number the
device would give?*

The physics being measured
--------------------------
``H = g3 X^3`` with ``X`` carrying both the coupler and the pump, so the cube
necessarily contains

    3 g3 eta(t)^2 s^dag e^{i (w_s - 2 w_p) t}

-- a LINEAR drive on the SNAIL at the subharmonic detuning. It has no
participation suppression, while the gate itself is ``6 g3 lam_a lam_b eta``,
down by ``lam^2 ~ 0.01``; only ``Delta_sub`` holds it off. Its forced response is
a coherent displacement

    |alpha| ~ 3 g3 eta^2 / Delta_sub ,

and a coherent state has weight on every Fock level, so as ``Delta_sub`` shrinks
the answer becomes truncation-controlled: adding coupler levels changes the
fidelity instead of confirming it. That is measured, not conjectural -- see
``docs/chirped-recursive-drag.md`` section 4, where the 5-vs-9-level spread
collapses 0.232 -> 0.0002 as ``w_s`` is walked away from ``2 w_p``. This module
turns that one-off table into a calibrated map with an explicit convergence
boundary, and checks it against the analytic ``|alpha|^2 + 4|alpha| + 1`` level
count a displaced state needs.

Why the PUMP moves, not the SNAIL
---------------------------------
``Delta_sub`` can be walked either by tuning the SNAIL (``w_s``) or by moving the
gate (``w_p = |w_b - w_a|``). This module moves the gate, which is the question
as asked -- "how far should the gate be detuned" -- and it is also the cleaner
axis: the iSWAP rate is ``6 g3 lam_a lam_b eta``, INDEPENDENT of ``w_p``, so at
fixed ``target_eta`` the nominal length ``t_g0 = 2A/eta*`` is identical in every
column. Nothing about the gate's speed or drive changes along the axis; only the
spurious channel's detuning does. Walking ``w_s`` instead would move the gate's
own Stark shift, the qubit-coupler participations' detunings and the subharmonic
together.

The partner qubit is placed at ``w_b = w_a + (w_s - Delta_sub)/2`` (``--branch
below`` mirrors it under ``w_a``), so the requested detuning is exact.

Other collisions cross the axis
-------------------------------
Moving ``w_p`` moves the gate past every OTHER pump-activated resonance too, and
a fidelity dip at one of those is not a truncation failure. :func:`collision_landmarks`
solves each resonance condition -- all of them affine in ``w_p`` once ``w_b`` is
tied to it -- and returns the ``Delta_sub`` at which it lands, so the figure draws
them as labelled rules and the CLI warns when a requested column sits on one. For
``4Gate4.5SNAIL`` (``w_a = 3.5``, ``w_s = 4.5``) the axis is clean between 0 and
1.0 GHz (``2 w_p = w_a``), then again to 2.5 (``a <-> s`` conversion, where
``w_b`` also lands on ``w_s``) and 3.5 (``b <-> s``).

Why every column is re-calibrated
---------------------------------
Each column re-fits the carrier offset (a shaped chevron) and then the length,
at fixed peak ``|eta|`` -- ``tune_up`` steps 3 and 4. The Stark shift moves with
the detuning, so scoring a whole axis against one calibration measures the
calibration going stale rather than the truncation: exactly the artifact called
out in the leakage analysis ("the falling F in the last two rows is unrelated and
benign ... those points need a re-tune, not a fix"). The Rabi/chirp steps (1 and
2) are NOT re-run; the carrier here is flat unless ``--chirp`` is given, which
keeps a column to ``wp_points`` parallel solves plus one length scan instead of a
full tune-up. Use ``tune_up_sweep`` when the chirp itself is the object of study.

Calibration happens once per column at ``--calib-levels`` (default: the largest
truncation in the grid), because the experiment calibrates against hardware, not
against a truncated model -- so a cell's error is truncation error alone.
``--recalibrate-per-cell`` answers the other reading of the question ("what would
a simulation run entirely at N levels have told me?") at one calibration per cell.

Usage
-----
Inspect the grid, the collisions it crosses and the cost, with no solves::

    python -m snail_solver.subharmonic_convergence --device 4Gate4.5SNAIL.json \
        --target-eta 1.2 --dry-run

Run it::

    python -m snail_solver.subharmonic_convergence --device 4Gate4.5SNAIL.json \
        --target-eta 1.2 --detunings 0.1:2.2:8:log --levels 3,5,7,9,11 \
        --jobs 16 --outdir results/subharm_4Gate4.5SNAIL \
        --plot figs/subharm_4Gate4.5SNAIL/convergence_map.png

Re-plot from a finished run, with no solves at all::

    python -m snail_solver.subharmonic_convergence --replot \
        results/subharm_4Gate4.5SNAIL/convergence.json --plot figs/final.png

Per-column calibrations and per-cell scores are cached under ``--outdir``, so an
interrupted run resumes and only the missing cells are solved.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Coupler truncations scored by default. The largest is the reference every
#: smaller one is measured against, so the grid is only as good as its top row.
DEFAULT_LEVELS: Tuple[int, ...] = (3, 5, 7, 9, 11)

#: Default |F(N) - F(N_ref)| below which a truncation is called converged. The
#: leakage analysis quoted 8e-4 as "5 levels agrees with 9"; 2e-3 is one step
#: looser so a converged cell is not decided by solver tolerance.
DEFAULT_TOL = 2.0e-3

#: Calibrated ``P(|01> -> |10>)`` below which a column's operating point is
#: called unhealthy. Its cells then measure a badly-calibrated pulse rather than
#: a truncation, so the spread down that column is not evidence either way. This
#: is the same ambiguity ``tune_up_sweep``'s calibration-health panel exists to
#: remove: "the gate got worse" vs "the calibration stopped being measurable".
DEFAULT_HEALTH_MIN = 0.9

_C_INK = "#52514e"
_C_COUPLER = "#c0392b"          # a landmark that excites the SNAIL
_C_QUBIT = "#7f7f7f"            # a landmark that does not
_C_PRED = "#2a78d6"             # the analytic prediction
_C_MEAS = "#1baf7a"             # a measured trace


def _plain(o: Any) -> Any:
    """json default: ndarrays and numpy scalars to plain Python."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ===========================================================================
# The axis: detuning <-> partner frequency
# ===========================================================================
def _freqs(config: Dict[str, Any]) -> Tuple[float, float, float]:
    """``(w_a, w_b, w_s)`` in GHz, with ``w_a`` the anchor this module holds fixed."""
    wa, wb = (float(v) for v in config["qubit_freqs_GHz"])
    return wa, wb, float(config["coupler_freq_GHz"])


def subharmonic_detuning_GHz(config: Dict[str, Any]) -> float:
    """``Delta_sub = w_s - 2 w_p`` for a config as it stands.

    Positive means the SNAIL sits ABOVE the pump's second harmonic. Zero is the
    resonance this whole module is measuring the distance from.
    """
    wa, wb, ws = _freqs(config)
    return ws - 2.0 * abs(wb - wa)


def wb_for_detuning(config: Dict[str, Any], delta_sub_GHz: float,
                    branch: str = "above") -> float:
    """Partner frequency that puts the gate at ``delta_sub_GHz`` from the subharmonic.

    ``w_p = (w_s - Delta_sub)/2`` and ``w_b = w_a +/- w_p``. ``branch="above"``
    (default) places the partner above the anchor, ``"below"`` under it -- the pump
    is ``|w_b - w_a|`` either way, so both branches give the same ``Delta_sub`` and
    a different frequency allocation.

    Raises
    ------
    ValueError
        If the implied ``w_p`` is not positive (``Delta_sub >= w_s``: the two
        qubits would be degenerate or inverted) or the implied ``w_b`` is not.
    """
    wa, _wb, ws = _freqs(config)
    w_p = 0.5 * (ws - float(delta_sub_GHz))
    if w_p <= 0.0:
        raise ValueError(
            f"Delta_sub={delta_sub_GHz:g} GHz needs w_p={w_p:g} GHz: at "
            f"Delta_sub >= w_s = {ws:g} the pump would be zero or negative, i.e. the "
            f"two qubits degenerate. The axis is bounded by the device's own w_s.")
    sign = {"above": +1.0, "below": -1.0}.get(str(branch))
    if sign is None:
        raise ValueError(f"branch={branch!r}: expected 'above' or 'below'")
    wb = wa + sign * w_p
    if wb <= 0.0:
        raise ValueError(
            f"Delta_sub={delta_sub_GHz:g} GHz on the 'below' branch puts "
            f"w_b={wb:g} GHz at or under zero; use branch='above'")
    return float(wb)


def config_at_detuning(config: Dict[str, Any], delta_sub_GHz: float, *,
                       levels: Optional[int] = None, branch: str = "above",
                       chirp_coeffs_GHz: Sequence[float] = (),
                       min_detuning_GHz: Optional[float] = None) -> Dict[str, Any]:
    """A COPY of `config` whose gate sits ``delta_sub_GHz`` from the subharmonic.

    Only ``qubit_freqs_GHz[1]`` (and, with `levels`, ``coupler_levels``) changes.

    ``chirp_coeffs_GHz`` is written into the copy as well as returned to the
    caller, because :func:`device_utils.build_coupler` falls back to
    ``config["chirp_coeffs_GHz"]`` when its own argument is None. Leaving a
    device-level chirp in a config whose ``w_b`` has moved would apply a chirp
    calibrated at a different pump -- the same trap ``tune_up_sweep.score_gate``
    documents for ``None`` vs ``[]``. The default ``()`` writes an explicit None.

    Raises
    ------
    ValueError
        If the implied pump is smaller than ``min_detuning_GHz`` (the device's own
        ``min_detuning_GHz`` by default): the gate would be inside its own
        linewidth of a direct qubit-qubit collision, so the point is not a gate.
    """
    wa, _wb, _ws = _freqs(config)
    wb = wb_for_detuning(config, delta_sub_GHz, branch)
    floor = (float(config.get("min_detuning_GHz", 0.05))
             if min_detuning_GHz is None else float(min_detuning_GHz))
    if abs(wb - wa) < floor:
        raise ValueError(
            f"Delta_sub={delta_sub_GHz:g} GHz implies w_p={abs(wb - wa):g} GHz, "
            f"under the {floor:g} GHz floor: the qubits are then within a linewidth "
            f"of each other and the 'gate' is a direct collision, not a pumped one.")
    out = dict(config)
    out["qubit_freqs_GHz"] = [wa, wb]
    if levels is not None:
        out["coupler_levels"] = int(levels)
    chirp = [float(c) for c in (chirp_coeffs_GHz or ())]
    out["chirp_coeffs_GHz"] = chirp or None
    return out


def delta_tag(delta_sub_GHz: float) -> str:
    """``0.35 -> 'd0p35'``, ``-0.2 -> 'dm0p2'`` -- a filename fragment, no dots."""
    return "d" + f"{float(delta_sub_GHz):g}".replace(".", "p").replace("-", "m")


def cell_tag(delta_sub_GHz: float, levels: int) -> str:
    """``(0.35, 7) -> 'd0p35_N7'``."""
    return f"{delta_tag(delta_sub_GHz)}_N{int(levels)}"


# ===========================================================================
# Grid parsing
# ===========================================================================
def parse_detunings(spec: str) -> List[float]:
    """Delta_sub grid in GHz: a comma list of scalars and/or ``lo:hi:n[:log]`` ranges.

    The log form is the useful default: the displacement is ``|alpha| ~ 1/Delta_sub``,
    so the interesting structure is decades, not equal steps. Log spacing needs both
    ends strictly positive (or both negative -- the axis is mirrored).

    Segments COMPOSE, which is how an axis gets extended without re-solving what is
    already cached::

        "0.15:0.8:8:log,1.15,1.35,1.6"

    Because a range segment reproduces the same floats every time, the eight log
    points above hit their existing cache files bit-for-bit while the three scalars
    are solved -- one document, one figure, no repeated work. Sorted on return, so
    a composed grid is still monotone on the axis.
    """
    text = str(spec).strip()
    if not text:
        raise ValueError(f"--detunings {spec!r}: no values")
    if "," in text:
        out: List[float] = []
        for piece in text.split(","):
            if piece.strip():
                out.extend(parse_detunings(piece))
        if not out:
            raise ValueError(f"--detunings {spec!r}: no values")
        return sorted(set(out), key=lambda v: (v < 0, abs(v)))
    if ":" not in text:
        return [float(text)]
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"--detunings {spec!r}: expected lo:hi:n[:log]")
    lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
    if n < 1:
        raise ValueError(f"--detunings {spec!r}: points must be >= 1")
    kind = (parts[3].lower() if len(parts) == 4 else "lin")
    if kind in ("lin", "linear"):
        return [float(v) for v in np.linspace(lo, hi, n)]
    if kind != "log":
        raise ValueError(f"--detunings {spec!r}: spacing must be 'lin' or 'log'")
    if lo == 0.0 or hi == 0.0 or (lo > 0.0) != (hi > 0.0):
        raise ValueError(
            f"--detunings {spec!r}: log spacing needs both ends non-zero and of the "
            f"same sign (Delta_sub = 0 IS the resonance -- approach it, don't sample it)")
    sign = 1.0 if lo > 0 else -1.0
    return [float(sign * v) for v in np.geomspace(abs(lo), abs(hi), n)]


def parse_levels(spec: str) -> List[int]:
    """``"3,5,7,9,11"`` or ``"3:11:5"`` -> sorted unique coupler truncations."""
    text = str(spec).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"--levels {spec!r}: expected lo:hi:n or a comma list")
        vals = [int(round(v)) for v in np.linspace(float(parts[0]), float(parts[1]),
                                                   int(parts[2]))]
    else:
        vals = [int(v) for v in text.split(",") if v.strip()]
    out = sorted(set(vals))
    if not out:
        raise ValueError(f"--levels {spec!r}: no values")
    if out[0] < 2:
        raise ValueError(f"--levels {spec!r}: a coupler needs at least 2 levels")
    if len(out) < 2:
        raise ValueError(
            f"--levels {spec!r}: convergence is a COMPARISON -- one truncation has "
            f"nothing to be measured against. Give at least two.")
    return out


# ===========================================================================
# Analytic expectations
# ===========================================================================
def displacement_alpha(config: Dict[str, Any], delta_sub_GHz: float,
                       peak_eta: float) -> float:
    """``|alpha| = 3 g3 eta^2 / |Delta_sub|`` -- the coherent displacement the
    subharmonic drive forces on the coupler.

    The ``3 g3 eta^2 s^dag`` term is the ``X^3`` cross-term with two pump letters
    and one coupler letter (multiplicity 3), and a linear drive detuned by
    ``Delta_sub`` displaces its mode by ``Omega/Delta``. Reproduces the 0.743 quoted
    in ``docs/chirped-recursive-drag.md`` at ``g3 = 0.06``, ``eta = 1.12``,
    ``Delta_sub = 0.304``. Infinite at exact resonance.
    """
    d = abs(float(delta_sub_GHz))
    g3 = float(config["g3_GHz"])
    if d == 0.0:
        return float("inf")
    return 3.0 * g3 * float(peak_eta) ** 2 / d


def predicted_levels(alpha: float, margin: float = 4.0) -> float:
    """Coupler levels a displacement of ``|alpha|`` needs: ``|alpha|^2 + m|alpha| + 1``.

    A coherent state's occupation is Poisson with mean ``|alpha|^2`` and standard
    deviation ``|alpha|``, so ``mean + m sigma`` with ``m ~ 4`` covers the ladder it
    actually populates, and the ``+1`` keeps a 1-level answer at zero drive. This is
    the curve the measured convergence boundary is tested against; it is a scaling
    argument, not a bound.
    """
    a = abs(float(alpha))
    if not np.isfinite(a):
        return float("inf")
    return a * a + float(margin) * a + 1.0


#: Processes whose resonance condition is affine in ``w_p``. Each entry gives the
#: net mode frequency the pump must supply as ``c0 + c1 * w_p`` (see
#: :func:`collision_landmarks`), a label, and whether the SNAIL is involved --
#: which is what separates "this dip is another coupler-exciting channel" from
#: "this dip is a qubit collision".
_PROCESSES = (
    ("SNAIL excitation", "SNAIL excitation", ("s",), True),
    ("qubit a excitation", r"qubit $a$ excitation", ("a",), False),
    ("qubit b excitation", r"qubit $b$ excitation", ("b",), False),
    ("a->s conversion", r"$a\!\to\!s$ conversion", ("s", "-a"), True),
    ("b->s conversion", r"$b\!\to\!s$ conversion", ("s", "-b"), True),
    ("a->b exchange", r"$a\!\to\!b$ exchange", ("b", "-a"), False),
    ("a+s pair creation", r"$a{+}s$ pair creation", ("a", "s"), True),
    ("b+s pair creation", r"$b{+}s$ pair creation", ("b", "s"), True),
    ("a+b pair creation", r"$a{+}b$ pair creation", ("a", "b"), False),
)


def collision_landmarks(config: Dict[str, Any], *, branch: str = "above",
                        n_pump_max: int = 2) -> List[Dict[str, Any]]:
    """Every ``Delta_sub`` at which some OTHER process becomes resonant.

    Along this axis ``w_b = w_a +/- w_p``, so each mode frequency is affine in the
    pump: ``w_a = (w_a, 0)``, ``w_b = (w_a, +/-1)``, ``w_s = (w_s, 0)`` as
    ``(c0, c1)`` with ``c0 + c1 w_p``. A process needing net frequency ``c0 + c1 w_p``
    from ``n`` pump quanta is resonant when ``+/- n w_p = c0 + c1 w_p``, i.e. at
    ``w_p = c0 / (+/- n - c1)`` -- one division, no search. The ``Delta_sub`` of that
    pump is ``w_s - 2 w_p``.

    ``2 w_p = w_s`` (SNAIL excitation at ``n = 2``) comes back at exactly
    ``Delta_sub = 0``, which is the axis origin and the consistency check on this
    whole construction.

    Returns
    -------
    list of dict
        ``delta_sub_GHz``, ``w_p_GHz``, ``w_b_GHz``, ``n_pump``, ``name`` (plain,
        for a terminal), ``label`` (mathtext, for the figure) and ``coupler``
        (bool), sorted by ``delta_sub_GHz``. Deduplicated on
        ``Delta_sub``, keeping the lowest pump order (the strongest process) and
        preferring a coupler-exciting label, since a coincidence there is the
        physically relevant one.
    """
    wa, _wb, ws = _freqs(config)
    sign = {"above": +1.0, "below": -1.0}[str(branch)]
    basis = {"a": (wa, 0.0), "b": (wa, sign), "s": (ws, 0.0)}

    found: Dict[int, Dict[str, Any]] = {}
    for name, label, letters, is_coupler in _PROCESSES:
        c0 = c1 = 0.0
        for letter in letters:
            neg = letter.startswith("-")
            k0, k1 = basis[letter[-1]]
            c0 += (-k0 if neg else k0)
            c1 += (-k1 if neg else k1)
        for n in range(1, int(n_pump_max) + 1):
            for s in (+1.0, -1.0):
                denom = s * n - c1
                if abs(denom) < 1e-12:              # the condition is w_p-independent
                    continue
                w_p = c0 / denom
                if w_p <= 1e-9:
                    continue
                delta = ws - 2.0 * w_p
                key = int(round(delta * 1e6))
                rec = {"delta_sub_GHz": float(delta), "w_p_GHz": float(w_p),
                       "w_b_GHz": float(wa + sign * w_p), "n_pump": int(n),
                       "name": name, "label": label,
                       "coupler": bool(is_coupler)}
                prev = found.get(key)
                if (prev is None or rec["n_pump"] < prev["n_pump"]
                        or (rec["n_pump"] == prev["n_pump"] and rec["coupler"]
                            and not prev["coupler"])):
                    found[key] = rec
    return sorted(found.values(), key=lambda r: r["delta_sub_GHz"])


def nearest_landmark(landmarks: Sequence[Dict[str, Any]], delta_sub_GHz: float,
                     *, skip_origin: bool = True) -> Optional[Dict[str, Any]]:
    """The landmark closest to `delta_sub_GHz`, with its signed distance.

    The subharmonic itself (``Delta_sub = 0``) is skipped by default: it is the
    axis, not a contaminant.
    """
    best = None
    for lm in landmarks:
        if skip_origin and abs(lm["delta_sub_GHz"]) < 1e-9:
            continue
        d = float(delta_sub_GHz) - float(lm["delta_sub_GHz"])
        if best is None or abs(d) < abs(best["distance_GHz"]):
            best = dict(lm, distance_GHz=float(d))
    return best


# ===========================================================================
# One column: re-fit the offset, then the length
# ===========================================================================
def calibrate_column(config: Dict[str, Any], delta_sub_GHz: float,
                     target_eta: float, *, levels: int, branch: str = "above",
                     chirp_coeffs_GHz: Sequence[float] = (),
                     wp_points: int = 41, wp_span_MHz: Optional[float] = None,
                     span_linewidths: float = 4.0, n_time: int = 161,
                     tg_points: int = 13, tg_lo: float = 0.7, tg_hi: float = 1.3,
                     passes: int = 1, solver: Optional[Dict[str, Any]] = None,
                     jobs: int = 1,
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Re-fit ``wp_offset`` (shaped chevron) then ``t_g`` (length scan) at this detuning.

    ``tune_up`` steps 3 and 4, at fixed peak ``|eta|``. Steps 1-2 (the Rabi table
    and the chirp built from it) are deliberately skipped: the carrier is flat
    unless `chirp_coeffs_GHz` is given, which is what makes a column affordable.
    Both steps see the SAME pulse the cells are later scored with, which is the
    only reason the fidelity in a cell is attributable to truncation.

    ``passes`` > 1 re-runs the pair. With the carrier flat and ``|eta|`` fixed the
    Stark shift is length-independent, so one pass IS the fixed point (the same
    argument ``tune_up`` uses to run its outer loop once with DRAG off); a second
    pass only re-centres the chevron window on the fitted length.

    Returns
    -------
    dict
        A ``tune_up``-shaped operating point -- ``target_eta``, ``t_g_ns``,
        ``amp_scale``, ``wp_offset_GHz``, ``chirp_coeffs_GHz``, ``spec_abs_GHz``,
        ``drag_beat_GHz`` -- so :func:`tune_up_sweep.score_gate` consumes it
        verbatim, plus ``w_b_GHz``, ``w_p_GHz``, ``peak_eta``, ``transfer``,
        ``railed``, ``offset_railed``, ``nfev`` and the window actually used.
    """
    from snail_solver import find_stark_resonance as FSR
    from snail_solver import tune_up as TU

    log = logger or logging.getLogger("subharmonic_convergence")
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    chirp = [float(c) for c in (chirp_coeffs_GHz or ())]
    cfg = config_at_detuning(config, delta_sub_GHz, levels=levels, branch=branch,
                             chirp_coeffs_GHz=chirp)
    wa, wb, _ws = _freqs(cfg)

    t_g0 = TU.nominal_t_g(cfg, target_eta)
    t_g, wp_offset = t_g0, 0.0
    nfev, offset_railed, transfer, railed = 0, False, float("nan"), False
    span_used = float("nan")

    for p in range(max(1, int(passes))):
        # Same linewidth sizing as tune_up step 3: a chevron is as wide as the
        # exchange rate, so a span fixed in MHz is wrong at some drive.
        span = (float(wp_span_MHz) if wp_span_MHz is not None
                else 2.0 * float(span_linewidths) * 1e3 / (2.0 * float(t_g)))
        span_used = float(span)
        offs = (np.linspace(-span / 2e3, span / 2e3, int(wp_points))
                + float(wp_offset))
        chev = FSR.scan(cfg, float(t_g),
                        TU.fixed_eta_amp_scale(cfg, float(t_g), target_eta),
                        offs, 1.05 * float(t_g), int(n_time), solver, n_jobs=jobs,
                        shape="raised_cosine", chirp_coeffs_GHz=chirp)
        wp_offset = float(chev["resonance_offset_GHz"])
        nfev += int(offs.size)
        step = float(offs[1] - offs[0]) if offs.size > 1 else 0.0
        offset_railed = bool(min(abs(wp_offset - offs[0]),
                                 abs(wp_offset - offs[-1])) <= step + 1e-12)

        L = TU.length_rabi(cfg, target_eta,
                           t_g0 * np.linspace(float(tg_lo), float(tg_hi),
                                              int(tg_points)),
                           wp_offset_GHz=wp_offset, chirp_coeffs_GHz=chirp,
                           solver=solver, logger=None)
        t_g, transfer = float(L["t_g_ns"]), float(L["transfer"])
        railed, nfev = bool(L["railed"]), nfev + int(L["nfev"])
        log.info(f"  Delta_sub={delta_sub_GHz:+.3f} GHz pass {p + 1}: "
                 f"wp_offset={wp_offset * 1e3:+.3f} MHz (span +/-{span / 2:.2f} MHz)"
                 f"{' RAILED' if offset_railed else ''}, t_g={t_g:.3f} ns, "
                 f"transfer={transfer:.5f}{' RAILED' if railed else ''}")

    amp = float(TU.fixed_eta_amp_scale(cfg, t_g, target_eta))
    return {"target_eta": float(target_eta), "t_g_ns": float(t_g),
            "amp_scale": amp, "wp_offset_GHz": float(wp_offset),
            "chirp_coeffs_GHz": chirp, "spec_abs_GHz": None,
            "drag_beat_GHz": None, "drag_n_pump": 1,
            "delta_sub_GHz": float(delta_sub_GHz), "branch": str(branch),
            "w_b_GHz": float(wb), "w_p_GHz": float(abs(wb - wa)),
            "peak_eta": float(TU.peak_eta_of(cfg, t_g, amp)),
            "t_g0_ns": float(t_g0), "transfer": transfer,
            "railed": railed, "offset_railed": offset_railed,
            "calib_levels": int(levels), "passes": int(max(1, passes)),
            "wp_span_MHz": span_used, "nfev": int(nfev)}


# ===========================================================================
# One cell: score the gate at a given truncation
# ===========================================================================
def coupler_occupation(config: Dict[str, Any], record: Dict[str, Any],
                       solver: Optional[Dict[str, Any]] = None) -> float:
    """``<n_s>`` at ``t_g`` on the ``|100> -> |010>`` trajectory.

    The direct physical readout of the displacement the subharmonic forces: it is
    what ``|alpha|^2`` predicts, and what the truncation has to hold. One extra
    solve per cell, against the four propagator columns the fidelity already costs.
    """
    from snail_solver.device_utils import build_coupler
    from snail_solver.spectroscopy import expected_number

    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    cpl, _w_p, _eta = build_coupler(
        config, float(record["t_g_ns"]), float(record["amp_scale"]),
        float(record["wp_offset_GHz"]), None, None,
        chirp_coeffs_GHz=[float(c) for c in (record.get("chirp_coeffs_GHz") or ())])
    psi = cpl.evolve_state([1, 0, 0], float(record["t_g_ns"]), **solver)
    return float(expected_number(np.abs(np.asarray(psi)) ** 2, cpl.dims, 2))


def score_cell(config: Dict[str, Any], delta_sub_GHz: float, levels: int,
               record: Dict[str, Any], *, branch: str = "above",
               occupation: bool = True, fit_virtual_z: bool = True,
               solver: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score the calibrated gate at ONE coupler truncation.

    Delegates to :func:`tune_up_sweep.score_gate` (``F_avg``, ``leakage``,
    ``transfer``, all with the amplitude re-derived from the fixed-|eta| algebra)
    and adds ``n_coupler``. The chirp is passed explicitly from the record, never
    left to the config fallback.
    """
    from snail_solver.tune_up_sweep import score_gate

    chirp = [float(c) for c in (record.get("chirp_coeffs_GHz") or ())]
    cfg = config_at_detuning(config, delta_sub_GHz, levels=levels, branch=branch,
                             chirp_coeffs_GHz=chirp)
    out = score_gate(cfg, record, chirp, solver=solver,
                     fit_virtual_z=fit_virtual_z)
    out["levels"] = int(levels)
    out["delta_sub_GHz"] = float(delta_sub_GHz)
    out["fit_virtual_z"] = bool(fit_virtual_z)
    out["branch"] = str(branch)
    out["n_coupler"] = (coupler_occupation(cfg, record, solver=solver)
                        if occupation else float("nan"))
    return out


# ===========================================================================
# Workers (module level, so they pickle)
# ===========================================================================
def _calib_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One column's calibration, exceptions captured as data."""
    try:
        t0 = time.perf_counter()
        rec = calibrate_column(payload["config"], payload["delta_sub_GHz"],
                               payload["target_eta"], **payload["kw"])
        rec["seconds"] = time.perf_counter() - t0
        return {"ok": True, "record": rec}
    except Exception as exc:                       # a bad column is a RESULT
        return {"ok": False, "error": {"type": type(exc).__name__,
                                       "stage": "calibrate",
                                       "message": str(exc)}}


def _cell_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One cell's score, exceptions captured as data."""
    try:
        t0 = time.perf_counter()
        cell = score_cell(payload["config"], payload["delta_sub_GHz"],
                          payload["levels"], payload["record"], **payload["kw"])
        cell["seconds"] = time.perf_counter() - t0
        return {"ok": True, "cell": cell}
    except Exception as exc:
        return {"ok": False, "levels": payload["levels"],
                "delta_sub_GHz": payload["delta_sub_GHz"],
                "error": {"type": type(exc).__name__, "stage": "score",
                          "message": str(exc)}}


def _same(a: Any, b: Any) -> bool:
    """Equality that survives a JSON round trip (ints as floats, tuples as lists)."""
    if isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, bool) or isinstance(b, bool):
        # Python calls True == 1.0, which would let a flag match a level count.
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if b is None or isinstance(b, str):
        return a == b
    if isinstance(b, (int, float)):
        try:
            return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return a == b


def _cache_load(path: Optional[str], expect: Dict[str, Any],
                logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """A cached record, but ONLY if it was produced under `expect`.

    The cache files are keyed by ``(Delta_sub, levels)`` alone, so a re-run into
    the same ``outdir`` at a different drive, branch or chirp would otherwise be
    handed the previous run's physics -- silently, with the stale numbers going
    into the map and the figure as if they had just been solved. Every key in
    `expect` must be present in the cached record and match it; anything else
    re-solves and overwrites.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            got = json.load(fh)
    except (OSError, ValueError):                # a truncated file is not a cache
        return None
    for key, want in expect.items():
        if key not in got or not _same(got[key], want):
            if logger:
                logger.info(f"  {os.path.basename(path)}: cached at "
                            f"{key}={got.get(key)!r}, this run wants {want!r} "
                            f"-- re-solving")
            return None
    return got


def _run_pool(fn, payloads: Sequence[Dict[str, Any]], workers: int) -> List[Any]:
    """`fn` over `payloads`, in a process pool when it buys anything.

    Order-preserving. Sequential at ``workers <= 1`` -- which is also the GPU path,
    where one big solve already uses the device and a pool would fight over it.
    """
    if int(workers) <= 1 or len(payloads) <= 1:
        return [fn(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        return list(ex.map(fn, payloads))


# ===========================================================================
# The map
# ===========================================================================
def run_convergence_map(config: Dict[str, Any], detunings: Sequence[float],
                        levels: Sequence[int], target_eta: float, *,
                        device_path: Optional[str] = None,
                        branch: str = "above",
                        ref_levels: Optional[int] = None,
                        calib_levels: Optional[int] = None,
                        recalibrate_per_cell: bool = False,
                        tol: float = DEFAULT_TOL,
                        chirp_coeffs_GHz: Sequence[float] = (),
                        wp_points: int = 41, wp_span_MHz: Optional[float] = None,
                        span_linewidths: float = 4.0, n_time: int = 161,
                        tg_points: int = 13, tg_lo: float = 0.7, tg_hi: float = 1.3,
                        passes: int = 1, occupation: bool = True,
                        fit_virtual_z: bool = True,
                        solver: Optional[Dict[str, Any]] = None,
                        jobs: int = 0, outdir: Optional[str] = None,
                        overwrite: bool = False, stop_on_error: bool = False,
                        logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Calibrate at every detuning, score at every truncation, measure the spread.

    Two phases, because they parallelise differently. Calibration is serial WITHIN
    a column (the length scan is a 1-D search) but independent BETWEEN columns, so
    columns go in the pool and each chevron runs single-threaded. Scoring is one
    independent solve per cell, so the cells go in the pool. A single column falls
    back to putting the whole pool inside its chevron.

    Every calibration and every cell is cached under ``outdir`` and re-read unless
    `overwrite`, so an interrupted run resumes -- a file is written only on success,
    as in ``run_sweep_zhou``.

    Returns
    -------
    dict
        The map document: ``device``, ``settings``, ``landmarks``, ``columns`` (one
        per detuning, each with its calibration and its ``cells``), ``boundary``
        (per truncation, the smallest ``|Delta_sub|`` from which it stays converged)
        and ``summary``.
    """
    from snail_solver import find_stark_resonance as FSR

    log = logger or logging.getLogger("subharmonic_convergence")
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    lv = sorted({int(v) for v in levels})
    ref = int(max(lv) if ref_levels is None else ref_levels)
    if ref not in lv:
        raise ValueError(f"ref_levels={ref} is not in the levels grid {lv}: every "
                         f"cell is measured against it, so it has to be scored")
    calib = int(max(lv) if calib_levels is None else calib_levels)
    deltas = [float(d) for d in detunings]
    chirp = [float(c) for c in (chirp_coeffs_GHz or ())]
    n_jobs = FSR._resolve_jobs(jobs)
    t0_all = time.perf_counter()

    landmarks = collision_landmarks(config, branch=branch)
    calib_dir = os.path.join(outdir, "calib") if outdir else None
    cell_dir = os.path.join(outdir, "cells") if outdir else None
    for d in (outdir, calib_dir, cell_dir):
        if d:
            os.makedirs(d, exist_ok=True)

    calib_kw = dict(branch=branch, chirp_coeffs_GHz=chirp, wp_points=wp_points,
                    wp_span_MHz=wp_span_MHz, span_linewidths=span_linewidths,
                    n_time=n_time, tg_points=tg_points, tg_lo=tg_lo, tg_hi=tg_hi,
                    passes=passes, solver=solver)
    cell_kw = dict(branch=branch, occupation=occupation,
                   fit_virtual_z=fit_virtual_z, solver=solver)

    # ---- columns, with their analytic expectations -----------------------
    columns: List[Dict[str, Any]] = []
    for d in deltas:
        col: Dict[str, Any] = {"delta_sub_GHz": d, "tag": delta_tag(d),
                               "landmark": nearest_landmark(landmarks, d),
                               "cells": [], "ok": False, "error": None}
        try:
            cfg = config_at_detuning(config, d, levels=ref, branch=branch,
                                     chirp_coeffs_GHz=chirp)
            wa, wb, _ws = _freqs(cfg)
            alpha = displacement_alpha(cfg, d, target_eta)
            col.update({"w_b_GHz": wb, "w_p_GHz": abs(wb - wa),
                        "alpha_pred": alpha,
                        "levels_pred": predicted_levels(alpha)})
        except ValueError as exc:
            col["error"] = {"type": "ValueError", "stage": "grid",
                            "message": str(exc)}
            log.warning(f"  Delta_sub={d:+.3f} GHz skipped: {exc}")
            if stop_on_error:
                raise
        columns.append(col)

    live = [c for c in columns if c["error"] is None]

    # ---- phase A: one calibration per column (or per cell) ---------------
    def _calib_payload(delta: float, lvls: int) -> Dict[str, Any]:
        return {"config": config, "delta_sub_GHz": delta,
                "target_eta": float(target_eta),
                "kw": dict(calib_kw, levels=int(lvls),
                           jobs=(n_jobs if len(live) <= 1 else 1))}

    #: What a cached CALIBRATION must have been produced under. These are the
    #: physics of a column; the grid resolutions (wp_points, n_time, tg_*) are
    #: recorded in the file for provenance but deliberately NOT gated on, so a
    #: cosmetic change of sampling does not throw away a day of solves.
    def _calib_expect(delta: float, lvls: int) -> Dict[str, Any]:
        return {"delta_sub_GHz": float(delta), "target_eta": float(target_eta),
                "branch": str(branch), "calib_levels": int(lvls),
                "chirp_coeffs_GHz": chirp, "passes": int(max(1, passes))}

    def _cell_expect(delta: float, lvls: int,
                     rec: Dict[str, Any]) -> Dict[str, Any]:
        # The operating point ties a cell to the calibration it was scored at, so
        # a re-calibrated column re-scores rather than reusing stale cells.
        return {"delta_sub_GHz": float(delta), "levels": int(lvls),
                "branch": str(branch), "fit_virtual_z": bool(fit_virtual_z),
                "t_g_ns": float(rec["t_g_ns"]),
                "amp_scale": float(rec["amp_scale"]),
                "wp_offset_GHz": float(rec["wp_offset_GHz"]),
                "chirp_coeffs_GHz": chirp}

    def _cached(path: Optional[str],
                expect: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None if overwrite else _cache_load(path, expect, log)

    if not recalibrate_per_cell:
        want, payloads = [], []
        for col in live:
            path = (os.path.join(calib_dir, f"{col['tag']}.json")
                    if calib_dir else None)
            got = _cached(path, _calib_expect(col["delta_sub_GHz"], calib))
            if got is not None:
                col["calibration"] = got
                col["ok"] = True
                log.info(f"  Delta_sub={col['delta_sub_GHz']:+.3f} GHz: "
                         f"calibration cached")
                continue
            want.append((col, path))
            payloads.append(_calib_payload(col["delta_sub_GHz"], calib))
        if payloads:
            log.info(f"phase A: calibrating {len(payloads)} column(s) at "
                     f"{calib} coupler levels "
                     f"({'pool of %d' % min(n_jobs, len(payloads)) if len(live) > 1 else 'serial, pool inside the chevron'})")
            got = _run_pool(_calib_worker, payloads,
                            min(n_jobs, len(payloads)) if len(live) > 1 else 1)
            for (col, path), res in zip(want, got):
                if not res["ok"]:
                    col["error"] = res["error"]
                    log.warning(f"  Delta_sub={col['delta_sub_GHz']:+.3f} GHz "
                                f"calibration FAILED: {res['error']['message']}")
                    if stop_on_error:
                        raise RuntimeError(res["error"]["message"])
                    continue
                col["calibration"] = res["record"]
                col["ok"] = True
                if path:
                    with open(path, "w") as fh:
                        json.dump(res["record"], fh, indent=2, default=_plain)
    else:
        # One calibration per CELL: the other reading of the question -- what a
        # simulation run entirely at N levels would have reported, calibration
        # error included.
        want, payloads = [], []
        for col in live:
            col["calibration_per_cell"] = {}
            for N in lv:
                path = (os.path.join(calib_dir, f"{cell_tag(col['delta_sub_GHz'], N)}.json")
                        if calib_dir else None)
                got = _cached(path, _calib_expect(col["delta_sub_GHz"], N))
                if got is not None:
                    col["calibration_per_cell"][str(N)] = got
                    continue
                want.append((col, N, path))
                payloads.append(_calib_payload(col["delta_sub_GHz"], N))
        if payloads:
            log.info(f"phase A: calibrating {len(payloads)} cell(s), one per "
                     f"(detuning, levels)")
            got = _run_pool(_calib_worker, payloads, min(n_jobs, len(payloads)))
            for (col, N, path), res in zip(want, got):
                if not res["ok"]:
                    log.warning(f"  Delta_sub={col['delta_sub_GHz']:+.3f} GHz "
                                f"N={N} calibration FAILED: "
                                f"{res['error']['message']}")
                    if stop_on_error:
                        raise RuntimeError(res["error"]["message"])
                    continue
                col["calibration_per_cell"][str(N)] = res["record"]
                if path:
                    with open(path, "w") as fh:
                        json.dump(res["record"], fh, indent=2, default=_plain)
        for col in live:
            recs = col["calibration_per_cell"]
            col["ok"] = str(ref) in recs
            if col["ok"]:
                col["calibration"] = recs[str(ref)]
            elif col["error"] is None:
                col["error"] = {"type": "RuntimeError", "stage": "calibrate",
                                "message": f"no calibration at the reference "
                                           f"{ref} levels"}

    # ---- phase B: one score per cell -------------------------------------
    want, payloads = [], []
    for col in [c for c in columns if c["ok"]]:
        for N in lv:
            rec = (col["calibration_per_cell"].get(str(N))
                   if recalibrate_per_cell else col["calibration"])
            if rec is None:
                continue
            path = (os.path.join(cell_dir, f"{cell_tag(col['delta_sub_GHz'], N)}.json")
                    if cell_dir else None)
            got = _cached(path, _cell_expect(col["delta_sub_GHz"], N, rec))
            if got is not None:
                col["cells"].append(got)
                continue
            want.append((col, path))
            payloads.append({"config": config, "delta_sub_GHz": col["delta_sub_GHz"],
                             "levels": int(N), "record": rec, "kw": cell_kw})
    if payloads:
        log.info(f"phase B: scoring {len(payloads)} cell(s) over "
                 f"{min(n_jobs, len(payloads))} worker(s)")
        got = _run_pool(_cell_worker, payloads, min(n_jobs, len(payloads)))
        for (col, path), res in zip(want, got):
            if not res["ok"]:
                col.setdefault("cell_errors", []).append(res)
                log.warning(f"  Delta_sub={col['delta_sub_GHz']:+.3f} GHz "
                            f"N={res['levels']} FAILED: {res['error']['message']}")
                if stop_on_error:
                    raise RuntimeError(res["error"]["message"])
                continue
            col["cells"].append(res["cell"])
            if path:
                with open(path, "w") as fh:
                    json.dump(res["cell"], fh, indent=2, default=_plain)

    # ---- the spread every truncation is judged on ------------------------
    for col in columns:
        col["cells"].sort(key=lambda c: c["levels"])
        by_N = {int(c["levels"]): c for c in col["cells"]}
        F_ref = by_N.get(ref, {}).get("F_avg")
        col["F_ref"] = (None if F_ref is None else float(F_ref))
        for c in col["cells"]:
            if F_ref is None:
                c["spread"], c["converged"] = None, None
            else:
                c["spread"] = float(abs(float(c["F_avg"]) - float(F_ref)))
                c["converged"] = bool(c["spread"] <= float(tol))

    doc = {
        "source": "subharmonic_convergence",
        "device": device_path,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "target_eta": float(target_eta), "branch": branch, "levels": lv,
            "ref_levels": ref, "calib_levels": calib, "tol": float(tol),
            "recalibrate_per_cell": bool(recalibrate_per_cell),
            "detunings_GHz": deltas, "chirp_coeffs_GHz": chirp,
            "wp_points": wp_points, "wp_span_MHz": wp_span_MHz,
            "span_linewidths": span_linewidths, "n_time": n_time,
            "tg_points": tg_points, "tg_lo": tg_lo, "tg_hi": tg_hi,
            "passes": passes, "occupation": bool(occupation),
            "fit_virtual_z": bool(fit_virtual_z), "jobs": n_jobs,
            "solver": solver, "qubit_levels": config.get("qubit_levels"),
            "device_delta_sub_GHz": float(subharmonic_detuning_GHz(config)),
            "w_a_GHz": _freqs(config)[0], "w_s_GHz": _freqs(config)[2],
            "g3_GHz": float(config["g3_GHz"]),
        },
        "landmarks": landmarks,
        "columns": columns,
    }
    doc["boundary"] = convergence_boundary(doc)
    n_ok = sum(1 for c in columns if c["ok"])
    doc["summary"] = {
        "n_columns": len(columns), "n_ok": n_ok,
        "n_failed": len(columns) - n_ok,
        "n_cells": sum(len(c["cells"]) for c in columns),
        "seconds": time.perf_counter() - t0_all,
    }
    return doc


def convergence_boundary(doc: Dict[str, Any],
                         tol: Optional[float] = None) -> Dict[str, Any]:
    """Per truncation, the smallest ``|Delta_sub|`` from which it stays converged.

    Defined from the FAR end inward: walk the columns in decreasing
    ``|Delta_sub|`` while every cell at that truncation agrees with the reference,
    and report the last one. A single converged cell sitting inside a
    non-converged run is not a boundary -- next to a collision the spread can dip
    through the tolerance by cancellation, and quoting that as "converged here"
    is exactly the truncation-as-regularizer trap this module exists to expose.

    Returns
    -------
    dict
        ``{str(levels): {"delta_sub_GHz": float or None, "n_converged": int,
        "n_scored": int}}``. ``None`` means the truncation never held anywhere on
        the sampled axis.
    """
    tol = float(doc["settings"]["tol"] if tol is None else tol)
    ref = int(doc["settings"]["ref_levels"])
    out: Dict[str, Any] = {}
    for N in [int(v) for v in doc["settings"]["levels"]]:
        pairs = []
        for col in doc["columns"]:
            for c in col["cells"]:
                if int(c["levels"]) == N and c.get("spread") is not None:
                    pairs.append((abs(float(col["delta_sub_GHz"])),
                                  float(c["spread"])))
        pairs.sort(key=lambda p: p[0])
        edge: Optional[float] = None
        n_conv = 0
        for absd, spread in reversed(pairs):        # from the far end inward
            if spread > tol:
                break
            edge, n_conv = absd, n_conv + 1
        out[str(N)] = {"delta_sub_GHz": edge, "n_converged": int(n_conv),
                       "n_scored": len(pairs),
                       "is_reference": bool(N == ref)}
    return out


# ===========================================================================
# The figure
# ===========================================================================
def plot_convergence_map(doc: Dict[str, Any],
                         out: str = "figs/convergence_map.png",
                         title: Optional[str] = None,
                         xscale: Optional[str] = None,
                         annotate: bool = True,
                         health_min: float = DEFAULT_HEALTH_MIN) -> str:
    r"""Coupler levels vs subharmonic detuning, coloured by fidelity.

    (a) The map. One cell per (``Delta_sub``, ``coupler_levels``), coloured by
        ``1 - F_avg`` on a log scale (the repo's fidelity-heatmap encoding, as in
        ``plot_results``/``plot_fidelity_map``: perceptually uniform, dark = good)
        and labelled with ``F_avg`` itself. Cells whose ``|F(N) - F(N_ref)|``
        exceeds the tolerance are hatched -- those are the truncations that are
        still lying -- and the staircase traces the convergence boundary (drawn
        only on a one-sided axis: the boundary is a distance ``|Delta_sub|``, so a
        grid straddling the resonance has one per side, and the hatching carries
        it instead). The
        reference row is marked, since it cannot disagree with itself. Dashed rules
        mark where OTHER processes become resonant along the axis (red when they
        excite the SNAIL), because a dip there is a collision, not a truncation
        failure. The blue curve is the analytic ``|alpha|^2 + 4|alpha| + 1`` level
        count. A red rug along the bottom edge marks a column whose
        CALIBRATION failed (``transfer < health_min``): its cells all score
        one bad operating point, so its spread says nothing about the
        truncation.

    (b) Coupler occupation, predicted against measured, on one axis in one unit
        (photons): ``|alpha|^2 = (3 g3 eta^2 / Delta_sub)^2`` and the ``<n_s>`` the
        reference truncation actually reaches at ``t_g``. Where these part company
        the displacement picture has stopped being the whole story.

    Raises
    ------
    ValueError
        If no column has a scored cell -- there is nothing to draw, and a blank
        figure would read as a converged one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    try:
        from snail_solver.plot_results import set_literature_style
        set_literature_style()
    except Exception:                                    # style is never fatal
        pass

    st = doc["settings"]
    lv = [int(v) for v in st["levels"]]
    ref = int(st["ref_levels"])
    tol = float(st["tol"])
    cols = [c for c in doc["columns"] if c.get("cells")]
    if not cols:
        raise ValueError("plot_convergence_map: no scored cells "
                         f"({len(doc['columns'])} columns, none with data)")
    cols.sort(key=lambda c: float(c["delta_sub_GHz"]))
    x = np.array([float(c["delta_sub_GHz"]) for c in cols], dtype=float)

    F = np.full((len(lv), len(cols)), np.nan)
    spread = np.full_like(F, np.nan)
    for j, col in enumerate(cols):
        for c in col["cells"]:
            i = lv.index(int(c["levels"]))
            F[i, j] = float(c["F_avg"])
            if c.get("spread") is not None:
                spread[i, j] = float(c["spread"])

    # x edges: midpoints, in whatever space the axis is drawn in, with the outer
    # edges mirrored so the first and last cells are as wide as their neighbours.
    if xscale is None:
        xscale = "log" if (np.all(x > 0) or np.all(x < 0)) and x.size > 1 else "linear"
    same_sign = bool(np.all(x > 0) or np.all(x < 0))
    use_log = (xscale == "log" and same_sign)
    sign = 1.0 if np.all(x >= 0) else -1.0

    def _edges(v: np.ndarray) -> np.ndarray:
        if v.size == 1:
            w = max(abs(v[0]) * 0.2, 0.05)
            return np.array([v[0] - w, v[0] + w])
        if use_log:
            u = np.log(np.abs(v))
            mid = 0.5 * (u[:-1] + u[1:])
            return sign * np.exp(np.concatenate(
                ([u[0] - (mid[0] - u[0])], mid, [u[-1] + (u[-1] - mid[-1])])))
        mid = 0.5 * (v[:-1] + v[1:])
        return np.concatenate(([v[0] - (mid[0] - v[0])], mid,
                               [v[-1] + (v[-1] - mid[-1])]))

    xe = _edges(x)
    ye = np.arange(len(lv) + 1, dtype=float) - 0.5

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(7.6, 6.6), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3.0, 1.0]})

    # -- (a) the map -------------------------------------------------------
    infid = np.clip(1.0 - F, 1e-5, 1.0)
    finite = infid[np.isfinite(infid)]
    vmin = max(float(np.nanmin(finite)) * 0.7, 1e-5) if finite.size else 1e-4
    vmax = min(max(float(np.nanmax(finite)) * 1.4, vmin * 10.0), 1.0)
    # a cell whose solve failed is grey, never a colour that reads as a fidelity
    cmap = plt.get_cmap("magma").with_extremes(bad="0.85")
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(infid), shading="flat",
                       cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=[ax, axn], shrink=0.62, pad=0.015, aspect=28,
                      location="right")
    cb.set_label(r"$1 - F_{\mathrm{avg}}$")

    if use_log:
        ax.set_xscale("log")
    ax.set_yticks(np.arange(len(lv)))
    ax.set_yticklabels([f"{N}" + ("  (ref)" if N == ref else "") for N in lv])
    ax.set_ylabel("coupler levels")
    ax.set_ylim(ye[0], ye[-1])
    ax.set_title(title or (
        r"Truncation convergence vs distance from the SNAIL subharmonic"
        f"  ($|\\eta^*|$ = {st['target_eta']:g}, tol = {tol:g})"))

    # cell labels + hatching for the ones that still disagree with the reference
    for i in range(len(lv)):
        for j in range(len(cols)):
            if not np.isfinite(F[i, j]):
                continue
            rgba = cmap(norm(infid[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            if annotate:
                ax.text(x[j], i, f"{F[i, j]:.3f}", ha="center", va="center",
                        fontsize=6.4, color=("white" if lum < 0.55 else "0.12"))
            bad = (np.isfinite(spread[i, j]) and spread[i, j] > tol)
            if bad:
                ax.add_patch(Rectangle(
                    (xe[j], ye[i]), xe[j + 1] - xe[j], 1.0, facecolor="none",
                    edgecolor=(0.55, 0.55, 0.55, 0.95), hatch="//////", lw=0.0,
                    zorder=2.5))

    # the boundary staircase: the left edge of the first converged cell per row
    halo = [pe.withStroke(linewidth=3.4, foreground="white")]
    verts: List[Tuple[float, float]] = []
    for i, N in enumerate(lv if same_sign else []):
        row = doc.get("boundary", {}).get(str(N), {})
        b = row.get("delta_sub_GHz")
        # The reference row cannot disagree with itself, so its "boundary" is the
        # whole axis and drawing it would imply a measurement that was never made.
        if b is None or row.get("is_reference"):
            continue
        j = int(np.argmin(np.abs(np.abs(x) - float(b))))
        left = xe[j] if sign > 0 else xe[j + 1]
        verts.append((left, ye[i]))
        verts.append((left, ye[i + 1]))
    if not same_sign:
        ax.annotate("axis straddles the resonance: the boundary is a distance "
                    "|$\\Delta_{\\rm sub}$|, one per side -- read the hatching",
                    (0.5, 1.0), xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=6.5, color=_C_INK)
    if len(verts) >= 2:
        vx = [v[0] for v in verts]
        vy = [v[1] for v in verts]
        ax.plot(vx, vy, "-", color="#111111", lw=1.8, zorder=4,
                path_effects=halo, solid_joinstyle="miter",
                label="convergence boundary")

    # where OTHER processes become resonant on this axis
    lo, hi = min(xe[0], xe[-1]), max(xe[0], xe[-1])
    for lm in doc.get("landmarks", []):
        d = float(lm["delta_sub_GHz"])
        if not (lo < d < hi) or abs(d) < 1e-9:
            continue
        colour = _C_COUPLER if lm.get("coupler") else _C_QUBIT
        for a in (ax, axn):
            a.axvline(d, ls="--", lw=1.0, color=colour, alpha=0.75, zorder=3)
        ax.annotate(f"{lm.get('label') or lm['name']} ({lm['n_pump']}p)",
                    (d, ye[-1]),
                    textcoords="offset points", xytext=(-3, -4), ha="right",
                    va="top", rotation=90, fontsize=6, color=colour,
                    # a chip, not a halo: the rule crosses cells at both ends of
                    # the colour ramp, and a white stroke reads as a smudge on the
                    # dark (good-fidelity) end.
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78,
                              pad=1.0))

    # the analytic level count, in the same categorical y coordinates
    eta = float(st["target_eta"])
    g3 = float(st["g3_GHz"])
    dd = (np.geomspace(abs(x[0]), abs(x[-1]), 200) * sign if use_log and x.size > 1
          else np.linspace(x[0], x[-1], 200))
    n_pred = np.array([predicted_levels(3.0 * g3 * eta ** 2 / max(abs(v), 1e-12))
                       for v in dd])
    y_pred = np.interp(n_pred, lv, np.arange(len(lv)),
                       left=np.nan, right=np.nan)
    ax.plot(dd, y_pred, "-", color=_C_PRED, lw=1.4, alpha=0.95, zorder=3.5,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
            label=r"$|\alpha|^2+4|\alpha|+1$")

    # A rug along the bottom edge for columns whose CALIBRATION failed. Without
    # it a scattered column reads as a truncation failure, when the cells are
    # really all scoring one bad operating point (a length scan that landed on
    # the wrong branch, say).
    sick = [j for j, c in enumerate(cols)
            if float((c.get("calibration") or {}).get("transfer", 1.0)) < health_min]
    for j in sick:
        ax.plot([xe[j], xe[j + 1]], [ye[0]] * 2, "-", color=_C_COUPLER, lw=3.2,
                solid_capstyle="butt", zorder=6, clip_on=False,
                label=(rf"calibrated transfer $< {health_min:g}$"
                       if j == sick[0] else None))

    # the device's own placement
    d_dev = float(st.get("device_delta_sub_GHz", np.nan))
    if np.isfinite(d_dev) and lo < d_dev < hi:
        ax.plot([d_dev], [ye[0]], "^", ms=8, color="#111111", clip_on=False,
                zorder=5, label="this device")

    map_handles, map_labels = ax.get_legend_handles_labels()
    map_handles = list(map_handles) + [
        Rectangle((0, 0), 1, 1, facecolor="none",
                  edgecolor=(0.45, 0.45, 0.45, 1.0), hatch="//////")]
    map_labels = list(map_labels) + [rf"$|F(N)-F({ref})| > {tol:g}$"]

    if annotate:
        for i, N in enumerate(lv):
            b = doc.get("boundary", {}).get(str(N), {}).get("delta_sub_GHz")
            txt = (r"$\geq$" + f"{b:.2f}" if b is not None else "never")
            ax.annotate(txt if N != ref else "ref", (1.005, i),
                        xycoords=("axes fraction", "data"), ha="left",
                        va="center", fontsize=6.4, color=_C_INK)

    # -- (b) coupler occupation, predicted vs measured ---------------------
    alpha2 = np.array([(3.0 * g3 * eta ** 2 / max(abs(v), 1e-12)) ** 2
                       for v in dd])
    axn.plot(dd, alpha2, "-", color=_C_PRED, lw=1.6,
             label=r"$|\alpha|^2 = (3g_3\eta^2/\Delta_{\rm sub})^2$")
    n_meas = np.array([
        next((float(c["n_coupler"]) for c in col["cells"]
              if int(c["levels"]) == ref and np.isfinite(float(c["n_coupler"]))),
             np.nan) for col in cols])
    if np.any(np.isfinite(n_meas)):
        axn.plot(x, n_meas, "-o", color=_C_MEAS, lw=1.4, ms=4.0,
                 label=rf"$\langle n_s\rangle$ at $t_g$, {ref} levels")
    axn.set_yscale("log")
    axn.set_ylabel("coupler photons")
    axn.set_xlabel(r"subharmonic detuning  $\Delta_{\rm sub} = \omega_s - 2\omega_p$"
                   "  (GHz)")
    axn.grid(True, which="both", alpha=0.22, lw=0.4)

    # ONE legend, outside both panels. In-axes it sat over the far columns --
    # the converged corner the reader goes to the figure for.
    nh, nl = axn.get_legend_handles_labels()
    fig.legend(map_handles + list(nh), map_labels + list(nl),
               loc="outside lower center", ncol=3, fontsize=7, frameon=False)

    # LAST, after every rule that can widen the shared x axis: a twiny copies the
    # limits it is given and does not track its parent, so building it earlier
    # slides every w_b label off its own column (the trap tune_up_sweep documents).
    axt = ax.twiny()
    axt.set_xscale(ax.get_xscale())
    axt.set_xlim(ax.get_xlim())
    axt.set_xticks(x)
    axt.set_xticklabels([f"{c.get('w_b_GHz', float('nan')):.2f}" for c in cols],
                        fontsize=6.6)
    axt.set_xlabel(r"partner frequency $\omega_b$ (GHz)", fontsize=8)
    axt.minorticks_off()

    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Reporting
# ===========================================================================
def describe_grid(config: Dict[str, Any], detunings: Sequence[float],
                  levels: Sequence[int], target_eta: float, *,
                  branch: str = "above", guard_GHz: float = 0.15,
                  wp_points: int = 41, tg_points: int = 13,
                  occupation: bool = True) -> str:
    """The grid, the collisions it crosses and what it will cost -- no solves.

    This is what ``--dry-run`` prints. It exists because the axis is not innocent:
    moving the pump walks the gate past other resonances, and a column sitting on
    one measures that collision rather than the truncation. It also checks that the
    chosen drive puts ``|alpha|`` in the range where truncation is even in question
    -- at small ``eta`` the displacement never approaches a photon and the map is
    flat by construction.
    """
    wa, wb0, ws = _freqs(config)
    lv = sorted({int(v) for v in levels})
    landmarks = collision_landmarks(config, branch=branch)
    lines = [
        f"device: w_a={wa:g}  w_b={wb0:g}  w_s={ws:g} GHz, "
        f"g3={float(config['g3_GHz']):g} GHz, "
        f"Delta_sub as it stands = {subharmonic_detuning_GHz(config):+.3f} GHz",
        f"axis: w_b = w_a {'+' if branch == 'above' else '-'} "
        f"(w_s - Delta_sub)/2,  target_eta = {target_eta:g},  levels = {lv}",
        "",
        f"  {'Delta_sub':>10} {'w_b':>7} {'w_p':>7} {'|alpha|':>8} "
        f"{'N_pred':>7}  nearest other resonance",
    ]
    n_ok = 0
    for d in detunings:
        try:
            cfg = config_at_detuning(config, d, branch=branch)
        except ValueError as exc:
            lines.append(f"  {d:>10.3f}    SKIPPED -- {exc}")
            continue
        n_ok += 1
        wa_, wb_, _ = _freqs(cfg)
        alpha = displacement_alpha(cfg, d, target_eta)
        lm = nearest_landmark(landmarks, d)
        note = "--"
        if lm is not None:
            flag = "  <== ON IT" if abs(lm["distance_GHz"]) < guard_GHz else ""
            note = (f"{lm['name']} ({lm['n_pump']}p) at "
                    f"{lm['delta_sub_GHz']:+.3f}, "
                    f"{abs(lm['distance_GHz']) * 1e3:.0f} MHz away{flag}")
        lines.append(f"  {d:>10.3f} {wb_:>7.3f} {abs(wb_ - wa_):>7.3f} "
                     f"{alpha:>8.3f} {predicted_levels(alpha):>7.1f}  {note}")

    alphas = []
    for d in detunings:
        try:
            alphas.append(displacement_alpha(
                config_at_detuning(config, d, branch=branch), d, target_eta))
        except ValueError:
            pass
    if alphas:
        lines += ["", f"|alpha| spans {min(alphas):.3f} to {max(alphas):.3f} "
                      f"over this grid."]
        if max(alphas) < 0.5:
            lines.append(
                "  WARNING: |alpha| stays well under 1 everywhere, so the coupler is "
                "never displaced far enough for the truncation to matter and the map "
                "will be flat by construction. Raise --target-eta or push the grid "
                "closer to the subharmonic; the interesting region is |alpha| ~ 1, "
                "i.e. Delta_sub ~ 3 g3 eta^2.")
        if min(alphas) > 3.0:
            lines.append(
                "  WARNING: |alpha| exceeds 3 everywhere, so |alpha|^2 > 9 photons is "
                "needed at every column and no truncation in the grid will converge. "
                "Extend the grid further from the subharmonic.")

    per_cell = 5 if occupation else 4        # 4 propagator columns (+1 for <n_s>)
    per_col = int(wp_points) + int(tg_points) + 19
    lines += ["", f"cost: {n_ok} column(s) x ~{per_col} calibration solves "
                  f"+ {n_ok * len(lv)} cell(s) x ~{per_cell} solves "
                  f"= ~{n_ok * per_col + n_ok * len(lv) * per_cell} exact solves.",
              "  Calibration parallelises over columns; scoring parallelises over "
              "cells. Both are cached, so a re-run only fills gaps.",
              "",
              f"other resonances on this axis (branch={branch}):"]
    for lm in landmarks:
        lines.append(f"  Delta_sub={lm['delta_sub_GHz']:+8.3f} GHz  "
                     f"w_p={lm['w_p_GHz']:6.3f}  w_b={lm['w_b_GHz']:6.3f}  "
                     f"{lm['n_pump']}-pump {lm['name']}"
                     f"{'  [excites the SNAIL]' if lm['coupler'] else ''}")
    return "\n".join(lines)


def print_map(doc: Dict[str, Any],
              health_min: float = DEFAULT_HEALTH_MIN) -> None:
    """:func:`format_map` to stdout."""
    print(format_map(doc, health_min=health_min))


def format_map(doc: Dict[str, Any],
               health_min: float = DEFAULT_HEALTH_MIN) -> str:
    """The map as a table, plus the boundary per truncation.

    Returned rather than printed so the CLI can put the same text in the log
    file: a run whose tables live only in a terminal scrollback has to be
    re-derived from the JSON to be read again.
    """
    st = doc["settings"]
    lv = [int(v) for v in st["levels"]]
    ref = int(st["ref_levels"])
    cols = sorted(doc["columns"], key=lambda c: float(c["delta_sub_GHz"]))

    out = [f"\n=== fidelity vs coupler truncation (ref = {ref} levels, "
           f"tol = {st['tol']:g}) ===",
           (f"  {'Delta_sub':>10} {'w_b':>7} {'|alpha|':>8} {'t_g/ns':>8} "
            f"{'transfer':>9} "
            + " ".join(f"{('F(' + str(N) + ')'):>9}" for N in lv)
            + f" {'max spread':>11}")]
    for col in cols:
        if not col.get("ok") or not col.get("cells"):
            msg = (col.get("error") or {}).get("message", "no cells")
            out.append(f"  {col['delta_sub_GHz']:>10.3f}    FAILED -- {msg[:70]}")
            continue
        by_N = {int(c["levels"]): c for c in col["cells"]}
        cal = col["calibration"]
        row = (f"  {col['delta_sub_GHz']:>10.3f} {col.get('w_b_GHz', np.nan):>7.3f} "
               f"{col.get('alpha_pred', np.nan):>8.3f} "
               f"{cal['t_g_ns']:>8.2f} "
               f"{float(cal.get('transfer', np.nan)):>8.4f}"
               f"{'!' if float(cal.get('transfer', 1.0)) < health_min else ' '} ")
        spreads = []
        for N in lv:
            c = by_N.get(N)
            if c is None:
                row += f"{'--':>9} "
                continue
            mark = "*" if (c.get("converged") is False) else " "
            row += f"{c['F_avg']:>8.5f}{mark} "
            if c.get("spread") is not None and N != ref:
                spreads.append(float(c["spread"]))
        row += f"{(max(spreads) if spreads else float('nan')):>11.2e}"
        out.append(row)
    out.append("  (* = disagrees with the reference truncation by more than "
               "the tolerance)")
    sick = [c for c in cols if c.get("ok") and c.get("cells")
            and float(c["calibration"].get("transfer", 1.0)) < health_min]
    if sick:
        out.append(f"  (! = calibrated transfer below {health_min:g}: this "
                   f"column's cells score a badly-calibrated pulse, so its "
                   f"spread is not evidence about the truncation. "
                   f"Delta_sub = "
                   + ", ".join(f"{c['delta_sub_GHz']:.3f}" for c in sick)
                   + " GHz -- check `railed` in its calibration and widen "
                     "--tg-lo/--tg-hi.)")

    out.append("\n=== convergence boundary ===")
    for N in lv:
        b = doc["boundary"][str(N)]
        if b.get("is_reference"):
            out.append(f"  {N:>3} levels: reference")
        elif b["delta_sub_GHz"] is None:
            out.append(f"  {N:>3} levels: NEVER converged on this axis "
                       f"(0/{b['n_scored']} columns)")
        else:
            out.append(f"  {N:>3} levels: converged for |Delta_sub| >= "
                       f"{b['delta_sub_GHz']:.3f} GHz "
                       f"({b['n_converged']}/{b['n_scored']} columns)")
    sm = doc["summary"]
    out.append(f"\n  {sm['n_ok']}/{sm['n_columns']} columns, {sm['n_cells']} "
               f"cells, {sm['seconds'] / 60:.1f} min")
    return "\n".join(out)


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.subharmonic_convergence",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="device JSON (bare name resolves under devices/); "
                         "required unless --replot")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="peak |eta| held fixed in every column. REQUIRED: the "
                         "whole effect scales as eta^2, so there is no defensible "
                         "default -- see --dry-run, which reports the |alpha| range "
                         "your choice implies")
    ap.add_argument("--detunings", default="0.1:2.2:8:log",
                    help="Delta_sub grid in GHz: lo:hi:n[:log] or a comma list "
                         "[0.1:2.2:8:log]")
    ap.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS),
                    help="coupler truncations to score, comma list or lo:hi:n "
                         f"[{','.join(str(v) for v in DEFAULT_LEVELS)}]")
    ap.add_argument("--ref-levels", type=int, default=None,
                    help="truncation every other one is measured against "
                         "[the largest in --levels]")
    ap.add_argument("--calib-levels", type=int, default=None,
                    help="truncation the per-column calibration runs at [the "
                         "largest in --levels]; the experiment calibrates against "
                         "hardware, not against a truncated model")
    ap.add_argument("--recalibrate-per-cell", action="store_true",
                    help="give every (detuning, levels) cell its own calibration: "
                         "what a simulation run entirely at N levels would have "
                         "reported, calibration error included")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"|F(N) - F(N_ref)| below which N is converged "
                         f"[{DEFAULT_TOL:g}]")
    ap.add_argument("--branch", choices=("above", "below"), default="above",
                    help="place the partner qubit above or below the anchor w_a "
                         "[above]")
    ap.add_argument("--chirp", default=None,
                    help="pump chirp (Legendre coefficients, GHz) to run in every "
                         "column; default is a FLAT carrier -- this module re-fits "
                         "the offset and the length, not the chirp")
    ap.add_argument("--wp-points", type=int, default=41,
                    help="offset points in the per-column chevron [41]")
    ap.add_argument("--wp-span-MHz", type=float, default=None,
                    help="fixed chevron span; default sizes it from the exchange "
                         "linewidth at each column's length")
    ap.add_argument("--span-linewidths", type=float, default=4.0)
    ap.add_argument("--n-time", type=int, default=161)
    ap.add_argument("--tg-points", type=int, default=13)
    ap.add_argument("--tg-lo", type=float, default=0.7)
    ap.add_argument("--tg-hi", type=float, default=1.3)
    ap.add_argument("--passes", type=int, default=1,
                    help="offset+length re-fits per column; one IS the fixed point "
                         "with a flat carrier at fixed |eta| [1]")
    ap.add_argument("--no-occupation", action="store_true",
                    help="skip the <n_s> readout (one solve per cell)")
    ap.add_argument("--no-fit-virtual-z", action="store_true",
                    help="do not fit out virtual-Z phases when scoring")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (0 -> SLURM_CPUS_PER_TASK or CPU count)")
    ap.add_argument("--gpu", action="store_true",
                    help="run via qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--outdir", default=None,
                    help="per-column and per-cell caches land here "
                         "[results/subharm_<device>]")
    ap.add_argument("--log", default=None,
                    help="progress log, colocated with the outputs by default "
                         "[<outdir>/run.log]; the grid and result tables are "
                         "written into it as well as to stdout")
    ap.add_argument("--out", default=None,
                    help="the map JSON [<outdir>/convergence.json]")
    ap.add_argument("--plot", nargs="?", const="figs/convergence_map.png",
                    default=None, help="the figure")
    ap.add_argument("--xscale", choices=("log", "linear"), default=None,
                    help="x axis of the figure [log when every column shares a sign]")
    ap.add_argument("--no-annotate", action="store_true",
                    help="no per-cell fidelity labels")
    ap.add_argument("--health-min", type=float, default=DEFAULT_HEALTH_MIN,
                    help="calibrated transfer below which a column is flagged as "
                         "scoring a badly-calibrated pulse, in the table and as a "
                         f"rug on the figure [{DEFAULT_HEALTH_MIN:g}]")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore the caches and re-solve")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid, the resonances it crosses and the cost, "
                         "then exit without solving")
    ap.add_argument("--replot", metavar="JSON", default=None,
                    help="regenerate --plot from a previous --out -- no solves")
    args = ap.parse_args()

    # --- the zero-solve path, before anything heavy is imported -----------
    if args.replot:
        with open(args.replot) as fh:
            doc = json.load(fh)
        print_map(doc, health_min=args.health_min)
        path = plot_convergence_map(
            doc, out=args.plot or "figs/convergence_map.png", xscale=args.xscale,
            annotate=not args.no_annotate, health_min=args.health_min)
        print(f"wrote {path}")
        return

    if not args.device:
        ap.error("--device is required unless --replot is given")
    if args.target_eta is None:
        ap.error("--target-eta is required: the subharmonic drive scales as eta^2, "
                 "so the map is meaningless without the drive it was measured at. "
                 "Try --dry-run to see what a choice implies.")

    from snail_solver.device_utils import load_device, parse_chirp_arg
    from snail_solver.paths import in_results, resolve_device

    device_path = resolve_device(args.device)
    config = load_device(device_path)
    deltas = parse_detunings(args.detunings)
    lv = parse_levels(args.levels)
    chirp = parse_chirp_arg(args.chirp) or []

    if args.dry_run:
        print(describe_grid(config, deltas, lv, args.target_eta,
                            branch=args.branch, wp_points=args.wp_points,
                            tg_points=args.tg_points,
                            occupation=not args.no_occupation))
        return

    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from snail_solver.log_utils import setup_run_logger

    stem = os.path.splitext(os.path.basename(args.device))[0]
    outdir = args.outdir or in_results(f"subharm_{stem}")
    os.makedirs(outdir, exist_ok=True)
    out_json = args.out or os.path.join(outdir, "convergence.json")
    log_path = args.log or os.path.join(outdir, "run.log")
    # Name the logger after the file, so two concurrent runs writing to different
    # logs do not share handlers and duplicate onto each other (log_utils).
    logger = setup_run_logger(log_path, f"subharmonic_convergence:{log_path}")

    grid = describe_grid(config, deltas, lv, args.target_eta, branch=args.branch,
                         wp_points=args.wp_points, tg_points=args.tg_points,
                         occupation=not args.no_occupation)
    logger.info(f"device={args.device}  target_eta={args.target_eta:g}\n{grid}")
    print(f"\noutdir={outdir}  log={log_path}")

    doc = run_convergence_map(
        config, deltas, lv, args.target_eta, device_path=device_path,
        branch=args.branch, ref_levels=args.ref_levels,
        calib_levels=args.calib_levels,
        recalibrate_per_cell=args.recalibrate_per_cell, tol=args.tol,
        chirp_coeffs_GHz=chirp, wp_points=args.wp_points,
        wp_span_MHz=args.wp_span_MHz, span_linewidths=args.span_linewidths,
        n_time=args.n_time, tg_points=args.tg_points, tg_lo=args.tg_lo,
        tg_hi=args.tg_hi, passes=args.passes,
        occupation=not args.no_occupation,
        fit_virtual_z=not args.no_fit_virtual_z,
        solver={"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps},
        jobs=args.jobs, outdir=outdir, overwrite=args.overwrite,
        stop_on_error=args.stop_on_error, logger=logger)

    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=2, default=_plain)
    print(f"\n  written {out_json}")
    report = format_map(doc, health_min=args.health_min)
    print(report)
    logger.info(report)

    if args.plot:
        # A map whose JSON is on disk has not failed, even if every column did:
        # the failures ARE the result. Do not turn "nothing to draw" into a
        # non-zero exit that looks like the run itself died.
        try:
            fig = plot_convergence_map(doc, out=args.plot, xscale=args.xscale,
                                       annotate=not args.no_annotate,
                                       health_min=args.health_min)
            print(f"  wrote {fig}")
            logger.info(f"wrote {fig}")
        except ValueError as exc:
            print(f"  no figure: {exc}")
            logger.warning(f"no figure: {exc}")


if __name__ == "__main__":
    main()
