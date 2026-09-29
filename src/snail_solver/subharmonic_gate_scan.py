r"""Walk the gate pump across a mode's own subharmonic, tuning up a pulse at every point.

``tune_up`` calibrates a subharmonic two-qubit gate at ONE pump frequency
``w_p = |w_b - w_a| + wp_offset_GHz``. This module scans the pump across qubit A's
subharmonic, ``w_p = w_a/2 + delta``, and designs a chirped recursive-DRAG pulse at
every offset, so "what does the calibration do as the pump's second harmonic lands on
a qubit" has an answer per column.

Geometry
--------
``w_a`` and the SNAIL stay at their device values; only ``qubit_freqs_GHz[1]`` moves::

    w_p = w_a/2 + delta;   w_b = w_a -/+ w_p        (branch "below" / "above")
    2 w_p - w_a = 2 delta                           <- qubit A's subharmonic beat
    Delta_sub = w_s - 2 w_p = (w_s - w_a) - 2 delta  <- the SNAIL subharmonic axis

``branch="below"`` (``w_p = w_a - w_b``) sends ``w_b -> w_a/2 = w_p`` as ``delta -> 0``,
so A's subharmonic and a direct drive on B collide at once. ``branch="above"``
(``w_b = 1.5 w_a + delta``) isolates A's subharmonic.

**``delta = 0`` is not a gate.** The A-subharmonic channel is exactly resonant there
(``g/|det| -> inf``, and the detuning is inside the pulse bandwidth ``1/t_g``), so the
default grid drops it, and any column carrying a non-perturbative channel is refused
unless ``--force``.

The eta scan
------------
``--amp-points`` over ``[eta_lo, eta_hi] * target_eta`` is the **Rabi amplitude scan that
determines the chirp** (``tune_up`` steps 1-2: measure ``k2, k4``, project to Legendre
coefficients). It is a calibration input, NOT an output axis: each column reports ONE
fidelity at its operating point. eta 0.5 -> 2.5 in 0.05 steps is::

    --target-eta 2.5 --eta-lo 0.2 --eta-hi 1.0 --amp-points 41

``eta_hi = 1.0`` keeps the shift law interpolated rather than extrapolated.

DRAG channels
-------------
DERIVED per column by ``spectator_audit.select_drag_channels`` (leakage, coupler and
mode-subharmonic categories always; remaining slots to the strongest correctable
parasite; everything unselected is reported). ``--dry-run`` prints that audit for every
column and solves nothing. ``envelope_m`` is pinned to ``--max-drag-channels`` grid-wide,
so the envelope vanishes to enough order for every column and ``t_g`` stays comparable.

Usage
-----
    python -m snail_solver.subharmonic_gate_scan --device 4Gate4.5SNAIL.json \
        --offsets=-0.1:0.1:21 --target-eta 2.5 --dry-run

    python -m snail_solver.subharmonic_gate_scan --device 4Gate4.5SNAIL.json \
        --offsets=-0.1:0.1:21 --target-eta 2.5 --amp-points 41 \
        --coupler-levels 9 --jobs 72 --out wpscan.h5 --plot figs/wpscan.png

``--offsets=`` needs the ``=``: argparse reads a leading ``-`` as a flag.

One HDF5 file per scan::

    wpscan.h5
      /scan                  <- rows, settings, summary, a copy of the device config
      /scan/figures
      /columns/dm0p05        <- that column's COMPLETE tune-up, in tune_up's --out schema

RESUMABLE: every successful column is cached under ``--outdir`` and re-read, so a killed
scan resumes by relaunching the identical command. ``--overwrite`` forces a re-solve.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Default offset grid (GHz): symmetric about the subharmonic, EXCLUDING the origin.
DEFAULT_OFFSETS = "-0.1:0.1:21"

#: |delta| under which a column is dropped as "the origin". Half the DRAG skip window at
#: the default 5 MHz, since the A-subharmonic beat is 2 delta.
ORIGIN_EPS_GHz = 2.5e-3


def _opt_float(x: Any) -> Optional[float]:
    return None if x is None else float(x)


# ===========================================================================
# The grid
# ===========================================================================
def parse_offsets(spec: str) -> List[float]:
    """Offset grid in GHz: a comma list of scalars and/or ``lo:hi:n`` ranges.

    Shares :func:`subharmonic_convergence.parse_detunings`, so a range reproduces the
    same floats every time and a refined grid hits the coarse grid's cache files.
    """
    from snail_solver.subharmonic_convergence import parse_detunings
    return parse_detunings(spec)


def columns_for(config: Dict[str, Any], offsets_GHz: Sequence[float], *,
                drop_origin: bool = True) -> List[Dict[str, Any]]:
    """``[{delta_GHz, w_p_GHz, ...}]`` for an offset grid (``w_p = w_a/2 + delta``).

    The origin is dropped by default: the A-subharmonic channel is exactly resonant
    there, which no pulse can fix.
    """
    from snail_solver.subharmonic_convergence import _freqs, detuning_for_wp
    wa, _wb, ws = _freqs(config)
    out: List[Dict[str, Any]] = []
    for d in offsets_GHz:
        d = float(d)
        if drop_origin and abs(d) < ORIGIN_EPS_GHz:
            continue
        w_p = 0.5 * wa + d
        out.append({"delta_GHz": d, "w_p_GHz": float(w_p),
                    "subharm_beat_MHz": -2.0e3 * d,
                    "delta_sub_GHz": float(detuning_for_wp(config, w_p))})
    return out


def column_tag(delta_GHz: float, target_eta: Optional[float] = None) -> str:
    """``(0.05, 1.2) -> 'd0p05_eta1p2'`` -- a filename fragment, no dots."""
    from snail_solver.subharmonic_convergence import delta_tag
    from snail_solver.tune_up_sweep import eta_tag
    tag = delta_tag(delta_GHz)
    return tag if target_eta is None else f"{tag}_{eta_tag(target_eta)}"


def channel_labels(chans, audit: Dict[str, Any]) -> list:
    """Pair played ``DragChannel``s with their audit rows, so each channel is NAMED.

    ``beat_GHz``/``n_pump`` alone do not say which parasite is corrected, and the
    shed-and-retry leaves a different survivor column to column. Joined on the
    selector's own 0.5 MHz beat bucket. A channel with no audit row is reported as
    such (the selector and audit disagreeing is worth seeing).
    """
    from snail_solver.spectator_audit import _audit_beat_key

    by_key: Dict[tuple, Dict[str, Any]] = {}
    for r in (audit.get("rows") or ()):
        if r.get("beat_GHz") is None:
            continue
        by_key.setdefault((_audit_beat_key(float(r["beat_GHz"])),
                           int(r.get("n_pump") or 0)), r)
    out = []
    for c in chans:
        r = by_key.get((_audit_beat_key(float(c.beat_GHz)), int(c.n_pump))) or {}
        out.append({"beat_GHz": float(c.beat_GHz), "n_pump": int(c.n_pump),
                    "n_photon": int(c.n_photon),
                    "label": r.get("process") or "(no matching audit row)",
                    "transition": r.get("transition"),
                    "category": r.get("category"),
                    "g_MHz": r.get("g_MHz"),
                    "detuning_MHz": r.get("detuning_MHz"),
                    "ratio": r.get("ratio"),
                    "verdict": r.get("verdict")})
    return out


def shard_columns(cols: Sequence[Dict[str, Any]], shard: int,
                  n_shards: int) -> list:
    """Every ``n_shards``-th column starting at `shard` -- one machine's share.

    STRIDED, not blocked: column cost varies from milliseconds (refused) to ~40 min,
    so a stride interleaves cheap and expensive columns into every shard. Shards are
    disjoint and cover the grid, so caches never race and the HDF5 files concatenate
    (rows are keyed on ``(delta, target_eta)``).
    """
    n = int(n_shards)
    i = int(shard)
    if n < 1:
        raise ValueError(f"n_shards must be >= 1, got {n}")
    if not 0 <= i < n:
        raise ValueError(f"shard must be in [0, {n}), got {i}")
    return list(cols)[i::n]


def scan_config(config: Dict[str, Any], *, max_drag_channels: int,
                envelope_m: Optional[int] = None) -> Dict[str, Any]:
    """The grid-wide base config: a sine_power envelope with enough vanishing edges.

    A raised cosine (every shipped device) supports exactly ONE derivative correction;
    more diverge as ``t^(-1/2)`` at the gate edges (see :mod:`snail_solver.drag`).
    ``envelope_m`` comes from the CAP, not the per-column channel count, so
    ``area_factor`` -- and hence ``t_g`` at fixed peak ``|eta|`` -- is the same for
    every column.

    An explicit `envelope_m` exists for an independently calibrated NO-DRAG baseline
    (``max_drag_channels=0`` would derive m=2 against the DRAG run's 3: same ``t_g``,
    different pulse SHAPE).
    """
    m = int(envelope_m) if envelope_m is not None else max(int(max_drag_channels), 2)
    if m < 1:
        raise ValueError(f"envelope_m must be >= 1, got {m}")
    return {**config, "envelope": "sine_power", "envelope_m": m}


# ===========================================================================
# Decoherence
# ===========================================================================
def coherence_penalty(t_g_ns: float, *, t1_us: Optional[float] = None,
                      t2_us: Optional[float] = None,
                      prefactor: float = 1.0,
                      n_qubits: int = 2) -> Dict[str, Any]:
    """First-order incoherent error for a gate of length `t_g_ns`. NOT a solve.

    The scan's ``F_avg`` comes from a CLOSED-system solve, which biases it toward weak
    drive (``t_g = 2A/eta``: a long gate pays nothing). This supplies the missing term::

        1/T_eff  = n_qubits * (1/T1 + 1/T2)          (whichever are given)
        eps      = 1 - exp(-prefactor * t_g / T_eff)

    `prefactor` is a KNOB (the exact value depends on the error model), and the raw
    ``t_g_over_T`` is reported so any other convention can be applied later. Good for
    RANKING drive strengths; for an absolute number use a real open-system solve.

    Parameters
    ----------
    t_g_ns : float
        Gate duration (ns).
    t1_us, t2_us : float, optional
        Relaxation and total dephasing times (us). Either may be omitted.
    prefactor : float, default 1.0
        Multiplies ``t_g / T_eff``.
    n_qubits : int, default 2
        Qubits exposed for the gate duration.

    Returns
    -------
    dict
        ``t_g_ns``, ``T_eff_us``, ``t_g_over_T``, ``eps_incoherent`` -- or all None
        when neither time is given.
    """
    if t1_us is None and t2_us is None:
        return {"t_g_ns": float(t_g_ns), "T_eff_us": None, "t_g_over_T": None,
                "eps_incoherent": None}
    rate = 0.0                                       # 1/us
    if t1_us:
        rate += 1.0 / float(t1_us)
    if t2_us:
        rate += 1.0 / float(t2_us)
    rate *= max(int(n_qubits), 1)
    t_g_us = float(t_g_ns) / 1e3
    ratio = float(prefactor) * t_g_us * rate
    return {"t_g_ns": float(t_g_ns),
            "T_eff_us": (1.0 / rate if rate else None),
            "t_g_over_T": float(t_g_us * rate),
            "eps_incoherent": float(1.0 - np.exp(-ratio))}


def total_infidelity(F_coh: float, eps_incoherent: Optional[float]) -> float:
    """``1 - F_coh * (1 - eps_incoherent)`` (independent to first order).

    With no incoherent estimate this is just the coherent infidelity.
    """
    if eps_incoherent is None:
        return float(1.0 - F_coh)
    return float(1.0 - float(F_coh) * (1.0 - float(eps_incoherent)))


# ===========================================================================
# One column
# ===========================================================================
def _settings_for(col: Dict[str, Any], settings: Dict[str, Any],
                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`settings` with ``target_eta`` pinned to this column's point on the eta axis.

    With ``ridge_grid`` (and no explicit span) also sizes the SHARED offset grid a
    chirp ridge needs: :func:`tune_up.plot_chirp_ridge` does not interpolate, so every
    Rabi row must sit on one offset axis, fixed at measurement time and dependent on
    `target_eta` -- hence per column.
    """
    out = {**settings, "target_eta": float(col["target_eta"])}
    if settings.get("ridge_grid") and settings.get("wp_span_MHz") is None:
        if config is None:
            raise ValueError("ridge_grid needs the config to size the shared span")
        from snail_solver.tune_up import ridge_span_MHz
        span, want = ridge_span_MHz(
            config, out["target_eta"],
            eta_lo=float(settings["eta_lo"]), eta_hi=float(settings["eta_hi"]),
            span_linewidths=float(settings["span_linewidths"]),
            wp_points=int(settings["wp_points"]))
        out["wp_span_MHz"] = float(span)
        # A fixed span undersamples the weakest row unless the point count grows.
        out["wp_points"] = max(int(settings["wp_points"]), int(want))
    return out


def _stale_chirp_free(got: Dict[str, Any], max_frac: float) -> bool:
    """True if a cached CHIRP-FREE row would not be chirp-free under `max_frac`.

    Not a `_column_expect` key, because every expect key must be PRESENT in the cached
    record, so adding one would invalidate every cached column. The threshold can only
    turn a ``no_measurable_shift`` fallback into a failure, and that can be re-checked
    from the stored law. Other chirp-free reasons (``stark_crossing``,
    ``chirp_not_converged``) have no trustworthy law at eta* and are never stale; a row
    with no reason predates them and can only be ``no_measurable_shift``. A railed ridge
    stores no k2/k4, so its excursion is unknown and it is always stale.
    """
    op = got.get("operating_point") or {}
    if not op.get("chirp_free"):
        return False
    if (op.get("chirp_free_reason") or "no_measurable_shift") != "no_measurable_shift":
        return False
    ch = got.get("chirp") or {}
    k2, k4 = ch.get("k2"), ch.get("k4")
    t_g = op.get("t_g_ns")
    eta = got.get("target_eta")
    if k2 is None or k4 is None or not t_g or eta is None:
        return True
    eta = float(eta)
    exc = abs(float(k2) * eta ** 2 + float(k4) * eta ** 4)
    return exc / (1e3 / (2.0 * float(t_g))) > float(max_frac)


def _column_expect(col: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Physics a cached column must have been solved under to be reused.

    Grid RESOLUTION is deliberately absent, so refining the offset grid reuses the
    columns it has. Everything that changes the pulse, the device, or the stored
    number is present. ``.get`` defaults mirror the CLI so a synthetic settings dict
    still keys.
    """
    return {"w_p_GHz": float(col["w_p_GHz"]),
            "target_eta": float(col["target_eta"]),
            "branch": str(settings["branch"]),
            "coupler_levels": int(settings["coupler_levels"]),
            "amp_points": int(settings["amp_points"]),
            "eta_lo": float(settings["eta_lo"]),
            "eta_hi": float(settings["eta_hi"]),
            "max_drag_channels": int(settings["max_drag_channels"]),
            "min_ratio": float(settings["min_ratio"]),
            "max_ratio": float(settings["max_ratio"]),
            "contrast_min": float(settings["contrast_min"]),
            "leak_max": settings.get("leak_max"),
            "probe_shape": str(settings["probe_shape"]),
            "moment_weighting": str(settings["moment_weighting"]),
            "envelope_m": int(settings["envelope_m"]),
            # Not refinements but different MEASUREMENTS of the same column: one
            # shared offset axis instead of adaptive spans, and which rows may
            # re-measure on a wider window (cannot be re-derived from the record).
            "ridge_grid": bool(settings.get("ridge_grid")),
            "span_growth": str(settings.get("span_growth", "both")),
            "max_compound_growths": int(settings.get("max_compound_growths", 3)),
            # Convergence budgets for BOTH relaxations: a bigger one can turn a
            # cached failure into a fit.
            "chirp_max_passes": int(settings.get("chirp_max_passes", 12)),
            "max_drag_iters": int(settings.get("max_drag_iters", 4)),
            # The length-fit window changes the answer when the optimum is outside it.
            "tg_lo": float(settings.get("tg_lo", 0.7)),
            "tg_hi": float(settings.get("tg_hi", 1.3)),
            "tg_points": int(settings.get("tg_points", 13)),
            # Change the pulse: a zeroed chirp or decoupled DRAG is a different gate.
            "zero_chirp_frac": float(settings.get("zero_chirp_frac", 0.0)),
            "chirp_free_fallback": bool(settings.get("chirp_free_fallback", False)),
            "couple_drag": bool(settings.get("couple_drag", True)),
            "drag_decouple_fallback": bool(
                settings.get("drag_decouple_fallback", False)),
            # Contract versions. score_drag: rows before 2026-09-16 scored a pulse
            # that played NO DRAG. length_extend: rows before 2026-09-17 report a
            # length-window BOUNDARY rather than a bracketed optimum. Bump either if
            # its contract changes again.
            "score_drag": 1,
            "length_extend": 1}


#: (message fragment, failure type, stage) for the two relaxations a tune-up runs.
#: Named apart because different knobs fix them: the chirp<->DRAG fixed point wants
#: ``--chirp-max-passes``, the chirp<->length loop wants ``--max-drag-iters``.
_DIVERGENCES = (("fixed point did not settle", "DragFixedPointDiverged", "chirp"),
                ("did not converge", "ChirpLengthLoopDiverged", "length"))


def _divergence(exc: BaseException) -> Optional[Tuple[str, str]]:
    """``(type, stage)`` if `exc` is a relaxation failing to converge, else None."""
    m = str(exc)
    for needle, kind, stage in _DIVERGENCES:
        if needle in m:
            return kind, stage
    return None


def _failure_type(exc: BaseException) -> str:
    """Name the failure so a row says WHICH relaxation gave up, not just that one did."""
    d = _divergence(exc)
    return d[0] if d else type(exc).__name__


def _failure_stage(exc: BaseException) -> str:
    """Which tune-up stage owns the failure, for the row's ``error.stage``."""
    d = _divergence(exc)
    return d[1] if d else "tune_up"


def audit_column(config: Dict[str, Any], col: Dict[str, Any],
                 settings: Dict[str, Any]) -> Tuple[tuple, Dict[str, Any]]:
    """The DRAG channels and the full audit for one column. No propagation.

    Algebraic only (``expand_terms`` plus one coupler build), which is what lets
    ``--dry-run`` report every column's channel set before anything is solved.
    """
    from snail_solver.spectator_audit import select_drag_channels
    from snail_solver.subharmonic_convergence import config_at_wp
    from snail_solver.tune_up import nominal_t_g

    cfg = config_at_wp(config, col["w_p_GHz"], branch=settings["branch"],
                       levels=settings["coupler_levels"])
    t_g0 = nominal_t_g(cfg, settings["target_eta"])
    channels, audit = select_drag_channels(
        cfg, t_g0, max_channels=settings["max_drag_channels"],
        min_ratio=settings["min_ratio"], max_ratio=settings["max_ratio"],
        spec_abs_GHz=None)
    audit["t_g0_ns"] = float(t_g0)
    return channels, audit


def _fail(row: Dict[str, Any], t0: float, kind: str, stage: str,
          message: str) -> Dict[str, Any]:
    row.update({"ok": False, "seconds": time.perf_counter() - t0,
                "error": {"type": kind, "stage": stage, "message": message}})
    return row


def solve_column(config: Dict[str, Any], col: Dict[str, Any],
                 settings: Dict[str, Any], *,
                 solver: Optional[Dict[str, Any]] = None,
                 jobs: int = 0, force: bool = False,
                 logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Calibrate and score ONE column. Returns the row (with its ``run_doc``).

    1. move ``w_b`` so the pump is ``w_p`` (``config_at_wp``, which also strips any
       device chirp calibrated at a different pump);
    2. derive the DRAG channels and audit them;
    3. ``run_tune_up`` -- the Rabi amplitude scan happens HERE, and its only outputs
       are the chirp, ``wp_offset`` and ``t_g``;
    4. score ONCE at the resulting operating point.

    Step 3 consumes the eta axis; step 4 produces a scalar.

    A column that cannot be calibrated is a RESULT, not an outage: a ``RabiFitError``
    keeps its chevrons, and a relaxation that fails to converge sheds the weakest DRAG
    channel and retries (up to ``drag_retries``) before the column is recorded as
    failed.
    """
    from snail_solver.subharmonic_convergence import config_at_wp, coupler_occupation
    from snail_solver.tune_up import RabiFitError, run_tune_up
    from snail_solver.tune_up_sweep import score_gate

    t0 = time.perf_counter()
    log = logger or logging.getLogger("wp_scan")
    channels, audit = audit_column(config, col, settings)
    row: Dict[str, Any] = dict(col)
    row.update({"branch": str(settings["branch"]),
                "t_g0_ns": audit["t_g0_ns"],
                "channel_audit": audit,
                "drag_channels": channel_labels(channels, audit),
                "n_drag_channels": len(channels),
                "total_error": audit["total_error"]})

    if audit["blocking"] and not force:
        return _fail(row, t0, "NonPerturbativeChannel", "audit", "; ".join(
            f"{b['name']} g={b['g_MHz']:.3f} MHz det={b['detuning_MHz']:.3f} MHz"
            for b in audit["blocking"]))

    cfg = config_at_wp(config, col["w_p_GHz"], branch=settings["branch"],
                       levels=settings["coupler_levels"])

    def _tune(chans):
        return run_tune_up(
            cfg, settings["target_eta"], drag_channels=list(chans),
            eta_lo=settings["eta_lo"], eta_hi=settings["eta_hi"],
            amp_points=settings["amp_points"],
            wp_points=settings["wp_points"], wp_span_MHz=settings["wp_span_MHz"],
            span_linewidths=settings["span_linewidths"], n_time=settings["n_time"],
            window_tg=settings["window_tg"], tg_points=settings["tg_points"],
            tg_lo=settings["tg_lo"], tg_hi=settings["tg_hi"],
            chirp_degree=settings["chirp_degree"],
            max_drag_iters=settings["max_drag_iters"],
            chirp_max_passes=settings.get("chirp_max_passes", 12),
            contrast_min=settings["contrast_min"],
            quartic_warn=settings["quartic_warn"],
            zero_chirp_frac=float(settings.get("zero_chirp_frac", 0.0)),
            chirp_free_fallback=bool(settings.get("chirp_free_fallback", False)),
            chirp_free_max_frac=float(settings.get("chirp_free_max_frac", 0.10)),
            max_span_growths=int(settings.get("max_span_growths", 3)),
            max_wp_points=int(settings.get("max_wp_points", 121)),
            span_growth=str(settings.get("span_growth", "railed")),
            max_compound_growths=int(settings.get("max_compound_growths", 1)),
            couple_drag=bool(settings.get("couple_drag", True)),
            drag_decouple_fallback=bool(
                settings.get("drag_decouple_fallback", False)),
            probe_shape=settings["probe_shape"],
            moment_weighting=settings["moment_weighting"],
            do_time_rabi=False, jobs=jobs, solver=solver, logger=logger,
            **settings.get("map_kw", {}))

    used = list(channels)
    try:
        out = _tune(used)
    except RabiFitError as exc:
        _fail(row, t0, "RabiFitError", "rabi", str(exc))
        row["run_doc"] = {"stages": {"rabi": exc.table}, "device": dict(cfg)}
        return row
    except Exception as exc:                      # noqa: BLE001 - recorded, not raised
        # One correction too many makes the chirp<->DRAG fixed point RUN AWAY, and
        # whether a depth is well posed depends on the operating point -- so shed the
        # weakest channel and retry. envelope_m (the grid-wide cap) is untouched;
        # m >= len(channels) still holds as channels are shed.
        retries = min(int(settings.get("drag_retries", 2)), max(len(used) - 1, 0))
        out = None
        if _divergence(exc) and retries:
            for _ in range(retries):
                used = used[:-1]
                log.info(f"  chirp<->DRAG fixed point diverged at {len(used) + 1} "
                         f"channels; retrying with {len(used)}")
                try:
                    out = _tune(used)
                    break
                except RabiFitError:
                    break
                except Exception as exc2:         # noqa: BLE001
                    if not _divergence(exc2):
                        raise
        if out is None:
            return _fail(row, t0, _failure_type(exc), _failure_stage(exc), str(exc))
        row["drag_channels"] = channel_labels(used, audit)
        row["n_drag_channels"] = len(used)
        row["drag_shed"] = len(channels) - len(used)

    rec = out["operating_point"]
    chirp = [float(c) for c in (rec.get("chirp_coeffs_GHz") or ())]
    scfg = config_at_wp(config, col["w_p_GHz"], branch=settings["branch"],
                        levels=settings["coupler_levels"], chirp_coeffs_GHz=chirp)
    # `drag_channels=used` (the post-shed set) is LOAD-BEARING: multi-channel DRAG
    # does not travel via rec["drag_beat_GHz"], so omitting it scores an UN-DRAGGED
    # pulse. The flat reference passes chirp `[]` (None would inherit the config
    # chirp) but drag_channels None -- `[]` and None are equivalent there.
    chirped = score_gate(scfg, rec, chirp, solver=solver, drag_channels=used)
    flat = score_gate(scfg, rec, [], solver=solver, drag_channels=None)

    stages = out["stages"]
    fit = (stages.get("rabi") or {}).get("fit") or {}
    chirp_stage = stages.get("chirp") or {}
    scored = ("F_avg", "leakage", "transfer", "t_g_ns", "n_drag_channels")
    row.update({
        "ok": True, "error": None, "seconds": time.perf_counter() - t0,
        "operating_point": rec,
        "chirp": {"coeffs_GHz": chirp,
                  "k2": fit.get("k2"), "k4": fit.get("k4"), "r2": fit.get("r2"),
                  "resid_MHz": fit.get("resid_MHz"),
                  "quartic_fraction": chirp_stage.get("quartic_fraction"),
                  "drag_correction_ratio": chirp_stage.get("drag_correction_ratio"),
                  "min_abs_detuning_GHz": chirp_stage.get("min_abs_detuning_GHz")},
        # t_g_ns / n_drag_channels: each trace's length, and whether it kept its DRAG.
        "fidelity": {k: chirped.get(k) for k in scored},
        "flat": {k: flat.get(k) for k in scored},
        "delta_F": ((chirped.get("F_avg") or 0.0) - (flat.get("F_avg") or 0.0)),
        # The only place gate LENGTH is charged for in a closed-system scan.
        "coherence": coherence_penalty(
            float(rec["t_g_ns"]), t1_us=settings.get("t1_us"),
            t2_us=settings.get("t2_us"),
            prefactor=float(settings.get("decoh_prefactor", 1.0))),
        "n_coupler": coupler_occupation(scfg, rec, solver=solver),
        "run_doc": {"operating_point": rec, "t_g0_ns": out["t_g0_ns"],
                    "stages": stages, "device": dict(scfg)},
    })
    # length_rabi extends its grid to bracket the maximum and reports `railed` only
    # when no interior optimum exists out to its cap; take its word for it.
    t_g0 = float(out["t_g0_ns"])
    t_g = float(rec["t_g_ns"])
    railed = bool(rec.get("t_g_railed"))
    row["t_g_railed"] = ("hi" if railed and t_g > t_g0 else
                         "lo" if railed else None)
    row["t_g_over_t_g0"] = float(rec.get("t_g_over_t_g0", t_g / t_g0))
    row["t_g_grid_span_t_g0"] = rec.get("t_g_grid_span_t_g0")
    row["t_g_extensions"] = int(rec.get("t_g_extensions", 0))
    if railed:
        log.info(f"  WARNING: t_g={t_g:.2f} ns ({t_g / t_g0:.2f} t_g0) is still on "
                 f"an edge after {row['t_g_extensions']} grid extension(s) over "
                 f"{row['t_g_grid_span_t_g0']} x t_g0 -- no interior full swap exists "
                 f"here, so this length is a bound, not a calibration")
    row["infidelity_total"] = total_infidelity(
        row["fidelity"]["F_avg"] or 0.0, row["coherence"]["eps_incoherent"])
    row["infidelity_coherent"] = float(1.0 - (row["fidelity"]["F_avg"] or 0.0))
    return row


def save_column_rabi(cache_path: str, row: Dict[str, Any],
                     logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Write this column's Rabi stage beside its JSON cache, from the WORKER.

    The chevrons are most of a column's cost, and every question about the FIT policy
    is answerable from them without re-solving. The parent's `_write_run` alone is not
    enough: the pool's results are consumed in submission order (so nothing is written
    until the first-submitted column returns), and a column served from cache is never
    re-written into a resumed run's output file. Writing here puts the arrays on disk
    before the row is returned.

    Compressed ``<cache>_rabi.npz`` with flat keys (``chevrons/00007/metric``), readable
    without this module. Never fatal: returns None on failure.
    """
    stages = ((row.get("run_doc") or {}).get("stages") or {})
    rabi = stages.get("rabi")
    if not rabi:
        return None
    out = os.path.splitext(cache_path)[0] + "_rabi.npz"
    try:
        flat: Dict[str, Any] = {}

        def _put(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _put(f"{prefix}/{k}" if prefix else str(k), v)
            elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
                for i, v in enumerate(obj):
                    _put(f"{prefix}/{i:05d}", v)
            elif obj is None:
                flat[prefix] = np.array(np.nan)
            else:
                flat[prefix] = np.asarray(obj)

        _put("", rabi)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez_compressed(out, **flat)
        return out
    except Exception as exc:                          # never lose a solved column
        if logger:
            logger.warning(f"  could not save the Rabi arrays ({type(exc).__name__}: "
                           f"{exc}); they will exist only if the parent writes them")
        return None


def _column_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One column, in a worker process (module level so it is picklable).

    Returns the row with its ``run_doc``; the PARENT does the HDF5 write, since several
    processes appending to one file would corrupt it. The JSON cache (success only)
    and the Rabi arrays (always -- a failed column's chevrons say why) are written
    here, so a pooled run is resumable if it dies.
    """
    col, settings = payload["col"], payload["settings"]
    # A log PER COLUMN (`tail -f`), since the parent is silent until a column lands.
    logger = None
    if payload.get("log_path"):
        from snail_solver.log_utils import setup_run_logger
        logger = setup_run_logger(payload["log_path"],
                                  f"wp_col:{payload['log_path']}")
        logger.info(f"delta={col['delta_GHz']:+.4f} GHz  w_p={col['w_p_GHz']:.4f}  "
                    f"Delta_sub={col['delta_sub_GHz']:+.4f}  "
                    f"A-subharm beat={col['subharm_beat_MHz']:+.1f} MHz")
    row = solve_column(payload["config"], col, settings,
                       solver=payload["solver"], jobs=payload["jobs"],
                       force=payload["force"], logger=logger)
    row.update(payload["expect"])
    row["cached"] = False
    if payload.get("cache_path"):
        save_column_rabi(payload["cache_path"], row, logger=logger)
        if row.get("ok"):
            keep = {k: v for k, v in row.items() if k != "run_doc"}
            with open(payload["cache_path"], "w") as fh:
                json.dump(keep, fh, default=float)
    return row


# ===========================================================================
# The scan
# ===========================================================================
def _load_cached(path: Optional[str], expect: Dict[str, Any],
                 settings: Dict[str, Any], overwrite: bool,
                 log: logging.Logger) -> Optional[Dict[str, Any]]:
    """A reusable cached column, or None (overwrite, mismatch, or stale chirp-free)."""
    from snail_solver.subharmonic_convergence import _cache_load
    cached = None if overwrite else _cache_load(path, expect, log)
    if cached is not None and _stale_chirp_free(
            cached, settings.get("chirp_free_max_frac", 0.10)):
        log.info("  cached CHIRP-FREE row sweeps more than "
                 "--chirp-free-max-frac of a half-linewidth, so a chirp "
                 "was measurable and was discarded -- re-solving")
        return None
    return cached


def run_wp_scan(config: Dict[str, Any], offsets_GHz: Sequence[float],
                target_etas: Sequence[float], *,
                device_path: Optional[str] = None,
                branch: str = "below",
                coupler_levels: Optional[int] = None,
                eta_lo: float = 0.2, eta_hi: float = 1.0, amp_points: int = 41,
                max_drag_channels: int = 3, min_ratio: float = 0.02,
                max_ratio: float = 0.3, drag_retries: int = 2,
                column_workers: int = 1,
                t1_us: Optional[float] = None, t2_us: Optional[float] = None,
                decoh_prefactor: float = 1.0,
                wp_points: int = 25, wp_span_MHz: Optional[float] = None,
                span_linewidths: float = 4.0, n_time: int = 161,
                window_tg: float = 2.0, tg_points: int = 13,
                tg_lo: float = 0.7, tg_hi: float = 1.3,
                chirp_degree: int = 8, max_drag_iters: int = 4,
                chirp_max_passes: int = 12,
                contrast_min: float = 0.35, quartic_warn: float = 0.25,
                leak_max: Optional[float] = None,
                zero_chirp_frac: float = 0.0,
                chirp_free_fallback: bool = False,
                chirp_free_max_frac: float = 0.10,
                max_span_growths: int = 3, max_wp_points: int = 121,
                span_growth: str = "railed", max_compound_growths: int = 1,
                couple_drag: bool = True,
                drag_decouple_fallback: bool = False,
                probe_shape: str = "constant", moment_weighting: str = "rabi",
                drop_origin: bool = True, column_figures: bool = True,
                ridge_grid: bool = False,
                shard: int = 0, n_shards: int = 1,
                envelope_m: Optional[int] = None,
                outdir: Optional[str] = None, sweep_path: Optional[str] = None,
                overwrite: bool = False, force: bool = False,
                jobs: int = 0, solver: Optional[Dict[str, Any]] = None,
                stop_on_error: bool = False,
                logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Tune up a chirped recursive-DRAG pulse at every offset from A's subharmonic.

    Two ways to spend cores:

    * ``column_workers = 1`` (default): columns run one at a time with `jobs` inside;
      step 1's ``amp_points x wp_points`` chevron solves saturate a machine alone.
    * ``column_workers > 1``: columns run in a pool. What a long scan wants, because
      ``tune_up`` step 4 (``length_rabi``) is serial and runs for minutes per column,
      leaving a `jobs`-parallel run's cores idle.

    Each column is cached under ``outdir`` on success and re-read unless `overwrite`,
    so an interrupted scan resumes.

    Returns
    -------
    dict
        ``device`` (a copy of the configuration every column ran with), ``device_path``,
        ``settings``, ``rows`` (one per column, each with ``ok`` and either scores or an
        ``error``), ``landmarks`` and ``summary``.
    """
    from snail_solver.h5_io import attach_figures, save_doc, split_address
    from snail_solver.subharmonic_convergence import (collision_landmarks,
                                                      nearest_landmark)

    log = logger or logging.getLogger("wp_scan")
    base = scan_config(config, max_drag_channels=max_drag_channels,
                       envelope_m=envelope_m)
    levels = int(coupler_levels if coupler_levels is not None
                 else base.get("coupler_levels", 7))
    etas = [float(e) for e in target_etas]
    settings: Dict[str, Any] = {
        "target_etas": etas, "branch": str(branch),
        "coupler_levels": levels, "eta_lo": float(eta_lo), "eta_hi": float(eta_hi),
        "amp_points": int(amp_points), "max_drag_channels": int(max_drag_channels),
        "min_ratio": float(min_ratio), "max_ratio": float(max_ratio),
        "envelope_m": int(base["envelope_m"]),
        "wp_points": int(wp_points),
        "wp_span_MHz": _opt_float(wp_span_MHz),
        "ridge_grid": bool(ridge_grid),
        "span_linewidths": float(span_linewidths), "n_time": int(n_time),
        "window_tg": float(window_tg), "tg_points": int(tg_points),
        "tg_lo": float(tg_lo), "tg_hi": float(tg_hi),
        "chirp_degree": int(chirp_degree), "max_drag_iters": int(max_drag_iters),
        "chirp_max_passes": int(chirp_max_passes),
        "contrast_min": float(contrast_min), "quartic_warn": float(quartic_warn),
        "zero_chirp_frac": float(zero_chirp_frac),
        "chirp_free_fallback": bool(chirp_free_fallback),
        "chirp_free_max_frac": float(chirp_free_max_frac),
        "max_span_growths": int(max_span_growths),
        "max_wp_points": int(max_wp_points),
        "span_growth": str(span_growth),
        "max_compound_growths": int(max_compound_growths),
        "couple_drag": bool(couple_drag),
        "drag_decouple_fallback": bool(drag_decouple_fallback),
        # The shaped ("gate") probe is what makes strong drive measurable: a flat pump
        # at |eta| = 1.2 leaks 0.245 and destroys the chevron, the shaped one 4e-3.
        "probe_shape": str(probe_shape),
        "moment_weighting": str(moment_weighting),
        # A constant-probe row leaks far more than the gate at strong drive, and
        # tune_up's default only rejects past 35%: tighten this past the leakage knee.
        "leak_max": _opt_float(leak_max),
        "map_kw": ({} if leak_max is None else {"leak_max": float(leak_max)}),
        "drop_origin": bool(drop_origin), "envelope": base["envelope"],
        # A sharded file holds PART of the grid; say which part.
        "shard": int(shard), "n_shards": int(n_shards),
        "drag_retries": int(drag_retries),
        "t1_us": _opt_float(t1_us),
        "t2_us": _opt_float(t2_us),
        "decoh_prefactor": float(decoh_prefactor),
        # The Rabi amplitude ladder is a FRACTION of each target_eta, so it is
        # recorded per eta.
        "eta_scan_by_target": {f"{e:g}": [float(x) for x in
                                          np.linspace(eta_lo, eta_hi,
                                                      int(amp_points)) * e]
                               for e in etas},
    }
    # (delta, target_eta) grid; delta outermost so a killed run leaves whole eta slices.
    base_cols = columns_for(base, offsets_GHz, drop_origin=drop_origin)
    cols = [{**c, "target_eta": e} for c in base_cols for e in etas]
    n_all = len(cols)
    cols = shard_columns(cols, shard, n_shards)
    if int(n_shards) > 1:
        log.info(f"shard {int(shard)}/{int(n_shards)}: {len(cols)} of {n_all} columns")
    landmarks = collision_landmarks(base, branch=branch)

    cache_dir = os.path.join(outdir, "columns") if outdir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    def _paths(col):
        tag = column_tag(col["delta_GHz"], col["target_eta"])
        path = os.path.join(cache_dir, f"col_{tag}.json") if cache_dir else None
        return tag, path

    def _write_run(row, col, tag):
        """Move a row's tune-up document into the scan file (parent process only)."""
        run_doc = row.pop("run_doc", None)
        if run_doc is not None and sweep_path:
            h5 = split_address(sweep_path)[0]
            row["run"] = save_doc(
                h5, run_doc,
                attrs={"delta_GHz": float(col["delta_GHz"]),
                       "w_p_GHz": float(col["w_p_GHz"]),
                       "status": "ok" if row.get("ok") else "failed"},
                group=f"columns/{tag}")
            # The chevron picture goes in beside its arrays; never fails the row.
            if column_figures:
                figs = render_column_figures(
                    {**run_doc, "delta_GHz": col["delta_GHz"],
                     "w_p_GHz": col["w_p_GHz"], "target_eta": col.get("target_eta")},
                    tag, figdir=(os.path.join(outdir, "column_figs") if outdir
                                 else os.path.join(os.path.dirname(h5) or ".",
                                                   "column_figs")),
                    ridge=True, logger=log)
                if figs:
                    attach_figures(f"{h5}:/columns/{tag}", figs)

    def _done():
        return _scan_doc(base, device_path, settings, base_cols, etas, landmarks,
                         rows, time.perf_counter() - t_start)

    if int(column_workers) > 1 and len(cols) > 1:
        from snail_solver.subharmonic_convergence import _run_pool
        pending, payloads = [], []
        for col in cols:
            tag, path = _paths(col)
            expect = _column_expect(col, settings)
            cached = _load_cached(path, expect, settings, overwrite, log)
            if cached is not None:
                cached["cached"] = True
                cached["nearest_landmark"] = nearest_landmark(landmarks,
                                                              col["delta_sub_GHz"])
                pending.append((col, tag, cached))
                continue
            payloads.append({"config": base, "col": col,
                             "settings": _settings_for(col, settings, base),
                             "solver": solver, "jobs": jobs, "force": force,
                             "expect": expect, "cache_path": path,
                             "log_path": (os.path.join(cache_dir, f"col_{tag}.log")
                                          if cache_dir else None)})
            pending.append((col, tag, None))
        log.info(f"{len(payloads)} column(s) to solve over {int(column_workers)} "
                 f"worker(s), {jobs} job(s) each; "
                 f"{len(pending) - len(payloads)} cached")
        if cache_dir and payloads:
            log.info(f"  per-column progress: tail -f {cache_dir}/col_*.log")
        solved = iter(_run_pool(_column_worker, payloads, int(column_workers)))
        for col, tag, cached in pending:
            row = cached if cached is not None else next(solved)
            if cached is None:
                row["nearest_landmark"] = nearest_landmark(landmarks,
                                                           col["delta_sub_GHz"])
                _write_run(row, col, tag)
            f = row.get("fidelity")
            log.info(f"  delta={col['delta_GHz']:+.4f}: "
                     + (f"F={f['F_avg']:.5f} leak={f['leakage']:.2e}" if f
                        else f"FAILED {row.get('error', {}).get('stage', '?')}")
                     + ("  (cached)" if cached is not None else ""))
            rows.append(row)
        return _done()

    for i, col in enumerate(cols):
        tag, path = _paths(col)
        log.info(f"=== [{i + 1}/{len(cols)}] delta={col['delta_GHz']:+.4f} GHz  "
                 f"eta*={col['target_eta']:g}  w_p={col['w_p_GHz']:.4f}  "
                 f"Delta_sub={col['delta_sub_GHz']:+.4f}  "
                 f"A-subharm beat={col['subharm_beat_MHz']:+.1f} MHz")

        expect = _column_expect(col, settings)
        cached = _load_cached(path, expect, settings, overwrite, log)
        if cached is not None:
            log.info("  cached")
            cached["cached"] = True
            rows.append(cached)
            continue

        row = solve_column(base, col, _settings_for(col, settings, base), solver=solver,
                           jobs=jobs, force=force, logger=log)
        row.update(expect)
        row["cached"] = False
        row["nearest_landmark"] = nearest_landmark(landmarks, col["delta_sub_GHz"])
        _write_run(row, col, tag)

        if row.get("ok"):
            f = row["fidelity"]
            log.info(f"  F={f['F_avg']:.5f}  leak={f['leakage']:.2e}  "
                     f"t_g={row['operating_point']['t_g_ns']:.1f}ns  "
                     + (f"1-F_tot={row['infidelity_total']:.2e}  "
                        if row['coherence']['eps_incoherent'] is not None else "")
                     + f"n_s={row['n_coupler']:.2e}  "
                     f"{row['n_drag_channels']} chan  {row['seconds']:.1f}s")
            # Name the channels actually PLAYED (the survivor set varies by column).
            for ch in (row.get("drag_channels") or ()):
                log.info(f"    drag: {ch['beat_GHz']:+.4f} GHz k={ch['n_pump']} "
                         f"[{ch.get('category') or '?'}] {ch.get('label')}"
                         + (f"  g={ch['g_MHz']:.2f} MHz" if ch.get("g_MHz") is not None
                            else "")
                         + (f"  ratio={ch['ratio']:.3f}" if ch.get("ratio") is not None
                            else ""))
            if path:                              # only on success
                with open(path, "w") as fh:
                    json.dump(row, fh, default=float)
        else:
            log.info(f"  FAILED [{row['error']['stage']}] {row['error']['message'][:120]}")
        rows.append(row)
        if stop_on_error and not row.get("ok"):
            break

    return _done()


def column_figure_title(row_or_doc: Dict[str, Any], tag: str) -> str:
    """Label a column figure with the operating point it belongs to, not its tag."""
    d = row_or_doc or {}
    op = d.get("operating_point") or {}
    bits = [f"delta={float(d['delta_GHz']) * 1e3:+.0f} MHz"] if "delta_GHz" in d else []
    if "w_p_GHz" in d:
        bits.append(f"w_p={float(d['w_p_GHz']):.4f} GHz")
    eta = d.get("target_eta", op.get("target_eta"))
    if eta is not None:
        bits.append(f"target_eta={float(eta):g}")
    probe = ((d.get("stages") or {}).get("rabi") or {}).get("fit") or {}
    if probe.get("probe_shape"):
        w = probe.get("moment_weighting")
        bits.append(f"probe={probe['probe_shape']}" + (f"/{w}" if w else ""))
    return f"{tag}  ({', '.join(bits)})" if bits else tag


def render_column_figures(run_doc: Dict[str, Any], tag: str, figdir: str, *,
                          ridge: bool = True,
                          logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """Render ONE column's chevron map and chirp ridge from its stored stages.

    Best-effort: the chevron figure needs only ``stages.rabi`` (so a column whose FIT
    failed still gets one -- the column whose chevrons matter most); the ridge also
    needs the chirp and a fitted length and is skipped without them; a rendering error
    is logged and swallowed.

    Returns ``{"rabi": path}`` plus ``{"chirp_ridge": path}`` when drawn, ready for
    :func:`h5_io.attach_figures`.
    """
    log = logger or logging.getLogger("wp_scan")
    stages = ((run_doc or {}).get("stages")) or {}
    table = stages.get("rabi")
    figs: Dict[str, str] = {}
    if not table:
        return figs
    os.makedirs(figdir, exist_ok=True)
    title = column_figure_title(run_doc, tag)
    from snail_solver.tune_up import plot_chirp_ridge, plot_rabi_table
    try:
        figs["rabi"] = plot_rabi_table(
            table, os.path.join(figdir, f"{tag}_chevrons.png"), title=title)
    except Exception as exc:                                  # never fatal
        log.warning(f"  {tag}: chevron figure failed ({type(exc).__name__}: {exc})")
    op = (run_doc or {}).get("operating_point") or {}
    chirp = stages.get("chirp")
    if ridge and chirp and op.get("t_g_ns"):
        try:
            figs["chirp_ridge"] = plot_chirp_ridge(
                table, chirp, float(op.get("wp_offset_GHz") or 0.0),
                float(op["t_g_ns"]),
                os.path.join(figdir, f"{tag}_ridge.png"), title=title)
        except Exception as exc:                              # never fatal
            log.warning(f"  {tag}: ridge figure failed ({type(exc).__name__}: {exc})")
    return figs


def attach_all_column_figures(path: str, *, figdir: Optional[str] = None,
                              ridge: bool = True, only: Optional[Sequence[str]] = None,
                              logger: Optional[logging.Logger] = None) -> int:
    """Render and embed every column's figures in an EXISTING scan file. No solves.

    Backfills an older file or re-renders after a plotting change. Returns how many
    figures were embedded.
    """
    from snail_solver.h5_io import attach_figures, load_doc
    log = logger or logging.getLogger("wp_scan")
    doc = load_doc(path)
    columns = doc.get("columns") or {}
    if not columns:
        log.warning(f"{path}: no /columns group -- nothing to plot")
        return 0
    rows = {}
    for r in ((doc.get("scan") or doc).get("rows") or []):
        try:
            rows[column_tag(float(r["delta_GHz"]), float(r["target_eta"]))] = r
        except (KeyError, TypeError, ValueError):
            continue
    # Per-FILE subdirectory: two scans of the same grid share column tags, and would
    # overwrite each other's loose PNGs.
    stem = os.path.splitext(os.path.basename(os.path.abspath(path)))[0]
    figdir = figdir or os.path.join(os.path.dirname(os.path.abspath(path)) or ".",
                                    "column_figs", stem)
    total = 0
    for tag in sorted(columns):
        if only and tag not in set(only):
            continue
        run_doc = dict(columns[tag])
        # The grid coordinates live on the scan row; merge them in for the title.
        for k in ("delta_GHz", "w_p_GHz", "target_eta"):
            if k not in run_doc and k in rows.get(tag, {}):
                run_doc[k] = rows[tag][k]
        figs = render_column_figures(run_doc, tag, figdir, ridge=ridge, logger=log)
        if not figs:
            log.info(f"  {tag}: no rabi stage -- skipped")
            continue
        n = attach_figures(f"{path}:/columns/{tag}", figs)
        total += n
        log.info(f"  {tag}: embedded {n} figure(s) ({', '.join(sorted(figs))})")
    log.info(f"{total} figure(s) embedded in {path}")
    return total


def _scan_doc(base, device_path, settings, base_cols, etas, landmarks, rows,
              seconds: float) -> Dict[str, Any]:
    """Assemble the scan document, ranking the best point on TOTAL infidelity.

    Ranking on ``F_avg`` alone always prefers the weakest drive (the closed-system
    solve never charges for a longer gate), so ``infidelity_total`` is used when
    coherence times were given.
    """
    def _coh_inf(r):
        if r.get("infidelity_coherent") is not None:
            return float(r["infidelity_coherent"])
        return 1.0 - float((r.get("fidelity") or {}).get("F_avg") or 0.0)

    def _rank(r):
        return (r["infidelity_total"] if r.get("infidelity_total") is not None
                else _coh_inf(r))

    ok = [r for r in rows if r.get("ok")]
    best = min(ok, key=_rank) if ok else None
    per_eta = {}
    for e in etas:
        slice_ = [r for r in ok if abs(float(r.get("target_eta", e)) - e) < 1e-12]
        if not slice_:
            continue
        b = min(slice_, key=_rank)
        per_eta[f"{e:g}"] = {
            "n_ok": len(slice_),
            "best_delta_GHz": float(b["delta_GHz"]),
            "t_g_ns": float((b.get("operating_point") or {}).get("t_g_ns",
                                                                 float("nan"))),
            "F_avg": float((b.get("fidelity") or {}).get("F_avg") or 0.0),
            "infidelity_coherent": _coh_inf(b),
            "eps_incoherent": (b.get("coherence") or {}).get("eps_incoherent"),
            "infidelity_total": b.get("infidelity_total"),
        }
    return {
        "source": "subharmonic_gate_scan",
        "device": dict(base), "device_path": (str(device_path) if device_path else ""),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": settings,
        "offsets_GHz": [float(c["delta_GHz"]) for c in base_cols],
        "target_etas": [float(e) for e in etas],
        "landmarks": landmarks, "rows": rows,
        "by_target_eta": per_eta,
        "summary": {"n_columns": len(rows), "n_ok": len(ok),
                    "n_failed": len(rows) - len(ok), "seconds": float(seconds),
                    "best_delta_GHz": (best["delta_GHz"] if best else None),
                    "best_target_eta": (best.get("target_eta") if best else None),
                    "best_infidelity_total": (best.get("infidelity_total") if best
                                              else None),
                    "best_infidelity": (_rank(best) if best else None)},
    }


def rescore_open_system(doc: Dict[str, Any], *, t1_us: float,
                        t2_us: Optional[float] = None,
                        coupler_t1_us: Optional[float] = None,
                        top: int = 3, fit_virtual_z: bool = True,
                        solver: Optional[Dict[str, Any]] = None,
                        logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Re-score the best `top` columns of a finished scan with a real open-system solve.

    Replaces the first-order incoherent ESTIMATE with ``mesolve`` (16 solves per point
    on a ``dim^2`` density matrix) on the few points the ranking picked out. Mutates
    and returns `doc`: each rescored row gains ``open_system``, as does the document.
    """
    from snail_solver.device_utils import build_coupler
    from snail_solver.open_system import collapse_ops, open_iswap_fidelity
    from snail_solver.subharmonic_convergence import config_at_wp
    from snail_solver.tune_up import fixed_eta_amp_scale

    log = logger or logging.getLogger("wp_scan")
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    base = doc["device"]
    settings = doc["settings"]
    times = {"t1_us": float(t1_us), "t2_us": _opt_float(t2_us),
             "coupler_t1_us": _opt_float(coupler_t1_us)}

    def _rank(r):
        v = r.get("infidelity_total")
        return v if v is not None else r.get("infidelity_coherent", 1.0)

    ok = sorted((r for r in doc["rows"] if r.get("ok")), key=_rank)[:max(int(top), 1)]
    log.info(f"open-system rescore of {len(ok)} point(s), T1={t1_us} us "
             f"T2={t2_us} us: 16 mesolve runs each")
    for r in ok:
        rec = r["operating_point"]
        chirp = [float(c) for c in (rec.get("chirp_coeffs_GHz") or ())]
        cfg = config_at_wp(base, r["w_p_GHz"], branch=settings["branch"],
                           levels=settings["coupler_levels"], chirp_coeffs_GHz=chirp)
        t_g = float(rec["t_g_ns"])
        # amp_scale re-derived, never read off the record, so the pulse is the
        # calibrated one.
        amp = fixed_eta_amp_scale(cfg, t_g, float(r["target_eta"]))
        cpl, _w_p, _peak = build_coupler(cfg, t_g, amp,
                                         float(rec["wp_offset_GHz"]),
                                         chirp_coeffs_GHz=chirp)
        ops = collapse_ops(cpl, t1_us=t1_us, t2_us=t2_us,
                           coupler_t1_us=coupler_t1_us)
        out = open_iswap_fidelity(cpl, 0, 1, t_g, c_ops=ops,
                                  fit_virtual_z=fit_virtual_z, **solver)
        r["open_system"] = {**out, **times,
                            "infidelity": float(1.0 - out["F_avg"]),
                            "estimate_infidelity": r.get("infidelity_total")}
        est = r.get("infidelity_total")
        log.info(f"  delta={r['delta_GHz']:+.4f} eta*={r['target_eta']:g} "
                 f"t_g={t_g:.1f}ns  1-F_open={1 - out['F_avg']:.3e}"
                 + (f"  (estimate was {est:.3e})" if est is not None else ""))
    doc["open_system"] = {**times, "n_rescored": len(ok), "top": int(top)}
    return doc


def describe_grid(config: Dict[str, Any], offsets_GHz: Sequence[float],
                  target_etas: Sequence[float], *, branch: str = "below",
                  coupler_levels: Optional[int] = None,
                  max_drag_channels: int = 3, min_ratio: float = 0.02,
                  max_ratio: float = 0.3,
                  eta_lo: float = 0.2, eta_hi: float = 1.0, amp_points: int = 41,
                  wp_points: int = 25, drop_origin: bool = True,
                  envelope_m: Optional[int] = None,
                  audit: bool = True) -> str:
    """The ``--dry-run`` report: geometry, cost, and every column's channel audit.

    Solves nothing -- read it before spending anything. With `audit` the report is
    printed as it streams and ``""`` is returned; without, the geometry is returned.
    """
    from snail_solver.spectator_audit import (print_channel_audit,
                                              select_drag_channels)
    from snail_solver.subharmonic_convergence import (_freqs, collision_landmarks,
                                                      config_at_wp,
                                                      nearest_landmark)
    from snail_solver.tune_up import nominal_t_g

    base = scan_config(config, max_drag_channels=max_drag_channels,
                       envelope_m=envelope_m)
    levels = int(coupler_levels if coupler_levels is not None
                 else base.get("coupler_levels", 7))
    tetas = [float(e) for e in target_etas]
    settings = {"target_eta": tetas[0], "branch": str(branch),
                "coupler_levels": levels, "eta_lo": float(eta_lo),
                "eta_hi": float(eta_hi), "amp_points": int(amp_points),
                "max_drag_channels": int(max_drag_channels),
                "min_ratio": float(min_ratio), "max_ratio": float(max_ratio)}
    wa, wb, ws = _freqs(base)
    base_cols = columns_for(base, offsets_GHz, drop_origin=drop_origin)
    cols = [{**c, "target_eta": e} for c in base_cols for e in tetas]
    etas = np.linspace(eta_lo, eta_hi, int(amp_points)) * tetas[0]
    landmarks = collision_landmarks(base, branch=branch)

    lines = [
        "subharmonic gate scan -- dry run (nothing is solved)",
        f"  device      w_a={wa:g}  w_b={wb:g} (moves)  w_s={ws:g} GHz, "
        f"{levels} coupler levels",
        f"  axis        w_p = w_a/2 + delta = {0.5 * wa:g} + delta   branch={branch}",
        f"              Delta_sub = (w_s - w_a) - 2 delta = {ws - wa:g} - 2 delta",
        "              A-subharmonic beat = w_a - 2 w_p = -2 delta",
        f"  columns     {len(base_cols)} of {len(list(offsets_GHz))} offsets "
        f"x {len(tetas)} target eta = {len(cols)}"
        + ("  (origin dropped: not a gate)" if drop_origin else ""),
        "  target eta  " + ", ".join(f"{e:g} (t_g={nominal_t_g(base, e):.1f}ns)"
                                      for e in tetas),
        f"  eta scan    {len(etas)} points per column, a FRACTION "
        f"[{eta_lo:g}, {eta_hi:g}] of each target eta "
        f"-- calibration input, NOT an output axis",
        f"  envelope    {base['envelope']} m={base['envelope_m']} "
        f"(>= --max-drag-channels, fixed grid-wide)",
        f"  cost        ~{len(cols) * int(amp_points) * int(wp_points)} chevron solves "
        f"for the Rabi tables alone ({len(cols)} x {amp_points} x {wp_points}); "
        f"step 4 then runs 2*n_channels serial length scans per column",
        "",
        "  landmarks on this axis (other processes that go resonant):",
    ]
    for lm in landmarks:
        w_p_off = lm["w_p_GHz"] - 0.5 * wa
        lines.append(f"    Delta_sub={lm['delta_sub_GHz']:+8.4f}  w_p={lm['w_p_GHz']:.4f} "
                     f"(delta={w_p_off:+.4f})  k={lm['n_pump']}  "
                     f"{'coupler ' if lm['coupler'] else '        '}{lm['name']}")
    lines.append("")

    lines.append("  perturbative feasibility, worst parasitic g/|det| per column")
    lines.append("  (<0.3 DRAG effective, 0.3-1 left uncorrected, >=1 NOT "
                 "perturbative -> refused)")
    hdr = "".join(f"{c['delta_GHz'] * 1e3:>8.0f}" for c in base_cols)
    lines.append(f"    {'eta*':>5} {'t_g(ns)':>8}  {hdr}   MHz offset")
    for e in tetas:
        vals = []
        for c in base_cols:
            try:
                cfg = config_at_wp(base, c["w_p_GHz"], branch=branch, levels=levels)
                _ch, au = select_drag_channels(cfg, nominal_t_g(cfg, e),
                                               max_channels=max_drag_channels,
                                               min_ratio=min_ratio,
                                               max_ratio=max_ratio)
                # NOT filtered on isfinite: an exactly resonant channel (ratio = inf)
                # IS the worst case, and dropping it would report the column clean.
                vals.append(max((r["ratio"] for r in au["rows"]
                                 if r["category"] != "target"), default=0.0))
            except Exception:                        # noqa: BLE001
                vals.append(float("nan"))
        lines.append(f"    {e:>5g} {nominal_t_g(base, e):>8.1f}  "
                     + "".join(("     inf" if not np.isfinite(v) else f"{v:8.2f}")
                               for v in vals))
        lines.append(f"    {'':>5} {'':>8}  "
                     + "".join(("     ok " if v < min_ratio * 15 else "   marg "
                                if v < 1.0 else "   RESON" if not np.isfinite(v)
                                else "   FAIL ") for v in vals))
    lines.append("")

    if not audit:
        return "\n".join(lines)

    print("\n".join(lines))
    n_block = 0
    for col in cols:
        try:
            _ch, au = audit_column(base, col, _settings_for(col, settings, base))
        except Exception as exc:                  # noqa: BLE001 - report, keep going
            print(f"  delta={col['delta_GHz']:+.4f}: audit failed: {exc}")
            continue
        lm = nearest_landmark(landmarks, col["delta_sub_GHz"])
        print(f"\n  --- delta={col['delta_GHz']:+.4f} GHz  "
              f"eta*={col['target_eta']:g}  w_p={col['w_p_GHz']:.4f}  "
              f"Delta_sub={col['delta_sub_GHz']:+.4f}  "
              f"A-subharm beat={col['subharm_beat_MHz']:+.1f} MHz"
              + (f"  [nearest: {lm['name']} at {lm['detuning_GHz']:+.4f} GHz in w_p]"
                 if lm else ""))
        print_channel_audit(au)
        n_block += bool(au["blocking"])
    if n_block:
        print(f"\n  !! {n_block}/{len(cols)} columns carry a NON-PERTURBATIVE channel; "
              f"they will be refused without --force.")
    return ""


# ===========================================================================
# Figure
# ===========================================================================
def plot_wp_scan(doc: Dict[str, Any], out: str = "figs/wp_scan.png") -> str:
    """Infidelity, leakage, coupler occupation and chirp residual vs offset, per eta.

    The top panel draws coherent infidelity (solid) with the TOTAL including the
    incoherent estimate (dashed) and each eta's coherence floor (dotted): the coherent
    curve alone always favours the weakest drive.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in doc["rows"] if r.get("ok")]
    bad = [r for r in doc["rows"] if not r.get("ok")]
    if not rows:
        raise ValueError("no successful column to plot")
    etas = sorted({float(r["target_eta"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    colours = {e: cmap(0.12 + 0.76 * i / max(len(etas) - 1, 1))
               for i, e in enumerate(etas)}
    has_decoh = any(r["coherence"]["eps_incoherent"] is not None for r in rows)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(8.0, 11.0), sharex=True)
    for e in etas:
        sl = sorted((r for r in rows if abs(float(r["target_eta"]) - e) < 1e-12),
                    key=lambda r: r["delta_GHz"])
        d = np.array([r["delta_GHz"] for r in sl]) * 1e3
        col = colours[e]
        t_g = sl[0]["operating_point"]["t_g_ns"]
        ax[0].plot(d, [r["infidelity_coherent"] for r in sl], "o-", color=col,
                   label=fr"$\eta^*$={e:g}  ($t_g$={t_g:.0f} ns)")
        if has_decoh and sl[0]["coherence"]["eps_incoherent"] is not None:
            ax[0].plot(d, [r["infidelity_total"] for r in sl], "s--", color=col,
                       alpha=0.75)
            ax[0].axhline(sl[0]["coherence"]["eps_incoherent"], color=col,
                          lw=0.8, ls=":", alpha=0.6)
        ax[1].plot(d, [r["fidelity"]["leakage"] for r in sl], "o-", color=col)
        ax[2].plot(d, [r.get("n_coupler", np.nan) for r in sl], "o-", color=col)
        ax[3].plot(d, [(r["chirp"] or {}).get("resid_MHz", np.nan) for r in sl],
                   "o-", color=col)

    ax[0].set_yscale("log"); ax[0].set_ylabel("infidelity")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title("gate quality across qubit A's subharmonic  "
                    r"($\omega_p=\omega_a/2+\delta$, branch "
                    f"{doc['settings']['branch']})"
                    + ("\nsolid: coherent   dashed: + incoherent estimate   "
                       "dotted: coherence floor" if has_decoh else ""))
    ax[1].set_yscale("log"); ax[1].set_ylabel("leakage")
    ax[2].set_yscale("log"); ax[2].set_ylabel(r"$\langle n_s\rangle$ at $t_g$")
    ax[3].set_ylabel("chirp resid (MHz)")
    ax[3].set_xlabel(r"pump offset from the subharmonic  $\delta$  (MHz)")
    for a in ax:
        a.grid(alpha=0.3)
        a.axvline(0.0, color="#b22222", lw=1.0, ls=":")
    for r in bad:
        for a in ax:
            a.axvline(r["delta_GHz"] * 1e3, color="#b22222", alpha=0.10, lw=3.0)
    ax[0].text(0.0, 0.02, " subharmonic\n (not a gate)",
               transform=ax[0].get_xaxis_transform(), fontsize=7,
               color="#b22222", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    """CLI entry point."""
    from snail_solver.stark_chirp import MOMENT_WEIGHTINGS

    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.subharmonic_gate_scan", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not required for --replot*: a finished scan carries its own device copy.
    ap.add_argument("--device", default=None,
                    help="device JSON (bare name resolves under devices/); not "
                         "needed with --replot or --replot-columns")
    ap.add_argument("--target-etas", default=None,
                    help="peak |eta| values to scan, as \"0.6,0.8,1.0,1.2\" or "
                         "\"0.6:1.2:4\"; each sets t_g = 2A/eta*. The subharmonic "
                         "coupling grows as eta^2 at a detuning set by delta, so past "
                         "eta ~ 1 a +/-100 MHz scan goes NON-perturbative. Check "
                         "with --dry-run first.")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="a single peak |eta| (shorthand for --target-etas with one "
                         "value)")
    ap.add_argument("--t1-us", type=float, default=None,
                    help="relaxation time (us). The solve is CLOSED-system, so without "
                         "this the scan charges nothing for gate length and is biased "
                         "toward weak drive (t_g = 2A/eta)")
    ap.add_argument("--t2-us", type=float, default=None,
                    help="total dephasing time (us); see --t1-us")
    ap.add_argument("--decoh-prefactor", type=float, default=1.0,
                    help="coefficient on t_g/T_eff in the incoherent estimate [1.0]. "
                         "A first-order RANKING aid, not a solve -- the exact "
                         "prefactor depends on the error model, so it is a knob")
    ap.add_argument("--offsets", default=DEFAULT_OFFSETS,
                    help=f"offset grid in GHz from w_a/2: comma list of scalars and/or "
                         f"lo:hi:n ranges [{DEFAULT_OFFSETS}]. NEEDS the '=' form "
                         f"(--offsets=-0.1:0.1:21): a leading '-' reads as a flag")
    ap.add_argument("--branch", choices=("below", "above"), default="below",
                    help="'below' is w_p = w_a - w_b (w_b -> w_p at the origin); "
                         "'above' is w_p = w_b - w_a, which isolates A's subharmonic "
                         "[below]")
    ap.add_argument("--keep-origin", action="store_true",
                    help="keep delta = 0, where the A-subharmonic channel is exactly "
                         "resonant. Needs --force to actually solve")
    ap.add_argument("--eta-lo", type=float, default=0.2,
                    help="low end of the Rabi amplitude scan, as a FRACTION of "
                         "--target-eta [0.2]")
    ap.add_argument("--eta-hi", type=float, default=1.0,
                    help="high end, as a fraction of --target-eta. 1.0 is the pulse's "
                         "own peak; above it the shift law is extrapolated [1.0]")
    ap.add_argument("--amp-points", type=int, default=41,
                    help="points in the Rabi amplitude scan that determines the chirp. "
                         "With the defaults and --target-eta 2.5 this is eta "
                         "0.50..2.50 in 0.05 steps [41]")
    ap.add_argument("--max-drag-channels", type=int, default=3,
                    help="cap on the recursion depth, and the grid-wide sine_power m. "
                         "3 is the deepest measured as well-posed; at 4 the "
                         "chirp<->DRAG fixed point diverged on 4Gate4.5SNAIL [3]")
    ap.add_argument("--drag-retries", type=int, default=2,
                    help="if the chirp<->DRAG fixed point diverges, shed the weakest "
                         "channel and retry, up to this many times, rather than losing "
                         "the column. 0 disables [2]")
    ap.add_argument("--min-ratio", type=float, default=0.02,
                    help="g/|det| under which a channel is negligible and stops being "
                         "mandatory [0.02]")
    ap.add_argument("--max-ratio", type=float, default=0.3,
                    help="g/|det| at or above which a channel is left UNCORRECTED "
                         "(its quadrature would make the chirp<->DRAG fixed point "
                         "diverge). Reported in the audit [0.3]")
    ap.add_argument("--force", action="store_true",
                    help="solve columns that carry a non-perturbative channel anyway")
    ap.add_argument("--coupler-levels", type=int, default=None,
                    help="override the device's coupler truncation")
    ap.add_argument("--wp-points", type=int, default=25)
    ap.add_argument("--wp-span-MHz", type=float, default=None)
    ap.add_argument("--span-linewidths", type=float, default=4.0)
    ap.add_argument("--n-time", type=int, default=161)
    ap.add_argument("--window-tg", type=float, default=2.0)
    ap.add_argument("--tg-points", type=int, default=13)
    ap.add_argument("--tg-lo", type=float, default=0.7)
    ap.add_argument("--tg-hi", type=float, default=1.3)
    ap.add_argument("--chirp-degree", type=int, default=8)
    ap.add_argument("--max-drag-iters", type=int, default=4)
    ap.add_argument("--chirp-max-passes", type=int, default=12,
                    help="passes for the chirp<->DRAG FIXED POINT [12]. A "
                         "converging point at strong drive can need 20-80; at 12 "
                         "those are reported as diverged. Invalidates a cached "
                         "column, since it can turn a failure into a fit")
    ap.add_argument("--contrast-min", type=float, default=0.35,
                    help="drop Rabi rows whose chevron contrast falls below this [0.35]")
    ap.add_argument("--leak-max", type=float, default=None,
                    help="drop Rabi rows leaking more than this. tune_up's default "
                         "0.35 is far too permissive for the shift-curve fit: a row "
                         "leaking 25%% is not measuring the Stark shift. Past the "
                         "leakage knee (|eta| ~ 0.85 on 4Gate4.5SNAIL) use ~0.05 and "
                         "watch extrapolation_ratio and quartic_fraction")
    ap.add_argument("--probe-shape", choices=("constant", "gate"), default="constant",
                    help="step 1's probe per column. 'gate' plays the configured "
                         "envelope at each PEAK |eta| instead of a flat pump: ~50x "
                         "less leakage and no extrapolation, at the cost of a moment "
                         "deconvolution. Needed for target_eta > 1")
    ap.add_argument("--moment-weighting", choices=MOMENT_WEIGHTINGS,
                    default="rabi",
                    help="deconvolution convention for --probe-shape gate; a 2x lever "
                         "on k2. 'rabi' [default] is the derived one; re-pin it with "
                         "'tune_up --cross-check-moments' at weak drive")
    ap.add_argument("--quartic-warn", type=float, default=0.25,
                    help="warn when the quartic term is this fraction of the quadratic "
                         "[0.25]; the guard that matters when the fit extrapolates")
    ap.add_argument("--jobs", type=int, default=0, help="0 = all cores")
    ap.add_argument("--column-workers", type=int, default=1,
                    help="solve this many COLUMNS at once, each with --jobs inside "
                         "[1]. tune_up step 4 (length_rabi) is single-threaded, so a "
                         "--jobs-only run idles most cores through it. Try "
                         "--column-workers 8 --jobs 8 on 72 cores.")
    ap.add_argument("--gpu", action="store_true",
                    help="qutip-jax/diffrax (forces --jobs 1). Usually a LOSS here: "
                         "this Hilbert space is ~10^2 states, far below the CPU/GPU "
                         "crossover")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--outdir", default=None,
                    help="per-column cache directory [results/wpscan_<device>]")
    ap.add_argument("--out", default=None, help="HDF5 scan file (or .json summary)")
    ap.add_argument("--plot", nargs="?", const="figs/wp_scan.png", default=None)
    ap.add_argument("--replot", metavar="FILE", default=None,
                    help="re-render the figure from a saved scan, no solving")
    ap.add_argument("--replot-columns", metavar="FILE", default=None,
                    help="render EVERY column's chevron map and chirp ridge from a "
                         "saved scan and embed them in it, no solving. Backfills a "
                         "file written before figures were stored, or re-renders one "
                         "after a plotting change")
    ap.add_argument("--ridge-grid", action="store_true",
                    help="measure every Rabi row on ONE shared offset axis so each "
                         "column's chirp RIDGE can be drawn (--wp-points grows "
                         "automatically). A different measurement from the default "
                         "per-row adaptive span, and cannot be added afterwards")
    ap.add_argument("--no-column-figures", action="store_true",
                    help="do not store each column's chevron/ridge figures in the "
                         "scan file")
    ap.add_argument("--chirp-free-fallback", action="store_true",
                    help="when the shift law cannot be fitted, calibrate a CHIRP-FREE "
                         "gate (delta0 sets the carrier) instead of discarding the "
                         "column. Excluding these near-zero-shift columns biases what "
                         "chirping appears to buy. GATED on --chirp-free-max-frac.")
    ap.add_argument("--chirp-free-max-frac", type=float, default=0.10,
                    help="largest chirp excursion, as a fraction of the resonance "
                         "half-width 1/(2 t_g), that --chirp-free-fallback may treat "
                         "as no chirp. Below 10%% the fit residual exceeds the "
                         "excursion (no signal); above 20%% the law is real. Columns "
                         "above the threshold RAISE so the measurement gets fixed.")
    ap.add_argument("--span-growth", choices=("railed", "both", "off"),
                    default="railed",
                    help="WHICH Rabi rows may re-measure on a wider pump-offset "
                         "window. 'railed' (default): only a row whose fitted centre "
                         "sits at the window edge. 'both' adds the too-wide trigger "
                         "(hwhm > 0.4*span, x3), measured to LOSE rows (a broad line "
                         "is not an unmeasured line). 'off' keeps a fixed window.")
    ap.add_argument("--max-compound-growths", type=int, default=1,
                    help="how many times ONE row may grow in a row (default 1); "
                         "rows that grew twice or more were rarely kept.")
    ap.add_argument("--max-span-growths", type=int, default=3,
                    help="how many times a Rabi row may GROW its pump-offset window "
                         "when the ridge rails against the edge (x2 each, offset count "
                         "grown to keep the step). 0 aborts the column instead. "
                         "Ignored under an explicit --wp-span-MHz.")
    ap.add_argument("--max-wp-points", type=int, default=121,
                    help="ceiling on the grown offset count per Rabi row (each "
                         "offset is a solve).")
    ap.add_argument("--zero-chirp-frac", type=float, default=0.0,
                    help="when the shift-law fit fails r2 BUT the chirp would sweep "
                         "less than this fraction of the resonance half-width "
                         "1/(2 t_g), proceed with ZERO chirp (the shift is absorbed "
                         "into delta0) instead of discarding the column. r2 is "
                         "relative, so a near-zero shift fails it even with pristine "
                         "chevrons. Keep it small (0.15 is generous). 0 means "
                         "DEFAULT TO --chirp-free-max-frac, not disabled.")
    ap.add_argument("--drag-decouple-fallback", action="store_true",
                    help="when the coupled chirp<->DRAG fixed point DIVERGES at a "
                         "column, fall back to decoupled DRAG there instead of losing "
                         "it; flagged by operating_point.drag_decoupled.")
    ap.add_argument("--decouple-drag", action="store_true",
                    help="build the chirp from the bare envelope and apply DRAG on "
                         "top (the first Picard iterate) instead of iterating to a "
                         "fixed point: cannot diverge, and the chirp is IDENTICAL to "
                         "the DRAG-off one. Leaves the quadrature's own Stark shift "
                         "uncancelled (see neglected_shift_frac).")
    ap.add_argument("--envelope-m", type=int, default=None,
                    help="override the sine_power vanishing order (else from "
                         "--max-drag-channels), e.g. so a NO-DRAG baseline plays the "
                         "same pulse shape as the DRAG run.")
    ap.add_argument("--shard", metavar="I/N", default=None,
                    help="run only this machine's share of the grid, e.g. --shard "
                         "0/4 (every N-th column, so cheap and expensive columns "
                         "interleave). Shards are disjoint and their HDF5 files "
                         "concatenate. Give each shard its own --out, and its own "
                         "--outdir unless the machines share a filesystem.")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore cached columns and re-solve")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the geometry, the cost and EVERY column's channel "
                         "audit, then stop. Run this first")
    ap.add_argument("--no-audit", action="store_true",
                    help="with --dry-run, skip the per-column audit (geometry only)")
    ap.add_argument("--open-system", type=int, nargs="?", const=3, default=None,
                    metavar="TOP",
                    help="after the scan, re-score the best TOP points (default 3) "
                         "with a REAL open-system solve (mesolve, 16 per point). "
                         "Needs --t1-us")
    ap.add_argument("--coupler-t1-us", type=float, default=None,
                    help="coupler/SNAIL loss for --open-system. Worth including near "
                         "a subharmonic, where the coupler carries real population")
    ap.add_argument("--log", default=None,
                    help="log file [<outdir>/wp_scan.log]")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    if args.replot_columns:
        # Needs no grid, device or drive: bail out before those checks.
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        attach_all_column_figures(args.replot_columns)
        return

    if args.replot:
        from snail_solver.h5_io import load_doc as _load_doc
        plot_wp_scan(_load_doc(args.replot), args.plot or "figs/wp_scan.png")
        return

    if (args.target_etas is None) == (args.target_eta is None):
        ap.error("pass exactly one of --target-etas or --target-eta")
    if args.target_etas is not None:
        from snail_solver.tune_up_sweep import parse_etas
        target_etas = parse_etas(args.target_etas)
    else:
        target_etas = [float(args.target_eta)]

    from snail_solver.h5_io import attach_figures, save_doc, save_tree
    from snail_solver.log_utils import setup_run_logger
    from snail_solver.paths import in_results, resolve_device

    if not args.device:
        ap.error("--device is required (except with --replot/--replot-columns)")

    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from snail_solver.device_utils import load_device
    device_path = resolve_device(args.device)
    config = load_device(device_path)
    offsets = parse_offsets(args.offsets)

    # fit_shift_curve needs >= 4 usable rows, and rows are DROPPED (contrast/leakage)
    # at the high drive this scan targets: refuse now, not after every Rabi table.
    if args.amp_points < 4:
        ap.error(f"--amp-points {args.amp_points} cannot fit a shift curve: it needs "
                 f">= 4 usable rows, and rows are dropped for low contrast or leakage. "
                 f"Use 5 or more (the requested eta 0.5..2.5 in 0.05 steps is 41).")
    if args.amp_points < 6:
        print(f"note: --amp-points {args.amp_points} leaves no margin -- the shift-curve "
              f"fit needs 4 usable rows and drops them for contrast/leakage.")

    shard, n_shards = 0, 1
    if args.shard:
        try:
            _a, _b = str(args.shard).split("/")
            shard, n_shards = int(_a), int(_b)
        except Exception:
            ap.error(f"--shard wants I/N (e.g. 0/4), got {args.shard!r}")
        if not (n_shards >= 1 and 0 <= shard < n_shards):
            ap.error(f"--shard {args.shard}: need 0 <= I < N and N >= 1")

    if args.dry_run:
        text = describe_grid(config, offsets, target_etas, branch=args.branch,
                             coupler_levels=args.coupler_levels,
                             max_drag_channels=args.max_drag_channels,
                             min_ratio=args.min_ratio, max_ratio=args.max_ratio,
                             eta_lo=args.eta_lo,
                             eta_hi=args.eta_hi, amp_points=args.amp_points,
                             wp_points=args.wp_points,
                             envelope_m=args.envelope_m,
                             drop_origin=not args.keep_origin,
                             audit=not args.no_audit)
        if text:                    # --no-audit returns the geometry unprinted
            print(text)
        if n_shards > 1:
            n_cols = len(columns_for(config, offsets,
                                     drop_origin=not args.keep_origin))
            mine = len(shard_columns(
                [(c, e) for c in range(n_cols) for e in target_etas],
                shard, n_shards))
            print(f"  shard       {shard}/{n_shards} would solve {mine} of "
                  f"{n_cols * len(target_etas)} columns (strided)")
        return

    stem = os.path.splitext(os.path.basename(device_path))[0]
    outdir = args.outdir or in_results(f"wpscan_{stem}")
    os.makedirs(outdir, exist_ok=True)
    # Logger name derived from the log FILE, so two scans never share handlers.
    log_path = args.log or os.path.join(outdir, "wp_scan.log")
    logger = setup_run_logger(log_path, f"wp_scan:{log_path}")

    out_path = in_results(args.out) if args.out else None
    holds_runs = bool(out_path) and not str(out_path).endswith(".json")
    attrs = {"tool": "snail_solver.subharmonic_gate_scan",
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "command": "python -m snail_solver.subharmonic_gate_scan "
                        + " ".join(shlex.quote(a) for a in sys.argv[1:]),
             "device_path": str(device_path), "host": platform.node()}
    if holds_runs:
        # Truncate ONCE, up front, so a rerun cannot inherit a previous scan's columns.
        save_tree(out_path, {}, attrs=attrs)

    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    doc = run_wp_scan(
        config, offsets, target_etas, device_path=device_path,
        branch=args.branch, coupler_levels=args.coupler_levels,
        eta_lo=args.eta_lo, eta_hi=args.eta_hi, amp_points=args.amp_points,
        max_drag_channels=args.max_drag_channels, min_ratio=args.min_ratio,
        max_ratio=args.max_ratio, drag_retries=args.drag_retries,
        contrast_min=args.contrast_min, quartic_warn=args.quartic_warn,
        leak_max=args.leak_max, probe_shape=args.probe_shape,
        moment_weighting=args.moment_weighting,
        column_figures=not args.no_column_figures,
        ridge_grid=args.ridge_grid,
        shard=shard, n_shards=n_shards, envelope_m=args.envelope_m,
        zero_chirp_frac=args.zero_chirp_frac,
        chirp_free_fallback=args.chirp_free_fallback,
        chirp_free_max_frac=args.chirp_free_max_frac,
        max_span_growths=args.max_span_growths,
        max_wp_points=args.max_wp_points,
        span_growth=args.span_growth,
        max_compound_growths=args.max_compound_growths,
        couple_drag=not args.decouple_drag,
        drag_decouple_fallback=args.drag_decouple_fallback,
        column_workers=args.column_workers,
        t1_us=args.t1_us, t2_us=args.t2_us,
        decoh_prefactor=args.decoh_prefactor,
        wp_points=args.wp_points, wp_span_MHz=args.wp_span_MHz,
        span_linewidths=args.span_linewidths, n_time=args.n_time,
        window_tg=args.window_tg, tg_points=args.tg_points,
        tg_lo=args.tg_lo, tg_hi=args.tg_hi, chirp_degree=args.chirp_degree,
        max_drag_iters=args.max_drag_iters,
        chirp_max_passes=args.chirp_max_passes,
        drop_origin=not args.keep_origin, outdir=outdir,
        sweep_path=(out_path if holds_runs else None),
        overwrite=args.overwrite, force=args.force, jobs=args.jobs,
        solver=solver, stop_on_error=args.stop_on_error, logger=logger)

    s = doc["summary"]
    print(f"\n{s['n_ok']}/{s['n_columns']} columns calibrated in {s['seconds']:.0f}s")
    if doc["by_target_eta"]:
        print(f"  {'eta*':>6} {'t_g(ns)':>8} {'best delta':>11} {'1-F_coh':>10} "
              f"{'eps_incoh':>10} {'1-F_total':>10}")
        for key, v in doc["by_target_eta"].items():
            inc = v["eps_incoherent"]
            tot = v["infidelity_total"]
            print(f"  {key:>6} {v['t_g_ns']:8.1f} {v['best_delta_GHz']:+11.4f} "
                  f"{v['infidelity_coherent']:10.3e} "
                  + (f"{inc:10.3e} " if inc is not None else f"{'--':>10} ")
                  + (f"{tot:10.3e}" if tot is not None else f"{'--':>10}"))
    if s["best_infidelity"] is not None:
        kind = ("1-F_total" if s["best_infidelity_total"] is not None else "1-F_coh")
        print(f"best overall: eta*={s['best_target_eta']:g} "
              f"delta={s['best_delta_GHz']:+.4f} GHz  "
              f"{kind}={s['best_infidelity']:.3e}")

    if args.open_system is not None:
        if args.t1_us is None:
            ap.error("--open-system needs --t1-us (and usually --t2-us)")
        # Save FIRST: the rescore can outlast the scan itself (once 2 days), and it
        # only ADDS an `open_system` block, so the summary must not wait on it.
        if out_path:
            pre = save_doc(out_path, doc, attrs=attrs,
                           group=("scan" if holds_runs else None))
            print(f"  written {pre} (before the open-system rescore)")
        doc = rescore_open_system(doc, t1_us=args.t1_us, t2_us=args.t2_us,
                                  coupler_t1_us=args.coupler_t1_us,
                                  top=args.open_system, solver=solver,
                                  logger=logger)

    figures: Dict[str, str] = {}
    if args.plot and s["n_ok"]:
        figures["wp_scan"] = plot_wp_scan(doc, args.plot)
    if out_path:
        written = save_doc(out_path, doc, attrs=attrs,
                           group=("scan" if holds_runs else None))
        print(f"  written {written}")
        if figures:
            attach_figures(written, figures)


if __name__ == "__main__":
    main()
