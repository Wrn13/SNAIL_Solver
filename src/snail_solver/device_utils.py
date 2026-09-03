"""
device_utils.py
===============

Shared device I/O, spectator-free gate construction, and a pure 1-D maximizer,
used by the calibration and Stark-resonance tools.

`load_device` merges a device JSON over run_sweep_zhou.DEFAULT_CONFIG. `build_coupler`
constructs the 3-mode (qubit a, qubit b, coupler) gate with the pump normalized to a
full iSWAP and scaled by amp_scale (anharmonicity included). `transfer_probability`
is the one-trajectory swap proxy used as a fast search objective. `maximize_1d` is a
deterministic grid+zoom optimizer (numpy only, unit-testable without QuTiP).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI: float = 2.0 * np.pi


def load_device(path: str) -> Dict[str, Any]:
    """Load a device JSON merged over run_sweep_zhou.DEFAULT_CONFIG.

    Parameters
    ----------
    path : str
        Path to the device JSON (same schema as run_sweep_zhou's --device file).

    Returns
    -------
    dict
        The merged configuration (device values override the defaults).
    """
    from snail_solver.run_sweep_zhou import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    with open(path) as f:
        config.update(json.load(f))
    return config


def parse_chirp_arg(text: Optional[str]) -> Optional[List[float]]:
    """Parse a ``--chirp-GHz`` CLI value into Legendre coefficients.

    Shared by every tool that takes the flag, so they cannot drift apart on what an
    empty string means.

    The distinction that matters is None vs ``[]``:

    * ``None`` (flag absent) -- "say nothing", leave whatever the device config or a
      resolved operating point supplies.
    * ``""`` (flag given, empty) -- an explicit "no chirp", which OVERRIDES a
      configured or saved chirp. Returns ``[]`` rather than None so a caller can tell
      the two apart and report the override instead of silently cancelling a
      calibrated chirp.

    Parameters
    ----------
    text : str or None
        Comma-separated coefficients of delta(t)/2pi in GHz, e.g. ``"0,0,-0.004"``.

    Returns
    -------
    list of float, or None
        None when `text` is None; otherwise the (possibly empty) coefficient list.
    """
    if text is None:
        return None
    return [float(x) for x in str(text).split(",") if x.strip()]


def describe_chirp(coeffs: Optional[Sequence[float]]) -> str:
    """One-line human description of a chirp, for CLI confirmation lines."""
    if not coeffs:
        return "none"
    lead = ", ".join(f"c{k}={c:+g}" for k, c in enumerate(coeffs) if c)
    return f"delta(t)/2pi = {list(map(float, coeffs))} GHz [{lead or 'all zero'}]"


def target_eta_area(g3_GHz: float, lam_a: float, lam_b: float) -> float:
    """Pulse area integral|eta|dt (ns) the analytic normalization targets for a full
    iSWAP: (pi/2) / (6 g3 lambda_a lambda_b), g3 in rad/ns.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.

    Returns
    -------
    float
        Required integral of |eta| over the gate (ns).
    """
    return (np.pi / 2) / (6 * (g3_GHz * TWO_PI) * lam_a * lam_b)


def auto_t_g(g3_GHz: float, lam_a: float, lam_b: float, target_eta: float) -> float:
    """Gate time (ns) for which a raised-cosine full-iSWAP pump has peak
    |eta| = target_eta. Hann window: integral|eta|dt = eta_peak * t_g/2, so
    t_g = 2 * area / target_eta.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.
    target_eta : float
        Desired peak |eta|.

    Returns
    -------
    float
        Gate duration (ns).
    """
    if target_eta <= 0.0:
        raise ValueError("target_eta must be positive.")
    return 2.0 * target_eta_area(g3_GHz, lam_a, lam_b) / target_eta


#: Smallest |Delta(t)| (GHz) a DRAG quadrature may reach before it is judged
#: singular. Mirrors ``sweep_common._drag_skip_GHz`` so the explicit and swept paths
#: agree on where DRAG stops being meaningful.
DRAG_FLOOR_GHz: float = 5e-4


def check_drag_detuning(tone, floor_GHz: float = DRAG_FLOOR_GHz) -> float:
    """Raise if a chirp drives the DRAG beat through (or near) zero mid-pulse.

    On a chirped tone ``Delta(t) = Delta_0 - k delta(t)`` can cross zero DURING the
    gate even when ``Delta_0`` is comfortably large -- the quadrature then diverges
    somewhere in the middle of the pulse, which is invisible if you only inspect
    ``delta_drag_GHz``. This is the check that turns that into an error.

    Explicit, single-point callers (`build_coupler`, tune-up, GRAPE) should let this
    raise. Sweeps should instead pre-check and DISABLE DRAG for the offending point
    (as they already do for a small static beat), so one bad point cannot kill a scan.

    Returns
    -------
    float
        ``min_t |Delta(t)|`` in GHz (``inf`` when DRAG is off).
    """
    floor_rad = float(min(floor_GHz, DRAG_FLOOR_GHz)) * TWO_PI
    floors = tone.drag_detuning_floors()
    got = min(floors) if floors else float("inf")
    if got < floor_rad:
        channels = tone.drag_channels_resolved()
        worst = int(np.argmin(floors))
        if len(channels) == 1:
            which = (f"Delta_0 = {float(channels[0].beat_GHz) * 1e3:.3f} MHz, "
                     f"drag_n_pump = {channels[0].n_pump}")
        else:
            # name the offender: with several channels the failing one is not
            # otherwise identifiable from the aggregate minimum
            which = (f"channel {worst + 1}/{len(channels)} "
                     f"(Delta_0 = {float(channels[worst].beat_GHz) * 1e3:.3f} MHz, "
                     f"n_pump = {channels[worst].n_pump}, "
                     f"n_photon = {channels[worst].n_photon}); all channels "
                     + ", ".join(f"{float(c.beat_GHz) * 1e3:+.1f}@k{c.n_pump}"
                                 f"->{f / TWO_PI * 1e3:.3f} MHz"
                                 for c, f in zip(channels, floors)))
        raise ValueError(
            f"DRAG beat passes through zero during the pulse: min|Delta(t)| = "
            f"{got / TWO_PI * 1e3:.4f} MHz < {floor_rad / TWO_PI * 1e3:.4f} MHz, with "
            f"{which}, chirp = "
            f"{None if tone.chirp is None else list(tone.chirp.coeffs_GHz)} GHz. "
            f"The chirp is sweeping the pump onto the process DRAG is meant to "
            f"suppress. Reduce the chirp, pick a further-detuned beat, or set "
            f"drag_n_pump=0 if that beat is genuinely pump-independent.")
    return got / TWO_PI


def drag_correction_ratio(tone, n: int = 257) -> float:
    """How large the DRAG correction is relative to the pulse it corrects.

    ``max_t |eta_corrected - eta_base| / max_t |eta_base|``, with the chirp phase
    excluded (it is a pure phase and would swamp the comparison).

    ``min|Delta(t)|`` is necessary but NOT sufficient for a recursive pulse. The
    perturbative ``F^(n)`` is only valid while ``|Omega'/(Omega Delta)| << 1`` at every
    level, and with several nestings that product can exceed 1 -- at which point the
    "correction" is larger than the pulse and the expansion has stopped meaning
    anything, even though every individual beat is comfortably far from zero. Nothing
    else in the codebase would surface that.

    Callers should WARN, not raise: like the rest of this pipeline, it reports.
    Returns 0.0 when DRAG is off.
    """
    channels = tone.drag_channels_resolved()
    if not channels:
        return 0.0
    from snail_solver import drag as _drag
    env = tone.envelope
    t_g = float(getattr(env, "t_g", 0.0)) or 1.0
    # interior samples: the envelope vanishes at the endpoints, where the ratio is
    # either 0/0 or (for a base shape that is too shallow) unbounded by construction
    ts = np.linspace(0.0, t_g, int(n) + 2)[1:-1]
    base = np.asarray(env.value_at(ts, np), dtype=complex)
    if tone.is_legacy_drag:
        corrected = base - 1j * np.asarray(env.deriv_at(ts, np)) / np.asarray(
            tone.drag_detuning(ts, np))
    else:
        order = _drag.required_order(channels)
        corrected = np.asarray(_drag.apply_drag(
            env.jet_at(ts, order, np),
            [tone.channel_detuning_jet(c, ts, order, np) for c in channels],
            channels, np), dtype=complex)
    denom = float(np.max(np.abs(base)))
    return float(np.max(np.abs(corrected - base)) / denom) if denom else float("inf")


def build_coupler(config: Dict[str, Any], t_g: float, amp_scale: float,
                  wp_offset_GHz: float, spec_abs_GHz: Optional[float] = None,
                  drag_beat_GHz: Optional[float] = None,
                  chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                  drag_n_pump: int = 1, drag_channels=None,
                  correction_warn: float = 0.3,
                  logger=None):
    """Build the (qubit a, qubit b, coupler[, spectator]) gate with the pump
    normalized to a full iSWAP and scaled by amp_scale (anharmonicity included).

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    amp_scale : float
        Multiplicative correction on the normalized pump amplitude.
    wp_offset_GHz : float
        Offset added to the pump frequency w_b - w_a (GHz).
    spec_abs_GHz : float, optional
        If given, add a 4th spectator mode at this ABSOLUTE frequency (participation
        lam_b, ``spec_levels`` levels, ``anharm_spec_GHz``) so the tune-up sees the
        spectator, i.e. a hardware-style per-point calibration. None -> bare (a, b) pair.
    drag_beat_GHz : float, optional
        If given, apply a DRAG quadrature tuned to this beat (GHz) on the pump, so the
        calibration matches a DRAG-on gate. None -> no DRAG.
    chirp_coeffs_GHz : sequence of float, optional
        Legendre coefficients of a time-dependent pump-frequency offset delta(t)
        (GHz), applied ON TOP of the constant `wp_offset_GHz`. None or all-zero
        leaves the tone un-chirped and the solver path unchanged. Defaults to
        ``config["chirp_coeffs_GHz"]``. See :class:`envelope.Chirp`.
    drag_n_pump : int, default 1
        Pump quanta carried by the process DRAG suppresses, which sets how the beat
        moves under a chirp: ``Delta(t) = drag_beat_GHz - drag_n_pump * delta(t)``.
        1 for a one-pump collision, 2 for a subharmonic one, 0 for a static
        (pump-independent) beat. Irrelevant without a chirp.
    drag_channels : sequence of DragChannel, optional
        Several processes to suppress at once, via recursive multi-derivative DRAG
        (see :mod:`snail_solver.drag`). OVERRIDES `drag_beat_GHz`/`drag_n_pump`,
        which remain the one-channel shorthand. None (default) leaves the tone on
        the historical first-order path.
    correction_warn : float, default 0.3
        Log a warning when :func:`drag_correction_ratio` exceeds this -- the
        perturbative expansion has stopped being small. Never raises.
    logger : logging.Logger, optional
        Where that warning goes; silent if omitted.

    Returns
    -------
    (ZhouCoupler, float, float)
        The coupler, its pump frequency w_p (GHz), and the resulting peak |eta|.

    Raises
    ------
    ValueError
        If a chirp drives the DRAG beat through zero during the pulse; see
        :func:`check_drag_detuning`.
    """
    from snail_solver.envelope import ENVELOPE_KINDS
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse, make_chirp

    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    ws = float(config["coupler_freq_GHz"])
    w_p_GHz = abs(wb - wa) + wp_offset_GHz
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    freqs = [wa, wb, ws]
    participations = {0: float(config["lam_a"]), 1: float(config["lam_b"])}
    anharm = {0: aq, 1: aq}
    if spec_abs_GHz is not None:                    # add the spectator as a 4th mode
        freqs.append(float(spec_abs_GHz))
        levels.append(int(config.get("spec_levels", 3)))
        participations[3] = float(config["lam_b"])              # spectator participation = lam_b
        anharm[3] = float(config.get("anharm_spec_GHz", 0.0))
    cpl = ZhouCoupler(mode_freqs_GHz=freqs, coupler_index=2,
                      participations=participations, nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz=anharm)
    EnvCls = ENVELOPE_KINDS.get(config["envelope"], ConstantPulse)
    if chirp_coeffs_GHz is None:
        chirp_coeffs_GHz = config.get("chirp_coeffs_GHz") or None
    env_kw = {}
    if EnvCls.__name__ == "SinePowerRamp":         # Eq. (13) shape parameters
        # The rise is a FRACTION of t_g, not a time in ns, so the envelope in
        # normalized gate time stays t_g-independent -- the property tune_up's
        # amplitude/length decoupling depends on. See `tune_up.area_factor`.
        env_kw = {"m": int(config.get("envelope_m", 3)),
                  "t_rise": float(config.get("envelope_rise_frac", 0.5)) * t_g}
    tone = PumpTone(w_p_GHz=w_p_GHz, envelope=EnvCls(amp=1.0, t_g=t_g, **env_kw),
                    is_eta=True,
                    drag=(drag_beat_GHz is not None),
                    delta_drag_GHz=(drag_beat_GHz if drag_beat_GHz is not None else 0.0),
                    chirp=make_chirp(chirp_coeffs_GHz, t_g),
                    drag_n_pump=int(drag_n_pump),
                    drag_channels=(list(drag_channels) if drag_channels else None))
    check_drag_detuning(tone)          # a chirp must not sweep the pump onto the beat
    # A far-from-zero beat is not on its own enough for a RECURSIVE pulse; see
    # drag_correction_ratio. Warn only -- this pipeline reports, it does not refuse.
    if logger is not None and not tone.is_legacy_drag and tone.drag_channels_resolved():
        ratio = drag_correction_ratio(tone)
        if ratio > float(correction_warn):
            logger.info(
                f"  WARNING: the DRAG correction is {100 * ratio:.0f}% of the pulse "
                f"it corrects (> {100 * correction_warn:.0f}%), over "
                f"{len(tone.drag_channels_resolved())} channels. The perturbative "
                f"expansion is no longer small, so the composed pulse is a guess, "
                f"not a correction -- use fewer channels or further-detuned beats.")
    cpl.set_pump(tone, normalize_iswap=(0, 1))
    cpl.scale_pump_amplitude(amp_scale)
    return cpl, w_p_GHz, cpl.peak_eta()


def transfer_probability(config: Dict[str, Any], t_g: float, amp_scale: float,
                         wp_offset_GHz: float, solver: Dict[str, Any],
                         spec_abs_GHz: Optional[float] = None,
                         drag_beat_GHz: Optional[float] = None,
                         chirp_coeffs_GHz: Optional[Sequence[float]] = None,
                         drag_n_pump: int = 1, drag_channels=None) -> float:
    """Single-shot swap probability P(|01> -> |10>) at t_g (QuTiP sesolve): a fast
    one-trajectory proxy for the rotation angle, used as a search objective.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    amp_scale : float
        Pump-amplitude correction to test.
    wp_offset_GHz : float
        Pump-frequency offset to test (GHz).
    solver : dict
        QuTiP integrator options (atol, rtol, nsteps).
    spec_abs_GHz : float, optional
        Spectator absolute frequency (GHz); adds the spectator mode (ground) to the
        Hilbert space so the probe sees it. None -> bare (a, b) pair.
    drag_beat_GHz : float, optional
        DRAG beat (GHz) for the probe pump. None -> no DRAG.
    chirp_coeffs_GHz : sequence of float, optional
        Pump chirp (GHz). Defaults to ``config["chirp_coeffs_GHz"]`` via
        `build_coupler`; pass it explicitly to probe a chirp the config does not
        carry. Without this the search objective would disagree with the gate it is
        calibrating.
    drag_n_pump : int, default 1
        Pump quanta carried by the process DRAG suppresses; sets how the beat moves
        under a chirp. See `build_coupler`.
    drag_channels : sequence of DragChannel, optional
        Several processes to suppress at once, via recursive multi-derivative DRAG.
        OVERRIDES `drag_beat_GHz`/`drag_n_pump`, which remain the one-channel
        shorthand. Forwarded verbatim to `build_coupler`, so a search objective and
        the gate it calibrates see the SAME pulse -- the whole point of this
        function. Omitting it was a real bug: `tune_up.length_rabi` passes it, so
        the length fit raised TypeError on every tune-up.

    Returns
    -------
    float
        P(|10>) starting from |01>, with any spectator left in its ground state.
    """
    cpl, _w_p, _eta = build_coupler(config, t_g, amp_scale, wp_offset_GHz,
                                    spec_abs_GHz, drag_beat_GHz,
                                    chirp_coeffs_GHz=chirp_coeffs_GHz,
                                    drag_n_pump=drag_n_pump,
                                    drag_channels=drag_channels)
    tail = [0] if spec_abs_GHz is not None else []       # spectator stays in |0>
    state = cpl.evolve_state([1, 0, 0] + tail, t_g, **solver)
    return float(np.abs(state[cpl.fock_index([0, 1, 0] + tail)]) ** 2)


def maximize_1d(func: Callable[[float], float], lo: float, hi: float,
                n_points: int = 7, n_refine: int = 2) -> Tuple[float, float, int]:
    """Maximize a unimodal `func` on [lo, hi] by a coarse grid plus successive
    zoom-ins around the best point. Deterministic and cache-backed.

    Parameters
    ----------
    func : callable(float) -> float
        Objective to MAXIMIZE.
    lo, hi : float
        Search bounds.
    n_points : int, default 7
        Grid points per refinement round.
    n_refine : int, default 2
        Zoom-in rounds after the initial grid.

    Returns
    -------
    (float, float, int)
        Best x, best func(x), and the number of distinct evaluations.
    """
    cache: Dict[float, float] = {}

    def evaluate(x: float) -> float:
        key = round(x, 10)
        if key not in cache:
            cache[key] = func(x)
        return cache[key]

    best_x, best_f = lo, -np.inf
    for _ in range(n_refine + 1):
        grid = np.linspace(lo, hi, n_points)
        for x in grid:
            value = evaluate(float(x))
            if value > best_f:
                best_f, best_x = value, float(x)
        step = (hi - lo) / (n_points - 1)
        lo, hi = best_x - step, best_x + step          # zoom to +/- one step
    return best_x, best_f, len(cache)