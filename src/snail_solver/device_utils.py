"""Shared device I/O, gate construction and a 1-D maximizer for the calibration tools.

`load_device` merges a device JSON over run_sweep_zhou.DEFAULT_CONFIG; `build_coupler`
builds the (a, b, coupler[, spectator]) gate with the pump normalized to a full iSWAP;
`transfer_probability` is the one-trajectory swap proxy used as a search objective;
`maximize_1d` is a deterministic grid+zoom optimizer (numpy only).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI: float = 2.0 * np.pi


def load_device(path: str) -> Dict[str, Any]:
    """Load a device JSON (run_sweep_zhou --device schema) merged over DEFAULT_CONFIG."""
    from snail_solver.run_sweep_zhou import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    with open(path) as f:
        config.update(json.load(f))
    return config


def parse_chirp_arg(text: Optional[str]) -> Optional[List[float]]:
    """Parse a ``--chirp-GHz`` value (delta(t)/2pi Legendre coefficients in GHz,
    e.g. ``"0,0,-0.004"``), shared by every tool that takes the flag.

    None (flag absent) -> None: keep whatever the config/operating point supplies.
    ``""`` (flag given, empty) -> ``[]``: an explicit "no chirp" that OVERRIDES a
    configured chirp, distinguishable from None so callers can report the override.
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
    """Pulse area integral|eta|dt (ns) for a full iSWAP: (pi/2) / (6 g3 lam_a lam_b),
    with g3 (given in GHz) converted to rad/ns."""
    return (np.pi / 2) / (6 * (g3_GHz * TWO_PI) * lam_a * lam_b)


def auto_t_g(g3_GHz: float, lam_a: float, lam_b: float, target_eta: float) -> float:
    """Gate time (ns) at which a raised-cosine full-iSWAP pump peaks at |eta| =
    target_eta. Hann window: area = eta_peak * t_g/2, so t_g = 2 * area / target_eta."""
    if target_eta <= 0.0:
        raise ValueError("target_eta must be positive.")
    return 2.0 * target_eta_area(g3_GHz, lam_a, lam_b) / target_eta


#: Smallest |Delta(t)| (GHz) a DRAG quadrature may reach before it is judged singular;
#: mirrors ``sweep_common._drag_skip_GHz`` so explicit and swept paths agree.
DRAG_FLOOR_GHz: float = 5e-4


def carrier_shifted(drag_beat_GHz: Optional[float], drag_n_pump: int,
                    drag_channels: Optional[Sequence[Any]],
                    carrier_offset_GHz: float) -> Tuple[Optional[float], Optional[list]]:
    """DRAG beats moved by the pump's CARRIER offset: ``Delta_0 - n_pump * offset``.

    The beats are audited at the nominal pump; the gate plays it at
    ``nominal + carrier_offset_GHz`` with the zero-mean chirp on top, so a process
    carrying ``n_pump`` pump quanta beats at ``Delta_0 - n_pump (offset + delta(t))``.
    ``n_pump = 0`` (pump-independent) beats do not move. Returns the legacy
    single-channel beat and the channel list, each ``None`` when absent.
    """
    import dataclasses
    off = float(carrier_offset_GHz or 0.0)
    beat = (None if drag_beat_GHz is None
            else float(drag_beat_GHz) - int(drag_n_pump) * off)
    chans = (None if not drag_channels else
             [dataclasses.replace(c, beat_GHz=float(c.beat_GHz) - int(c.n_pump) * off)
              for c in drag_channels])
    return beat, chans


def check_drag_detuning(tone, floor_GHz: float = DRAG_FLOOR_GHz) -> float:
    """Raise ValueError if a chirp drives a DRAG beat through (or near) zero mid-pulse.

    On a chirped tone ``Delta(t) = Delta_0 - k delta(t)`` can cross zero DURING the
    gate even when ``Delta_0`` is large, so the quadrature diverges mid-pulse.
    Single-point callers (`build_coupler`, tune-up, GRAPE) let this raise; sweeps
    pre-check and disable DRAG for the offending point instead.

    Returns ``min_t |Delta(t)|`` in GHz (``inf`` when DRAG is off).
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
        else:                                       # name the offending channel
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
    """``max_t |eta_corrected - eta_base| / max_t |eta_base|`` (chirp phase excluded).

    ``min|Delta(t)|`` is necessary but not sufficient for a recursive pulse: the
    perturbative ``F^(n)`` needs ``|Omega'/(Omega Delta)| << 1`` at every level, and
    with several nestings the "correction" can exceed the pulse even when every beat
    is far from zero. Callers should warn, not raise. Returns 0.0 when DRAG is off.
    """
    channels = tone.drag_channels_resolved()
    if not channels:
        return 0.0
    from snail_solver import drag as _drag
    env = tone.envelope
    t_g = float(getattr(env, "t_g", 0.0)) or 1.0
    # interior samples only: at the endpoints the envelope vanishes and the ratio is 0/0
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
    t_g, amp_scale, wp_offset_GHz : float
        Gate duration (ns), pump-amplitude correction, and offset added to the pump
        frequency |w_b - w_a| (GHz).
    spec_abs_GHz : float, optional
        Add a 4th spectator mode at this ABSOLUTE frequency (participation lam_b,
        ``spec_levels`` levels, ``anharm_spec_GHz``). None -> bare (a, b) pair.
    drag_beat_GHz : float, optional
        Apply a DRAG quadrature tuned to this beat (GHz). None -> no DRAG.
    chirp_coeffs_GHz : sequence of float, optional
        Legendre coefficients of delta(t) (GHz), on top of `wp_offset_GHz`. Defaults
        to ``config["chirp_coeffs_GHz"]``; None/all-zero leaves the tone un-chirped.
    drag_n_pump : int, default 1
        Pump quanta of the suppressed process: ``Delta(t) = drag_beat_GHz -
        drag_n_pump * delta(t)`` (1 one-pump, 2 subharmonic, 0 static beat).
    drag_channels : sequence of DragChannel, optional
        Recursive multi-channel DRAG (:mod:`snail_solver.drag`); OVERRIDES
        `drag_beat_GHz`/`drag_n_pump`. None keeps the first-order path.
    correction_warn : float, default 0.3
        Log a warning via `logger` (if given) when :func:`drag_correction_ratio`
        exceeds this. Never raises.

    Returns
    -------
    (ZhouCoupler, float, float)
        The coupler, its pump frequency w_p (GHz), and the resulting peak |eta|.
        Raises ValueError if a chirp drives the DRAG beat through zero
        (:func:`check_drag_detuning`).
    """
    from snail_solver.envelope import ENVELOPE_KINDS
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, ConstantPulse, make_chirp

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
        # the rise is a FRACTION of t_g, keeping the normalized envelope t_g-independent
        # (tune_up's amplitude/length decoupling relies on it; see `tune_up.area_factor`)
        env_kw = {"m": int(config.get("envelope_m", 3)),
                  "t_rise": float(config.get("envelope_rise_frac", 0.5)) * t_g}
    # DRAG beats follow the carrier (they were audited at the nominal pump)
    drag_beat_GHz, drag_channels = carrier_shifted(drag_beat_GHz, drag_n_pump,
                                                   drag_channels, wp_offset_GHz)
    tone = PumpTone(w_p_GHz=w_p_GHz, envelope=EnvCls(amp=1.0, t_g=t_g, **env_kw),
                    is_eta=True,
                    drag=(drag_beat_GHz is not None),
                    delta_drag_GHz=(drag_beat_GHz if drag_beat_GHz is not None else 0.0),
                    chirp=make_chirp(chirp_coeffs_GHz, t_g),
                    drag_n_pump=int(drag_n_pump),
                    drag_channels=(list(drag_channels) if drag_channels else None))
    check_drag_detuning(tone)          # a chirp must not sweep the pump onto the beat
    # a recursive pulse can still be non-perturbative: warn (drag_correction_ratio)
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
    """Single-shot swap probability P(|01> -> |10>) at t_g (QuTiP sesolve), a fast
    one-trajectory proxy for the rotation angle used as a search objective.

    `solver` holds QuTiP integrator options (atol, rtol, nsteps). Every pulse argument
    is forwarded verbatim to `build_coupler`, so the objective sees the same pulse as
    the gate it calibrates. A spectator, if any, starts and is read in |0>.
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
    """Maximize a unimodal `func` on [lo, hi]: an `n_points` grid, then `n_refine`
    zoom-ins to +/- one step around the best point. Deterministic and cache-backed.

    Returns (best x, best func(x), number of distinct evaluations).
    """
    cache: Dict[float, float] = {}

    def evaluate(x: float) -> float:
        key = round(x, 10)
        if key not in cache:
            cache[key] = func(x)
        return cache[key]

    best_x, best_f = lo, -np.inf
    for _ in range(n_refine + 1):
        for x in np.linspace(lo, hi, n_points):
            value = evaluate(float(x))
            if value > best_f:
                best_f, best_x = value, float(x)
        step = (hi - lo) / (n_points - 1)
        lo, hi = best_x - step, best_x + step          # zoom to +/- one step
    return best_x, best_f, len(cache)
