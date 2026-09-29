"""Spectator sweep: fixed a-b pair, a single spectator swept in frequency.

build_grid() builds the Delta = w_b - w_spec grid; run_spectator_point()
evaluates one point (analytic collision + full iSWAP fidelity).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

from snail_solver.sweep_common import (
    Point, TWO_PI, _GRAPE_BLANKS, _calibrate_point, _chirp_of, _drag_beat_if_ok,
    _drag_ok_with_chirp, _drag_skip_GHz, _grape_augment, _nearest_collision,
    _nonlinearities, _pump_quanta_of, _solver_opts, _stark_offset_GHz,
)


def build_grid(specfreqs: Sequence[float], drags: Sequence[bool]) -> List[Point]:
    """Cartesian grid over spectator detuning Delta = w_b - w_spec (GHz) x DRAG flag,
    ordered spec_freq -> drag and indexed 0..M-1."""
    pairs = [(sf, d) for sf in specfreqs for d in drags]
    return [Point(index=i, spec_freq_GHz=float(sf), drag=bool(d))
            for i, (sf, d) in enumerate(pairs)]

def run_spectator_point(pt: Point, config: Dict[str, Any]) -> Dict[str, Any]:
    """One spectator-sweep point: analytic collision prediction always, plus the full
    QuTiP gate simulation when ``config['integrate']``.

    Returns the result row: analytic fields (beat, eta_peak, effective rates, ...) and,
    if integrated, F_avg / leakage / occupations / the 4x4 ``U_proj``.
    """
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    t0 = time.time()
    a, b, coupler, spec = 0, 1, 2, 3
    anchor = int(config["anchor"])
    integrate = config.get("integrate", True)

    # frequencies (rad/ns); only the spectator moves: w_spec = w_b - Delta
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float) * TWO_PI)
    ws = config["coupler_freq_GHz"] * TWO_PI
    w_p = abs(wb - wa)                              # iSWAP pump = |detuning| (fixed)
    w_spec = wb - pt.spec_freq_GHz * TWO_PI
    wa_GHz, wb_GHz, ws_GHz = wa / TWO_PI, wb / TWO_PI, ws / TWO_PI
    wspec_GHz = w_spec / TWO_PI
    w_p_nom_GHz = w_p / TWO_PI

    # Per-point calibration (integrated run only), spectator loaded at w_spec:
    #   calibrate_points -> amplitude + Stark tune-up with DRAG on if this point uses
    #                       it; sets amp_scale AND wp_offset for this point.
    #   stark_drive      -> frequency only: shift w_p to the spectator-aware Stark
    #                       resonance (with stark_match_pulse: the DRAG-on resonance).
    # DRAG beats come from the pre-Stark pump (the ~MHz shift is negligible there).
    amp_scale_used = float(config.get("amp_scale", 1.0))
    wp_offset_used_GHz = float(config.get("wp_offset_GHz", 0.0))
    stark_offset_GHz = 0.0
    _chevron = None
    if bool(integrate) and bool(config.get("calibrate_points", False)):
        _cn = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_nom_GHz)
        _cb, _ck = _cn[1], _pump_quanta_of(_cn[2])
        rec = _calibrate_point(config, wa_GHz, wb_GHz, wspec_GHz,
                               _drag_beat_if_ok(config, bool(pt.drag), _cb, _ck), _ck)
        amp_scale_used = float(rec["amp_scale"])
        wp_offset_used_GHz = float(rec["wp_offset_GHz"])          # measured from nominal
    elif bool(integrate) and bool(config.get("stark_drive", False)):
        _cn = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz,
                                 w_p_nom_GHz + wp_offset_used_GHz)
        _cb, _ck = _cn[1], _pump_quanta_of(_cn[2])
        _chevron = _stark_offset_GHz(config, wa_GHz, wb_GHz,
                                     float(config["t_g_ns"]), amp_scale_used,
                                     _solver_opts(config), spec_abs_GHz=wspec_GHz,
                                     drag_beat_GHz=_drag_beat_if_ok(config, pt.drag, _cb, _ck),
                                     drag_n_pump=_ck)
        stark_offset_GHz = float(_chevron["resonance_offset_GHz"])
        wp_offset_used_GHz += stark_offset_GHz
    w_p_GHz = w_p_nom_GHz + wp_offset_used_GHz               # calibrated / configured pump

    # spectator: a single 3-level anharmonic transmon with participation lam_b
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"]), int(config["spec_levels"])]
    aq = float(config.get("anharm_qubit_GHz", 0.0))
    lam_b = float(config["lam_b"])
    cpl = ZhouCoupler(
        mode_freqs_GHz=[wa_GHz, wb_GHz, ws_GHz, wspec_GHz],
        coupler_index=coupler,
        participations={a: float(config["lam_a"]), b: lam_b, spec: lam_b},
        nonlinearities=_nonlinearities(config),
        levels=levels,
        anharmonicities_GHz={a: aq, b: aq, spec: float(config.get("anharm_spec_GHz", 0.0))},
    )

    # DRAG tunes to the NEAREST collision across channels (as in the target sweep), so
    # the spectator may sit below OR above w_b. Singular as beat -> 0 (needs allocation,
    # not DRAG), so it is disabled -- not raised -- there or when a chirp crosses it.
    _nearest = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_GHz)
    beat_GHz = _nearest[1]
    drag_n_pump = _pump_quanta_of(_nearest[2])     # chirp: Delta(t) = beat - k*delta(t)
    use_drag = bool(pt.drag)
    status_drag = "ok"
    if pt.drag and not _drag_ok_with_chirp(config, beat_GHz, drag_n_pump,
                                           _chirp_of(config), float(config["t_g_ns"])):
        use_drag = False
        status_drag = ("drag_skipped_resonant_spectator"
                       if abs(beat_GHz) < _drag_skip_GHz(config)
                       else "drag_skipped_chirp_crosses_beat")

    # pump at w_b - w_a, amplitude normalized to a full iSWAP on (a,b)
    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    from snail_solver.zhou_coupler import make_chirp
    env = EnvCls(amp=1.0, t_g=float(config["t_g_ns"]))
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=env, is_eta=True,
                          drag=use_drag,
                          delta_drag_GHz=(beat_GHz if use_drag else None),
                          chirp=make_chirp(_chirp_of(config), float(config["t_g_ns"])),
                          drag_n_pump=drag_n_pump),
                 normalize_iswap=(a, b))
    cpl.scale_pump_amplitude(amp_scale_used)       # 1.0 = raw analytic pi/2 normalization

    # ANALYTIC collision prediction (Eq. 62). For a single-tone pure-g3 coupler the
    # only in-band collision is the one-pump exchange with the anchor.
    eta_peak = cpl.peak_eta()
    g_iswap = cpl.iswap_rate(a, b)                  # 6 g3 la lb |eta|
    g_spec = cpl.effective_rate([anchor, spec], n=3, C=6)   # 6 g3 l_anchor l_spec |eta|

    out = {
        "index": pt.index,
        "spec_freq_GHz": pt.spec_freq_GHz,
        "lam_spec": round(lam_b, 4),           # spectator participation (= lam_b)
        "drag": bool(pt.drag),
        "drag_applied": bool(use_drag),
        "beat_GHz": round(float(beat_GHz), 6),
        "nearest_kind": _nearest[2],
        "nearest_target": _nearest[3],
        "eta_peak": round(float(eta_peak), 5),
        "g_iswap_eff_MHz": round(float(g_iswap / TWO_PI * 1e3), 4),
        "g_spec_eff_MHz": round(float(g_spec / TWO_PI * 1e3), 4),
        "w_p_GHz": round(float(w_p_GHz), 6),
        "stark_offset_MHz": round(float(stark_offset_GHz) * 1e3, 4),
        "amp_scale_used": round(float(amp_scale_used), 5),
        "wp_offset_used_MHz": round(float(wp_offset_used_GHz) * 1e3, 4),
        "w_spec_GHz": round(float(wspec_GHz), 6),
        "t_g_ns": float(config["t_g_ns"]),
        "status": status_drag if status_drag != "ok" else "analytic",
        "F_avg": "", "leakage": "", "n_spec": "", "n_coupler": "", "p_transfer": "",
        **_GRAPE_BLANKS,
        "U_proj": None,
    }
    if _chevron is not None:
        out["_chevron"] = _chevron          # persisted by save_point, plotted by mode=chevrons

    if not integrate:
        out["wall_s"] = time.time() - t0
        return out

    # FULL non-perturbative evolution (QuTiP, exact Hamiltonian)
    t_g = float(config["t_g_ns"])
    solver = _solver_opts(config)

    # spectator diagnostic: excite qubit a, see where population ends up
    pops = np.abs(cpl.evolve_state([1, 0, 0, 0], t_g, **solver)) ** 2
    out["n_spec"] = float(cpl.mean_occupation(pops, spec))
    out["n_coupler"] = float(cpl.mean_occupation(pops, coupler))
    out["p_transfer"] = float(pops[cpl.fock_index([0, 1, 0, 0])])

    # leakage-aware iSWAP fidelity on the target pair (4 trajectories)
    F_avg, leakage, U_proj = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True,
                                                **solver)
    out["F_avg"] = float(F_avg)
    out["leakage"] = float(leakage)
    out["U_proj"] = U_proj
    out["status"] = status_drag        # "ok", or the drag-skip note if it fired

    if config.get("grape"):            # baseline = the applied gate
        _grape_augment(out, cpl, a, b, config,
                       drag_beat_GHz=(beat_GHz if use_drag else None),
                       nearest_beat_GHz=beat_GHz)

    out["wall_s"] = time.time() - t0
    return out
