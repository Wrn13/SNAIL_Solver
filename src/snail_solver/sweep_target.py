"""Target (frequency-allocation) sweep: fixed w_a and coupler; w_b and a
spectator swept. build_target_grid() builds the grid; _run_target_point()
evaluates one point. Supports the bare-gate no_spectator mode.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

from snail_solver.sweep_common import (
    Point, TWO_PI, _GRAPE_BLANKS, _calibrate_point, _chirp_of, _drag_beat_if_ok,
    _drag_skip_GHz, _grape_augment, _nearest_collision, _nonlinearities,
    _pump_quanta_of, _solver_opts, _stark_offset_GHz,
)


def build_target_grid(wa_GHz: float, wb_list: Sequence[float], spec_list: Sequence[float],
                      drags: Sequence[bool],
                      min_detuning_GHz: float = 0.05) -> List[Point]:
    """Allocation grid: fixed w_a (and w_s), Cartesian w_b x w_spec (ABSOLUTE) x drag,
    ordered w_b -> w_spec -> drag and indexed 0..M-1. Placements with
    |w_b - w_a| < ``min_detuning_GHz`` (pump too slow) are dropped; raises ValueError
    if nothing is left."""
    points: List[Point] = []
    for wb in wb_list:
        if abs(float(wb) - float(wa_GHz)) < float(min_detuning_GHz):
            continue
        for spec in spec_list:
            for drag in drags:
                points.append(Point(index=len(points), kind="target",
                                    wa_GHz=float(wa_GHz), wb_GHz=float(wb),
                                    spec_abs_GHz=float(spec), drag=bool(drag)))
    if not points:
        raise ValueError("empty target grid; check --wb-GHz/--spec-GHz and min_detuning_GHz.")
    return points

def _run_target_point(pt: Point, config: Dict[str, Any]) -> Dict[str, Any]:
    """Allocation point: fixed w_a (``qubit_freqs_GHz[0]``) and w_s
    (``coupler_freq_GHz``); partner at ``pt.wb_GHz`` and one spectator (participation
    ``lam_b``) at absolute ``pt.spec_abs_GHz``. Analytic collision search always; full
    iSWAP fidelity when ``config['integrate']``. Same row contract as `run_point`.
    """
    from snail_solver.zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    t0 = time.time()
    a, b, coupler, spec = 0, 1, 2, 3
    wa_GHz = float(config["qubit_freqs_GHz"][0])       # fixed
    ws_GHz = float(config["coupler_freq_GHz"])         # fixed SNAIL
    wb_GHz = float(pt.wb_GHz)                           # swept
    wspec_GHz = float(pt.spec_abs_GHz)                  # swept (absolute)
    w_p_GHz = abs(wb_GHz - wa_GHz) + float(config.get("wp_offset_GHz", 0.0))
    no_spec = bool(config.get("no_spectator", False))  # true 3-mode bare a-b-coupler gate
    lam_spec = 0.0 if no_spec else float(config["lam_b"])   # spectator participation
    spec_abs = None if no_spec else wspec_GHz
    integrate = bool(config.get("integrate", True))
    drag_always = bool(config.get("drag_always", False))

    # Per-point calibration (w_b and the spectator move the optimum point to point);
    # integrated run only:
    #   calibrate_points -> amplitude + Stark tune-up (calibrate_gate) with the
    #                       spectator loaded and DRAG on if the point runs it;
    #   stark_drive      -> frequency only: shift w_p to the Stark resonance
    #                       (DRAG-matched chevron only with drag_always).
    amp_scale_used = float(config.get("amp_scale", 1.0))
    wp_offset_used_GHz = float(config.get("wp_offset_GHz", 0.0))
    _chevron = None
    if integrate and bool(config.get("calibrate_points", False)):
        _cn = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz,
                                 abs(wb_GHz - wa_GHz))
        _cb, _ck = _cn[1], _pump_quanta_of(_cn[2])
        _use_drag = drag_always or bool(pt.drag)
        rec = _calibrate_point(config, wa_GHz, wb_GHz, spec_abs,
                               _drag_beat_if_ok(config, _use_drag, _cb, _ck), _ck)
        amp_scale_used = float(rec["amp_scale"])
        wp_offset_used_GHz = float(rec["wp_offset_GHz"])
        w_p_GHz = abs(wb_GHz - wa_GHz) + wp_offset_used_GHz
    elif integrate and bool(config.get("stark_drive", False)):
        _cn = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_GHz)
        _cb, _ck = _cn[1], _pump_quanta_of(_cn[2])
        _chevron = _stark_offset_GHz(config, wa_GHz, wb_GHz,
                                     float(config["t_g_ns"]), amp_scale_used,
                                     _solver_opts(config), spec_abs_GHz=spec_abs,
                                     drag_beat_GHz=_drag_beat_if_ok(config, drag_always,
                                                                    _cb, _ck),
                                     drag_n_pump=_ck)
        wp_offset_used_GHz += float(_chevron["resonance_offset_GHz"])
        w_p_GHz = abs(wb_GHz - wa_GHz) + wp_offset_used_GHz

    # Rates/eta are level-independent, so the analytic-only build uses 2 levels.
    if integrate:
        q_lv, c_lv = int(config["qubit_levels"]), int(config["coupler_levels"])
        s_lv = int(config["spec_levels"])
    else:
        q_lv = c_lv = s_lv = 2

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    if no_spec:                        # true 3-mode bare gate [a, b, coupler]
        freqs_GHz = [wa_GHz, wb_GHz, ws_GHz]
        participations = {a: float(config["lam_a"]), b: float(config["lam_b"])}
        levels = [q_lv, q_lv, c_lv]
        anharm = {a: aq, b: aq}
    else:
        freqs_GHz = [wa_GHz, wb_GHz, ws_GHz, wspec_GHz]
        participations = {a: float(config["lam_a"]), b: float(config["lam_b"]),
                          spec: lam_spec}
        levels = [q_lv, q_lv, c_lv, s_lv]
        anharm = {a: aq, b: aq, spec: float(config.get("anharm_spec_GHz", 0.0))}

    cpl = ZhouCoupler(mode_freqs_GHz=freqs_GHz, coupler_index=coupler,
                      participations=participations,
                      nonlinearities=_nonlinearities(config), levels=levels,
                      anharmonicities_GHz=anharm)

    # nearest collision (spectator vs {a, b}, subharmonics), vs the calibrated pump
    nearest = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_GHz)
    drag_beat = nearest[1]
    drag_n_pump = _pump_quanta_of(nearest[2])   # chirp: Delta(t) = beat - k*delta(t)
    # DRAG-compare window (off-resonant but close); |beat| rounded to 1 kHz so
    # threshold placements classify deterministically.
    beat_abs = round(abs(drag_beat), 6)
    thr_GHz = float(config.get("drag_compare_below_MHz", 100.0)) / 1000.0
    drag_compare = bool(config.get("drag_compare", False))
    in_window = (_drag_skip_GHz(config) < beat_abs < thr_GHz)

    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse

    def _configure(use_drag: bool) -> None:
        """(Re)set the pump from scratch -- fresh unit envelope (normalization not
        cumulative) and the configured chirp -- with DRAG on/off at the nearest beat."""
        from snail_solver.zhou_coupler import make_chirp
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz,
                              envelope=EnvCls(amp=1.0, t_g=float(config["t_g_ns"])),
                              is_eta=True, drag=use_drag,
                              delta_drag_GHz=(drag_beat if use_drag else None),
                              chirp=make_chirp(_chirp_of(config), float(config["t_g_ns"])),
                              drag_n_pump=drag_n_pump),
                     normalize_iswap=(a, b))
        cpl.scale_pump_amplitude(amp_scale_used)

    # Baseline pump: DRAG-off under --drag-compare (DRAG-on is added in the window);
    # otherwise the point's own flag, skipped on-resonance (needs allocation, not DRAG).
    if drag_compare:
        base_drag = False
        status_drag = "drag_compare" if in_window else "ok"
    else:
        base_drag = bool(pt.drag) or drag_always
        status_drag = "ok"
        if base_drag and abs(drag_beat) < _drag_skip_GHz(config):
            base_drag = False
            status_drag = "drag_skipped_resonant_collision"

    _configure(base_drag)
    eta_peak = cpl.peak_eta()
    g_iswap = cpl.iswap_rate(a, b)
    if no_spec or nearest[2] == "subharm":
        g_coll = float("nan")     # the cubic pair rate models neither case
    else:
        g_coll = cpl.effective_rate([nearest[4], spec], n=3, C=6)   # 6 g3 l_q l_spec |eta|

    out = {
        "index": pt.index, "kind": "target",
        "wa_GHz": round(wa_GHz, 6), "wb_GHz": round(wb_GHz, 6),
        "w_snail_GHz": round(ws_GHz, 6),
        "spec_GHz": ("" if no_spec else round(wspec_GHz, 6)),
        "detuning_GHz": round(wb_GHz - wa_GHz, 6), "w_p_GHz": round(w_p_GHz, 6),
        "stark_offset_MHz": round((wp_offset_used_GHz
                                   - float(config.get("wp_offset_GHz", 0.0))) * 1e3, 4),
        "amp_scale_used": round(float(amp_scale_used), 5),
        "wp_offset_used_MHz": round(float(wp_offset_used_GHz) * 1e3, 4),
        "lam_spec": round(lam_spec, 4),
        "drag": bool(pt.drag), "drag_applied": bool(base_drag),
        "drag_compare_window": bool(drag_compare and in_window),
        "eta_peak": round(float(eta_peak), 5),
        "g_iswap_eff_MHz": round(float(g_iswap / TWO_PI * 1e3), 4),
        "nearest_beat_GHz": round(nearest[1], 6),
        "nearest_kind": nearest[2],
        "nearest_target": nearest[3],
        "g_collision_MHz": round(float(g_coll / TWO_PI * 1e3), 4),
        "t_g_ns": float(config["t_g_ns"]),
        "status": status_drag if status_drag != "ok" else "analytic",
        "F_avg": "", "leakage": "", "n_spec": "", "n_coupler": "", "p_transfer": "",
        "F_avg_drag": "", "leakage_drag": "", "dF_drag": "",
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

    n_modes = 3 if no_spec else 4
    occ0 = [0] * n_modes; occ0[a] = 1
    pops = np.abs(cpl.evolve_state(occ0, t_g, **solver)) ** 2
    occ_b = [0] * n_modes; occ_b[b] = 1
    out["n_spec"] = "" if no_spec else float(cpl.mean_occupation(pops, spec))
    out["n_coupler"] = float(cpl.mean_occupation(pops, coupler))
    out["p_transfer"] = float(pops[cpl.fock_index(occ_b)])

    F_avg, leakage, U_proj = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True, **solver)
    out["F_avg"] = float(F_avg)
    out["leakage"] = float(leakage)
    out["U_proj"] = U_proj
    out["status"] = status_drag

    # DRAG comparison: rerun the SAME placement with DRAG on, in the window only
    if drag_compare and in_window:
        _configure(True)
        F_d, leak_d, U_d = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True, **solver)
        out["F_avg_drag"] = float(F_d)
        out["leakage_drag"] = float(leak_d)
        out["dF_drag"] = float(F_d - F_avg)
        if F_d >= F_avg:                       # keep the better propagator + flag
            out["U_proj"] = U_d
            out["drag_applied"] = True

    # GRAPE (opt-in); baseline = the applied gate (base_drag)
    if config.get("grape"):
        _grape_augment(out, cpl, a, b, config,
                       drag_beat_GHz=(drag_beat if base_drag else None),
                       nearest_beat_GHz=drag_beat)

    out["wall_s"] = time.time() - t0
    return out
