"""Shared loaders for the DRAG-sign study on the 5 MHz ridge-tracked curve.

Rebuilds, for one column of ``results/drag_curve_5MHz_2026-09-22``, everything the
spectra (`plot_pulse_spectra.py`) and the drive scan (`drag_sign_amp_scan.py`) need,
from the STORED records only (nothing is re-calibrated here):

    bare     passA bare trace: no chirp, no DRAG, its own refit length
    chirp    passA_ridge operating point: the ridge chirp, DRAG off
    drag     passC_ridge operating point: ridge chirp + the subharm-leak DRAG set,
             beats AS PLAYED (the pre-fix audit sign)
    fixed    the passC operating point with each channel's beat re-read from the
             fixed audit (``spectator_audit.interaction_channels``'s ``beat_MHz``)

The ridge-tracked curve files are ``curves_*_ridge_tracked.json``; none of the
columns in -185..-160 MHz were in the tracker re-solve, so ``passA_ridge`` /
``passC_ridge`` ``columns/*.json`` are their records.
"""
import json
import os

import numpy as np

RES = "results/drag_curve_5MHz_2026-09-22"
TWO_PI = 2.0 * np.pi


def col_tag(delta_MHz: float) -> str:
    from snail_solver.subharmonic_gate_scan import column_tag
    return column_tag(float(delta_MHz) * 1e-3, 1.3)


def scan_doc(pass_dir: str = "passC_ridge"):
    """(device, settings) of the stored scan: the config every column was solved in."""
    from snail_solver.h5_io import load_doc
    d = load_doc(os.path.join(RES, pass_dir, "eta1p3_full.h5"))
    s = d.get("scan") or d
    return s["device"], s["settings"]


def _curve_row(name: str, delta_MHz: float):
    rows = json.load(open(os.path.join(RES, name)))
    return next(r for r in rows if round(r["delta_GHz"] * 1e3) == round(delta_MHz))


def stored_channels(rec):
    from snail_solver.envelope import DragChannel
    return [DragChannel(beat_GHz=float(c["beat_GHz"]), n_pump=int(c["n_pump"]),
                        n_photon=int(c.get("n_photon", c["n_pump"])),
                        quotient_rule=True)
            for c in (rec.get("drag_channels") or [])]


def column(delta_MHz: float):
    """Everything about one column: configs, the four variants, the fixed audit."""
    import dataclasses
    from snail_solver import subharmonic_gate_scan as GS
    from snail_solver.subharmonic_convergence import config_at_wp

    device, settings = scan_doc()
    tag = col_tag(delta_MHz)
    A = json.load(open(os.path.join(RES, "passA_ridge", "columns", f"col_{tag}.json")))
    C = json.load(open(os.path.join(RES, "passC_ridge", "columns", f"col_{tag}.json")))
    bareA = _curve_row("curves_nodrag_ridge_tracked.json", delta_MHz)["traces"]["bare"]
    levels = int(C["coupler_levels"])

    def cfg_of(rec):
        return config_at_wp(device, rec["w_p_GHz"], branch=settings["branch"],
                            levels=levels,
                            chirp_coeffs_GHz=rec["operating_point"]["chirp_coeffs_GHz"])

    # the FIXED audit at this column, the same call the scan makes
    col = GS.columns_for(device, [delta_MHz * 1e-3], drop_origin=False)[0]
    chans_fixed, audit = GS.audit_column(device, col, {**settings,
                                                       "coupler_levels": levels,
                                                       "target_eta": 1.3})
    by_label = {r["transition"]: r for r in audit["rows"]}
    stored = stored_channels(C)
    fixed = []
    for c, row in zip(stored, C["drag_channels"]):
        r = by_label[row["transition"]]
        fixed.append(dataclasses.replace(c, beat_GHz=float(r["beat_GHz"])))

    opA, opC = A["operating_point"], C["operating_point"]
    variants = {
        "bare":  dict(cfg=cfg_of(A), t_g=float(bareA["t_g_ns"]),
                      amp=float(bareA["amp_scale"]), wp=float(opA["wp_offset_GHz"]),
                      chirp=[], chans=None),
        "chirp": dict(cfg=cfg_of(A), t_g=float(opA["t_g_ns"]),
                      amp=float(opA["amp_scale"]), wp=float(opA["wp_offset_GHz"]),
                      chirp=list(opA["chirp_coeffs_GHz"]), chans=None),
        "drag":  dict(cfg=cfg_of(C), t_g=float(opC["t_g_ns"]),
                      amp=float(opC["amp_scale"]), wp=float(opC["wp_offset_GHz"]),
                      chirp=list(opC["chirp_coeffs_GHz"]), chans=stored),
        "fixed": dict(cfg=cfg_of(C), t_g=float(opC["t_g_ns"]),
                      amp=float(opC["amp_scale"]), wp=float(opC["wp_offset_GHz"]),
                      chirp=list(opC["chirp_coeffs_GHz"]), chans=fixed),
    }
    return {"delta_MHz": float(delta_MHz), "A": A, "C": C, "variants": variants,
            "audit": audit, "chans_fixed_selected": chans_fixed,
            "settings": settings, "device": device,
            "w_p_GHz": float(C["w_p_GHz"]), "levels": levels}


def eta_nominal(v, n_time: int = 4001):
    """``(t, eta(t))`` of a variant in the NOMINAL pump frame ``e^{-i w_nom t}``.

    `ZhouCoupler._eta_at` is what the solver plays about the tone's own carrier
    ``w_nom + wp_offset``; folding the offset back in puts every variant on one
    absolute frequency axis, where the processes sit still.
    """
    from snail_solver.device_utils import build_coupler
    cpl, _w_p, _pk = build_coupler(v["cfg"], v["t_g"], v["amp"], v["wp"], None, None,
                                   chirp_coeffs_GHz=v["chirp"],
                                   drag_channels=v["chans"])
    tone = cpl._pump_tones[0]
    t = np.linspace(0.0, v["t_g"], int(n_time))
    eta = np.asarray(cpl._eta_at(tone, t, np), dtype=complex)
    return t, eta * np.exp(-1j * TWO_PI * v["wp"] * t)


def spectrum(t, x, nu_GHz):
    """``S(nu) = int x(t) e^{+i 2 pi nu t} dt``: the drive component a process with
    DRAG beat ``Delta = 2 pi nu`` sees (first-order amplitude ``~ g S(Delta)``)."""
    dt = np.gradient(t)
    return np.exp(1j * TWO_PI * np.outer(nu_GHz, t)) @ (x * dt)
