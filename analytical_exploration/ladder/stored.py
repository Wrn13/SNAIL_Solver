"""The target: figs_ridge_tracked/table.txt's `1-F bare` and `1-F chirp`, per column.

Both series come from ``curves_nodrag_ridge_tracked_flagged.json`` (pass A ridge, no
DRAG). ``chirp`` is that run's ``chirp+DRAG`` trace, which plays no DRAG in pass A, and
``bare`` is its own-length-refitted flat pulse. The `1-F drag` column of the table
comes from pass C and is out of scope.

Each column's operating point (chirp, carrier offset, length) is read from the pass A
ridge scans. The tracked re-solves supersede ``full`` at the delta they re-ran,
``tracked2`` supersedes both, and every pick is checked against the trace's own
t_g / amp_scale.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(ROOT, "results", "drag_curve_5MHz_2026-09-22")
CURVES = os.path.join(RUN, "curves_nodrag_ridge_tracked_flagged.json")
TABLE = os.path.join(RUN, "figs_ridge_tracked", "table.txt")
SCANS = ("eta1p3_full", "eta1p3_tracked", "eta1p3_tracked2")    # later supersedes earlier


def _scan(tag: str):
    from snail_solver.h5_io import load_doc
    d = load_doc(os.path.join(RUN, "passA_ridge", f"{tag}.h5"))
    s = d.get("scan") or d
    return s["rows"], s["device"]


def base_device() -> Dict[str, Any]:
    """The scan's merged device config (sine_power m=3); levels come from each row."""
    return _scan(SCANS[0])[1]


def read_table(path: str = TABLE) -> Dict[float, Dict[str, float]]:
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            d = float(parts[0])
        except ValueError:
            continue
        def num(v):
            try:
                return float(v)
            except ValueError:
                return float("nan")
        out[round(d * 1e-3, 6)] = {"bare": num(parts[5]), "chirp": num(parts[6]),
                                   "tg_bare": num(parts[3])}
    return out


def _key(d: float) -> float:
    return round(float(d), 6)


def columns(check_table: bool = True) -> List[Dict[str, Any]]:
    """One dict per column: delta, w_p, coupler_levels, op (operating point), stored
    1-F bare/chirp, their lengths, and the flags (chirp_excluded, chirp_free, ...)."""
    rows: Dict[float, Dict[str, Any]] = {}
    for tag in SCANS:
        for r in _scan(tag)[0]:
            rows[_key(r["delta_GHz"])] = dict(r, _scan=tag)
    curves = json.load(open(CURVES))
    table = read_table() if check_table else {}
    out = []
    for c in curves:
        k = _key(c["delta_GHz"])
        r = rows.get(k)
        tr = c.get("traces") or {}
        ch, ba = tr.get("chirp+DRAG") or {}, tr.get("bare") or {}
        rec: Dict[str, Any] = {
            "delta_GHz": float(c["delta_GHz"]), "w_p_GHz": float(c["w_p_GHz"]),
            "branch": c["branch"], "target_eta": float(c["target_eta"]),
            "ok": bool(c.get("ok")) and r is not None and bool(r.get("ok")),
            "chirp_excluded": c.get("chirp_excluded"), "chirp_free": c.get("chirp_free"),
            "chirp_free_reason": c.get("chirp_free_reason"),
            "coupler_truncated": c.get("coupler_truncated"), "n_coupler": c.get("n_coupler"),
            "inf_chirp": (1.0 - ch["F_avg"]) if "F_avg" in ch else None,
            "inf_bare": (1.0 - ba["F_avg"]) if "F_avg" in ba else None,
            "tg_chirp": ch.get("t_g_ns"), "tg_bare": ba.get("t_g_ns"),
            "scan": r.get("_scan") if r else None,
        }
        if r is not None:
            op = r.get("operating_point") or {}
            rec["op"] = op
            rec["coupler_levels"] = int(r.get("coupler_levels") or 9)
            rec["chirp_coeffs_GHz"] = [float(v) for v in ((r.get("chirp") or {}).get("coeffs_GHz") or [])]
            if ch and op and abs(float(op.get("t_g_ns", -1)) - float(ch["t_g_ns"])) > 1e-6:
                raise ValueError(f"delta={k}: scan row t_g {op.get('t_g_ns')} != trace "
                                 f"{ch['t_g_ns']} ({rec['scan']})")
        if table and k in table:
            t = table[k]
            for name in ("bare", "chirp"):
                v = rec[f"inf_{name}"]
                if v is not None and abs(v - t[name]) > 1e-4 * max(t[name], 1e-6) + 5e-9:
                    raise ValueError(f"delta={k}: {name} 1-F {v:.6g} != table {t[name]:.6g}")
        out.append(rec)
    return out


def config_for(rec: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The exact config the scorer used for this column (curve_drag_vs_bare._one)."""
    from snail_solver.subharmonic_convergence import config_at_wp
    return config_at_wp(base or base_device(), rec["w_p_GHz"], branch=rec["branch"],
                        levels=int(rec["coupler_levels"]),
                        chirp_coeffs_GHz=rec.get("chirp_coeffs_GHz") or [])
