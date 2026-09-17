"""Provisional curves straight from the scan rows, no re-scoring.

The scan already stores two scores per column: `fidelity` (chirp on, DRAG played) and
`flat` (chirp off, DRAG off). Both reuse the SAME fitted t_g -- the bare trace does not
get its own length here -- so this is a weaker baseline than the final Phase B pass,
but it needs no solves and shows the curve shape today.

Also marks the columns whose length fit RAILED on its search window, since those are
boundary values rather than optima.
"""
import json
import sys
from snail_solver.h5_io import load_doc
from snail_solver.subharmonic_gate_scan import coherence_penalty, total_infidelity

OUT = sys.argv[1]
TAGS = sys.argv[2:]

rows_out = []
for tag in TAGS:
    d = load_doc(f"results/drag_curve_2026-09-16/{tag}.h5")
    scan = d.get("scan") or d
    st = scan.get("settings") or {}
    t1, t2 = st.get("t1_us"), st.get("t2_us")
    lo, hi = float(st.get("tg_lo", 0.7)), float(st.get("tg_hi", 1.3))
    for r in scan.get("rows", []):
        t_g0 = r.get("t_g0_ns")
        out = {"delta_GHz": r["delta_GHz"], "target_eta": r["target_eta"],
               "branch": r.get("branch"), "ok": bool(r.get("ok")),
               "n_channels_designed": len(r.get("drag_channels") or []),
               "drag_channels": r.get("drag_channels"),
               "error": r.get("error"), "traces": {}}
        if not r.get("ok"):
            out["traces"] = {"bare": None, "chirp+DRAG": None}
            rows_out.append(out)
            continue
        t_g = float(r["operating_point"]["t_g_ns"])
        rail = None
        if t_g0:
            if abs(t_g - hi * float(t_g0)) < 1e-3 * float(t_g0):
                rail = "hi"
            elif abs(t_g - lo * float(t_g0)) < 1e-3 * float(t_g0):
                rail = "lo"
        out["t_g_railed"] = rail
        coh = coherence_penalty(t_g, t1_us=t1, t2_us=t2)
        for name, src in (("chirp+DRAG", r["fidelity"]), ("bare", r["flat"])):
            F = float(src["F_avg"])
            out["traces"][name] = {
                "F_avg": F, "leakage": src.get("leakage"),
                "transfer": src.get("transfer"), "t_g_ns": t_g,
                "n_drag_played": src.get("n_drag_channels",
                                         r["n_drag_channels"] if name != "bare" else 0),
                "infidelity_coherent": 1.0 - F,
                "eps_incoherent": coh["eps_incoherent"],
                "infidelity_total": (total_infidelity(F, coh["eps_incoherent"])
                                     if coh["eps_incoherent"] is not None else None)}
        rows_out.append(out)

rows_out.sort(key=lambda r: (r["target_eta"], r["delta_GHz"]))
json.dump(rows_out, open(OUT, "w"), indent=1)
n_ok = sum(1 for r in rows_out if r["ok"])
n_rail = sum(1 for r in rows_out if r.get("t_g_railed"))
print(f"wrote {OUT}: {len(rows_out)} rows, {n_ok} calibrated, {n_rail} length-railed")
