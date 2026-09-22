"""Two traces per column: the calibrated chirp+DRAG gate, and the bare pulse.

    chirp+DRAG   chirp on, DRAG played, the length the calibration fitted for it
    bare         no chirp, no DRAG, its OWN length refitted

This is the comparison the curve figure plots. Both traces keep the calibrated carrier
offset `wp_offset_GHz`, so the ONLY difference between them is the two corrections --
carrier tuning is common to both and is not credited to DRAG.

Why a separate pass rather than reading the scan's own numbers:

* the scan's `_column_expect` keys on 18 physics knobs and NOTHING about scoring, so
  putting a scoring flag there would force every cached column to re-solve its whole
  calibration for a change that happens after it;
* only the bare trace needs a length refit (~24 serial solves). The chirp+DRAG trace's
  `t_g` was already fitted for it by step 4 of the calibration, so refitting both would
  double the cost of this pass for nothing.

Failed columns are carried through as `ok: false` rows, never dropped: the curve has to
show where the drive ceiling bit rather than interpolate across it.

Usage:  curve_drag_vs_bare.py SCAN.h5[,SCAN2.h5...] OUT.json [WORKERS]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SRC_LIST = sys.argv[1].split(",")
OUT = sys.argv[2]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12

# (name, chirp on, DRAG on, refit its own length)
VARIANTS = (("chirp+DRAG", True, True, False),
            ("bare", False, False, True))


def _chirp_quality(row, quartic_warn=0.25):
    """Whether the chirp this column carries rests on a law that converged.

    `r2` can be 0.998 while `quartic_fraction` is 20 -- a confident fit to a
    series that is not a series, because `delta = k2 eta^2 (1 + (k4/k2) eta^2)`
    is a TRUNCATION and `quartic_fraction` is how big the last kept term is next
    to the first. When it is not small the unmeasured eta^6 term is plausibly the
    same size, so the chirp is a candidate, not a calibration.

    `perturbative_ok` is computed rather than read: the scan stores
    `quartic_fraction` on the row but not the verdict. `min_abs_detuning_GHz` is
    the floor the chirp<->DRAG fixed point reached, which is the quantity that
    goes to zero when the loop chases its own denominator.
    """
    ch = row.get("chirp")
    if not isinstance(ch, dict):
        ch = {}
    q = ch.get("quartic_fraction")
    q = float(q) if q is not None else None
    free = ch.get("coeffs_GHz") is not None and len(ch.get("coeffs_GHz") or []) == 0
    return {"quartic_fraction": q,
            "perturbative_ok": (None if q is None else bool(q < quartic_warn)),
            "min_abs_detuning_GHz": ch.get("min_abs_detuning_GHz"),
            "shift_law_r2": ch.get("r2"),
            "chirp_free": bool(free)}


def _one(job):
    """Score ONE (column, variant). Runs in a worker process."""
    import warnings
    warnings.filterwarnings("ignore")
    from snail_solver.envelope import DragChannel
    from snail_solver.subharmonic_convergence import config_at_wp
    from snail_solver.tune_up_sweep import score_gate

    r, base = job["row"], job["device"]
    name, on_chirp, on_drag, refit = job["variant"]
    chirp = [float(x) for x in r["chirp"]["coeffs_GHz"]]
    chans = [DragChannel(beat_GHz=float(c["beat_GHz"]), n_pump=int(c["n_pump"]),
                         n_photon=int(c.get("n_photon", c["n_pump"])),
                         quotient_rule=True)
             for c in (r.get("drag_channels") or [])]
    # `[]` for no chirp (None would inherit the config's); None for no DRAG (`[]` is
    # falsy at device_utils.py:317 and so indistinguishable from None there).
    cfg = config_at_wp(base, r["w_p_GHz"], branch=r["branch"],
                       levels=int(r["coupler_levels"]), chirp_coeffs_GHz=chirp)
    t0 = time.perf_counter()
    s = score_gate(cfg, r["operating_point"],
                   chirp if on_chirp else [],
                   drag_channels=(chans if on_drag else None),
                   refit_length=refit, tg_points=job["tg_points"],
                   tg_lo=job["tg_lo"], tg_hi=job["tg_hi"])
    return {"key": job["key"], "variant": name,
            "seconds": time.perf_counter() - t0,
            "F_avg": s["F_avg"], "leakage": s["leakage"], "transfer": s["transfer"],
            "t_g_ns": s["t_g_ns"], "amp_scale": s["amp_scale"],
            "n_drag_played": s["n_drag_channels"], "refit_length": s["refit_length"]}


def _scan_of(src):
    """(rows, device, settings) from a scan HDF5 or a rows.json salvage."""
    if src.endswith(".json"):
        return json.load(open(src)), None, {}
    from snail_solver.h5_io import load_doc
    d = load_doc(src)
    scan = d.get("scan") or d
    return scan.get("rows", []), scan.get("device"), (scan.get("settings") or {})


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from snail_solver.subharmonic_gate_scan import coherence_penalty, total_infidelity

    # Reuse anything already scored: this pass costs minutes per solve, and widening
    # the eta set should not re-pay for the eta already done. A cached trace is keyed
    # on (delta, eta, variant) and only reused when it is actually populated.
    cached = {}
    if os.path.exists(OUT):
        try:
            for r in json.load(open(OUT)):
                k = f"{r['delta_GHz']:+.4f}|{r['target_eta']:.2f}"
                for v, t in (r.get("traces") or {}).items():
                    if t is not None:
                        cached[(k, v)] = t
            print(f"reusing {len(cached)} already-scored traces from {OUT}",
                  flush=True)
        except Exception as exc:                       # a corrupt cache is not fatal
            print(f"(ignoring unreadable {OUT}: {exc})", flush=True)

    jobs, meta, skipped = [], {}, []
    for src in SRC_LIST:
        rows, dev, settings = _scan_of(src)
        if dev is None:
            raise SystemExit(f"{src}: no device copy in the scan doc; pass the HDF5")
        t1, t2 = settings.get("t1_us"), settings.get("t2_us")
        pre = float(settings.get("decoh_prefactor", 1.0) or 1.0)
        tgp = int(settings.get("tg_points", 9) or 9)
        tlo = float(settings.get("tg_lo", 0.7) or 0.7)
        thi = float(settings.get("tg_hi", 1.3) or 1.3)
        for r in rows:
            key = f"{r['delta_GHz']:+.4f}|{r['target_eta']:.2f}"
            meta[key] = {"delta_GHz": r["delta_GHz"], "target_eta": r["target_eta"],
                         "branch": r.get("branch"), "ok": bool(r.get("ok")),
                         "w_p_GHz": r.get("w_p_GHz"),
                         "delta_sub_GHz": r.get("delta_sub_GHz"),
                         "n_channels_designed": len(r.get("drag_channels") or []),
                         "drag_channels": r.get("drag_channels"),
                         "drag_shed": r.get("drag_shed"),
                         "t1_us": t1, "t2_us": t2, "decoh_prefactor": pre,
                         "error": r.get("error"),
                         # Whether the chirp this column carries rests on a shift
                         # law that converged. r2 can be 0.998 while
                         # quartic_fraction is 20 -- a confident fit to a series
                         # that is not a series. Carried through so the analysis
                         # can separate those columns instead of averaging a
                         # meaningless chirp into the gain.
                         **_chirp_quality(r, float(settings.get("quartic_warn", 0.25) or 0.25))}
            if not r.get("ok"):
                skipped.append(key)
                continue
            for v in VARIANTS:
                if (key, v[0]) in cached:
                    continue
                jobs.append({"key": key, "row": r, "device": dev, "variant": v,
                             "tg_points": tgp, "tg_lo": tlo, "tg_hi": thi})

    print(f"{len(meta)} column(s): {len(meta) - len(skipped)} calibrated x "
          f"{len(VARIANTS)} variants; {len(jobs)} solves to run on {WORKERS} workers "
          f"({len(cached)} reused, {len(skipped)} uncalibrated -> ok=false)",
          flush=True)

    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for out in ex.map(_one, jobs):
            res.setdefault(out["key"], {})[out["variant"]] = out
            done += 1
            print(f"  [{done}/{len(jobs)}] {out['key']} {out['variant']:11s} "
                  f"1-F={1 - out['F_avg']:.4e} t_g={out['t_g_ns']:.1f}ns "
                  f"played={out['n_drag_played']} ({out['seconds']:.0f}s)", flush=True)

    payload = []
    for k, m in meta.items():
        row = dict(m)
        row["traces"] = {}
        for name, *_ in VARIANTS:
            s = (res.get(k) or {}).get(name) or cached.get((k, name))
            if s is None:
                row["traces"][name] = None
                continue
            # Charge each trace for ITS OWN length. Applying one eps to both would hand
            # the longer trace a free pass on decoherence, which is exactly the term
            # that decides whether a correction was worth its gate time.
            coh = coherence_penalty(float(s["t_g_ns"]), t1_us=m["t1_us"],
                                    t2_us=m["t2_us"], prefactor=m["decoh_prefactor"])
            row["traces"][name] = {
                **s,
                "infidelity_coherent": 1.0 - float(s["F_avg"]),
                "eps_incoherent": coh["eps_incoherent"],
                "infidelity_total": (total_infidelity(s["F_avg"],
                                                      coh["eps_incoherent"])
                                     if coh["eps_incoherent"] is not None else None)}
        payload.append(row)

    payload.sort(key=lambda r: (r["target_eta"], r["delta_GHz"]))
    json.dump(payload, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  ({len(payload)} rows)", flush=True)
