#!/usr/bin/env python3
"""Re-interpret a finished scan's Stark ridges under alternative fit policies.

    scripts/refit_ridges.py RUN.h5 [RUN2.h5 ...] [--policies a,b,c] [--csv OUT]

READ-ONLY and SOLVES NOTHING. Every number comes from `stages/rabi/chevrons`, which
the run wrote for every column -- failed ones too, since a failed column attaches its
partial table to the exception and `_write_run` stores it.

The Rabi sweep is 69-91% of a column's cost (2026-09-25 grid) and everything after it
is numpy, so "would another law have fitted this ridge?" is minutes over 189 columns,
while "would another measurement have?" is days. Settle the first before touching the
pipeline.

r2 rises trivially by fitting fewer rows over a shorter range and extrapolating to
eta*, so the table also prints `extrap` (eta* over the highest drive fitted) and
`last` (highest kept term over the lowest at eta*; generalizes `quartic_fraction`,
warn 0.25). A law is only usable when all three are good.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from snail_solver import ridge_refit as RR          # noqa: E402

#: `r2_min` in `rabi_shift_table`, and `quartic_warn` in `run_tune_up`.
R2_MIN = 0.9
LAST_TERM_WARN = 0.25
#: eta* is inside the measured rows when this is <= 1; above it the law is read
#: outside the data that produced it, which no r2 can detect.
EXTRAP_MAX = 1.0


def scan_context(path: str) -> Dict[str, Any]:
    """The config and probe settings the run was made under, rebuilt from the file so
    a report cannot describe a different device from the one measured. Only the
    ENVELOPE affects the probe moments; the settings are carried for printing.
    """
    from snail_solver.h5_io import load_tree, split_address
    from snail_solver.sweep_common import DEFAULT_CONFIG

    file_path, _ = split_address(path)
    scan = load_tree(file_path, group="scan")
    settings = dict(scan.get("settings") or {})
    device = dict(scan.get("device") or {})

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in device.items() if k in DEFAULT_CONFIG})
    # `--envelope-m` is a SCAN setting (pass A's baseline is pinned to 3), so it wins
    # over whatever the device file carried.
    if settings.get("envelope_m") is not None:
        cfg["envelope_m"] = int(settings["envelope_m"])
        if not str(cfg.get("envelope") or "").strip():
            cfg["envelope"] = "sine_power"
    etas = np.atleast_1d(np.asarray(scan.get("target_etas", [1.3]), dtype=float))
    return {"config": cfg, "settings": settings,
            "probe_shape": str(settings.get("probe_shape") or "constant"),
            "moment_weighting": str(settings.get("moment_weighting") or "rabi"),
            "target_eta": float(etas[0]),
            "amp_points": int(settings.get("amp_points") or 0)}


def verdict(fit: Dict[str, Any], diag: Dict[str, Any]) -> str:
    """One word for whether this law may be chirped with."""
    if fit["r2_unweighted"] < R2_MIN:
        return "r2"
    if diag["last_term_fraction"] > LAST_TERM_WARN:
        return "truncated"
    if diag["extrapolation_ratio"] > EXTRAP_MAX + 1e-9:
        return "extrapolated"
    return "converged"


def run_one(path: str, policies: List[str], ctx: Dict[str, Any],
            narrow_points: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eta_star = ctx["target_eta"]
    for tag, status, delta in sorted(RR.iter_columns(path), key=lambda r: r[2]):
        try:
            table = RR.load_column_rabi(path, tag)
        except KeyError:
            rows.append({"file": os.path.basename(path), "tag": tag,
                         "delta_GHz": delta, "status": status,
                         "policy": "-", "verdict": "no_ridge"})
            continue
        stored = RR.load_column_stored_fit(path, tag)
        for name in policies:
            kw = RR.policy_kwargs(name, eta_star)
            if narrow_points is not None and "n_narrow" in kw:
                kw["n_narrow"] = narrow_points
            rec: Dict[str, Any] = {
                "file": os.path.basename(path), "tag": tag,
                "delta_GHz": delta, "status": status, "policy": name,
                "stored_r2": stored.get("r2"),
                "stored_exc": stored.get("chirp_excursion_frac_linewidth"),
            }
            try:
                out = RR.refit_column(
                    table, eta_star, config=ctx["config"],
                    probe_shape=ctx["probe_shape"],
                    moment_weighting=ctx["moment_weighting"], **kw)
            except Exception as exc:                 # a policy can legitimately fail
                rec.update(verdict="unfittable", note=f"{type(exc).__name__}: {exc}")
                rows.append(rec)
                continue
            f, d = out["fit"], out["diagnostics"]
            rec.update(
                r2=f["r2_unweighted"], n_used=f["n_used"], n_rows=out["n_rows"],
                kept=out["rows"].get("kept", 0),
                excursion_frac=d["excursion_frac_linewidth"],
                last_term=d["last_term_fraction"],
                extrap=d["extrapolation_ratio"],
                verdict=verdict(f, d),
                **{f"n_{k}": v for k, v in out["rows"].items() if k != "kept"})
            rows.append(rec)
    return rows


def summarize(rows: List[Dict[str, Any]], policies: List[str]) -> None:
    for status in ("ok", "failed"):
        sel = [r for r in rows if r["status"] == status]
        if not sel:
            continue
        n_cols = len({r["tag"] for r in sel})
        print(f"\n== {status} ({n_cols} columns) ==")
        print(f"   {'policy':16s} {'rows':>5s} {'r2 med':>7s} {'>=0.9':>6s} "
              f"{'conv':>5s} {'trunc':>6s} {'extrap':>7s} {'r2fail':>7s}")
        for name in policies:
            p = [r for r in sel if r["policy"] == name and "r2" in r]
            if not p:
                print(f"   {name:16s}     - (no column fitted)")
                continue
            r2 = np.array([r["r2"] for r in p])
            used = np.array([r["n_used"] for r in p])
            v = [r["verdict"] for r in p]
            print(f"   {name:16s} {np.median(used):5.0f} {np.median(r2):7.3f} "
                  f"{int((r2 >= R2_MIN).sum()):6d} "
                  f"{v.count('converged'):5d} {v.count('truncated'):6d} "
                  f"{v.count('extrapolated'):7d} {v.count('r2'):7d}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="+", help="scan .h5 files (read-only)")
    ap.add_argument("--policies", default="stored,k6,k8,narrow_fit_0.8,no_widen,no_widen_k6",
                    help=f"comma-separated; available: {','.join(sorted(RR.POLICIES))}")
    ap.add_argument("--narrow-points", type=int, default=None,
                    help="offsets per row before any span growth (default: the "
                         "run's --wp-points, read from the file)")
    ap.add_argument("--csv", default=None, help="write the per-column table here")
    ap.add_argument("--per-column", action="store_true",
                    help="also print every column, not just the summary")
    args = ap.parse_args(argv)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policies if p not in RR.POLICIES]
    if unknown:
        ap.error(f"unknown policies {unknown}; have {sorted(RR.POLICIES)}")

    all_rows: List[Dict[str, Any]] = []
    for path in args.h5:
        ctx = scan_context(path)
        narrow = args.narrow_points
        if narrow is None:
            narrow = int(ctx["settings"].get("wp_points") or 15)
        print(f"{os.path.basename(path)}: eta*={ctx['target_eta']:g} "
              f"probe={ctx['probe_shape']} moments={ctx['moment_weighting']} "
              f"envelope_m={ctx['config'].get('envelope_m')} "
              f"amp_points={ctx['amp_points']} wp_points={narrow}")
        all_rows += run_one(path, policies, ctx, narrow)

    if args.per_column:
        print(f"\n{'delta':>8s} {'status':>7s} {'policy':16s} {'r2':>6s} "
              f"{'used':>4s} {'exc':>6s} {'last':>6s} {'extr':>5s}  verdict")
        for r in all_rows:
            if "r2" not in r:
                print(f"{r['delta_GHz']:+8.3f} {r['status']:>7s} {r['policy']:16s}"
                      f"      -    -      -      -     -  {r['verdict']}")
                continue
            print(f"{r['delta_GHz']:+8.3f} {r['status']:>7s} {r['policy']:16s} "
                  f"{r['r2']:6.3f} {r['n_used']:4d} {r['excursion_frac']:6.2f} "
                  f"{r['last_term']:6.2f} {r['extrap']:5.2f}  {r['verdict']}")

    summarize(all_rows, policies)

    if args.csv:
        keys = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
