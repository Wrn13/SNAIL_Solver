#!/usr/bin/env python3
"""
post_chirp.py
=============

Does the calibrated chirp actually help, on the REAL shaped gate, and up to
what drive?

``tune_up.py`` builds a chirp from a CONSTANT-probe diagnostic -- a property of
the device, not of the pulse (see its module docstring). This module re-runs
the ACTUAL raised-cosine gate, with the calibrated chirp, as a chevron sweep
across drive strengths, and (by default) the identical gate with the chirp
switched off at the same length/amplitude/offset -- the ``chirp_ablation``
comparison ``run_tune_up`` already does at ONE point, generalised across drive
strength and across the pump-offset axis. This is the direct, gate-level
answer to "does the chirp work", complementing the constant-probe diagnostics
in ``tune_up.py``.

Every row's gate length is rescaled to keep it a genuine full swap at its own
drive (see ``tune_up.post_chirp_table``), so a row's contrast collapsing means
something is actually going wrong at that drive -- not that the row is just a
partial rotation.

Usage
-----
    python -m snail_solver.post_chirp \\
        --from-tuneup results/tuneup_probe_dev1_eta1p8_wide.h5 \\
        --eta-lo 0.5 --eta-hi 1.0 --amp-points 7 --wp-points 25 --jobs 16 \\
        --out post_chirp_dev1_eta1p8.h5 \\
        --plot figs/probe_dev1_eta1p8_wide/post_chirp_chevrons.png
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from snail_solver.h5_io import (attach_figures, is_hdf5, load_doc, save_doc,
                                split_address)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.post_chirp", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="device JSON (bare name resolves under devices/). Optional "
                         "with --from-tuneup, which carries its own copy of the "
                         "configuration the tune-up ran with -- pass one only to run "
                         "against a DIFFERENT device than the tune-up used, and the "
                         "context check below will tell you if it disagrees. Required "
                         "with --point (the point lives in that file)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-tuneup", metavar="FILE", default=None,
                     help="a tune_up --out file (HDF5, or pre-HDF5 JSON): reads "
                          "operating_point and stages.rabi. FILE:/runs/eta1p8 "
                          "reads one run out of a tune_up_sweep file")
    src.add_argument("--point", metavar="NAME", default=None,
                     help="a saved operating point's name in --device's own "
                          "JSON (see operating_points.save_point)")
    ap.add_argument("--eta-lo", type=float, default=0.5,
                    help="drive-strength row sweep, as a fraction of target_eta")
    ap.add_argument("--eta-hi", type=float, default=1.0)
    ap.add_argument("--amp-points", type=int, default=7)
    ap.add_argument("--wp-span-MHz", type=float, default=None,
                    help="fixed chevron offset span for every row; default sizes "
                         "each row from its own linewidth (see --span-linewidths)")
    ap.add_argument("--span-linewidths", type=float, default=4.0)
    ap.add_argument("--wp-points", type=int, default=25)
    ap.add_argument("--n-time", type=int, default=161)
    ap.add_argument("--no-compare-flat", action="store_true",
                    help="skip the flat-carrier (chirp off) comparison row")
    ap.add_argument("--reproject-chirp", action="store_true",
                    help="re-derive the chirp at EACH row's drive from the Rabi "
                         "table instead of using the calibrated chirp verbatim -- "
                         "answers 'would the procedure work here', not 'does the "
                         "calibrated gate work here'. Needs --from-tuneup (the "
                         "Rabi table isn't saved with a --point)")
    ap.add_argument("--chirp-degree", type=int, default=8)
    ap.add_argument("--coupler-levels", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--gpu", action="store_true",
                    help="run via qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--out", default=None,
                    help="write the result here, under results/ unless absolute; "
                         "HDF5 (a bare name gains .h5), or JSON on a .json suffix")
    ap.add_argument("--plot", nargs="?", const="figs/post_chirp_chevrons.png",
                    default=None)
    ap.add_argument("--replot", metavar="FILE", default=None,
                    help="skip the run entirely and regenerate --plot from a "
                         "previously written --out -- no new solves. HDF5 or JSON, "
                         "detected by content")
    args = ap.parse_args()

    if args.reproject_chirp and args.from_tuneup is None:
        ap.error("--reproject-chirp needs --from-tuneup (a saved --point has "
                 "no Rabi table to reproject from)")
    if args.point and not args.device:
        ap.error("--point needs --device: the point is stored in that device JSON")
    if args.gpu:
        from snail_solver import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from snail_solver.paths import resolve_device, in_results
    from snail_solver.tune_up import post_chirp_table, plot_post_chirp_table

    if args.replot:
        post = load_doc(args.replot)
        rabi_table = post.get("rabi_table")
        if args.plot:
            path = plot_post_chirp_table(post, out=args.plot, rabi_table=rabi_table)
            print(f"wrote {path}")
            # re-drawn from this file, so refresh the copy it carries
            if is_hdf5(split_address(args.replot)[0]):
                attach_figures(args.replot, {"post_chirp": path})
        return

    from snail_solver.device_utils import load_device
    from snail_solver.log_utils import setup_run_logger

    logger = setup_run_logger(None, "post_chirp")
    device_path = resolve_device(args.device) if args.device else None
    config = load_device(device_path) if device_path else None

    rabi_table: Optional[Dict[str, Any]] = None
    saved: Optional[Dict[str, Any]] = None
    if args.from_tuneup:
        saved = load_doc(args.from_tuneup)
        record = saved["operating_point"]
        rabi_table = saved["stages"]["rabi"]
        if config is None:
            # The run carries the configuration it was calibrated with. Using it is
            # strictly better than re-reading the device file: that file gets edited
            # (coupler_levels, frequencies) between the tune-up and this check, and
            # then this "validation" would be measuring a different device.
            config = saved.get("device")
            if config is None:
                ap.error(f"--device is required: {args.from_tuneup} was written "
                         f"before runs carried a copy of their device config")
            device_path = str(saved.get("device_path") or "") or None
            print(f"device: from the tune-up's own copy"
                  + (f" ({device_path})" if device_path else ""))
    else:
        from snail_solver.operating_points import get_point
        record = get_point(config, args.point)

    if args.coupler_levels is not None:
        config = {**config, "coupler_levels": int(args.coupler_levels)}

    from snail_solver.operating_points import check_context
    # t_g is pinned to the POINT's own calibrated length: post_chirp_table derives
    # every row's length from it directly, never from config["t_g_ns"] (a generic
    # starting guess), so that is not a context mismatch worth flagging here --
    # only the device's own frequencies (wa/wb/spec_abs) are.
    mismatches = check_context(record, config, t_g=float(record["t_g_ns"]))
    if mismatches:
        raise SystemExit(
            f"--device {args.device} does not match the operating point's "
            f"context:\n  " + "\n  ".join(mismatches))

    from snail_solver import find_stark_resonance as FSR
    print(f"device={args.device or device_path}  target_eta={record['target_eta']}  "
          f"jobs={FSR._resolve_jobs(args.jobs)}{' GPU' if args.gpu else ''}")

    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    post = post_chirp_table(
        config, record, eta_lo=args.eta_lo, eta_hi=args.eta_hi,
        amp_points=args.amp_points, wp_span_MHz=args.wp_span_MHz,
        wp_points=args.wp_points, span_linewidths=args.span_linewidths,
        n_time=args.n_time, compare_flat=not args.no_compare_flat,
        reproject_chirp=args.reproject_chirp, rabi_table=rabi_table,
        chirp_degree=args.chirp_degree, jobs=args.jobs, solver=solver,
        logger=logger)

    print("\n=== post-chirp result ===")
    print(f"  target_eta   = {record['target_eta']}")
    ok = post["residual_MHz"]
    import numpy as np
    finite = np.asarray(ok, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        print(f"  residual_MHz = mean {np.mean(np.abs(finite)):.3f}  "
              f"max {np.max(np.abs(finite)):.3f}")
    print(f"  transfer(chirped) = {post['transfer_chirped']}")
    if post["compare_flat"]:
        print(f"  transfer(flat)     = {post['transfer_flat']}")

    written = None
    if args.out:
        out_doc = dict(post)
        if rabi_table is not None:
            out_doc["rabi_table"] = rabi_table
        out_doc["device"] = config          # the configuration these solves ran with
        written = save_doc(in_results(args.out), out_doc,
                           attrs={"tool": "snail_solver.post_chirp",
                                  "device_path": str(device_path or ""),
                                  "from_tuneup": str(args.from_tuneup or ""),
                                  "point": str(args.point or "")})
        print(f"  written {written}")

    if args.plot:
        path = plot_post_chirp_table(post, out=args.plot, rabi_table=rabi_table)
        print(f"  wrote {path}")
        # the figure belongs with the sweep it draws, not only in figs/
        if written and is_hdf5(split_address(written)[0]):
            attach_figures(written, {"post_chirp": path})
            print(f"  embedded it in {written}")


if __name__ == "__main__":
    main()
