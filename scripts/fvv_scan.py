"""Run an unchanged pipeline entry point on an FVV effective model.

    fvv_scan.py --order 4 [--window 0.6] [--displaced] [--cache DIR] \
        -- subharmonic_gate_scan <its own arguments...>
    fvv_scan.py --order 4 ... -- curve_drag_vs_bare SCAN.h5 OUT.json [WORKERS]

Everything after ``--`` is that program's own command line. The only change is that
``ZhouCoupler.to_qutip_hamiltonian`` returns F(t) of the FVV model
(``analytical_exploration/floquet/propagator.py``). The tune-up, the ridge chirp, the
length scan and the scoring are the pipeline's own code. Point the program at a new
outdir, never at a stored run.
"""
import argparse
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytical_exploration"))

PROGRAMS = {
    "subharmonic_gate_scan": ("module", "snail_solver.subharmonic_gate_scan"),
    "curve_drag_vs_bare": ("path", os.path.join(ROOT, "scripts", "curve_drag_vs_bare.py")),
}


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit(__doc__)
    cut = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--window", type=float, default=0.6, help="secular window (GHz)")
    ap.add_argument("--displaced", action="store_true")
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "fvv_cache"))
    args = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    if not rest or rest[0] not in PROGRAMS:
        sys.exit(f"program must be one of {sorted(PROGRAMS)}")
    from floquet import propagator
    propagator.install(cache_dir=args.cache, order=args.order, window_GHz=args.window,
                       displaced=args.displaced)
    kind, target = PROGRAMS[rest[0]]
    sys.argv = [target] + rest[1:]
    print(f"[fvv] order={args.order} window={args.window} GHz displaced={args.displaced} "
          f"-> {rest[0]}", flush=True)
    if kind == "module":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
