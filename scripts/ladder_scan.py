"""Run an unchanged pipeline entry point on a Hamiltonian-ladder level.

    ladder_scan.py --level L5 [--order 4] [--window 0.6] [--q-final N] [--sw-cache DIR] \
        -- subharmonic_gate_scan  <its own arguments...>
    ladder_scan.py --level L5 ... -- curve_drag_vs_bare  SCAN.h5 OUT.json [WORKERS]

Everything after ``--`` is that program's own command line. The only change is that
``ZhouCoupler.to_qutip_hamiltonian`` returns the level's effective Hamiltonian
(``analytical_exploration/ladder/engine.py``): the tune-up, the ridge chirp, the length
scan and the scoring are the pipeline's own code, so a reduced fidelity is made the same
way as the stored one. Columns go to the outdir the program is given, never to the
stored run.
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
    ap.add_argument("--level", required=True)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--window", type=float, default=0.6, help="slow window (GHz)")
    ap.add_argument("--q-final", type=int, default=None)
    ap.add_argument("--sw-cache", default=None, help="share SW models across processes")
    args = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    if not rest or rest[0] not in PROGRAMS:
        sys.exit(f"program must be one of {sorted(PROGRAMS)}")

    from ladder import engine
    engine.install(args.level, order=args.order, window_GHz=args.window,
                   q_final=args.q_final, cache_dir=args.sw_cache)
    kind, target = PROGRAMS[rest[0]]
    sys.argv = [target] + rest[1:]
    print(f"[ladder] {args.level} order={args.order} window={args.window} GHz "
          f"q_final={args.q_final} -> {rest[0]}", flush=True)
    if kind == "module":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
