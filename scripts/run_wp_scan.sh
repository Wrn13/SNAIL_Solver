#!/usr/bin/env bash
# ============================================================================
# subharmonic_gate_scan, detached, on ONE box (no SLURM).
#
# Scans the gate pump across qubit A's subharmonic (w_p = w_a/2 + delta) and tunes
# up a chirped recursive-DRAG pulse at every offset. The scan is long (the Rabi
# amplitude scan that determines each column's chirp is amp_points x wp_points
# solves per column), so it is launched detached and survives the terminal.
#
# RUN --dry-run FIRST, ALWAYS. It solves nothing and prints every column's DRAG
# channel audit -- which channels will be corrected, which will not, and why:
#
#   uv run python -m snail_solver.subharmonic_gate_scan \
#       --device 4Gate4.5SNAIL.json --offsets=-0.1:0.1:21 --target-eta 2.5 --dry-run
#
# QUICK START
#   scripts/run_wp_scan.sh --device 4Gate4.5SNAIL.json --offsets=-0.1:0.1:21 \
#       --target-eta 2.5 --amp-points 41 --coupler-levels 9 \
#       --column-workers 8 --jobs 8 \
#       --out wpscan_full.h5 --plot figs/wpscan_full.png
#
# HOW TO SPEND THE CORES. Step 1 (the Rabi amplitude scan that builds the chirp)
# fans out over --jobs, but step 4 (length_rabi) takes no jobs at all -- it is an
# optimizer over gate length and runs SINGLE-THREADED, for minutes per column once
# the pulse carries a multi-channel recursion. A --jobs 72 run therefore idles ~70
# cores through step 4 of every column. Pair a moderate --jobs with --column-workers
# so the pool covers that stretch.
#
# It prints a PID and a log path. Watch it with `tail -f <log>`; stop it with
# `kill <pid>`.
#
# RESUMABLE. Every column is cached under --outdir the moment it succeeds, so a run
# that is killed (or a box that reboots) is resumed by relaunching the IDENTICAL
# command -- only the missing columns solve. --overwrite forces a re-solve.
#
# ENVIRONMENT
#   LOG       log file                 [results/wpscan_<UTC timestamp>.log]
#   LAUNCHER  how to run python        [uv run python]
#
# NOTE ON --offsets: it needs the '=' form (--offsets=-0.1:0.1:21). A bare leading
# '-' is read by argparse as a flag.
#
# NOTE ON THE GPU: don't. This Hilbert space is ~10^2 states, far below the
# CPU/GPU crossover, and --gpu forces --jobs 1. The 72 cores are the resource.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# One BLAS thread per process. NOT optional: the tune-up fans its chevron columns
# out over a process pool, and 72 workers each spawning 72 BLAS threads thrashes
# the machine instead of using it.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

LAUNCHER="${LAUNCHER:-uv run python}"
LOG="${LOG:-results/wpscan_$(date -u +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${LOG}")"

# setsid + nohup + </dev/null: survives the terminal closing and never blocks on
# stdin. The scan writes its own log too (--log), but this catches anything that
# escapes the logger, including a traceback during startup.
setsid nohup ${LAUNCHER} -m snail_solver.subharmonic_gate_scan "$@" \
    > "${LOG}" 2>&1 < /dev/null &
PID=$!
echo "pid ${PID}  ->  ${LOG}"
echo "  watch:  tail -f ${LOG}"
echo "  stop:   kill ${PID}"
echo "  resume: relaunch this exact command (cached columns are re-read)"
