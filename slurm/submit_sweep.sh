#!/bin/bash
# Submit a full sweep pipeline: prepare (here) -> point array -> collect -> plot,
# chained with --dependency=afterok so each stage waits for the previous one.
#
# `prepare` runs on the LOGIN NODE (it is fast and needs no QuTiP) because the
# array size depends on the grid it produces -- #SBATCH --array cannot be computed
# from inside the job.
#
# Usage:
#   ./slurm/submit_sweep.sh <KIND> <OUTDIR> -- <prepare args...>
#
# KIND is only used to pick the plotter: spectator | bare | target
#
# Examples ---------------------------------------------------------------------
# Every frequency list below is tuned to the DEVICE it names, so these run as-is.
# Retune them if you point them at a different device: the collision each sweep
# scans through sits at a device-specific frequency (see the note per example).
#
# isolated one-pump collision (the headline DRAG figure).
# 2Gate4.9SNAIL: w_a=3.8, w_b=5.5 -> w_p=1.7, so the one-pump collision is at
# |Delta|=1.7. Use the NEGATIVE (above-w_b) branch: on the +1.7 branch w_spec
# lands exactly on w_a, which makes it a direct collision instead of an isolated
# pump-assisted one. Note the `=`: argparse reads a leading `-` as a flag.
#   ./slurm/submit_sweep.sh spectator clean_v1 -- \
#       --sweep spectator --device 2Gate4.9SNAIL.json \
#       --drags false,true --specfreqs=-1.60,-1.65,-1.70,-1.75,-1.80
#
# bare 3-mode gate through the SNAIL subharmonic (needs 2 w_p to reach w_c).
# 1Gate4.2SNAIL: w_a=4.7, w_c=4.2 -> 2 w_p = w_c at w_b = w_a + w_c/2 = 6.8.
#   ./slurm/submit_sweep.sh bare snail_v1 -- \
#       --sweep target --no-spectator --device 1Gate4.2SNAIL.json \
#       --drag-subharmonic --subharmonic-modes s \
#       --drags false,true --wb-GHz 6.65,6.70,6.75,6.80,6.85,6.90,6.95
#
# spectator on the pump 2nd harmonic (w_spec = 2 w_p).
# evan_device: w_a=3.5, w_b=5.7 -> w_p=2.2, so w_spec=4.4 <=> Delta=1.30.
#   ./slurm/submit_sweep.sh spectator subharm_v1 -- \
#       --sweep spectator --device evan_device.json --drags false,true \
#       --drag-subharmonic --subharmonic-modes spec --specfreqs 1.28,1.30,1.32
#
# 2D frequency allocation map:
#   ./slurm/submit_sweep.sh target alloc_v1 -- \
#       --sweep target --device evan_device.json --drags false,true \
#       --wb-GHz 3.8,3.9,4.0 --spec-min-GHz 3.1 --spec-max-GHz 6.1 --spec-step-GHz 0.1
#
# Add `--operating-point <name>` to any of these to run at a CALIBRATED (amp_scale,
# wp_offset) instead of the analytic pi/2 normalization. It is left off above
# because the shipped devices carry no saved points -- write one first with
# snail_calibration_map.slurm, then list them with
#   uv run python -m snail_solver.operating_points --device <dev>
# ------------------------------------------------------------------------------
#
# Environment knobs:
#   LAUNCHER   how to run python           (default "uv run python")
#   CHUNK      points per array task       (default 1; raise when N > MaxArraySize)
#   THROTTLE   max concurrent array tasks  (default 32)
#   CPUS       --cpus-per-task for the sweep array (raise for --stark chevrons)
#   DRY        set to 1 to print the sbatch commands without submitting
set -euo pipefail

KIND="${1:?usage: submit_sweep.sh <spectator|bare|target> <OUTDIR> -- <prepare args...>}"
OUTDIR="${2:?missing OUTDIR}"
shift 2
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no prepare args given after --"; exit 2; }

LAUNCHER="${LAUNCHER:-uv run python}"
CHUNK="${CHUNK:-1}"
THROTTLE="${THROTTLE:-32}"
CPUS="${CPUS:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the repo root
cd "${HERE}"

case "${KIND}" in spectator|bare|target) ;; *)
  echo "KIND must be spectator|bare|target (got ${KIND})"; exit 2 ;; esac

echo "== prepare (login node) =="
${LAUNCHER} -m snail_solver.run_sweep_zhou prepare "$@" --outdir "${OUTDIR}"

GRID="results/${OUTDIR}/grid.json"
[ -f "${GRID}" ] || GRID="${OUTDIR}/grid.json"
[ -f "${GRID}" ] || { echo "cannot find grid.json for ${OUTDIR}"; exit 1; }

N=$(${LAUNCHER} -c "import json,sys; print(len(json.load(open(sys.argv[1]))['points']))" "${GRID}")
TASKS=$(( (N + CHUNK - 1) / CHUNK ))
LAST=$(( TASKS - 1 ))
echo "grid has ${N} point(s) -> array 0-${LAST} (CHUNK=${CHUNK}, throttle ${THROTTLE})"

SUB=(sbatch --parsable)
if [ "${DRY:-0}" = "1" ]; then
  echo "-- DRY RUN --"
  echo "OUTDIR=${OUTDIR} CHUNK=${CHUNK} sbatch --array=0-${LAST}%${THROTTLE} --cpus-per-task=${CPUS} slurm/snail_sweep.slurm"
  echo "OUTDIR=${OUTDIR} sbatch --dependency=afterok:<JID> slurm/snail_collect.slurm"
  echo "OUTDIR=${OUTDIR} KIND=${KIND} sbatch --dependency=afterok:<CID> slurm/snail_plot.slurm"
  exit 0
fi

JID=$(OUTDIR="${OUTDIR}" CHUNK="${CHUNK}" RUNNER=snail_solver.run_sweep_zhou \
      "${SUB[@]}" --array="0-${LAST}%${THROTTLE}" --cpus-per-task="${CPUS}" \
      slurm/snail_sweep.slurm)
echo "sweep   array job ${JID}"

CID=$(OUTDIR="${OUTDIR}" RUNNER=snail_solver.run_sweep_zhou \
      "${SUB[@]}" --dependency="afterok:${JID}" slurm/snail_collect.slurm)
echo "collect job       ${CID}  (after ${JID})"

PID=$(OUTDIR="${OUTDIR}" KIND="${KIND}" \
      "${SUB[@]}" --dependency="afterok:${CID}" slurm/snail_plot.slurm)
echo "plot    job       ${PID}  (after ${CID})"

cat <<EOF

submitted: sweep=${JID} collect=${CID} plot=${PID}
  squeue -j ${JID},${CID},${PID}
if tasks fail or time out, patch the gaps rather than rerunning everything:
  ${LAUNCHER} -m snail_solver.run_sweep_zhou missing --outdir ${OUTDIR}
  RESUME=results/${OUTDIR}/missing.txt OUTDIR=${OUTDIR} RUNNER=snail_solver.run_sweep_zhou \\
      sbatch --array=0-<M-1> slurm/snail_sweep.slurm
note: 'collect' with --dependency=afterok will NOT run if any array task fails;
      after a resume, run it manually or resubmit snail_collect.slurm.
EOF
