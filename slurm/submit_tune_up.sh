#!/bin/bash
# Fan a tune-up out over devices x target drive strengths, one job per combination.
#
# TARGET_ETA is the one number the tune-up cannot choose for you: it sets the drive,
# and hence the length through t_g0 = 2A/eta*. Larger eta* means a shorter gate and
# more leakage. Passing a LIST is the point of this script -- the jobs are
# independent, so you get the whole speed/leakage trade-off in one wall-clock pass
# and pick the operating point afterwards from the summary.
#
# Every solve is exact QuTiP (see snail_tune_up.slurm for why the Rabi step cannot
# use the reduced model), so there is no explore/confirm staging: one job per
# combination is the whole calculation.
#
# Usage:
#   ./slurm/submit_tune_up.sh
#   DEVICES="a.json,b.json" TARGET_ETAS="1.5,1.8,2.1" ./slurm/submit_tune_up.sh
#
# Environment knobs:
#   DEVICES      comma-separated device JSONs
#                            [1Gate4.2SNAIL.json,2Gate4.9SNAIL.json,evan_device.json]
#   TARGET_ETAS  comma-separated peak |eta| values                    [1.8]
#   SAVE_POINT   operating-point name; the target_eta is appended when more than
#                one is requested, so runs cannot overwrite each other      [tuneup]
#   AMP_POINTS / WP_POINTS / SPAN_LINEWIDTHS / TG_POINTS   grid overrides
#   DRAG_BEAT / DRAG_N_PUMP   exercise the chirp<->DRAG loop          [off / 1]
#   DRAG_SHIFT_POINTS  with DRAG_BEAT, also measure the DRAG-induced resonance shift
#                      vs drive strength and fit its power law (4 = 'DRAG only adds
#                      drive', already covered by the chirp)               [0=off]
#   SPEC_ABS     absolute spectator frequency                           [none]
#   CPUS         --cpus-per-task                                          [16]
#   DRY          set to 1 to print the sbatch commands without submitting
set -euo pipefail

DEVICES="${DEVICES:-1Gate4.2SNAIL.json,2Gate4.9SNAIL.json,evan_device.json}"
TARGET_ETAS="${TARGET_ETAS:-1.8}"
SAVE_POINT="${SAVE_POINT:-tuneup}"
CPUS="${CPUS:-16}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the repo root
cd "${HERE}"

PASS=""
for V in AMP_POINTS WP_POINTS WP_SPAN SPAN_LINEWIDTHS TG_POINTS WINDOW_TG N_TIME \
         CHIRP_DEGREE MAX_DRAG_ITERS DRAG_SHIFT_POINTS COUPLER_LEVELS \
         DRAG_BEAT DRAG_N_PUMP SPEC_ABS; do
  [ -n "${!V:-}" ] && PASS="${PASS} ${V}=${!V}"
done

IFS=',' read -ra DEVLIST <<< "${DEVICES}"
IFS=',' read -ra ETALIST <<< "${TARGET_ETAS}"
MULTI=$([ "${#ETALIST[@]}" -gt 1 ] && echo 1 || echo 0)

echo "devices:     ${DEVICES}"
echo "target_etas: ${TARGET_ETAS}"
echo "operating point '${SAVE_POINT}'$([ "${MULTI}" = 1 ] && echo " + _eta<value>")"
echo "overrides:  ${PASS:-<defaults>}"
echo

IDS=""
for DEV in "${DEVLIST[@]}"; do
  for ETA in "${ETALIST[@]}"; do
    TAG="${DEV%.json}_eta${ETA}"
    POINT="${SAVE_POINT}"
    [ "${MULTI}" = 1 ] && POINT="${SAVE_POINT}_eta${ETA}"

    if [ "${DRY:-0}" = "1" ]; then
      echo "DEVICE=${DEV} TARGET_ETA=${ETA} SAVE_POINT=${POINT} OVERWRITE=1${PASS} \\"
      echo "    sbatch --job-name=tu_${TAG} --cpus-per-task=${CPUS} slurm/snail_tune_up.slurm"
      continue
    fi

    JID=$(env DEVICE="${DEV}" TARGET_ETA="${ETA}" SAVE_POINT="${POINT}" OVERWRITE=1 \
               OUT="tuneup_${TAG}.json" ${PASS} \
          sbatch --parsable --job-name="tu_${TAG}" --cpus-per-task="${CPUS}" \
                 slurm/snail_tune_up.slurm)
    echo "${DEV} @ eta*=${ETA}: job ${JID} -> point '${POINT}'"
    IDS="${IDS:+${IDS},}${JID}"
  done
done

[ "${DRY:-0}" = "1" ] && exit 0

cat <<EOF

submitted: ${IDS}
  squeue -j ${IDS}

when they finish:
  uv run python -m snail_solver.operating_points --device <dev>   # list saved points
  cat results/tuneup_<dev>_eta<value>.json                        # full per-stage data

then run the gate at a chosen point, e.g.
  ./slurm/submit_sweep.sh spectator tuned_v1 -- --sweep spectator \\
      --device <dev> --drags false,true --operating-point ${SAVE_POINT} ...

CHECK BEFORE TRUSTING A POINT -- all three are hard failures, so a job that finished
is a job that passed them, but the numbers are worth reading:
  * the Rabi fit r2, and 'stark_span_MHz' vs 'resid_MHz'. The run refuses to build a
    chirp when the DRIVE-DEPENDENT shift is smaller than the fit residual: the static
    offset delta0 is still calibrated, there is simply no resolved Stark shift to
    track. That is a real answer about the device, not a failure of the run.
  * 'railed' on the length scan -- a maximum on the grid edge is not a maximum.
  * with DRAG on, that the chirp<->length loop CONVERGED (it raises if it did not).
EOF
