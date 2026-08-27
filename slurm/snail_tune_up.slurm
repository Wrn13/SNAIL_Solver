#!/bin/bash
#SBATCH --job-name=snail_tuneup
# NOTE: #SBATCH directives are parsed before any shell runs, so they cannot
# reference $DEVICE, and any directory in --output must pre-exist.
#SBATCH --output=snail_tuneup_%j.out
#SBATCH --error=snail_tuneup_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # the Rabi map and the length scan both fan out over
                                 # independent grid points; JOBS defaults to this.
#SBATCH --mem=32G
#SBATCH --time=08:00:00          # The Rabi sweep is AMP_POINTS x WP_POINTS exact
                                 #   sesolve trajectories, run once; the length scan
                                 #   adds ~TG_POINTS + 20 more. DRAG_BEAT adds an
                                 #   outer chirp<->length pass (<= MAX_DRAG_ITERS),
                                 #   which repeats only steps 2-4, not the sweep.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
#
# For GPU (qutip-jax/diffrax): set GPU=1 below, request a device, load CUDA, and
# note GPU forces --jobs 1 (one big solve, not a CPU offset-pool):
# #SBATCH --gres=gpu:1
# module load cuda   # and: uv pip install qutip-jax diffrax jax[cuda]
#
# =============================================================================
# Hardware-style gate tune-up for the SNAIL iSWAP  (snail_solver.tune_up)
# =============================================================================
# Runs the tune-up in the order a lab does it, rather than the order that is
# convenient to simulate:
#
#   0. (DRAG only) after the loop, the DRAG-induced shift is MEASURED directly and
#                  compared against what the quadrature model predicted -- see
#                  "CATCHING A DRAG-INDUCED STARK SHIFT" below.
#   1. Rabi        one CONSTANT-amplitude chevron per drive strength, giving the
#                  shift law delta(|eta|) = k2|eta|^2 + k4|eta|^4. Run ONCE -- it is a
#                  property of the device, not of the pulse.
#   2. chirp       evaluate that law along the pulse and project onto Legendre -> c_k
#   3. offset      residual from a SHAPED chevron with the chirp on -> wp_offset
#   4. length      sweep t_g at FIXED peak |eta|                    -> t_g_ns
#   with DRAG:     steps 2-4 iterate, because the DRAG quadrature (d eta/dt)/Delta(t)
#                  scales as 1/t_g and so makes the chirp length-dependent.
#
# WHY A CHEVRON AND NOT A CALIBRATION-MAP SLICE. Reading the resonance off a fixed-
# length (offset, amp_scale) map does not work: raising the amplitude over-rotates the
# gate, and an over-rotated pulse transfers most population slightly OFF resonance,
# because detuning is what removes the excess rotation. The per-row argmax then tracks
# rotation-angle error instead of the Stark shift. Measured on evan_device with the
# EXACT solver, peak transfer fell 0.76 -> 0.23 across the amplitude window, the ridge
# railed against the scan edge, and the fit came back at r2 = 0.58. A chevron scans
# TIME as well, so full contrast is reached on resonance at any amplitude.
#
# WHY THE AMPLITUDE IS FIXED, NOT SCANNED. The pulse AREA is pinned by the iSWAP
# normalization to A = (pi/2)/(6 g3 lam_a lam_b), which does NOT depend on t_g.
# For a Hann envelope area = amp t_g / 2, so holding the PEAK fixed means
#
#     amp_scale(t_g) = eta* t_g / (2A)     and     t_g0 = 2A / eta*
#
# and the envelope in normalized gate time, |eta(u)| = eta* cos^2(pi u / 2), stops
# depending on t_g entirely. That is the whole point: delta(u) and hence the chirp
# coefficients become length-independent, so the frequency calibration is DECOUPLED
# from the length calibration and "length is the only free parameter" is literally
# true. Scanning (offset, amp_scale) instead makes the two fight -- changing the
# amplitude changes the Stark shift changes the required offset -- which is the
# feedback calibration_map's own docstring blames for railing 1-D scans.
#
# TARGET_ETA is the ONE number you choose. It sets the drive, and hence the length
# through t_g0 = 2A/eta*. For the shipped devices (g3 = 0.06 GHz, lam = 0.1),
# A ~ 69.4 ns, so eta*=1.8 -> t_g0 ~ 77.2 ns and eta*=1.5 -> ~92.6 ns. Resubmit with
# a different TARGET_ETA to trade gate speed against leakage.
#
# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------
#   # coarse first pass
#   DEVICE=evan_device.json TARGET_ETA=1.8 AMP_POINTS=7 WP_POINTS=21 \
#       sbatch slurm/snail_tune_up.slurm
#
#   # full resolution, saved as an operating point
#   DEVICE=evan_device.json TARGET_ETA=1.8 SAVE_POINT=tuneup \
#       sbatch slurm/snail_tune_up.slurm
#
#   # exercise the chirp<->DRAG loop against a one-pump collision
#   DEVICE=evan_device.json TARGET_ETA=1.8 SPEC_ABS=4.6 DRAG_BEAT=-0.04 \
#       DRAG_N_PUMP=1 sbatch slurm/snail_tune_up.slurm
#
# EVERY solve here is exact QuTiP. There is no engine choice, and nothing to validate
# afterwards. The Rabi step measures a SUB-MHz ridge -- far more delicate than the
# population transfer the reduced model is built for, where a term dropped by the
# rotating-frame cutoff carries no transfer but does carry part of the second-order
# AC-Stark shift. Each chevron point is a single trajectory covering all readout
# times, so exactness is affordable: cost is AMP_POINTS x WP_POINTS solves.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   DEVICE      (required) device JSON (bare name resolves under devices/)
#   TARGET_ETA  (required) peak |eta| to hold fixed for the whole tune-up
#   ETA_LO/ETA_HI   Rabi amplitude window as a FRACTION of TARGET_ETA [0.4 / 1.3]
#   AMP_POINTS  drive-strength rows; each is one chevron                [9]
#   WP_SPAN     FIXED chevron offset span in MHz for every row. Unset by default:
#               each row is sized from its own linewidth instead      [auto]
#   SPAN_LINEWIDTHS  chevron half-span in estimated linewidths; rows whose fitted
#               width is too wide for their window are re-measured wider     [4]
#   WP_POINTS   chevron offset points                                  [25]
#   WINDOW_TG   chevron time window in SWAPS (window scales as 1/|eta|)   [2.0]
#   N_TIME      chevron readout times (one solve covers all)          [161]
#   TG_POINTS   length-scan grid points                              [13]
#   CHIRP_DEGREE  Legendre truncation (even terms only matter). delta(u) is an
#               EXACT degree-8 polynomial in cos(pi u/2), so the default of 8
#               reconstructs it exactly for free (algebra on the already-fit
#               k2/k4, not a new fit); below that, PLOT_RIDGE will show the
#               chirp overshoot the ridge near |eta| -> 0                  [8]
#   QUARTIC_WARN  warn (never raise) when the fitted quartic term exceeds this
#               fraction of the quadratic term at TARGET_ETA               [0.25]
#   DRAG_BEAT   beat of the process DRAG suppresses, in GHz; enables the
#               chirp<->DRAG iteration                             [none]
#   DRAG_N_PUMP pump quanta of that process: 1 one-pump, 2 subharmonic,
#               0 static (a chirp must NOT move a static beat)        [1]
#   MAX_DRAG_ITERS  cap on the self-consistency loop                  [4]
#   DRAG_SHIFT_POINTS  with DRAG_BEAT, also measure the DRAG-induced resonance
#               shift at this many drive strengths and fit its power law   [0=off]
#   SPEC_ABS    absolute spectator frequency (loads a spectator)   [none]
#   COUPLER_LEVELS  coupler truncation override                    [device]
#   SKIP_TIME_RABI  set to 1 to skip the constant-drive cross-check    [0]
#   JOBS        parallel workers (default: --cpus-per-task)
#   GPU         set to 1 to run via qutip-jax/diffrax (forces --jobs 1; also
#               uncomment #SBATCH --gres=gpu:1 above and load CUDA)      [0]
#   OUT         JSON path for the full record + per-stage data  [tuneup_<dev>.json]
#   PLOT        set to 0 to skip rendering the Rabi sweep                  [1]
#   PLOT_OUT    figure path; defaults to its OWN folder per device+eta so
#               fanned-out runs never overwrite each other's plot
#                                          [figs/tuneup_<dev>_eta<target_eta>/rabi_chevrons.png]
#   PLOT_RIDGE  set to 0 to skip the chirp-vs-ridge overlay (Fig. 4 /
#               arXiv:2306.10162 picture)                                  [1]
#   PLOT_RIDGE_OUT  figure path, same per-run folder as PLOT_OUT by default
#                                             [figs/tuneup_<dev>_eta<target_eta>/chirp_ridge.png]
#   SAVE_POINT  save the result into the device JSON under this name  [none]
#   OVERWRITE   set to 1 to replace an existing SAVE_POINT             [0]
#   LAUNCHER / CODE_DIR   as in snail_sweep.slurm (CODE_DIR = repo root)
#
# CATCHING A DRAG-INDUCED STARK SHIFT. The chirp is built from a CONSTANT probe,
# which has d eta/dt = 0 and is therefore blind to DRAG. DRAG is folded in
# analytically instead -- its quadrature raises the total drive, and the shift is
# assumed to follow the same measured law -- and that assumption is never tested by
# the steps that depend on it. So whenever DRAG_BEAT is set the run ALSO measures the
# shift directly, with shaped DRAG-off/DRAG-on chevrons, and reports the excess over
# the model prediction; a disagreement above 50% is flagged in the log. Set
# DRAG_SHIFT_POINTS=4 to get the shift versus DRIVE STRENGTH as well: the expected
# power law is |eta|^4 ("DRAG only adds drive", already covered by the chirp), so an
# exponent far from 4 identifies a mechanism the model does not contain. Both are
# recorded under stages.drag_measured / stages.drag_shift_table in the output JSON.
#
# READING THE OUTPUT: the result is a complete operating point -- t_g_ns, amp_scale,
# wp_offset_GHz, chirp_coeffs_GHz -- ready for `--operating-point` in the sweeps or
# as the starting point for snail_grape.slurm. Check three things before trusting
# it: the Rabi fit's r2 (the run HARD-FAILS below 0.9 rather than emitting a
# confident wrong chirp), `railed` on the length scan (a maximum on the grid edge is
# not a maximum), and, when DRAG is on, that the loop CONVERGED -- it raises rather
# than returning the last iterate.
# =============================================================================
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
TARGET_ETA="${TARGET_ETA:?set TARGET_ETA=<peak |eta| to hold fixed>}"
LAUNCHER="${LAUNCHER:-uv run python}"
ETA_LO="${ETA_LO:-0.4}"
ETA_HI="${ETA_HI:-1.3}"
AMP_POINTS="${AMP_POINTS:-9}"
WP_POINTS="${WP_POINTS:-25}"
SPAN_LINEWIDTHS="${SPAN_LINEWIDTHS:-4}"
DRAG_SHIFT_POINTS="${DRAG_SHIFT_POINTS:-0}"
WINDOW_TG="${WINDOW_TG:-2.0}"
N_TIME="${N_TIME:-161}"
TG_POINTS="${TG_POINTS:-13}"
CHIRP_DEGREE="${CHIRP_DEGREE:-8}"
QUARTIC_WARN="${QUARTIC_WARN:-0.25}"
DRAG_N_PUMP="${DRAG_N_PUMP:-1}"
MAX_DRAG_ITERS="${MAX_DRAG_ITERS:-4}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"   # GPU=1 -> qutip-jax/diffrax (forces --jobs 1)
OUT="${OUT:-tuneup_${DEVICE%.json}.json}"
PLOT="${PLOT:-1}"
# Own folder per device+eta, so DEVICE/TARGET_ETA alone -- with no other override --
# already keeps fanned-out runs (e.g. from submit_tune_up.sh) from clobbering each
# other's figure.
PLOT_OUT="${PLOT_OUT:-figs/tuneup_${DEVICE%.json}_eta${TARGET_ETA}/rabi_chevrons.png}"
PLOT_RIDGE="${PLOT_RIDGE:-1}"
PLOT_RIDGE_OUT="${PLOT_RIDGE_OUT:-figs/tuneup_${DEVICE%.json}_eta${TARGET_ETA}/chirp_ridge.png}"

OPTS=""
[ -n "${SPEC_ABS:-}"       ] && OPTS="${OPTS} --spec-abs-GHz ${SPEC_ABS}"
[ -n "${WP_SPAN:-}"        ] && OPTS="${OPTS} --wp-span-MHz ${WP_SPAN}"
[ -n "${DRAG_BEAT:-}"      ] && OPTS="${OPTS} --drag-beat-GHz ${DRAG_BEAT}"
[ -n "${COUPLER_LEVELS:-}" ] && OPTS="${OPTS} --coupler-levels ${COUPLER_LEVELS}"
[ -n "${GPU_FLAG}"         ] && OPTS="${OPTS} ${GPU_FLAG}"
[ "${SKIP_TIME_RABI:-0}" = "1" ] && OPTS="${OPTS} --skip-time-rabi"
[ "${PLOT}" = "1"          ] && OPTS="${OPTS} --plot ${PLOT_OUT}"
[ "${PLOT_RIDGE}" = "1"    ] && OPTS="${OPTS} --plot-ridge ${PLOT_RIDGE_OUT}"
[ -n "${SAVE_POINT:-}"     ] && OPTS="${OPTS} --save-point ${SAVE_POINT}"
[ "${OVERWRITE:-0}" = "1" ] && OPTS="${OPTS} --overwrite"

# One process that fans out internally -> leave BLAS a thread each so the worker
# pool is not oversubscribed.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

# module load python/3.12      # or: conda activate qsim
# source .venv/bin/activate    # (then set LAUNCHER=python)
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} device=${DEVICE} target_eta=${TARGET_ETA}"
echo "all-exact QuTiP; jobs=${JOBS}  gpu=${GPU:-0}"
echo "rabi ${AMP_POINTS} chevrons x ${WP_POINTS} offsets over |eta|/eta* in "\
"[${ETA_LO},${ETA_HI}]; span=${WP_SPAN:-${SPAN_LINEWIDTHS} linewidths}; "\
"window=${WINDOW_TG} swaps @ ${N_TIME} times; length grid=${TG_POINTS}"
echo "chirp degree=${CHIRP_DEGREE}  drag=${DRAG_BEAT:-<off>} k=${DRAG_N_PUMP} (<= ${MAX_DRAG_ITERS} passes)"
echo "opts=${OPTS:-<none>}  out=${OUT}"
echo "start=$(date -Is)"

srun ${LAUNCHER} -m snail_solver.tune_up \
    --device "${DEVICE}" --target-eta "${TARGET_ETA}" \
    --window-tg "${WINDOW_TG}" --n-time "${N_TIME}" \
    --eta-lo "${ETA_LO}" --eta-hi "${ETA_HI}" --amp-points "${AMP_POINTS}" \
    --wp-points "${WP_POINTS}" --span-linewidths "${SPAN_LINEWIDTHS}" \
    --drag-shift-points "${DRAG_SHIFT_POINTS}" \
    --tg-points "${TG_POINTS}" --chirp-degree "${CHIRP_DEGREE}" \
    --quartic-warn "${QUARTIC_WARN}" \
    --drag-n-pump "${DRAG_N_PUMP}" --max-drag-iters "${MAX_DRAG_ITERS}" \
    --jobs "${JOBS}" --out "${OUT}" \
    ${OPTS}

echo "done end=$(date -Is)"
