#!/bin/bash
#SBATCH --job-name=snail_subharm
# NOTE: #SBATCH directives are parsed before any shell runs, so they cannot
# reference $DEVICE, and any directory in --output must pre-exist.
#SBATCH --output=snail_subharm_%j.out
#SBATCH --error=snail_subharm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # calibration fans out over COLUMNS, scoring over
                                 # CELLS; JOBS defaults to this.
#SBATCH --mem=32G
#SBATCH --time=12:00:00          # ~(WP_POINTS + TG_POINTS + 19) exact solves per
                                 #   column, plus ~5 per cell. A cell at 11 coupler
                                 #   levels is a 99-dimensional propagator over four
                                 #   columns -- the dominant cost at the top of the
                                 #   grid, not at the bottom.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
#
# =============================================================================
# Truncation convergence vs distance from the SNAIL subharmonic
#   (snail_solver.subharmonic_convergence)
# =============================================================================
# Answers, experimentally: with a FIXED coupler dimension, how far does the gate
# have to sit from the SNAIL subharmonic before the simulation converges?
#
# The map is coupler_levels (y) against Delta_sub = w_s - 2 w_p (x), coloured by
# fidelity, with the cells that agree with the largest truncation outlined. Each
# column moves the PARTNER QUBIT -- w_b = w_a + (w_s - Delta_sub)/2 -- so the pump
# walks 2 w_p away from w_s. The iSWAP rate 6 g3 lam_a lam_b eta carries no w_p, so
# at fixed TARGET_ETA every column runs at the SAME nominal length and drive: the
# only thing that changes along the axis is the spurious channel's detuning.
#
# WHY THE MODEL STOPS CONVERGING. g3 X^3 contains 3 g3 eta^2 s^dag at detuning
# Delta_sub -- a linear drive on the SNAIL with NO participation suppression, where
# the gate itself is down by lam^2 ~ 0.01. It displaces the coupler by
# |alpha| ~ 3 g3 eta^2 / Delta_sub, and a coherent state has weight on every Fock
# level, so close in, the truncation IS the regularizer (docs/chirped-recursive-drag.md,
# section 4: the 5-vs-9-level spread collapses 0.232 -> 0.0002 as the subharmonic is
# detuned).
#
# RUN --dry-run FIRST, ALWAYS. Moving the pump also walks the gate past every other
# pump-activated resonance, and a column sitting on one measures that collision
# instead of the truncation. DRY=1 prints where those land on this device's axis,
# flags any column within 150 MHz of one, reports the |alpha| range TARGET_ETA
# implies (too weak a drive and the map is flat by construction) and totals the
# solves -- all without a single solve.
#
# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------
#   # what the grid would do, no solves
#   DEVICE=4Gate4.5SNAIL.json TARGET_ETA=1.2 DRY=1 ./slurm/snail_subharmonic.slurm
#
#   # the map
#   DEVICE=4Gate4.5SNAIL.json TARGET_ETA=1.2 sbatch slurm/snail_subharmonic.slurm
#
#   # a clean window on this device: inside 1.0 GHz, where 2 w_p = w_a lands
#   DEVICE=4Gate4.5SNAIL.json TARGET_ETA=1.2 DETUNINGS=0.08:0.8:8:log \
#       LEVELS=3,5,7,9,11 sbatch slurm/snail_subharmonic.slurm
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   DEVICE      (required) device JSON (bare name resolves under devices/)
#   TARGET_ETA  (required) peak |eta| held fixed in every column. The effect scales
#               as eta^2, so this is the number the map is ABOUT; there is no
#               default. |alpha| ~ 1 is where truncation is in question, i.e.
#               Delta_sub ~ 3 g3 eta^2.
#   DETUNINGS   Delta_sub grid in GHz, lo:hi:n[:log] or a comma list
#                                                          [0.1:2.2:8:log]
#   LEVELS      coupler truncations to score               [3,5,7,9,11]
#   REF_LEVELS  the truncation every other one is measured against  [max LEVELS]
#   CALIB_LEVELS  truncation the per-column calibration runs at     [max LEVELS]
#   TOL         |F(N) - F(N_ref)| below which N is called converged     [2e-3]
#   BRANCH      put the partner above or below w_a               [above]
#   PER_CELL    set to 1 to calibrate every (Delta_sub, N) cell separately: what a
#               simulation run entirely at N levels would have reported, its own
#               calibration error included. One calibration per CELL.      [0]
#   WP_POINTS   offset points in each column's shaped chevron          [41]
#   WP_SPAN     fixed chevron span in MHz; default sizes it from the exchange
#               linewidth at each column's length                     [auto]
#   TG_POINTS   length-scan grid points                               [13]
#   TG_LO/TG_HI length-scan window as a fraction of t_g0 = 2A/eta*  [0.7 / 1.3]
#               Widen it near the subharmonic: the coupler dressing moves the true
#               full swap, and a `railed` length is a bound, not a calibration.
#   N_TIME      chevron readout times (one solve covers all)         [161]
#   PASSES      offset+length re-fits per column; one IS the fixed point with a
#               flat carrier at fixed |eta|                             [1]
#   CHIRP       Legendre chirp (GHz) to run in every column   [flat carrier]
#   JOBS        parallel workers                     [--cpus-per-task]
#   GPU         set to 1 for qutip-jax/diffrax on EVERYTHING (forces --jobs 1) [0]
#   GPU_LEVELS_MIN  score cells at this many coupler levels and above on the GPU,
#               keeping the smaller ones in the CPU pool.        [off]
#               MEASURED ON THIS PROBLEM IT IS A LOSS: on a GH200 at t_g = 209 ns,
#               a 13-level cell took 761 s on the GPU against 128 s on one CPU
#               core -- 6x slower at 95% device utilisation, with F agreeing to
#               six digits. dim ~ 162 with a time-dependent coefficient is far
#               below the JAX/diffrax crossover, and the CPU alternative is JOBS
#               independent cells at once, not one core. Use it as a cross-check
#               of the CPU path, not to go faster.
#   OUTDIR      per-column and per-cell caches           [results/subharm_<dev>]
#   OUT         the map JSON                        [<OUTDIR>/convergence.json]
#   PLOT        set to 0 to skip the figure                             [1]
#   PLOT_OUT    figure path      [figs/subharm_<dev>_eta<eta>/convergence_map.png]
#   OVERWRITE   set to 1 to ignore the caches and re-solve               [0]
#   DRY         set to 1 to print the grid, the collisions and the cost, then exit
#   LAUNCHER / CODE_DIR   as in snail_sweep.slurm (CODE_DIR = repo root)
#
# RESUMABLE. Every column's calibration and every cell's score is written to
# OUTDIR the moment it succeeds, and re-read on the next run, so a job that hits
# the wall clock is resumed by resubmitting it verbatim -- only the missing cells
# are solved. OVERWRITE=1 is the way to force a re-solve.
#
# READING THE OUTPUT. The boundary table is the answer: "5 levels: converged for
# |Delta_sub| >= 0.53 GHz" is the number to design against. Check it against three
# things before quoting it: (1) whether the reference truncation itself is
# converged -- if the two largest N in the grid disagree, the whole column is
# uncalibrated and the boundary is a lower bound; (2) the landmark rules in the
# figure -- a dip on one is a collision, not a truncation failure; (3) `railed` in
# the per-column calibration, since a length optimum on the grid edge is not an
# optimum.
# =============================================================================
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
TARGET_ETA="${TARGET_ETA:?set TARGET_ETA=<peak |eta| the map is measured at>}"
LAUNCHER="${LAUNCHER:-uv run python}"
DETUNINGS="${DETUNINGS:-0.1:2.2:8:log}"
LEVELS="${LEVELS:-3,5,7,9,11}"
TOL="${TOL:-2e-3}"
BRANCH="${BRANCH:-above}"
WP_POINTS="${WP_POINTS:-41}"
TG_POINTS="${TG_POINTS:-13}"
TG_LO="${TG_LO:-0.7}"
TG_HI="${TG_HI:-1.3}"
N_TIME="${N_TIME:-161}"
PASSES="${PASSES:-1}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"
OUTDIR="${OUTDIR:-results/subharm_${DEVICE%.json}}"
OUT="${OUT:-${OUTDIR}/convergence.json}"
PLOT="${PLOT:-1}"
PLOT_OUT="${PLOT_OUT:-figs/subharm_${DEVICE%.json}_eta${TARGET_ETA}/convergence_map.png}"

OPTS=""
[ -n "${REF_LEVELS:-}"   ] && OPTS="${OPTS} --ref-levels ${REF_LEVELS}"
[ -n "${CALIB_LEVELS:-}" ] && OPTS="${OPTS} --calib-levels ${CALIB_LEVELS}"
[ -n "${WP_SPAN:-}"      ] && OPTS="${OPTS} --wp-span-MHz ${WP_SPAN}"
[ -n "${CHIRP:-}"        ] && OPTS="${OPTS} --chirp ${CHIRP}"
[ -n "${GPU_FLAG}"       ] && OPTS="${OPTS} ${GPU_FLAG}"
[ -n "${GPU_LEVELS_MIN:-}" ] && OPTS="${OPTS} --gpu-levels-min ${GPU_LEVELS_MIN}"
[ "${PER_CELL:-0}"  = "1" ] && OPTS="${OPTS} --recalibrate-per-cell"
[ "${OVERWRITE:-0}" = "1" ] && OPTS="${OPTS} --overwrite"
[ "${DRY:-0}"       = "1" ] && OPTS="${OPTS} --dry-run"
[ "${PLOT}"         = "1" ] && OPTS="${OPTS} --plot ${PLOT_OUT}"

cd "${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "device=${DEVICE}  target_eta=${TARGET_ETA}  detunings=${DETUNINGS}"
echo "levels=${LEVELS}  branch=${BRANCH}  tol=${TOL}  jobs=${JOBS}"
echo "outdir=${OUTDIR}"

# --detunings/--chirp can start with a '-' (the axis has two sides, and the
# 'below' branch mirrors it), which argparse reads as a flag without the '='.
${LAUNCHER} -m snail_solver.subharmonic_convergence \
    --device "${DEVICE}" \
    --target-eta "${TARGET_ETA}" \
    --detunings="${DETUNINGS}" \
    --levels "${LEVELS}" \
    --tol "${TOL}" \
    --branch "${BRANCH}" \
    --wp-points "${WP_POINTS}" \
    --tg-points "${TG_POINTS}" \
    --tg-lo "${TG_LO}" \
    --tg-hi "${TG_HI}" \
    --n-time "${N_TIME}" \
    --passes "${PASSES}" \
    --jobs "${JOBS}" \
    --outdir "${OUTDIR}" \
    --out "${OUT}" \
    ${OPTS}
