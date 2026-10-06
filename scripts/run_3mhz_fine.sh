#!/usr/bin/env bash
# The eta = 1.3 curve on a 3 MHz grid with fine (41-point) chevrons, end to end:
#
#   nohup setsid scripts/run_3mhz_fine.sh > results/drag_curve_3MHz_2026-10-02/driver.log 2>&1 &
#
#   grid     delta = -189 .. +159 MHz in 3 MHz steps (117 columns, 0 included)
#   pass A   measures the chevrons; bare + chirp only          -> passA_fine/
#   pass C   replays them; chirp + the fixed DRAG set           -> passC_fine/
#            (subharmonic 0->1, subharmonic 1->2, |2> leakage), the beats following
#            the pump carrier
#   then     Phase B + figures                                  -> figs_fine/
#
# The physics and every knob are scripts/run_5mhz_fine.sh's (ridge chirp, tracked
# ridge, step-3 residual capped at 1 linewidth); only OUT and the grid differ.
# Re-runnable: finished columns are cached and a finished pass is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
export OUT=${OUT:-results/drag_curve_3MHz_2026-10-02}
export OFFSETS=${OFFSETS:-'-0.189:0.159:117'}
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] 3 MHz fine-grid scan -> $OUT  offsets=$OFFSETS"
scripts/run_5mhz_fine.sh
echo "[$(date +%H:%M:%S)] scan finished; Phase B + figures"
scripts/finalize_5mhz_fine.sh > "$OUT/finalize.log" 2>&1
tail -5 "$OUT/finalize.log"
echo "[$(date +%H:%M:%S)] weekend run finished"
