#!/usr/bin/env bash
# Unattended: finish the tracker re-solve, then build figs_ridge_tracked/.
#
#   nohup setsid scripts/chain_5mhz_ridge_tracked.sh > OUT/chain_tracked.log 2>&1 &
#
#   1. wait for the 9-column re-solve (run_5mhz_ridge.sh TAG=tracked) to finish
#   2. re-solve pass A at delta = -5 MHz (TAG=tracked2): its tracked pass A started
#      before the tracker learned the rail rule, so its ridge railed
#   3. wait for finalize_5mhz_ridge.sh (figs_ridge/) so the two do not compete
#   4. scripts/finalize_5mhz_ridge_tracked.sh -> figs_ridge_tracked/
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}

echo "[$(date +%H:%M:%S)] waiting for the tracked re-solve"
until grep -q "all passes finished" "$OUT/ridge_tracked.driver.log" 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] re-solving pass A at delta = -5 MHz"
TAG=tracked2 EXTRA=--overwrite PASSES=A WORKERS=1 JOBS=12 OFFSETS='-0.005' \
    scripts/run_5mhz_ridge.sh

echo "[$(date +%H:%M:%S)] waiting for figs_ridge/"
until grep -q "done ->" "$OUT/finalize_ridge.log" 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] building figs_ridge_tracked/"
scripts/finalize_5mhz_ridge_tracked.sh
echo "[$(date +%H:%M:%S)] chain finished"
