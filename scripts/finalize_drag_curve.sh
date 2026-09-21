#!/usr/bin/env bash
# Phase B + C over every eta that has finished, into the run's own results/ dir.
#
# Re-runnable: Phase B re-scores from the stored operating points, so running this
# again after more eta land simply widens curves.json. Safe to run while the scan is
# still going -- it only reads the .h5 files that already exist.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

OUT=${OUT:-results/drag_curve_lev9_2026-09-17}
WORKERS=${WORKERS:-6}
DEVICE=${DEVICE:-devices/6Gate4.7SNAIL.json}

H5=$(ls "$OUT"/*.h5 2>/dev/null | tr '\n' ',' | sed 's/,$//')
[[ -z "$H5" ]] && { echo "no eta*.h5 in $OUT"; exit 1; }
echo "[$(date +%H:%M:%S)] Phase B over: $H5"
.venv/bin/python scripts/curve_drag_vs_bare.py "$H5" "$OUT/curves.json" "$WORKERS" \
    2>&1 | tee "$OUT/phaseB.log" | tail -3

echo "[$(date +%H:%M:%S)] Phase C"
.venv/bin/python scripts/plot_drag_curves.py "$OUT/curves.json" "$OUT" "$DEVICE"
.venv/bin/python scripts/plot_drag_gain.py "$OUT/curves.json" "$OUT/gain_vs_detuning.png"
# The channel diagram, from the best column that actually calibrated.
BEST=$(ls "$OUT"/columns/col_*.json 2>/dev/null | head -1)
[[ -n "$BEST" ]] && .venv/bin/python scripts/plot_channel_map.py "$BEST" \
    "$OUT/channel_map.png"
echo "[$(date +%H:%M:%S)] done -> $OUT"
ls -la "$OUT"/*.png "$OUT"/*.json "$OUT"/table.txt 2>/dev/null | sed 's/^/  /'
