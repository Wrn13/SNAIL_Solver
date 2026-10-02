#!/usr/bin/env bash
# Phase B + C for scripts/run_5mhz_ridge.sh. Re-runnable, and safe while the scan is
# still going (reads existing .h5 only; Phase B reuses scored traces).
#
#   scripts/finalize_5mhz_ridge.sh
#
#   passA_ridge -> curves_nodrag_ridge.json       bare + chirp only (the baseline)
#   passC_ridge -> curves_subharmleak_ridge.json  chirp + the fixed DRAG set
#
# Flagged with --passes so the ridge run's chirp-free verdicts never mix with the
# law runs', then the three-series bars in figs_ridge/.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
WORKERS=${WORKERS:-12}
DEVICE=${DEVICE:-devices/6Gate4.7SNAIL.json}
FIGS="$OUT/figs_ridge"

phase_b() {                                   # phase_b DIR OUTJSON
    local dir="$1" outjson="$2" h5
    h5=$(ls "$dir"/*_full.h5 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$h5" ]]; then
        echo "[$(date +%H:%M:%S)] no .h5 in $dir -- skipping"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] Phase B  $dir"
    "$PY" scripts/curve_drag_vs_bare.py "$h5" "$outjson" "$WORKERS" \
        2>&1 | tee "$dir/phaseB.log" | tail -3
}

phase_b "$OUT/passA_ridge" "$OUT/curves_nodrag_ridge.json"      || exit 1
phase_b "$OUT/passC_ridge" "$OUT/curves_subharmleak_ridge.json" || exit 1

"$PY" scripts/backfill_chirp_exclusions.py "$OUT" \
    --passes curves_nodrag_ridge,curves_subharmleak_ridge

mkdir -p "$FIGS"
echo "[$(date +%H:%M:%S)] Phase C"
"$PY" scripts/plot_drag_curves.py --bars-only \
    "$OUT/curves_subharmleak_ridge_flagged.json" "$FIGS" "$DEVICE" \
    "$OUT/curves_nodrag_ridge_flagged.json"
"$PY" scripts/plot_drag_gain.py "$OUT/curves_subharmleak_ridge_flagged.json" \
    "$FIGS/gain_vs_detuning.png" "$DEVICE"
"$PY" scripts/paired_gain_table.py "$OUT/curves_nodrag_ridge_flagged.json" \
    "$OUT/curves_subharmleak_ridge_flagged.json" | tee "$FIGS/paired_gain.txt"

echo "[$(date +%H:%M:%S)] done -> $FIGS"
ls -la "$FIGS" | sed 's/^/  /'
