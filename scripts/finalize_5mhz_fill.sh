#!/usr/bin/env bash
# Phase B + C for scripts/run_5mhz_fill.sh. Re-runnable, and safe while the scan is
# still going: it only reads existing .h5 files, and Phase B reuses every trace
# already in its output json.
#
#   scripts/finalize_5mhz_fill.sh
#
#   passA_nodrag      -> curves_nodrag.json       bare + chirp only (the baseline)
#   passC_subharmleak -> curves_subharmleak.json  chirp + the fixed DRAG set
#
# then the chirp-exclusion flags, and the three-series bars in figs_subharmleak/.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
WORKERS=${WORKERS:-12}
DEVICE=${DEVICE:-devices/6Gate4.7SNAIL.json}
FIGS="$OUT/figs_subharmleak"

phase_b() {                                   # phase_b DIR OUTJSON
    local dir="$1" outjson="$2" h5
    h5=$(ls "$dir"/*.h5 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$h5" ]]; then
        echo "[$(date +%H:%M:%S)] no .h5 in $dir -- skipping"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] Phase B  $dir"
    "$PY" scripts/curve_drag_vs_bare.py "$h5" "$outjson" "$WORKERS" \
        2>&1 | tee "$dir/phaseB.log" | tail -3
}

phase_b "$OUT/passA_nodrag"      "$OUT/curves_nodrag.json"      || exit 1
phase_b "$OUT/passC_subharmleak" "$OUT/curves_subharmleak.json" || exit 1

"$PY" scripts/backfill_chirp_exclusions.py "$OUT"

mkdir -p "$FIGS"
echo "[$(date +%H:%M:%S)] Phase C"
"$PY" scripts/plot_drag_curves.py --bars-only "$OUT/curves_subharmleak_flagged.json" \
    "$FIGS" "$DEVICE" "$OUT/curves_nodrag_flagged.json"
"$PY" scripts/plot_drag_gain.py "$OUT/curves_subharmleak_flagged.json" \
    "$FIGS/gain_vs_detuning.png" "$DEVICE"
"$PY" scripts/paired_gain_table.py "$OUT/curves_nodrag_flagged.json" \
    "$OUT/curves_subharmleak_flagged.json" | tee "$FIGS/paired_gain.txt"

echo "[$(date +%H:%M:%S)] done -> $FIGS"
ls -la "$FIGS" | sed 's/^/  /'
