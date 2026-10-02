#!/usr/bin/env bash
# Phase B + C for scripts/run_5mhz_fine.sh. Re-runnable, and safe while the scan is
# still going (reads existing .h5 only; Phase B reuses scored traces).
#
#   scripts/finalize_5mhz_fine.sh
#
#   passA_fine -> curves_nodrag_fine.json       bare + chirp only (the baseline)
#   passC_fine -> curves_subharmleak_fine.json  chirp + the fixed DRAG set
#
# Flagged with --passes so this run's chirp-free verdicts never mix with the other
# runs', then the three-series bars (and caveats.txt) in figs_fine/.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
WORKERS=${WORKERS:-12}
DEVICE=${DEVICE:-devices/6Gate4.7SNAIL.json}
FIGS="$OUT/figs_fine"

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

phase_b "$OUT/passA_fine" "$OUT/curves_nodrag_fine.json"      || exit 1
phase_b "$OUT/passC_fine" "$OUT/curves_subharmleak_fine.json" || exit 1

"$PY" scripts/backfill_chirp_exclusions.py "$OUT" \
    --passes curves_nodrag_fine,curves_subharmleak_fine

mkdir -p "$FIGS"
echo "[$(date +%H:%M:%S)] Phase C"
"$PY" scripts/plot_drag_curves.py --bars-only \
    "$OUT/curves_subharmleak_fine_flagged.json" "$FIGS" "$DEVICE" \
    "$OUT/curves_nodrag_fine_flagged.json"
"$PY" scripts/plot_drag_gain.py "$OUT/curves_subharmleak_fine_flagged.json" \
    "$FIGS/gain_vs_detuning.png" "$DEVICE"
"$PY" scripts/paired_gain_table.py "$OUT/curves_nodrag_fine_flagged.json" \
    "$OUT/curves_subharmleak_fine_flagged.json" | tee "$FIGS/paired_gain.txt"

echo "[$(date +%H:%M:%S)] done -> $FIGS"
ls -la "$FIGS" | sed 's/^/  /'
