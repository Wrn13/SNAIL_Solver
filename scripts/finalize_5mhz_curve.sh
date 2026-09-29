#!/usr/bin/env bash
# Phase B + C for the 5 MHz curve. Re-runnable and safe to run while the scan is
# still going: it only reads the .h5 files that already exist, and Phase B reuses
# any trace already in its output json (keyed on delta, eta, variant).
#
#   scripts/finalize_5mhz_curve.sh
#
# The two passes are finalized SEPARATELY and stay separate, because they carry the
# same (delta, eta, variant) keys from different calibrations:
#
#   passA_nodrag -> curves_nodrag.json   "chirp+DRAG" trace = chirp only (no channels)
#                                        "bare"       trace = the authoritative bare
#   passB_drag   -> curves_drag.json     "chirp+DRAG" trace = the real thing
#
# plot_drag_curves.py then takes curves_drag.json as the series and
# curves_nodrag.json as the 4th positional BASELINE.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
WORKERS=${WORKERS:-12}
DEVICE=${DEVICE:-devices/6Gate4.7SNAIL.json}

phase_b() {                                   # phase_b DIR OUTJSON
    local dir="$1" outjson="$2"
    local h5
    h5=$(ls "$dir"/*.h5 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$h5" ]]; then
        echo "[$(date +%H:%M:%S)] no .h5 in $dir -- skipping"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] Phase B  $dir"
    "$PY" scripts/curve_drag_vs_bare.py "$h5" "$outjson" "$WORKERS" \
        2>&1 | tee "$dir/phaseB.log" | tail -3
}

phase_b "$OUT/passA_nodrag" "$OUT/curves_nodrag.json" ; A=$?
phase_b "$OUT/passB_drag"   "$OUT/curves_drag.json"   ; B=$?

if [[ $B -ne 0 ]]; then
    echo "no chirp+DRAG pass yet; nothing to plot against"
    exit 0
fi

BASE=""
[[ $A -eq 0 ]] && BASE="$OUT/curves_nodrag.json"

echo "[$(date +%H:%M:%S)] Phase C"
# --bars-only: the trend curves were fits over columns whose structure is PHYSICS
# (two A poles about -30, plus a monotone SNAIL background), so a smooth line
# through them implies a continuity the data does not have.
"$PY" scripts/plot_drag_curves.py --bars-only "$OUT/curves_drag.json" "$OUT/figs" \
    "$DEVICE" $BASE
"$PY" scripts/plot_drag_gain.py "$OUT/curves_drag.json" "$OUT/gain_vs_detuning.png"

BEST=$(ls "$OUT"/passB_drag/columns/col_*.json 2>/dev/null | head -1)
[[ -n "$BEST" ]] && "$PY" scripts/plot_channel_map.py "$BEST" "$OUT/channel_map.png"

# The paired comparison. Medians over DIFFERENT column sets are not comparable, so
# every ratio below is computed over columns where both series exist.
"$PY" scripts/paired_gain_table.py "$OUT/curves_nodrag.json" "$OUT/curves_drag.json" \
    | tee "$OUT/paired_gain.txt"

echo "[$(date +%H:%M:%S)] done -> $OUT"
ls -la "$OUT"/*.png "$OUT"/*.json "$OUT"/*.txt 2>/dev/null | sed 's/^/  /'
