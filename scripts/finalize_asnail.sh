#!/usr/bin/env bash
# Phase B + figures for scripts/run_asnail.sh. Re-runnable; reads existing .h5 only.
#
#   OUT=results/asnail_10MHz_2026-10-26 scripts/finalize_asnail.sh
#
#   passA_fine -> curves_nodrag_fine.json        bare + chirp
#   passC_fine -> curves_subharmleak_fine.json   chirp + DRAG (the RANKED set here; the
#                 name is the one backfill_chirp_exclusions.py keys pass C on)
# then the flagged copies and $OUT/figs/asnail_map.png (+ table.txt).
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:?set OUT}
WORKERS=${WORKERS:-12}

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

passes=curves_nodrag_fine
phase_b "$OUT/passA_fine" "$OUT/curves_nodrag_fine.json" || exit 1
phase_b "$OUT/passC_fine" "$OUT/curves_subharmleak_fine.json" \
    && passes=$passes,curves_subharmleak_fine

# Exits non-zero when no column is chirp-free; the plot then reads the unflagged files.
"$PY" scripts/backfill_chirp_exclusions.py "$OUT" --passes "$passes" \
    || echo "(backfill wrote nothing; plotting unflagged curves)"

mkdir -p "$OUT/figs"
"$PY" scripts/plot_asnail_map.py "$OUT" "$OUT/figs/asnail_map" | tee "$OUT/figs/table.txt"
echo "[$(date +%H:%M:%S)] done -> $OUT/figs"
