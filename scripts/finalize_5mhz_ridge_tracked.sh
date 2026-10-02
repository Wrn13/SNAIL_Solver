#!/usr/bin/env bash
# The ridge-chirp figures AFTER the tracker re-solve, as a SEPARATE chart:
# figs_ridge/ (from finalize_5mhz_ridge.sh) is left as it is.
#
#   scripts/finalize_5mhz_ridge_tracked.sh
#
# The ridge tracker (ridge_chirp.track_ridge) extended the measured ridge at the
# columns in TRACKED below, and those were re-solved into eta1p3_tracked*.h5 beside
# eta1p3_full.h5. Phase B merges the files (a later file's solved row wins, so the
# re-solve supersedes the full run there) into NEW curves files:
#
#   passA_ridge -> curves_nodrag_ridge_tracked.json
#   passC_ridge -> curves_subharmleak_ridge_tracked.json
#
# seeded from curves_*_ridge.json with the TRACKED columns' traces removed, so
# only those are re-scored. Figures, caveats.txt and tables go to figs_ridge_tracked/.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
WORKERS=${WORKERS:-12}
DEVICE=${DEVICE:-devices/6Gate4.7SNAIL.json}
FIGS="$OUT/figs_ridge_tracked"
# delta (MHz) of the re-solved columns
TRACKED=${TRACKED:-"-70 -65 -10 -5 5 10 15 20 25"}

seed() {                                      # seed SRC.json DST.json
    "$PY" - "$1" "$2" "$TRACKED" <<'PYEOF'
import json, os, sys
src, dst, tracked = sys.argv[1], sys.argv[2], {int(v) for v in sys.argv[3].split()}
rows = json.load(open(src)) if os.path.exists(src) else []
keep = [r for r in rows if round(r["delta_GHz"] * 1e3) not in tracked]
json.dump(keep, open(dst, "w"), indent=1)
print(f"seeded {dst}: {len(keep)}/{len(rows)} rows kept, tracked columns re-scored")
PYEOF
}

phase_b() {                                   # phase_b DIR OUTJSON
    local dir="$1" outjson="$2" h5
    # sorted: eta1p3_full < eta1p3_tracked < eta1p3_tracked2, so re-solves win
    h5=$(ls "$dir"/*.h5 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')
    [[ -z "$h5" ]] && { echo "no .h5 in $dir"; return 1; }
    echo "[$(date +%H:%M:%S)] Phase B  $dir  ($h5)"
    "$PY" scripts/curve_drag_vs_bare.py "$h5" "$outjson" "$WORKERS" \
        2>&1 | tee "$dir/phaseB_tracked.log" | tail -3
}

seed "$OUT/curves_nodrag_ridge.json"      "$OUT/curves_nodrag_ridge_tracked.json"
seed "$OUT/curves_subharmleak_ridge.json" "$OUT/curves_subharmleak_ridge_tracked.json"
phase_b "$OUT/passA_ridge" "$OUT/curves_nodrag_ridge_tracked.json"      || exit 1
phase_b "$OUT/passC_ridge" "$OUT/curves_subharmleak_ridge_tracked.json" || exit 1

"$PY" scripts/backfill_chirp_exclusions.py "$OUT" \
    --passes curves_nodrag_ridge_tracked,curves_subharmleak_ridge_tracked

mkdir -p "$FIGS"
echo "[$(date +%H:%M:%S)] Phase C"
"$PY" scripts/plot_drag_curves.py --bars-only \
    "$OUT/curves_subharmleak_ridge_tracked_flagged.json" "$FIGS" "$DEVICE" \
    "$OUT/curves_nodrag_ridge_tracked_flagged.json"
"$PY" scripts/plot_drag_gain.py "$OUT/curves_subharmleak_ridge_tracked_flagged.json" \
    "$FIGS/gain_vs_detuning.png" "$DEVICE"
"$PY" scripts/paired_gain_table.py "$OUT/curves_nodrag_ridge_tracked_flagged.json" \
    "$OUT/curves_subharmleak_ridge_tracked_flagged.json" | tee "$FIGS/paired_gain.txt"

echo "[$(date +%H:%M:%S)] done -> $FIGS"
ls -la "$FIGS" | sed 's/^/  /'
