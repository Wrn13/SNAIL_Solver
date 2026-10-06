#!/usr/bin/env bash
# The eta = 1.3 5 MHz curve RE-MEASURED on a finer offset grid, with the chirp built
# from the measured ridge (tracked through rejected rows).
#
#   scripts/run_5mhz_fine.sh               # then: scripts/finalize_5mhz_fine.sh
#
# Why: the stored chevrons have 15 offset points ~2 MHz apart against a 2-3 MHz peak
# half-width, so near the resonances (-70..+35 MHz) the centre fit is coarse and a
# competing peak in the window can still hide the ridge. --wp-points 41 puts ~0.6-
# 0.8 MHz between points.
#
#   pass A  (--max-drag-channels 0 --envelope-m 3) -> passA_fine/   MEASURES the
#           chevrons (Rabi, DRAG off) at the fine grid; bare + chirp only
#   pass C  (--drag-set subharm-leak)              -> passC_fine/   chirp + fixed DRAG,
#           REPLAYING pass A's fine chevrons (the Rabi stage plays no DRAG, so one
#           measurement serves both passes)
#
# Cost (from the 15-point run's stored per-column times): the Rabi stage scales with
# the offset count, ~1.1 h -> ~3 h per column at 41 points, so pass A is roughly
# 71 x 3 h / 12 workers x 1.5 overhead ~ 25 h; pass C replays, ~7 h. About 1.5 days
# in all. Interruptible: finished columns are cached per column, and a re-run of this
# script resumes (a pass with a .done sentinel is skipped).
#
# Step 3's constant carrier correction is capped at RESIDUAL_CAP (1) linewidth, so a
# weak peak away from the measured ridge cannot drag the whole chirp off it.
#
# Knobs: WP_POINTS (41), OFFSETS (all 71 ticks), ETA (1.3), WORKERS x JOBS (12 x 6),
# PASSES ("A C"), RESIDUAL_CAP (1), TAG (fine). Pass C needs pass A's chevrons, so
# run A first. A pilot on a few columns (TAG=pilot OFFSETS=...) writes its own .h5;
# the full run then serves those columns from the shared per-column cache.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
LEVELS=${LEVELS:-9}
DEVICE=${DEVICE:-6Gate4.7SNAIL.json}
AMP=${AMP:-41}
WP_POINTS=${WP_POINTS:-41}
WORKERS=${WORKERS:-12}
JOBS=${JOBS:-6}
ETA=${ETA:-1.3}
PASSES=${PASSES:-"A C"}
OFFSETS=${OFFSETS:-'-0.19:0.16:71'}
RESIDUAL_CAP=${RESIDUAL_CAP:-1}
TAG=${TAG:-fine}

# Same physics as scripts/run_5mhz_ridge.sh except the offset grid.
COMMON=(--device "$DEVICE" --branch above --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points "$WP_POINTS" --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback --drag-decouple-fallback
        --keep-origin --target-eta "$ETA" --chirp-source ridge
        --residual-cap "$RESIDUAL_CAP"
        --column-workers "$WORKERS" --jobs "$JOBS")

run_pass() {                                   # run_pass DIR TAG FLAGS...
    local dir="$1" tag="$2"; shift 2
    mkdir -p "$dir"
    # Sentinel, not the .h5: the scan truncates its HDF5 up front.
    if [ -f "$dir/$tag.done" ]; then
        echo "[$(date +%H:%M:%S)] skip $dir/$tag (already finished)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start $dir/$tag  wp-points=$WP_POINTS  offsets=$OFFSETS"
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" "$@" \
        --offsets="$OFFSETS" --outdir "$dir" --out "$dir/$tag.h5" \
        --log "$dir/$tag.log" --plot "$dir/$tag.png"
    rc=$?
    [ "$rc" -eq 0 ] && touch "$dir/$tag.done"
    echo "[$(date +%H:%M:%S)] done  $dir/$tag  (exit $rc)"
    return "$rc"
}

etag="eta$(printf '%g' "$ETA" | tr '.' 'p')"
echo "[$(date +%H:%M:%S)] 5 MHz fine-grid run -> $OUT  eta=$ETA  (${WORKERS}x${JOBS})"
for pass in $PASSES; do
    case "$pass" in
        A) run_pass "$OUT/passA_fine" "${etag}_${TAG}" \
               --max-drag-channels 0 --envelope-m 3 ;;
        C) if ! ls "$OUT"/passA_fine/columns/*_rabi.npz >/dev/null 2>&1; then
               echo "pass C replays pass A's fine chevrons; run pass A first" >&2
               exit 1
           fi
           run_pass "$OUT/passC_fine" "${etag}_${TAG}" \
               --max-drag-channels 3 --envelope-m 3 --drag-set subharm-leak \
               --replay-rabi "$OUT/passA_fine" ;;
        *) echo "unknown pass $pass" >&2; exit 1 ;;
    esac
done
echo "[$(date +%H:%M:%S)] all passes finished; now run scripts/finalize_5mhz_fine.sh"
