#!/usr/bin/env bash
# ONE box's share of the eta = 1.3 curve over delta = -250 .. +250 MHz in 1 MHz steps
# (501 columns, delta = 0 kept). Launched on every box by scripts/launch_1mhz_wide.sh;
# by hand it is
#
#   SHARD=1 NSHARDS=4 scripts/run_1mhz_wide.sh
#
# The physics and every knob are scripts/run_5mhz_fine.sh's: ridge chirp tracked
# through rejected rows, 41-point chevrons, step-3 residual capped at 1 linewidth.
#
#   pass A  (--max-drag-channels 0 --envelope-m 3) -> passA_fine/   measures the
#           chevrons; bare + chirp only
#   pass C  (--drag-set subharm-leak)              -> passC_fine/   chirp + DRAG with
#           the corrected beat signs and Stark shifts, REPLAYING this box's pass A
#           chevrons
#
# --shard takes every NSHARDS-th column, so pass C on shard i replays exactly the
# chevrons pass A on shard i measured -- keep the host -> shard mapping fixed between
# passes (the launcher does). Each shard writes its own .h5; the finalize globs them.
# Re-runnable: finished columns are cached and a pass with a .done sentinel is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

SHARD=${SHARD:?set SHARD, e.g. SHARD=0}
NSHARDS=${NSHARDS:?set NSHARDS, e.g. NSHARDS=4}
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_1MHz_2026-10-23}
LEVELS=${LEVELS:-9}
DEVICE=${DEVICE:-6Gate4.7SNAIL.json}
AMP=${AMP:-41}
WP_POINTS=${WP_POINTS:-41}
JOBS=${JOBS:-6}
WORKERS=${WORKERS:-$(( $(nproc) / JOBS ))}
ETA=${ETA:-1.3}
PASSES=${PASSES:-"A C"}
OFFSETS=${OFFSETS:-'-0.25:0.25:501'}
RESIDUAL_CAP=${RESIDUAL_CAP:-1}

COMMON=(--device "$DEVICE" --branch above --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points "$WP_POINTS" --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback --drag-decouple-fallback
        --keep-origin --target-eta "$ETA" --chirp-source ridge
        --residual-cap "$RESIDUAL_CAP"
        --shard "$SHARD/$NSHARDS"
        --column-workers "$WORKERS" --jobs "$JOBS")

TAG="eta$(printf '%g' "$ETA" | tr '.' 'p')_wide_s${SHARD}of${NSHARDS}"

run_pass() {                                   # run_pass DIR FLAGS...
    local dir="$1"; shift
    mkdir -p "$dir"
    # Sentinel, not the .h5: the scan truncates its HDF5 up front.
    if [ -f "$dir/$TAG.done" ]; then
        echo "[$(date +%H:%M:%S)] skip $dir/$TAG (already finished)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start $dir/$TAG  shard $SHARD/$NSHARDS  offsets=$OFFSETS"
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" "$@" \
        --offsets="$OFFSETS" --outdir "$dir" --out "$dir/$TAG.h5" \
        --log "$dir/$TAG.log" --plot "$dir/$TAG.png"
    rc=$?
    [ "$rc" -eq 0 ] && touch "$dir/$TAG.done"
    echo "[$(date +%H:%M:%S)] done  $dir/$TAG  (exit $rc)"
    return "$rc"
}

echo "[$(date +%H:%M:%S)] 1 MHz wide run on $(hostname) -> $OUT  eta=$ETA  (${WORKERS}x${JOBS})"
for pass in $PASSES; do
    case "$pass" in
        A) run_pass "$OUT/passA_fine" --max-drag-channels 0 --envelope-m 3 || exit 1 ;;
        C) if ! ls "$OUT"/passA_fine/columns/*_rabi.npz >/dev/null 2>&1; then
               echo "pass C replays pass A's chevrons; run pass A first" >&2
               exit 1
           fi
           run_pass "$OUT/passC_fine" \
               --max-drag-channels 3 --envelope-m 3 --drag-set subharm-leak \
               --replay-rabi "$OUT/passA_fine" || exit 1 ;;
        *) echo "unknown pass $pass" >&2; exit 1 ;;
    esac
done
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS finished"
