#!/usr/bin/env bash
# ONE box's share of the A-SNAIL exchange scan: w_a = 4.0, w_s = 4.7 GHz,
# w_b = 3.3 + delta, w_p = w_a - w_b = 0.7 - delta, so delta = w_s - w_a - w_p is the
# 1-pump a s^dag beat (snail_solver.asnail_gate_scan). Launched on every box by
# scripts/launch_asnail.sh; by hand it is
#
#   SHARD=1 NSHARDS=4 MODE=coarse scripts/run_asnail.sh
#
#   MODE=coarse  delta = -200 .. +200 MHz in 10 MHz steps x eta* = 1.0 .. 1.5 (0.1)
#                -> results/asnail_10MHz_2026-10-26            246 columns
#   MODE=fine    delta = -200 .. +200 MHz in  1 MHz steps at ONE eta* (ETA=1.3 ...)
#                -> results/asnail_1MHz_eta<ETA>               401 columns
#
# Every other knob is scripts/run_1mhz_wide.sh's (ridge chirp, 41-point chevrons,
# step-3 residual cap 1 linewidth, origin kept):
#
#   pass A  (--max-drag-channels 0)  -> passA_fine/   bare + chirp only
#   pass C  (--drag-set ranked)      -> passC_fine/   chirp + DRAG, REPLAYING pass A's
#           chevrons. Ranked, not subharm-leak: on this axis the forced A-subharmonic
#           channels sit >1.3 GHz away, and the ranked set picks the a->s exchange.
#
# Keep the host -> shard mapping fixed between passes (the launcher does).
# Re-runnable: finished columns are cached and a pass with a .done sentinel is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

SHARD=${SHARD:?set SHARD, e.g. SHARD=0}
NSHARDS=${NSHARDS:?set NSHARDS, e.g. NSHARDS=4}
MODE=${MODE:-coarse}
case "$MODE" in
    coarse) OFFSETS=${OFFSETS:-'-0.2:0.2:41'}
            ETAS=${ETAS:-'1.0:1.5:6'}
            OUT=${OUT:-results/asnail_10MHz_2026-10-26}
            TAG="coarse_s${SHARD}of${NSHARDS}" ;;
    fine)   ETA=${ETA:?MODE=fine needs ETA, e.g. ETA=1.3}
            OFFSETS=${OFFSETS:-'-0.2:0.2:401'}
            ETAS=$ETA
            ETA_TAG="eta$(printf '%g' "$ETA" | tr '.' 'p')"
            OUT=${OUT:-results/asnail_1MHz_$ETA_TAG}
            TAG="${ETA_TAG}_s${SHARD}of${NSHARDS}" ;;
    *) echo "MODE=$MODE: expected coarse or fine" >&2; exit 2 ;;
esac
PY=${PY:-.venv/bin/python}
LEVELS=${LEVELS:-9}
DEVICE=${DEVICE:-devices/4.0Gate4.7SNAIL_asnail.json}
AMP=${AMP:-41}
WP_POINTS=${WP_POINTS:-41}
JOBS=${JOBS:-6}
# --all: plain nproc honours the OMP_NUM_THREADS=1 exported above and returns 1.
WORKERS=${WORKERS:-$(( $(nproc --all) / JOBS ))}
PASSES=${PASSES:-"A C"}
RESIDUAL_CAP=${RESIDUAL_CAP:-1}

COMMON=(--device "$DEVICE" --branch below --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points "$WP_POINTS" --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback --drag-decouple-fallback
        --keep-origin --target-etas "$ETAS" --chirp-source ridge
        --residual-cap "$RESIDUAL_CAP"
        --shard "$SHARD/$NSHARDS"
        --column-workers "$WORKERS" --jobs "$JOBS")

run_pass() {                                   # run_pass DIR FLAGS...
    local dir="$1"; shift
    mkdir -p "$dir"
    # Sentinel, not the .h5: the scan truncates its HDF5 up front.
    if [ -f "$dir/$TAG.done" ]; then
        echo "[$(date +%H:%M:%S)] skip $dir/$TAG (already finished)"
        return 0
    fi
    local cmd=("$PY" -m snail_solver.asnail_gate_scan "${COMMON[@]}" "$@"
               --offsets="$OFFSETS" --outdir "$dir" --out "$dir/$TAG.h5"
               --log "$dir/$TAG.log" --plot "$dir/$TAG.png")
    # No provenance, no run: the scan's own HDF5 records subharmonic_gate_scan's
    # command line, not this module's, so this record is the reproducible one.
    PY=$PY scripts/provenance.sh "$dir" "$TAG" -- "${cmd[@]@Q}" || {
        echo "provenance failed for $dir/$TAG -- not starting" >&2; return 1; }
    echo "[$(date +%H:%M:%S)] start $dir/$TAG  shard $SHARD/$NSHARDS  offsets=$OFFSETS  etas=$ETAS"
    "${cmd[@]}"
    rc=$?
    [ "$rc" -eq 0 ] && touch "$dir/$TAG.done"
    echo "[$(date +%H:%M:%S)] done  $dir/$TAG  (exit $rc)"
    return "$rc"
}

echo "[$(date +%H:%M:%S)] A-SNAIL $MODE run on $(hostname) -> $OUT  (${WORKERS}x${JOBS})"
for pass in $PASSES; do
    case "$pass" in
        A) run_pass "$OUT/passA_fine" --max-drag-channels 0 --envelope-m 3 || exit 1 ;;
        C) if ! ls "$OUT"/passA_fine/columns/*_rabi.npz >/dev/null 2>&1; then
               echo "pass C replays pass A's chevrons; run pass A first" >&2
               exit 1
           fi
           run_pass "$OUT/passC_fine" \
               --max-drag-channels 3 --envelope-m 3 --drag-set ranked \
               --replay-rabi "$OUT/passA_fine" || exit 1 ;;
        *) echo "unknown pass $pass" >&2; exit 1 ;;
    esac
done
echo "[$(date +%H:%M:%S)] shard $SHARD/$NSHARDS finished"
