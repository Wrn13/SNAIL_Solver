#!/usr/bin/env bash
# The eta = 1.3 5 MHz curve again, with the chirp built from the MEASURED RIDGE
# (--chirp-source ridge) instead of the k2/k4 truncation -- WITHOUT re-measuring.
#
#   scripts/run_5mhz_ridge.sh
#
# Why: the law path calibrated 21 pass-C columns (+35..+140 MHz) chirp-free as
# `stark_crossing` -- an avoided crossing moves through the drive sweep and steps the
# ridge by 0.2-4 MHz, well inside the ~4.7 MHz half-linewidth -- and the truncated
# series did not converge at most of the rest. The ridge law (ridge_chirp, "direct":
# the smoothed ridge divided by the shaped probe's M2) follows both.
#
# The Rabi sweep (steps 1, ~1.1 h of a column's ~1.9 h) is NOT repeated:
# --replay-rabi takes each column's stored col_<tag>_rabi.npz, pass C's first, then
# pass A's fill. Only delta = -25 and -15 have none (pass C died there before saving;
# pass A predates saving them) and are measured from scratch.
#
#   pass A  (--max-drag-channels 0 --envelope-m 3) -> passA_ridge/   chirp only + bare
#   pass C  (--drag-set subharm-leak)              -> passC_ridge/   chirp + fixed DRAG
#
# Same physics flags as scripts/run_5mhz_fill.sh. Then: scripts/finalize_5mhz_ridge.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_5MHz_2026-09-22}
LEVELS=${LEVELS:-9}
DEVICE=${DEVICE:-6Gate4.7SNAIL.json}
AMP=${AMP:-41}
WORKERS=${WORKERS:-12}
JOBS=${JOBS:-6}
ETA=${ETA:-1.3}
PASSES=${PASSES:-"A C"}
OFFSETS=${OFFSETS:-'-0.19:0.16:71'}
# A partial re-solve (e.g. after a ridge-law change) goes to its own file, so the
# finished run is never truncated: TAG=tracked EXTRA=--overwrite OFFSETS=...
TAG=${TAG:-full}
EXTRA=${EXTRA:-}
REPLAY="$OUT/passC_subharmleak,$OUT/passA_nodrag"

COMMON=(--device "$DEVICE" --branch above --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points 15 --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback --drag-decouple-fallback
        --keep-origin --target-eta "$ETA"
        --chirp-source ridge --replay-rabi "$REPLAY"
        --column-workers "$WORKERS" --jobs "$JOBS")

run_pass() {                                   # run_pass DIR TAG FLAGS...
    local dir="$1" tag="$2"; shift 2
    mkdir -p "$dir"
    if [ -f "$dir/$tag.done" ]; then
        echo "[$(date +%H:%M:%S)] skip $dir/$tag (already finished)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start $dir/$tag  offsets=$OFFSETS"
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" "$@" \
        --offsets="$OFFSETS" $EXTRA --outdir "$dir" --out "$dir/$tag.h5" \
        --log "$dir/$tag.log" --plot "$dir/$tag.png"
    rc=$?
    [ "$rc" -eq 0 ] && touch "$dir/$tag.done"
    echo "[$(date +%H:%M:%S)] done  $dir/$tag  (exit $rc)"
}

etag="eta$(printf '%g' "$ETA" | tr '.' 'p')"
echo "[$(date +%H:%M:%S)] 5 MHz ridge-chirp replay -> $OUT  eta=$ETA  (${WORKERS}x${JOBS})"
for pass in $PASSES; do
    case "$pass" in
        A) run_pass "$OUT/passA_ridge" "${etag}_${TAG}" \
               --max-drag-channels 0 --envelope-m 3 ;;
        C) run_pass "$OUT/passC_ridge" "${etag}_${TAG}" \
               --max-drag-channels 3 --envelope-m 3 --drag-set subharm-leak ;;
        *) echo "unknown pass $pass" >&2; exit 1 ;;
    esac
done
echo "[$(date +%H:%M:%S)] all passes finished; now run scripts/finalize_5mhz_ridge.sh"
