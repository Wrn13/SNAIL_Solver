#!/usr/bin/env bash
# One machine's share of a grid. Run the SAME command on every box, changing only
# SHARD; each takes a strided subset, so they finish together rather than one box
# running hours after the rest.
#
#   box 0:  SHARD=0 NSHARDS=4 ETA=1.3 scripts/run_shard.sh
#   box 1:  SHARD=1 NSHARDS=4 ETA=1.3 scripts/run_shard.sh
#   ...
#
# Shards are disjoint and cover the grid exactly, so the per-column caches cannot
# race even on a shared filesystem. Each shard writes its OWN .h5; the analysis pass
# takes a comma-separated list and keys rows on (delta, eta), so merging is just
# passing all of them:
#
#   scripts/finalize_drag_curve.sh          # globs eta*.h5 in $OUT
#
# If the boxes do NOT share a filesystem, collect the .h5 files into one directory
# first -- that is all the merge needs.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

SHARD=${SHARD:?set SHARD, e.g. SHARD=0}
NSHARDS=${NSHARDS:?set NSHARDS, e.g. NSHARDS=4}
ETA=${ETA:?set ETA, e.g. ETA=1.3}
OUT=${OUT:-results/drag_curve_shards}
LEVELS=${LEVELS:-9}
DEVICE=${DEVICE:-6Gate4.7SNAIL.json}
# --max-drag-channels 0 gives the independently calibrated NO-DRAG baseline: the
# chirp is then derived without assuming a correction will be played, which the
# `bare` trace of a DRAG run is not (its length and carrier come from a DRAG-aware
# fixed point).
CHANNELS=${CHANNELS:-3}
WORKERS=${WORKERS:-8}
JOBS=${JOBS:-6}
OFFSETS=${OFFSETS:--0.2:0.2:41,-0.4,-0.35,-0.3,0.3,0.35,0.4}

TAG="eta$(printf '%g' "$ETA" | tr '.' 'p')_ch${CHANNELS}_lev${LEVELS}_s${SHARD}of${NSHARDS}"
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] $TAG on $(hostname)"

exec .venv/bin/python -m snail_solver.subharmonic_gate_scan \
    --device "$DEVICE" --branch above --probe-shape gate \
    --offsets="$OFFSETS" --target-eta "$ETA" \
    --max-drag-channels "$CHANNELS" --coupler-levels "$LEVELS" \
    --amp-points 9 --wp-points 15 --tg-points 9 \
    --chirp-max-passes 200 --max-drag-iters 12 \
    --t1-us 50 --t2-us 50 \
    --shard "$SHARD/$NSHARDS" \
    --column-workers "$WORKERS" --jobs "$JOBS" \
    --outdir "$OUT" \
    --out "$OUT/$TAG.h5" \
    --log "$OUT/$TAG.log" \
    --plot "$OUT/$TAG.png"
