#!/usr/bin/env bash
# Every 5 MHz tick of the 5 MHz curve at eta = 1.3, with a FIXED DRAG set.
#
#   scripts/run_5mhz_fill.sh
#
# Two passes into results/drag_curve_5MHz_2026-09-22, eta = 1.3 only:
#
#   pass A fill  (--max-drag-channels 0 --envelope-m 3) -> passA_nodrag/eta1p3_fill.h5
#       only the ticks pass A never calibrated: delta = -70..-50 and -10..+10
#       (the old driver ran them unforced so the audit refused them, and dropped
#       delta = 0). Chirp only + bare do not depend on the DRAG set, so the other
#       61 columns stand.
#
#   pass C       (--drag-set subharm-leak)              -> passC_subharmleak/
#       chirp + DRAG at EVERY tick in [-190, +160], delta = 0 included, playing
#       exactly the A |0>-|1> subharmonic, the A |1>->|2> subharmonic and the |2>
#       leakage -- whatever g/|det| says. Only the 5 MHz skip window drops one (A
#       subharmonic at delta = 0, A |1>->|2> at delta = -60), and at delta = -30 the
#       two subharmonics share one beat, so one substitution covers both. passB_drag
#       (ranked DRAG: mostly the A-SNAIL spectator) is left as it was.
#
# Nothing is refused on the audit any more: a non-perturbative column is calibrated
# and flagged `audit_nonperturbative`.
#
# Runtime, from the stored per-column seconds: pass C ~137 column-hours / 12 workers
# x 1.55 observed overhead = ~18 h; the fill ~1.5-2 h; ~20 h in all. If that would
# exceed the budget, run the holes only:
#   PASSC_OFFSETS='-0.07:-0.05:5,-0.01:0.01:5' scripts/run_5mhz_fill.sh
#
# Then: scripts/finalize_5mhz_fill.sh
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
FILL_OFFSETS=${FILL_OFFSETS:-'-0.07:-0.05:5,-0.01:0.01:5'}      # 10 columns
PASSC_OFFSETS=${PASSC_OFFSETS:-'-0.19:0.16:71'}                  # 71 columns

# Same physics as scripts/run_5mhz_curve.sh, so the fill is comparable to what it
# fills; --keep-origin so delta = 0 is a column.
COMMON=(--device "$DEVICE" --branch above --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points 15 --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback --drag-decouple-fallback
        --keep-origin --target-eta "$ETA"
        --column-workers "$WORKERS" --jobs "$JOBS")

run_pass() {                                   # run_pass DIR TAG OFFSETS FLAGS...
    local dir="$1" tag="$2" offs="$3"; shift 3
    mkdir -p "$dir"
    # Sentinel, not the .h5: main() truncates the HDF5 up front, so the file exists
    # from the first second and an interrupted pass would look finished.
    if [ -f "$dir/$tag.done" ]; then
        echo "[$(date +%H:%M:%S)] skip $dir/$tag (already finished)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start $dir/$tag  offsets=$offs"
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" "$@" \
        --offsets="$offs" --outdir "$dir" --out "$dir/$tag.h5" \
        --log "$dir/$tag.log" --plot "$dir/$tag.png"
    rc=$?
    [ "$rc" -eq 0 ] && touch "$dir/$tag.done"
    echo "[$(date +%H:%M:%S)] done  $dir/$tag  (exit $rc)"
}

etag="eta$(printf '%g' "$ETA" | tr '.' 'p')"
echo "[$(date +%H:%M:%S)] 5 MHz fill -> $OUT  eta=$ETA  (${WORKERS}x${JOBS} procs)"
for pass in $PASSES; do
    case "$pass" in
        A) run_pass "$OUT/passA_nodrag" "${etag}_fill" "$FILL_OFFSETS" \
               --max-drag-channels 0 --envelope-m 3 ;;
        C) run_pass "$OUT/passC_subharmleak" "${etag}_full" "$PASSC_OFFSETS" \
               --max-drag-channels 3 --envelope-m 3 --drag-set subharm-leak ;;
        *) echo "unknown pass $pass" >&2; exit 1 ;;
    esac
done
echo "[$(date +%H:%M:%S)] all passes finished; now run scripts/finalize_5mhz_fill.sh"
