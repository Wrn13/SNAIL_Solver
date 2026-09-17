#!/usr/bin/env bash
# The bare-vs-chirp+DRAG curve across qubit A's subharmonic, ABOVE branch.
#
# One run per target eta, sequential so the box is never oversubscribed (8 column
# workers x 6 jobs = 48 of 72 cores; someone else is on this machine). All runs share
# one --outdir so the per-column cache is reused and a kill/relaunch resumes free.
#
# Deliberately NOT passed: --open-system (it hung a previous run for 2 days confirming
# a 1% correction to a 37% error; the analytic coherence penalty is free) and
# --ridge-grid (13.4x on wp_points, unaffordable at this column count).
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

# Overridable so a truncation or window change gets its OWN directory rather than
# colliding with an earlier grid's cache. The cache key now covers the length-fit
# window and the length/scoring contracts, so a stale column cannot be served -- but
# a separate directory also keeps the old data readable for comparison.
OUT=${OUT:-results/drag_curve_2026-09-16}
LEVELS=${LEVELS:-7}
PY=.venv/bin/python
mkdir -p "$OUT"

# 10 MHz spacing inside +-200 MHz, then coarse. delta=0 is dropped by the scan
# (the A-subharmonic beat is inside the pulse bandwidth there); +-250 is absent
# because three beats alias at that offset.
FINE='-0.2:0.2:41,-0.4,-0.35,-0.3,0.3,0.35,0.4'
# eta 1.4/1.6 are sampled only on the negative wing -- the frontier
# delta <= 600 - 300 eta^2 says that is the only place they can live.
WING='-0.06,-0.08,-0.1,-0.12,-0.14,-0.16,-0.18,-0.2,-0.3,-0.35,-0.4'

COMMON=(--device 6Gate4.7SNAIL.json --branch above
        --probe-shape gate --max-drag-channels 3
        --coupler-levels "$LEVELS" --amp-points 9 --wp-points 15 --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --column-workers 8 --jobs 6 --outdir "$OUT")

run_one() {          # run_one <eta> <offsets>
    local eta="$1" offs="$2" tag
    tag="eta$(printf '%g' "$eta" | tr '.' 'p')"
    if [[ -f "$OUT/$tag.h5" ]]; then
        echo "[$(date +%H:%M:%S)] $tag already has $OUT/$tag.h5 -- skipping"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] === eta=$eta ($tag) ==="
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" \
        --offsets="$offs" --target-eta "$eta" \
        --out "$OUT/$tag.h5" --log "$OUT/$tag.log" --plot "$OUT/$tag.png"
    echo "[$(date +%H:%M:%S)] eta=$eta exit=$?"
}

# eta 1.3 first: it is the primary curve, so a usable figure exists earliest. At
# --coupler-levels 9 each fine eta is ~9-11 h, so the first three fill the budget;
# 1.4/1.6/1.2 run only if time remains and can be stopped at any point.
run_one 1.3 "$FINE"
run_one 1.5 "$FINE"
run_one 1.7 "$FINE"
run_one 1.4 "$WING"
run_one 1.6 "$WING"
run_one 1.2 "$FINE"
echo "[$(date +%H:%M:%S)] ALL DONE"
