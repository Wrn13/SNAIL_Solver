#!/usr/bin/env bash
# The 5 MHz fidelity curve on the ABOVE branch: bare vs chirp vs chirp+DRAG,
# delta in [-160, +160] MHz, at eta = 1.3 and 1.5.
#
#   scripts/run_5mhz_curve.sh
#
# Two passes per eta, into SEPARATE outdirs:
#
#   pass A (--max-drag-channels 0)  -> the chirp-only series AND the independently
#                                      calibrated bare series
#   pass B (--max-drag-channels 3)  -> the chirp+DRAG series
#
# They must not share a directory: finalize globs *.h5 and the two passes carry the
# same (delta, eta, variant) keys, so one would silently overwrite the other.
#
# Pass A NEEDS --envelope-m 3. Without it m derives to max(0, 2) = 2 and the
# "baseline" is a different pulse shape from the DRAG run -- which is not a
# baseline. (scripts/run_shard.sh omits this; do not copy it.)
#
# Two mandatory channels cross zero detuning inside this window:
#
#   A subharmonic          g = 30.4 MHz   det = -+2 delta          zero at delta = 0
#   A |1>-><2> subharmonic g = 43.0 MHz   det = 2(delta + 60 MHz)  zero at delta = -60
#
# DRAG is perturbative in g/Delta, so it cannot be valid at either. Columns OUTSIDE
# +-10 MHz of those two points are solved with --force; the nine inside run unforced
# so they record an explicit NonPerturbativeChannel row and the plot marks the
# exclusion instead of silently missing the points.
#
# Order is eta=1.3 A, eta=1.3 B, eta=1.5 A, eta=1.5 B, so a complete three-series
# comparison at 1.3 lands first and the job stays useful if it is stopped early.
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
ETAS=${ETAS:-"1.3 1.5"}
# Pass A never enters the chirp<->DRAG fixed point, so it is not gated on whatever
# pass B is doing: PASSES=A can be started while B is still being validated.
PASSES=${PASSES:-"A B"}

# 5 MHz spacing over [-160, +160]. delta = 0 is dropped by columns_for.
# FORCED omits +-10 MHz of delta = 0 and of delta = -60; UNFORCED is exactly those.
FORCED='-0.16:-0.075:18,-0.045:-0.015:7,0.015:0.16:30'      # 55 columns
UNFORCED='-0.07:-0.05:5,-0.01:-0.005:2,0.005:0.01:2'        #  9 columns

COMMON=(--device "$DEVICE" --branch above --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points 15 --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback
        --column-workers "$WORKERS" --jobs "$JOBS")

# pass name -> extra flags. Pass A carries --envelope-m 3 (see header).
pass_flags() {
    case "$1" in
        A) echo "--max-drag-channels 0 --envelope-m 3" ;;
        B) echo "--max-drag-channels 3" ;;
        *) echo "unknown pass $1" >&2; return 1 ;;
    esac
}

run_one() {                                   # run_one PASS ETA SETNAME OFFSETS [--force]
    local pass="$1" eta="$2" setname="$3" offs="$4" force="${5:-}"
    local dir="$OUT/pass${pass}_$([ "$pass" = A ] && echo nodrag || echo drag)"
    local tag="eta$(printf '%g' "$eta" | tr '.' 'p')_${setname}"
    mkdir -p "$dir"
    if [ -f "$dir/$tag.h5" ]; then
        echo "[$(date +%H:%M:%S)] skip $pass/$tag (h5 exists)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start pass $pass  eta=$eta  $setname  ${force:-(no force)}"
    # shellcheck disable=SC2046,SC2086
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" $(pass_flags "$pass") \
        --offsets="$offs" --target-eta "$eta" $force \
        --outdir "$dir" --out "$dir/$tag.h5" \
        --log "$dir/$tag.log" --plot "$dir/$tag.png"
    echo "[$(date +%H:%M:%S)] done  pass $pass  eta=$eta  $setname  (exit $?)"
}

mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] 5 MHz curve -> $OUT  (amp-points=$AMP, ${WORKERS}x${JOBS} procs)"
"$PY" -c 'import subprocess' >/dev/null 2>&1 || { echo "no $PY" >&2; exit 1; }

for eta in $ETAS; do
    for pass in $PASSES; do
        run_one "$pass" "$eta" forced   "$FORCED"   --force
        run_one "$pass" "$eta" excluded "$UNFORCED"
    done
done

echo "[$(date +%H:%M:%S)] all passes finished; now run scripts/finalize_5mhz_curve.sh"
