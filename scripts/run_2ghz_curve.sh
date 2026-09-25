#!/usr/bin/env bash
# The 2 GHz fidelity curve on the ABOVE branch: bare vs chirp vs chirp+DRAG,
# w_p in [0.5, 2.5] GHz at 10 MHz, eta = 1.3 only.
#
#   scripts/run_2ghz_curve.sh
#
# w_p = w_a/2 + delta = 1.75 + delta, so the requested pump range is
# delta in [-1.25, +0.75] GHz: 201 grid points, 200 solved (delta = 0 dropped).
#
# Same two-pass structure as run_5mhz_curve.sh -- see its header for why pass A
# NEEDS --envelope-m 3 and why the passes must not share an outdir.
#
# FOUR first-order channels cross zero detuning in this window, two more than the
# +-160 MHz grid ever reached:
#
#   A subharmonic          g =  30.4 MHz   zero at delta =    0
#   A |1>-><2> subharmonic g =  43.0 MHz   zero at delta =  -60
#   SNAIL subharmonic      g = 304.2 MHz   zero at delta = +600     <-- new
#   A-SNAIL spectator                      zero at delta = -550     <-- new
#
# Columns within +-10 MHz of any of them run UNFORCED so they record an explicit
# NonPerturbativeChannel row; everything else is forced.
#
# A fifth, 3 w_p = w_s at -183 MHz, is SECOND ORDER in g3 and invisible to the
# first-order audit, so it is deliberately left forced -- it is a feature we want
# measured, and the 5 MHz grid already found structure either side of it.
#
# KNOWN LIMITATION, stated rather than discovered later: --coupler-levels 9 is
# sized for a coupler that stays near its ground state. Approaching delta = +600
# the SNAIL subharmonic goes resonant and the coupler saturates (0.85 photons was
# already measured at delta = +200 on the old grid), so columns within roughly
# 100 MHz of +600 are TRUNCATION-LIMITED and will read optimistic. Treat them as
# locating the resonance, not as fidelities.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/drag_curve_2GHz_2026-09-25}
LEVELS=${LEVELS:-9}
DEVICE=${DEVICE:-6Gate4.7SNAIL.json}
AMP=${AMP:-41}
WORKERS=${WORKERS:-14}
JOBS=${JOBS:-5}
ETAS=${ETAS:-"1.3"}
PASSES=${PASSES:-"A B"}

FORCED='-1.25,-1.24,-1.23,-1.22,-1.21,-1.2,-1.19,-1.18,-1.17,-1.16,-1.15,-1.14,-1.13,-1.12,-1.11,-1.1,-1.09,-1.08,-1.07,-1.06,-1.05,-1.04,-1.03,-1.02,-1.01,-1,-0.99,-0.98,-0.97,-0.96,-0.95,-0.94,-0.93,-0.92,-0.91,-0.9,-0.89,-0.88,-0.87,-0.86,-0.85,-0.84,-0.83,-0.82,-0.81,-0.8,-0.79,-0.78,-0.77,-0.76,-0.75,-0.74,-0.73,-0.72,-0.71,-0.7,-0.69,-0.68,-0.67,-0.66,-0.65,-0.64,-0.63,-0.62,-0.61,-0.6,-0.59,-0.58,-0.57,-0.53,-0.52,-0.51,-0.5,-0.49,-0.48,-0.47,-0.46,-0.45,-0.44,-0.43,-0.42,-0.41,-0.4,-0.39,-0.38,-0.37,-0.36,-0.35,-0.34,-0.33,-0.32,-0.31,-0.3,-0.29,-0.28,-0.27,-0.26,-0.25,-0.24,-0.23,-0.22,-0.21,-0.2,-0.19,-0.18,-0.17,-0.16,-0.15,-0.14,-0.13,-0.12,-0.11,-0.1,-0.09,-0.08,-0.04,-0.03,-0.02,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75'
UNFORCED='-0.56,-0.55,-0.54,-0.07,-0.06,-0.05,-0.01,0,0.01,0.59,0.6,0.61'

COMMON=(--device "$DEVICE" --branch above --probe-shape gate
        --coupler-levels "$LEVELS" --amp-points "$AMP"
        --wp-points 15 --tg-points 9
        --chirp-max-passes 200 --max-drag-iters 12
        --t1-us 50 --t2-us 50
        --chirp-free-fallback --drag-decouple-fallback
        --column-workers "$WORKERS" --jobs "$JOBS")

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
    if [ -f "$dir/$tag.done" ]; then
        echo "[$(date +%H:%M:%S)] skip $pass/$tag (already finished)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] start pass $pass  eta=$eta  $setname  ${force:-(no force)}"
    # shellcheck disable=SC2046,SC2086
    "$PY" -m snail_solver.subharmonic_gate_scan "${COMMON[@]}" $(pass_flags "$pass") \
        --offsets="$offs" --target-eta "$eta" $force \
        --outdir "$dir" --out "$dir/$tag.h5" \
        --log "$dir/$tag.log" --plot "$dir/$tag.png"
    rc=$?
    [ "$rc" -eq 0 ] && touch "$dir/$tag.done"
    echo "[$(date +%H:%M:%S)] done  pass $pass  eta=$eta  $setname  (exit $rc)"
}

mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] 2 GHz curve -> $OUT  (amp-points=$AMP, ${WORKERS}x${JOBS} procs)"
"$PY" -c 'import subprocess' >/dev/null 2>&1 || { echo "no $PY" >&2; exit 1; }

for eta in $ETAS; do
    for pass in $PASSES; do
        run_one "$pass" "$eta" excluded "$UNFORCED"
        run_one "$pass" "$eta" forced   "$FORCED"   --force
    done
done

echo "[$(date +%H:%M:%S)] all passes finished; now run scripts/finalize_5mhz_curve.sh with OUT=$OUT"
