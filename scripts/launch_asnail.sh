#!/usr/bin/env bash
# The A-SNAIL exchange scan (scripts/run_asnail.sh) spread over lcs-akj-05 .. 08.
# Shard i runs on the (i+5)-th box: s0 -> lcs-akj-05, s1 -> 06, s2 -> 07, s3 -> 08.
#
#   scripts/launch_asnail.sh check|sync|status|wait|stop
#   scripts/launch_asnail.sh launch      # pass A then pass C on every box, in tmux
#   scripts/launch_asnail.sh finalize    # Phase B + figures on the merged shards
#   scripts/launch_asnail.sh all         # check, sync, launch, wait, finalize
#
# MODE=coarse (default): 10 MHz x eta* 1.0 .. 1.5.  MODE=fine ETA=1.3: 1 MHz, one eta*.
# Run `all` detached:
#
#   nohup setsid scripts/launch_asnail.sh all > results/asnail_10MHz_2026-10-26.driver.log 2>&1 &
#   MODE=fine ETA=1.3 nohup setsid scripts/launch_asnail.sh all > results/asnail_1MHz_eta1p3.driver.log 2>&1 &
#
# One MODE at a time: both use the tmux session name below.
set -uo pipefail
cd "$(dirname "$0")/.."

export MODE=${MODE:-coarse}
case "$MODE" in
    coarse) export OUT=${OUT:-results/asnail_10MHz_2026-10-26} ;;
    fine)   : "${ETA:?MODE=fine needs ETA, e.g. ETA=1.3}"
            export OUT=${OUT:-results/asnail_1MHz_eta$(printf '%g' "$ETA" | tr '.' 'p')} ;;
    *) echo "MODE=$MODE: expected coarse or fine" >&2; exit 2 ;;
esac
export NAME=${NAME:-asnail}
export COLLECT=$OUT
export POLL=${POLL:-300}

hosts=()
for n in 05 06 07 08; do
    h=lcs-akj-$n.syr.edu
    [[ "${h%%.*}" == "$(hostname -s)" ]] && h=$(hostname)
    hosts+=("$h")
done
export HOSTS="${hosts[*]}"

GRID=scripts/grid.sh
ts() { date +%H:%M:%S; }
REMOTE="MODE=$MODE ETA=${ETA:-} OUT=$OUT PASSES='${PASSES:-A C}' scripts/run_asnail.sh"

finalize() {
    echo "[$(ts)] Phase B + figures -> $OUT/figs"
    OUT=$OUT WORKERS=${FINAL_WORKERS:-$(nproc --all)} scripts/finalize_asnail.sh \
        > "$OUT/finalize.log" 2>&1
    tail -5 "$OUT/finalize.log"
}

sub=${1:-}
case "$sub" in
    check|sync|status|collect|wait|stop) "$GRID" "$sub" ;;
    launch) "$GRID" launch "$REMOTE" ;;
    finalize) finalize ;;
    all)
        echo "[$(ts)] A-SNAIL $MODE scan -> $OUT on $HOSTS"
        "$GRID" check || exit 1
        "$GRID" sync || exit 1
        "$GRID" launch "$REMOTE" || exit 1
        "$GRID" wait || { echo "[$(ts)] a shard failed; fix it and re-launch (cached columns resume)" >&2; exit 1; }
        finalize
        echo "[$(ts)] finished" ;;
    *) sed -n '2,17p' "$0" >&2; exit 2 ;;
esac
