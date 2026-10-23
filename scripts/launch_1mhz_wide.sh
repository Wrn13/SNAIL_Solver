#!/usr/bin/env bash
# The 1 MHz, -250 .. +250 MHz eta = 1.3 curve spread over lcs-akj-05 .. 08.
# Shard i runs on the (i+5)-th box: s0 -> lcs-akj-05, s1 -> 06, s2 -> 07, s3 -> 08.
#
#   scripts/launch_1mhz_wide.sh check       # ssh, tmux, venv on every box
#   scripts/launch_1mhz_wide.sh sync        # push the working tree (uncommitted fixes too)
#   scripts/launch_1mhz_wide.sh launch      # pass A then pass C on every box, in tmux
#   scripts/launch_1mhz_wide.sh status      # per box: running / exit code + last log line
#   scripts/launch_1mhz_wide.sh wait        # block until all exit, then collect $OUT
#   scripts/launch_1mhz_wide.sh finalize    # Phase B + figures on the merged shards
#   scripts/launch_1mhz_wide.sh all         # check, sync, launch, wait, finalize
#   scripts/launch_1mhz_wide.sh stop        # kill the job everywhere
#
# `all` blocks for ~2 days, so run it detached:
#
#   nohup setsid scripts/launch_1mhz_wide.sh all > results/drag_curve_1MHz_2026-10-23.driver.log 2>&1 &
#
# A thin wrapper over scripts/grid.sh (see its header); only $OUT is collected, so the
# boxes' other results/ are left alone. SYNC_VENV=1 also pushes .venv, fine between
# identical images. This box is addressed by its own hostname so grid.sh runs it in
# place instead of ssh-ing to itself.
set -uo pipefail
cd "$(dirname "$0")/.."

export OUT=${OUT:-results/drag_curve_1MHz_2026-10-23}
export NAME=${NAME:-wide1mhz}
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

finalize() {
    echo "[$(ts)] Phase B + figures -> $OUT/figs_fine"
    OUT=$OUT WORKERS=${FINAL_WORKERS:-$(nproc)} scripts/finalize_5mhz_fine.sh \
        > "$OUT/finalize.log" 2>&1
    tail -5 "$OUT/finalize.log"
}

sub=${1:-}
case "$sub" in
    check|sync|status|collect|wait|stop) "$GRID" "$sub" ;;
    launch) "$GRID" launch "OUT=$OUT scripts/run_1mhz_wide.sh" ;;
    finalize) finalize ;;
    all)
        echo "[$(ts)] 1 MHz wide curve -> $OUT on $HOSTS"
        "$GRID" check || exit 1
        "$GRID" sync || exit 1
        "$GRID" launch "OUT=$OUT scripts/run_1mhz_wide.sh" || exit 1
        "$GRID" wait || { echo "[$(ts)] a shard failed; fix it and re-launch (cached columns resume)" >&2; exit 1; }
        finalize
        echo "[$(ts)] finished" ;;
    *) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
