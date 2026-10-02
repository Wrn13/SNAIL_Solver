#!/usr/bin/env bash
# Spread one sharded job over several boxes by ssh -- no scheduler.
#
#   export HOSTS="gh1 gh2 gh3 gh4"         # or list them in scripts/grid_hosts
#   scripts/grid.sh check                   # ssh, tmux, venv on every box
#   scripts/grid.sh sync                    # push the working tree (not results/)
#   scripts/grid.sh launch ETA=1.3 scripts/run_shard.sh
#   scripts/grid.sh status                  # running / exit code + last log line
#   scripts/grid.sh wait                    # block until all exit, then collect
#   scripts/grid.sh collect                 # pull results/ back from every box
#   scripts/grid.sh stop                    # kill the job everywhere
#
# Box i of N (in HOSTS order) runs the command with SHARD=i NSHARDS=N GRID_HOST=<box>
# in its environment, inside a detached tmux session, so a dropped ssh does not kill
# it. scripts/run_shard.sh already reads SHARD/NSHARDS and takes a strided share of
# the columns, so the boxes finish together; any other command can use them too:
#
#   scripts/grid.sh launch .venv/bin/python -m snail_solver.subharmonic_gate_scan \
#       ... --shard '$SHARD/$NSHARDS' --out 'results/x/s$SHARD.h5'
#
# (single quotes: $SHARD must expand on the remote box, not here.)
#
# The boxes do not share a filesystem, so each writes its own results/ and `collect`
# rsyncs them home. Shards are disjoint and their files are named by shard, so the
# pulls never collide; --update keeps a newer local copy. A host that is this machine
# (localhost or its hostname) runs in place and is skipped by sync/collect.
#
# Env: HOSTS, REMOTE_DIR (default: this repo's path), NAME (tmux session, default
# grid), COLLECT (dirs to pull, default results), POLL (wait interval, s),
# SYNC_VENV=1 (also push .venv -- fine between identical GH200 images).
set -uo pipefail
cd "$(dirname "$0")/.."

REPO=$PWD
REMOTE_DIR=${REMOTE_DIR:-$REPO}
NAME=${NAME:-grid}
COLLECT=${COLLECT:-results}
POLL=${POLL:-60}
LOGDIR=results/grid_logs
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)

if [[ -z "${HOSTS:-}" && -f scripts/grid_hosts ]]; then
    HOSTS=$(sed 's/#.*//' scripts/grid_hosts | xargs)
fi
read -ra HOST_LIST <<< "${HOSTS:?set HOSTS=\"h1 h2 ...\" or write scripts/grid_hosts}"
N=${#HOST_LIST[@]}

ts() { date +%H:%M:%S; }

is_local() {
    [[ "$1" == localhost || "$1" == "$(hostname)" || "$1" == "$(hostname -s)" ]]
}

on() {                                         # on HOST 'shell string'
    if is_local "$1"; then
        bash -c "$2"
    else
        "${SSH[@]}" "$1" "$2"
    fi
}

# Same tree on both ends: nothing to copy.
same_tree() { is_local "$1" && [[ "$REMOTE_DIR" == "$REPO" ]]; }

rdir=$(printf %q "$REMOTE_DIR")

cmd_check() {
    local h bad=0
    for h in "${HOST_LIST[@]}"; do
        if out=$(on "$h" "cd $rdir 2>/dev/null || { echo 'no $REMOTE_DIR'; exit 1; }
                 command -v tmux >/dev/null || { echo 'no tmux'; exit 1; }
                 .venv/bin/python -c 'import snail_solver' 2>/dev/null \
                     || { echo 'venv missing or snail_solver not importable'; exit 1; }
                 echo \"\$(nproc) cores, \$(nvidia-smi -L 2>/dev/null | wc -l) GPU\"" 2>&1)
        then
            echo "  ok   $h: $out"
        else
            echo "  FAIL $h: $out"; bad=1
        fi
    done
    return $bad
}

cmd_sync() {
    local h ex=(--exclude 'results/' --exclude '__pycache__/' --exclude '.git/')
    [[ "${SYNC_VENV:-0}" == 1 ]] || ex+=(--exclude '.venv/')
    for h in "${HOST_LIST[@]}"; do
        same_tree "$h" && { echo "[$(ts)] $h: local, skip"; continue; }
        echo "[$(ts)] sync -> $h:$REMOTE_DIR"
        on "$h" "mkdir -p $rdir" || return 1
        rsync -az -e "${SSH[*]}" "${ex[@]}" ./ "$h:$REMOTE_DIR/" || return 1
    done
}

# One session per shard, so two shards can share a box (or a box can be listed twice).
sess() { echo "${NAME}_s$1"; }
running() { on "${HOST_LIST[$1]}" "tmux has-session -t $(sess "$1") 2>/dev/null"; }

cmd_launch() {
    (( $# )) || { echo "launch needs a command" >&2; return 1; }
    local cmd h i inner
    # One arg is taken as a shell string; several are quoted word by word, except
    # that a leading VAR=value stays an assignment.
    if (( $# == 1 )); then cmd=$1; else cmd=$(printf '%q ' "$@"); fi
    for i in "${!HOST_LIST[@]}"; do        # all-or-nothing: no half-launched grid
        if running "$i"; then
            echo "${HOST_LIST[$i]} already runs '$(sess "$i")' -- stop it or set NAME=" >&2
            return 1
        fi
    done
    for i in "${!HOST_LIST[@]}"; do
        h=${HOST_LIST[$i]}
        local log="$LOGDIR/$NAME.s$i.log" ex="$LOGDIR/$NAME.s$i.exit"
        inner="export SHARD=$i NSHARDS=$N GRID_HOST=$h; $cmd >$log 2>&1; echo \$? >$ex"
        on "$h" "cd $rdir && mkdir -p $LOGDIR && rm -f $ex &&
                 tmux new-session -d -s $(sess "$i") bash -c $(printf %q "$inner")" \
            || { echo "[$(ts)] launch FAILED on $h" >&2; return 1; }
        echo "[$(ts)] shard $i/$N -> $h   (log $REMOTE_DIR/$log)"
    done
}

# One line per box: "<state> <last log line>", state = running | exit=<code> | idle.
host_state() {
    local i=$1
    on "${HOST_LIST[$i]}" "cd $rdir 2>/dev/null || exit 0
        if tmux has-session -t $(sess "$i") 2>/dev/null; then s=running
        elif [ -f $LOGDIR/$NAME.s$i.exit ]; then s=exit=\$(cat $LOGDIR/$NAME.s$i.exit)
        else s=idle; fi
        echo \"\$s \$(tail -n1 $LOGDIR/$NAME.s$i.log 2>/dev/null | cut -c1-110)\"" \
        2>/dev/null || echo "unreachable"
}

cmd_status() {
    local i
    for i in "${!HOST_LIST[@]}"; do
        printf '  s%-2d %-16s %s\n' "$i" "${HOST_LIST[$i]}" "$(host_state "$i")"
    done
}

cmd_collect() {
    local h d
    for h in "${HOST_LIST[@]}"; do
        same_tree "$h" && continue
        for d in $COLLECT; do
            echo "[$(ts)] collect $h:$REMOTE_DIR/$d"
            mkdir -p "$d"
            rsync -az --update -e "${SSH[*]}" "$h:$REMOTE_DIR/$d/" "$d/" \
                || echo "  (rsync from $h failed; re-run collect)" >&2
        done
    done
}

cmd_wait() {
    local i st busy fail
    while :; do
        busy=0; fail=0
        for i in "${!HOST_LIST[@]}"; do
            st=$(host_state "$i"); st=${st%% *}
            case "$st" in
                running|unreachable) busy=1 ;;
                exit=0) ;;
                *) fail=1 ;;
            esac
        done
        (( busy )) || break
        sleep "$POLL"
    done
    cmd_status
    cmd_collect
    (( fail == 0 )) || { echo "[$(ts)] some shards failed -- see $LOGDIR/" >&2; return 1; }
    echo "[$(ts)] all $N shards finished and collected"
}

cmd_stop() {
    local i
    for i in "${!HOST_LIST[@]}"; do
        on "${HOST_LIST[$i]}" "tmux kill-session -t $(sess "$i") 2>/dev/null" \
            && echo "[$(ts)] stopped s$i on ${HOST_LIST[$i]}" \
            || echo "[$(ts)] s$i on ${HOST_LIST[$i]}: nothing running"
    done
}

sub=${1:-}; shift || true
case "$sub" in
    check|sync|launch|status|collect|wait|stop) "cmd_$sub" "$@" ;;
    *) sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
