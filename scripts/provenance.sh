#!/usr/bin/env bash
# Write a provenance record for a run into DIR, or fail -- a run script calls this
# BEFORE it solves anything and exits if it fails, so no result exists without one.
#
#   scripts/provenance.sh DIR NAME [-- COMMAND...]
#
# Writes DIR/provenance/NAME.json and DIR/provenance/NAME.manifest:
#   - host, user, UTC time, the COMMAND given (the run's own command line), and $PWD
#   - code_sha256: one hash over every source file (src/, scripts/, devices/,
#     pyproject.toml), with the per-file list in the .manifest. The lcs-akj boxes have
#     no .git, so this hash is what ties a remote shard to the home box's commit:
#     equal hashes = identical code.
#   - where .git exists: commit, branch, the uncommitted diff (DIR/provenance/NAME.diff)
#     and the untracked source files, which the hash covers but the commit does not
#     (archived as DIR/provenance/NAME.untracked.tar.gz). Commit + diff + archive
#     rebuild the exact tree; check it against the .manifest.
#   - python version and the installed packages (every distribution in the venv).
set -euo pipefail
cd "$(dirname "$0")/.."

DIR=${1:?usage: provenance.sh DIR NAME [-- COMMAND...]}
NAME=${2:?usage: provenance.sh DIR NAME [-- COMMAND...]}
shift 2
[[ "${1:-}" == "--" ]] && shift
PY=${PY:-.venv/bin/python}
P="$DIR/provenance"
mkdir -p "$P"

find src scripts devices pyproject.toml -type f \
        \( -name '*.py' -o -name '*.sh' -o -name '*.json' -o -name '*.toml' \) \
        -not -path '*/__pycache__/*' -print0 \
    | sort -z | xargs -0 sha256sum > "$P/$NAME.manifest"
code_sha=$(sha256sum < "$P/$NAME.manifest" | cut -d' ' -f1)

commit= branch= dirty= untracked=
if git rev-parse --git-dir >/dev/null 2>&1; then
    commit=$(git rev-parse HEAD)
    branch=$(git rev-parse --abbrev-ref HEAD)
    git diff HEAD > "$P/$NAME.diff"
    [[ -s "$P/$NAME.diff" ]] && dirty=true || { dirty=false; rm -f "$P/$NAME.diff"; }
    untracked=$(git ls-files --others --exclude-standard -- src scripts devices)
    # The commit cannot rebuild these, so they are archived alongside.
    [[ -n "$untracked" ]] && git ls-files -z --others --exclude-standard -- src scripts devices \
        | tar czf "$P/$NAME.untracked.tar.gz" --null -T -
fi

COMMAND="$*" CODE_SHA="$code_sha" COMMIT="$commit" BRANCH="$branch" DIRTY="$dirty" \
UNTRACKED="$untracked" NAME="$NAME" "$PY" - > "$P/$NAME.json" <<'EOF'
import json, os, platform, sys, time
from importlib.metadata import distributions
e = os.environ
git = None
if e["COMMIT"]:
    git = {"commit": e["COMMIT"], "branch": e["BRANCH"], "dirty": e["DIRTY"] == "true",
           "diff": f"{e['NAME']}.diff" if e["DIRTY"] == "true" else None,
           "untracked_archive": f"{e['NAME']}.untracked.tar.gz" if e["UNTRACKED"] else None,
           "untracked": [u for u in e["UNTRACKED"].splitlines() if u]}
freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
json.dump({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "user": e.get("USER"), "cwd": os.getcwd(),
           "command": e["COMMAND"], "code_sha256": e["CODE_SHA"],
           "manifest": f"{e['NAME']}.manifest", "git": git,
           "python": sys.version.split()[0], "packages": freeze},
          sys.stdout, indent=1)
EOF
echo "provenance -> $P/$NAME.json  (code $code_sha${commit:+, commit ${commit:0:7}${dirty:+ dirty=$dirty}})"
