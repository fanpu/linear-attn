#!/usr/bin/env bash
# Commit ONLY one project directory to the current branch (main), serialized across agents.
#
#   _shared/commit.sh 03-saxe-dynamics "saxe: analytic vs measured mode curves"
#
# Uses `git commit --only -- <dir>`, so anything else staged in the shared index (the user's own
# work, other agents' files) is left untouched. Never pushes.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <project-dir> <message>" >&2; exit 2; }
THEORY="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$THEORY/$1"
[ -d "$DIR" ] || { echo "no such project dir: $DIR" >&2; exit 2; }
cd "$THEORY"
# Refuse oversized files (git history is forever): anything > 25 MB must be shrunk or ignored.
big=$(find "$1" -type f -size +25M -not -path "*/cache/*" -not -path "*/_preview/*" | head -5)
[ -z "$big" ] || { echo "refusing: files > 25MB (shrink or move to cache/):" >&2; echo "$big" >&2; exit 1; }
exec 9>"$THEORY/_shared/.git.lock"
flock 9
git add -- "$1"
if git diff --cached --quiet -- "$1"; then echo "nothing to commit in $1"; exit 0; fi
git commit -q --only -m "theory/$2" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- "$1"
git log --oneline -1
