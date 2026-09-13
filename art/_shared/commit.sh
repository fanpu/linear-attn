#!/usr/bin/env bash
# Commit one art project's directory on the current branch, safely under concurrency.
#
#   art/_shared/commit.sh <project-dir> "<commit message>"         # commits art/<project-dir>/
#   art/_shared/commit.sh hardware/<project-dir> "<commit message>" # any path with a slash is used as-is
#
# - Serializes git operations across agents with a lock (no index.lock races).
# - Commits ONLY that project directory (never the user's other uncommitted work).
# - Skips files > 20 MB (listed on stderr; mention them in your README instead).
set -euo pipefail
ROOT=/home/fzeng/ml/research
proj=${1:?project dir}; msg=${2:?commit message}
MAX=$((20 * 1024 * 1024))
cd "$ROOT"
case "$proj" in */*) P="$proj" ;; *) P="art/$proj" ;; esac
[ -d "$P" ] || { echo "no such project dir: $P" >&2; exit 1; }

exec 9>"$ROOT/art/_shared/.git.lock"
flock 9

while IFS= read -r -d '' f; do
  if [ "$(stat -c %s "$f")" -gt "$MAX" ]; then
    echo "[commit.sh] skipping >20MB: $f" >&2
  else
    git add -- "$f"
  fi
done < <(git ls-files -z --others --modified --exclude-standard -- "$P")
git ls-files -z --deleted -- "$P" | xargs -0 -r git rm --cached --quiet --

if git diff --cached --quiet -- "$P"; then
  echo "[commit.sh] nothing to commit in $P"
  exit 0
fi
git commit --quiet -m "$msg" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -- "$P"
git log -1 --oneline
