#!/usr/bin/env bash
# Compile dl-alchemy .typ files to PDF (CPU only; no GPU).
#   template/build.sh unit1-basics/handout.typ [more.typ ...]
# There is no typst binary on this box; the PyPI `typst` package bundles the
# compiler and fonts, and uv runs it without touching any project venv.
# Root is dl-alchemy/, so documents import the template as "/template/template.typ".
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for f in "$@"; do
  src="$(realpath "$f")"
  uv run --quiet --no-project --with typst python - "$src" "$root" <<'EOF'
import sys, typst
src, root = sys.argv[1], sys.argv[2]
out = src[:-4] + ".pdf"
typst.compile(src, output=out, root=root)
print(out)
EOF
done
