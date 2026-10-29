#!/usr/bin/env bash
# Compile training-dynamics .typ files to PDF (CPU only).
#   template/build.sh a0-prereqs/handout.typ a0-prereqs/solutions/solutions.typ [...]
# Uses the typst binary if one is on PATH (or at ~/.local/bin/typst); otherwise the PyPI
# `typst` package, which bundles the compiler and fonts, run by uv without touching the
# course venv (as dl-alchemy/template/build.sh does). Root is the course directory, so
# documents import "/template/template.typ" and cite "/template/refs.bib".
# Fails if a compiled document still contains a ⟨placeholder⟩ from #ph[...].
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
typst_bin="$(command -v typst || true)"
if [[ -z "$typst_bin" && -x "$HOME/.local/bin/typst" ]]; then typst_bin="$HOME/.local/bin/typst"; fi
for f in "$@"; do
  src="$(realpath "$f")"
  out="${src%.typ}.pdf"
  if [[ -n "$typst_bin" ]]; then
    "$typst_bin" compile --root "$root" "$src" "$out"
  else
    uv run --quiet --no-project --with typst python - "$src" "$root" "$out" <<'PY'
import sys, typst
src, root, out = sys.argv[1:4]
typst.compile(src, output=out, root=root)
PY
  fi
  if command -v pdftotext >/dev/null && pdftotext "$out" - | grep -q "⟨"; then
    echo "error: $out still contains a #ph[...] placeholder" >&2
    exit 1
  fi
  echo "$out"
done
