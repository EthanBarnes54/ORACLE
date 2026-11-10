#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Build the ORACLE user manual PDF (Linux/macOS + TeX Live or Tectonic).
#  Output: build/manual.pdf
# ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

if command -v latexmk &>/dev/null; then
    latexmk -pdf -interaction=nonstopmode -outdir=build manual.tex
elif command -v tectonic &>/dev/null; then
    mkdir -p build && tectonic --outdir build manual.tex
else
    echo "ERROR: neither latexmk nor tectonic is on PATH."
    echo "Install TeX Live / MacTeX, or Tectonic (brew install tectonic)."
    exit 1
fi
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    echo "Build FAILED. See build/manual.log for details."
    exit "$rc"
fi

echo
echo "==========================================================="
echo "  PDF built: build/manual.pdf"
echo "==========================================================="
