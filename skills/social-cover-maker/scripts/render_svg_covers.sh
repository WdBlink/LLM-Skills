#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory-containing-svg-covers>" >&2
  exit 2
fi

dir="$1"
if [[ ! -d "$dir" ]]; then
  echo "Not a directory: $dir" >&2
  exit 2
fi

if ! command -v xmllint >/dev/null 2>&1; then
  echo "Missing dependency: xmllint" >&2
  exit 1
fi

if ! command -v sips >/dev/null 2>&1; then
  echo "Missing dependency: sips" >&2
  exit 1
fi

svg_list="$(mktemp)"
find "$dir" -maxdepth 1 -type f -name '*.svg' | sort > "$svg_list"
if [[ ! -s "$svg_list" ]]; then
  rm -f "$svg_list"
  echo "No SVG files found in $dir" >&2
  exit 1
fi

while IFS= read -r svg; do
  png="${svg%.svg}.png"
  xmllint --noout "$svg"
  sips -s format png "$svg" --out "$png" >/dev/null

  expected="$(python3 - "$svg" <<'PY'
from __future__ import annotations
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r"<svg\b[^>]*\bwidth=\"([0-9]+)\"[^>]*\bheight=\"([0-9]+)\"", text)
if not m:
    raise SystemExit("missing integer width/height on root svg")
print(f"{m.group(1)}x{m.group(2)}")
PY
)"

  actual="$(python3 - "$png" <<'PY'
from __future__ import annotations
import subprocess
import sys

out = subprocess.check_output(
    ["sips", "-g", "pixelWidth", "-g", "pixelHeight", sys.argv[1]],
    text=True,
)
width = height = None
for line in out.splitlines():
    line = line.strip()
    if line.startswith("pixelWidth:"):
        width = line.split(":", 1)[1].strip()
    if line.startswith("pixelHeight:"):
        height = line.split(":", 1)[1].strip()
if not width or not height:
    raise SystemExit("could not read png dimensions")
print(f"{width}x{height}")
PY
)"

  if [[ "$expected" != "$actual" ]]; then
    echo "Dimension mismatch for $png: expected $expected, got $actual" >&2
    exit 1
  fi

  echo "rendered $(basename "$png") $actual"
done < "$svg_list"
rm -f "$svg_list"
