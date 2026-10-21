#!/usr/bin/env python3
"""
Smoke test for glm-ocr-pdf2md.

Pipeline:
  1. Build a 3-page synthetic PDF (tests/make_fixture.py).
  2. Run convert_pdf.py on it with --keep-images.
  3. Assert the produced .md:
     - exists
     - starts with `# ` (book title heading)
     - has at least one `<!-- p.N -->` marker per input page
     - has the auto-detected TOC (`## 目录` or `## 目录 (auto-detected)`)
     - has at least one Chinese character in the body

Run:
  python3 tests/test_pipeline.py

Requires the model to be available (skip this test with GLM_OCR_SKIP=1).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
FIXTURE = SKILL_DIR / "tests" / "fixture_book.pdf"
SCRIPT = SKILL_DIR / "scripts" / "convert_pdf.py"
MAKE = SKILL_DIR / "tests" / "make_fixture.py"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    if os.environ.get("GLM_OCR_SKIP"):
        print("GLM_OCR_SKIP=1  →  skipping pipeline test")
        return 0

    # 1. Make fixture
    if FIXTURE.exists():
        FIXTURE.unlink()
    r = run([sys.executable, str(MAKE), "--output", str(FIXTURE)])
    if r.returncode != 0:
        print("make_fixture failed:\n", r.stdout, r.stderr, file=sys.stderr)
        return 1
    assert FIXTURE.exists(), "fixture PDF not produced"

    # 2. Run converter
    with tempfile.TemporaryDirectory() as tmp:
        out_md = Path(tmp) / "out.md"
        work = Path(tmp) / "work"
        r = run([
            sys.executable, str(SCRIPT),
            str(FIXTURE),
            "--output", str(out_md),
            "--work-dir", str(work),
            "--device", "auto",
            "--max-edge", "768",
            "--no-resume",
            "--keep-images",
        ])
        print(r.stdout)
        if r.returncode != 0:
            print("convert_pdf failed:\n", r.stderr, file=sys.stderr)
            return 1
        if not out_md.exists():
            print(f"output missing: {out_md}", file=sys.stderr)
            return 1

        text = out_md.read_text(encoding="utf-8")
        print(f"--- output ({len(text)} chars) ---")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print("--- end preview ---")

        # 3. Assertions
        assert text.startswith("# "), "missing top-level # heading"
        page_markers = re.findall(r"<!--\s*p\.(\d+)\s*/", text)
        assert len(page_markers) >= 3, f"expected >=3 page markers, got {len(page_markers)}"
        assert "目录" in text, "missing auto-detected TOC"
        assert re.search(r"[一-鿿]", text), "no Chinese characters in output"

        # sanity: page markers should be 1..N
        nums = sorted({int(x) for x in page_markers})
        assert nums == list(range(1, 4)), f"page markers are {nums}, expected 1..3"

    # cleanup fixture + workdirs
    if FIXTURE.exists():
        FIXTURE.unlink()
    img_cache = FIXTURE.with_suffix(".work")
    if img_cache.exists():
        shutil.rmtree(img_cache, ignore_errors=True)
    print("\n✅ pipeline smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
