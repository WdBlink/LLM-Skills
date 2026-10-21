#!/usr/bin/env python3
"""
Regression test for the canonical evaluation book.

Loads `evals/学术写作如何讲好故事.md` (produced from a real 153-page run)
and asserts the structural invariants a downstream skill like book-to-skill
or minimax-docx would depend on.

Run:
  python3 tests/test_eval_book.py

Set GLM_OCR_SKIP=1 to skip.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL = SKILL_DIR / "evals" / "学术写作如何讲好故事.md"


def main() -> int:
    if os.environ.get("GLM_OCR_SKIP"):
        print("GLM_OCR_SKIP=1  →  skipping eval-book regression test")
        return 0
    if not EVAL.exists():
        print(f"eval fixture missing: {EVAL}")
        print("Run the full pipeline against the canonical book first.")
        return 1

    text = EVAL.read_text(encoding="utf-8")
    print(f"eval size: {len(text)/1024:.1f} KB  ({text.count(chr(10))} lines)")

    # 1. top-level H1
    assert text.startswith("# "), "missing H1 title"

    # 2. OCR'd meta line
    assert re.search(r"^> OCR'd with", text, re.M), "missing OCR'd meta line"

    # 3. Auto TOC
    assert "## 目录" in text, "missing auto-detected TOC"
    toc_block = text.split("## 目录", 1)[1].split("---", 1)[0]
    toc_lines = [l for l in toc_block.splitlines() if l.startswith("- [")]
    print(f"TOC entries: {len(toc_lines)}")
    assert len(toc_lines) >= 5, f"TOC too sparse: {len(toc_lines)} entries"

    # 4. Page markers  (e.g. <!-- p.1 -->  or  <!-- p.42 / src=... -->)
    page_markers = re.findall(r"<!--\s*p\.(\d+)(?:\s|/|>|$)", text)
    nums = sorted({int(x) for x in page_markers})
    assert len(nums) >= 100, f"too few page markers: {len(nums)}"
    assert nums[0] == 1 and nums[-1] == len(nums), \
        f"page markers are not contiguous: {nums[0]}..{nums[-1]} ({len(nums)} total)"
    print(f"page markers: {len(nums)}  ({nums[0]}..{nums[-1]})")

    # 5. Chapter pages reference plausible first-pages
    chapter_titles = re.findall(r"^- \[(.+?)\]\(#.+?\) ……… p\.(\d+)$", toc_block, re.M)
    assert len(chapter_titles) >= 5, "no chapter title -> page mapping"
    for title, pg in chapter_titles[:5]:
        print(f"  {title}  →  p.{pg}")

    # 6. Chinese content present
    han = re.findall(r"[一-鿿]", text)
    print(f"Chinese characters: {len(han)}")
    assert len(han) > 10_000, f"suspiciously few Chinese characters: {len(han)}"

    # 7. English / ISBN / table fragments survive
    assert "ISBN" in text, "lost ISBN / English metadata"
    assert "CIP" in text or "在版编目" in text, "lost CIP catalog info"

    # 8. First chapter is at a sane page
    first_chapter_pg = int(chapter_titles[0][1])
    assert 1 <= first_chapter_pg <= 30, \
        f"first chapter is at p.{first_chapter_pg} (expected 1..30)"

    print("\n✅ eval-book regression test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
