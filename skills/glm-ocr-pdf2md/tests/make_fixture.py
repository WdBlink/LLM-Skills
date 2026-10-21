#!/usr/bin/env python3
"""
Generate a tiny synthetic PDF for testing the glm-ocr-pdf2md pipeline.

The fixture has 3 pages, each a single rendered Chinese / English heading
plus a few lines. It is intentionally small so the full pipeline (PDF →
PNG → OCR → Markdown) can be smoke-tested in under a minute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


PAGES = [
    {
        "title": "第一章  引言",
        "body": "本章介绍本书的核心命题：每一篇学术文章都在讲一个故事。\n\n"
                "研究者需要掌握的不仅是论证逻辑，更是叙事的节奏。\n"
                "Good writing is good storytelling in disguise.",
    },
    {
        "title": "Chapter 2  Plot",
        "body": "Every article has a beginning, a middle, and an end. The\n"
                "introduction is the hook, the methods are the rising action,\n"
                "and the discussion is the resolution.\n\n"
                "故事的弧线与论文的结构高度同构。",
    },
    {
        "title": "第三章  结论",
        "body": "在结论部分，作者需要把发现放回到更大的知识脉络中。\n"
                "Conclusions should answer the 'so what' question every\n"
                "reviewer silently asks.",
    },
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("tests/fixture_book.pdf"))
    p.add_argument("--dpi", type=int, default=120, help="rendering DPI")
    args = p.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    for page in PAGES:
        page_obj = doc.new_page(width=595, height=842)  # A4
        # Title
        page_obj.insert_text(
            (60, 80), page["title"],
            fontname="china-s", fontsize=18,
        )
        # Body
        y = 130
        for line in page["body"].split("\n"):
            page_obj.insert_text(
                (60, y), line,
                fontname="china-s", fontsize=12,
            )
            y += 22
    doc.save(args.output)
    doc.close()
    print(f"wrote {args.output}  ({args.output.stat().st_size} bytes, {len(PAGES)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
