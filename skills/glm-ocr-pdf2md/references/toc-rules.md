# TOC synthesis rules

After every page has been OCR'd, the script scans the per-page Markdown for
chapter headings. The patterns, in order of preference, are:

1. `^#{1,3} 第[一二三四五六七八九十百千零0-9]+章 …`   →  Chinese academic books
2. `^#{1,3} 第[0-9]+章 …`                              →  Chinese numeric chapters
3. `^#{1,3} Chapter [0-9IVXLCDM]+ …`                   →  English books
4. `^#{1,3} [一二三四五六七八九十]+、…`                →  Chinese ordinal sections

A chapter is recorded at the first page where its full title appears. The
title is slugified into a Markdown anchor and emitted under `## 目录 (auto-detected)`.

## Failure modes

- GLM-OCR sometimes drops the `章` character on title pages → the page still
  appears under that page's body, just not in the TOC. To force a hit, edit the
  page's Markdown to add `## ` prefix.
- A book with **only unnumbered chapter titles** (e.g. 序言 / 前言 / 引言 / 鸣谢)
  will produce a sparse TOC. Run `--pages 30` to confirm, then either accept it
  or hand-edit the result.
- A book where the first chapter is named "绪论" but later chapters are "第 N 章 ..."
  will split the TOC. Re-run the synthesizer with a relaxed pattern if needed.

The synthesis is intentionally permissive — adding more noise to the TOC is
preferred over missing a real chapter.
