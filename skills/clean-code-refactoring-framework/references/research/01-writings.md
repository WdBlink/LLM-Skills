# 01 Writings

## Mode

本地语料优先，且本次实际为纯本地书籍蒸馏。未使用网络搜索。

## Primary Sources

1. 《代码整洁之道》
   - Local file: `LLM-Wiki/raw/books/代码整洁之道 (马丁(Robert C. Martin)) (z-library.sk, 1lib.sk, z-lib.sk).epub`
   - Extraction method: EPUB zipfile fallback.
   - Evidence range: E0001-E0074.
   - High-value topics: readability economics, naming, functions, comments, error handling, boundaries, tests, classes, systems, concurrency, case-study cleanup.

2. 《重构：改善既有代码的设计》第2版
   - Local file: `LLM-Wiki/raw/books/重构：改善既有代码的设计（第2版）（异步图书） (未知) (z-library.sk, 1lib.sk, z-lib.sk).mobi`
   - Extraction method: MOBI HTML extraction, converted to Markdown via pandoc.
   - Evidence range: E0074-E0149.
   - High-value topics: behavior-preserving refactoring, two hats, when to refactor, self-testing code, code smells, refactoring catalog.

## Repeated Core Claims

- Code is optimized for reading and modification over time, not for first draft convenience.
- Names, functions, tests, and module boundaries are all mechanisms for making intent visible.
- Refactoring must preserve observable behavior and proceed in small steps.
- Tests are a precondition for confident change.
- Bad smells are symptoms that route to specific refactoring moves.

## Confidence

High for book-derived rules and workflows. Medium for synthesis across both books where wording differs but operational direction aligns.
