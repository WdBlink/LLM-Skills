---
name: transcript-source-compiler
description: Convert long-form speech or sharing-session transcripts into evidence-backed LLM-Wiki source records plus one or more readable articles. Use when the input is already transcribed and the user wants full information preservation, fine-grained evidence ids, and human-readable article materials. Does not transcribe audio.
---

# Transcript Source Compiler

Use this skill for already-transcribed talks, lectures, salons, technical
shares, and long-form speech materials. The goal is not a short summary. The
goal is full, auditable compilation into source material and readable articles.

## Core Contract

- Input: `.md`, `.txt`, `.srt`, `.vtt`, a directory of these files, or pasted transcript text.
- Output: a run directory containing `source-pack.md`, `evidence-fragments.jsonl`, `source-record.md`, `article-main.md` or `article-index.md` plus topic-specific article files, `article-map.md`, and `uncertain-items.md`.
- Preserve the original transcript unchanged in `source-pack.md`.
- Every substantive information item must cite one or more evidence ids.
- Every evidence fragment must have a coverage-table destination.
- Readable articles must be compiled from the source record and article map, not invented from memory.
- Do not transcribe audio, search the web, or write directly into `wiki/` unless the user separately asks for an ingest step.

## Preparation

For pasted transcript text, first save the text unchanged into a temporary
`.md` or `.txt` file, then run the preparation script on that file. Do not
clean, rewrite, summarize, or normalize the pasted transcript before saving it.

For file or directory input, run:

```bash
python3 ~/.codex/skills/transcript-source-compiler/scripts/prepare_transcript_source.py \
  /path/to/transcript-or-directory \
  --output-dir /path/to/generated/transcript-source-compiler/run-slug \
  --title "材料标题"
```

If working from the project copy instead of the installed skill copy, use:

```bash
python3 skills/transcript-source-compiler/scripts/prepare_transcript_source.py \
  /path/to/transcript-or-directory \
  --output-dir generated/transcript-source-compiler/run-slug \
  --title "材料标题"
```

Then read `source-pack.md` and `evidence-fragments.jsonl`.

## Compilation Procedure

### 1. Understand The Source

Identify the talk topic, intended audience, speaker role if explicit, central
problem, major themes, and any obvious chronology.

Do not repair unclear terms from memory. Mark uncertain names, products,
numbers, or phrases as `[听写不确定]`, `[上下文不足]`, or `[原文未说明]`.

### 2. Extract Information Items

Traverse all evidence fragments. Extract all substantive items:

- 观点
- 概念
- 定义
- 方法
- 步骤
- 案例
- 判断
- 数字
- 术语
- 人物
- 产品
- 项目
- 因果关系
- 限制条件
- 风险
- 反例
- 待核对内容

Use this item format inside `source-record.md` where useful:

```markdown
- I0001 | 类型: 观点 | 置信度: high | 证据: E0001, E0003
  - 内容: ...
  - 原文片段: "..."
```

Merge duplicates only when they state the same claim. Preserve all supporting
evidence ids.

### 3. Build `source-record.md`

Fill these sections:

```markdown
# 标题

## 来源信息
## 内容概览
## 主题脉络
## 主要观点
## 关键概念
## 方法框架
## 案例与例子
## 可沉淀到 Wiki 的知识点
## 矛盾与不确定项
## 待核对
## 证据覆盖表
```

The source record is the factual base. It may be longer than a normal article.
Completeness and evidence traceability are more important than elegance.

### 4. Build Readable Article Materials

Create `article-main.md` by compiling from the information items. The article
should have a title, introduction, clear sections, and conclusion. It may
reorder material and remove filler, but it must not add unsupported facts or
drop examples, limits, numbers, terms, or counterexamples.

If the source clearly contains several independent themes, create
`article-index.md` and topic-specific files such as:

```text
article-01-主题A.md
article-02-主题B.md
article-03-主题C.md
```

Keep details that do not fit the main flow in sections such as `补充信息`,
`术语与例子`, or `待核对`.

### 5. Build `article-map.md`

Map every article paragraph to information items and evidence ids:

```markdown
| Paragraph | Article | Items | Evidence | Coverage Note |
|---|---|---|---|---|
| P0001 | article-main.md | I0001, I0002 | E0001, E0003 | 合并表达，没有新增事实 |
```

### 6. Verify

Run:

```bash
python3 ~/.codex/skills/transcript-source-compiler/scripts/validate_transcript_source.py \
  /path/to/generated/transcript-source-compiler/run-slug
```

If working from the project copy, run:

```bash
python3 skills/transcript-source-compiler/scripts/validate_transcript_source.py \
  generated/transcript-source-compiler/run-slug
```

Fix any validation errors before delivering.

Manual verification is still required after the validator passes. The validator
checks structure and references; it does not prove semantic completeness or the
absence of hallucination. Review article paragraphs against `article-map.md`,
information items, and evidence ids, and confirm that examples, numbers, limits,
and counterexamples from the source record were preserved or explicitly marked
as excluded.

## Coverage Table Rules

Every `E-id` must appear in the source-record coverage table.

Valid statuses:

- `extracted`: converted into one or more information items.
- `duplicate`: repeats or reinforces another extracted item.
- `filler`: oral scaffolding, greeting, transition, or non-substantive phrase.
- `unclear`: transcription defect or context too incomplete to structure safely.

Example:

```markdown
| Evidence | Destination | Status | Linked Items | Note |
|---|---|---|---|---|
| E0001 | 主要观点 | extracted | I0001 |  |
| E0002 | 主要观点 | duplicate | I0001 | 重复强调 |
| E0003 | 无 | filler |  | 口语过渡 |
| E0004 | 待核对 | unclear |  | 产品名听写不确定 |
```

## Delivery

Return the output paths and a short uncertainty summary. Do not paste the full
article unless the user asks.
