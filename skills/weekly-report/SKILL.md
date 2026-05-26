---
name: weekly-report
description: Generate Chinese weekly work report DOCX files from recent work logs or a provided log file. Use when the user asks to create, test, troubleshoot, or adapt a portable weekly report workflow for `#工作日志` notes, MiniMax/OpenAI-compatible LLM summaries, Word weekly report templates, fallback summaries, or plan-completion wording in any environment.
---

# Weekly Report

Use this skill to generate Chinese weekly report DOCX files from work logs in a
portable way. Do not assume a specific local checkout, absolute path, or
project entrypoint exists in the target environment.

## Quick Start

Use the bundled self-contained script:

```bash
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py \
  --log-dir /path/to/work-logs \
  --output-dir /path/to/reports \
  --days 7
```

When no Word template is available, the script creates a basic DOCX report. If
a template exists, pass it explicitly:

```bash
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py \
  --log-dir /path/to/work-logs \
  --template /path/to/template.docx \
  --output-dir /path/to/reports
```

## Workflow

1. Confirm whether the user wants to generate a report, smoke-test the
   generator, or adapt the workflow to another environment.
2. Use `scripts/generate_weekly_report.py --dry-run` first if paths are
   uncertain, especially template, output, and log paths.
3. For normal generation, pass `--log-dir` when the user wants recent
   `#工作日志` notes to drive the report, or `--logs-file` when the source is a
   single file.
4. Pass `--include-untagged` only when the source material does not use
   `#工作日志`.
5. Pass `--no-api` for deterministic fallback output or smoke tests. Otherwise
   configure an OpenAI-compatible endpoint through environment variables.
6. Preserve the low-AI style: "计划完成情况" should map one-to-one to
   plan/task items, avoid template-heavy headings, and avoid invented metrics
   or outcomes.

## Commands

Generate without a template:

```bash
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py \
  --log-dir /path/to/logs \
  --output-dir /path/to/reports
```

Generate from a specific log folder and force fallback mode without any LLM API
call:

```bash
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py \
  --log-dir /path/to/logs \
  --output-dir /path/to/reports \
  --days 7 \
  --no-api
```

Run a path/configuration check without writing a report:

```bash
python3 ~/.codex/skills/weekly-report/scripts/generate_weekly_report.py \
  --log-dir /path/to/logs \
  --output-dir /path/to/reports \
  --template /path/to/template.docx \
  --dry-run
```

## Environment

Read `references/workflow.md` when adapting the workflow or debugging template
filling. The generator uses:

- `python-docx` for DOCX reading/writing.
- `requests` only when calling an LLM API.
- `WEEKLY_REPORT_*`, `MINIMAX_*`, `OPENAI_*`, or `SILICONFLOW_*` variables for
  OpenAI-compatible chat-completions APIs.

Validate changes with a dry-run and a temporary fixture generation. Avoid
writing into a user's real report directory during smoke tests unless they ask
for an actual report.
