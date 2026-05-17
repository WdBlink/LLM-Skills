# Book to Skill

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

Convert a PDF, EPUB, Markdown, or text book into one or more traceable, executable Agent Skills. This skill is for compiling a book into reusable agent behavior — workflows, decision rules, checklists, templates, gotchas, references, and evals — rather than producing a generic summary.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/book-to-skill
rsync -a skills/book-to-skill/ ~/.codex/skills/book-to-skill/
```

Claude Code users can install to `~/.claude/skills/book-to-skill/` with the same `rsync` pattern.

## Requirements

- Python 3.9+
- Optional PDF tools: `pdftotext` from poppler, `PyPDF2`, `pdfminer.six`, or `docling`
- Optional EPUB tools: `ebooklib` and `beautifulsoup4`

The extractor falls back to stdlib where possible and gives actionable errors when extraction is not possible.

## Usage

Prepare a source run:

```bash
python3 skills/book-to-skill/scripts/extract_book.py \
  /path/to/book.pdf \
  --output-dir generated/book-to-skill/my-book \
  --mode auto
```

Then follow the skill instructions to fill the work files and generate:

```text
generated/book-to-skill/my-book/output/<skill-slug>/
├── SKILL.md
├── README.md
├── references/
├── templates/
└── evals/evals.json
```

Validate the generated skill:

```bash
python3 skills/book-to-skill/scripts/validate_book_skill.py \
  generated/book-to-skill/my-book/output/<skill-slug>
```

## Design Principles

| Principle | Meaning |
|---|---|
| Structure, not summary | Extract frameworks, procedures, decision rules, anti-patterns, and templates. |
| Traceability | Major claims map to source chunks in `references/source-map.md`. |
| Progressive disclosure | Main `SKILL.md` stays compact; detailed material lives in `references/`. |
| Task boundaries | Prefer one skill per stable task scenario; recommend derived skills when a book spans many workflows. |
| Evals required | Every generated skill includes `evals/evals.json` for trigger and quality checks. |

## What This Improves Over Basic Book-to-Skill Converters

This skill takes inspiration from the public `virgiliojr94/book-to-skill` project, especially PDF/EPUB extraction, chapter-level digestion, and progressive chapter files. It adds stricter source mapping, required evals, templates, task-boundary decisions, and validation for long-term skill maintenance.

## License

MIT
