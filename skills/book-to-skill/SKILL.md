---
name: book-to-skill
description: Convert a PDF, EPUB, Markdown, or text book into one or more traceable, executable Agent Skills. Use when the user wants to turn a book's knowledge into reusable skills, not just summarize it. Produces a compact SKILL.md, progressive reference files, source-map evidence, templates, and evals.
argument-hint: <book-path> [skill-slug]
---

# Book to Skill

Turn a book into reusable agent capability. Do not dump the book into context and do not produce a generic summary. Compile the book into task-oriented skill files with traceability and evals.

## Core Contract

- Input: `.pdf`, `.epub`, `.md`, `.txt`, or a directory of these files.
- Output: a generated skill directory containing `SKILL.md`, `references/`, `templates/`, and `evals/`.
- Preserve source traceability: important claims, rules, and examples must map back to chapter/page/location/evidence ids.
- Prefer executable behavior over book-report prose: write decision rules, workflows, checklists, gotchas, and output templates.
- Use progressive disclosure: keep the generated `SKILL.md` concise; move long chapter digests and examples into `references/`.
- If the book contains several independent workflows, create a book knowledge skill plus derived task-skill recommendations instead of forcing everything into one giant skill.
- Validate the generated skill before delivering.

## Quick Start

From an installed skill copy:

```bash
python3 ~/.codex/skills/book-to-skill/scripts/extract_book.py \
  /path/to/book.pdf \
  --output-dir generated/book-to-skill/book-slug \
  --mode auto
```

From a repository checkout:

```bash
python3 skills/book-to-skill/scripts/extract_book.py \
  /path/to/book.pdf \
  --output-dir generated/book-to-skill/book-slug \
  --mode auto
```

The extraction script creates:

- `source/full_text.md` — extracted source text, normalized but not summarized.
- `source/chunks.jsonl` — evidence chunks with stable ids such as `E0001`.
- `source/metadata.json` — source stats, extraction method, detected headings.
- `work/chapter-digests.md` — fill this during analysis.
- `work/concepts.md` — fill this during analysis.
- `work/procedures.md` — fill this during analysis.
- `work/source-map.md` — evidence-backed mapping from extracted knowledge to source ids.
- `output/<skill-slug>/` — destination for the final skill.

## Workflow

### 1. Extract and inspect

Run `extract_book.py`. Then read `source/metadata.json` and inspect `source/chunks.jsonl` plus the first/last parts of `source/full_text.md`.

Classify the book:

| Type | Best output |
|---|---|
| Single clear method | One procedural skill |
| Broad technical reference | One knowledge skill with reference files |
| Multiple independent workflows | One book skill plus derived task-skill recommendations |
| Mostly narrative/history/philosophy | LLM-Wiki/notes first, skill only for actionable sub-methods |

If extraction quality is poor, stop and explain what to install or try, for example `docling`, `pdftotext`, `ebooklib`, OCR, or a cleaner EPUB/PDF.

### 2. Build evidence-backed digests

Traverse `source/chunks.jsonl`. Extract into `work/` using these files:

- `chapter-digests.md`: chapter/section summaries with evidence ids.
- `concepts.md`: terms, definitions, mental models, and relationships.
- `procedures.md`: workflows, heuristics, decision rules, checklists, anti-patterns, templates, and examples.
- `source-map.md`: every high-value rule or example maps to chapter/page/location and evidence ids.

Use this item shape when useful:

```markdown
- K0001 | type: decision-rule | confidence: high | evidence: E0012, E0013
  - Name: <exact book term if present>
  - Rule: Use X when Y; avoid Z because W.
  - Source: Chapter 3 / p. 57 if available
```

Do not create unsupported rules. If a rule is a synthesis across multiple fragments, list all supporting evidence ids.

### 3. Decide skill boundaries

Before generating files, decide whether to create one or several skills.

Create **one skill** when the book serves one recurring task, e.g. customer interviewing, code review, negotiation prep.

Create **book knowledge skill + derived skill recommendations** when the book has multiple independent tasks. Put the derived recommendations in `references/derived-skills.md` unless the user explicitly asks you to create all of them now.

A generated skill should correspond to a stable task scenario, not merely a book title.

### 4. Generate the target skill

Create `output/<skill-slug>/` with:

```text
<skill-slug>/
├── SKILL.md
├── README.md
├── references/
│   ├── source-map.md
│   ├── chapter-digests.md
│   ├── concepts.md
│   ├── procedures.md
│   ├── examples.md
│   ├── gotchas.md
│   └── derived-skills.md        # when relevant
├── templates/
│   ├── task-brief.md
│   ├── checklist.md
│   └── output-template.md
└── evals/
    └── evals.json
```

Generated `SKILL.md` rules:

- Keep it under ~500 lines and preferably under 4,000 tokens.
- Front-load the most important behavior.
- Include activation conditions, core workflow, decision rules, gotchas, output format, and validation loop.
- Link to reference files and tell the agent when to read each one.
- Do not include long chapter summaries, decorative quotes, or raw copied book text.
- Use practitioner language: `Use X when Y`, `Check A before B`, `If C, do D`.

### 5. Generate evals

Create `evals/evals.json` with at least six cases:

1. Should trigger for a task directly covered by the book.
2. Should trigger for a paraphrased task with no book title mentioned.
3. Should read `references/source-map.md` or chapter digest for a specific claim.
4. Should follow a generated checklist/template.
5. Should not trigger for an unrelated task.
6. Should refuse or qualify when the user asks for content not supported by the book.

Each eval must include `id`, `prompt`, `expected`, and `checks`.

### 6. Validate

Run:

```bash
python3 ~/.codex/skills/book-to-skill/scripts/validate_book_skill.py \
  generated/book-to-skill/book-slug/output/<skill-slug>
```

From a repo checkout:

```bash
python3 skills/book-to-skill/scripts/validate_book_skill.py \
  generated/book-to-skill/book-slug/output/<skill-slug>
```

Fix validation errors. Warnings are acceptable only if you explain why.

## Quality Bar

A good generated skill is:

- **Traceable** — major rules cite source ids and chapter/page/location where available.
- **Actionable** — it changes what the agent does, not just what the agent knows.
- **Compact** — core behavior fits in `SKILL.md`; details live in references.
- **Navigable** — topic/chapter indexes tell the agent which reference file to read.
- **Testable** — evals cover trigger, non-trigger, behavior, and source-grounding.
- **Splittable** — multiple workflows become multiple skill recommendations.

## Common Failure Modes

- **Summary dump:** chapter summaries exist but no decision rules or workflow. Fix by extracting procedures and checklists.
- **No evidence map:** claims cannot be traced to source chunks. Fix `references/source-map.md`.
- **One-book-one-skill overreach:** huge books become unfocused. Split into task skills.
- **Overloaded SKILL.md:** too much chapter content in the main file. Move details to `references/`.
- **No evals:** the skill looks plausible but cannot be tested. Add `evals/evals.json`.
- **Copyright leakage:** do not copy long passages. Use short evidence snippets only when necessary and prefer paraphrase.
