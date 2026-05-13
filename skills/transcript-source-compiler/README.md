# Transcript Source Compiler

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

Convert long-form speech transcripts into auditable LLM-Wiki source records and one or more readable articles. It is for already-transcribed talks, lectures, salons, and sharing-session materials where completeness and evidence traceability matter more than short summarization.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/transcript-source-compiler
rsync -a skills/transcript-source-compiler/ ~/.codex/skills/transcript-source-compiler/
```

Requires: Python 3.8+.

## Usage

Prepare a transcript source pack:

```bash
python3 skills/transcript-source-compiler/scripts/prepare_transcript_source.py \
  /path/to/transcript-or-directory \
  --output-dir generated/transcript-source-compiler/run-slug \
  --title "材料标题"
```

Then compile the generated files manually with the skill instructions:

| Output | Purpose |
|--------|---------|
| `source-pack.md` | Immutable copy of the source text plus chunk index |
| `evidence-fragments.jsonl` | Evidence ids used for traceable extraction |
| `source-record.md` | Structured source record and coverage table |
| `article-main.md` | Main human-readable article |
| `article-index.md` | Index for multi-article output when needed |
| `article-map.md` | Paragraph-to-evidence mapping |
| `uncertain-items.md` | Unclear terms, missing context, and verification notes |

Validate the compiled run:

```bash
python3 skills/transcript-source-compiler/scripts/validate_transcript_source.py \
  generated/transcript-source-compiler/run-slug
```

## Workflow

| Step | Description |
|------|-------------|
| Prepare | Preserve raw transcript text and split it into evidence fragments |
| Extract | Convert every substantive claim into evidence-backed information items |
| Compile | Build `source-record.md` as the factual base |
| Write | Produce one complete article or several topic-specific articles |
| Map | Link each article paragraph back to item ids and evidence ids |
| Verify | Run the validator and manually review semantic coverage |

## License

MIT
