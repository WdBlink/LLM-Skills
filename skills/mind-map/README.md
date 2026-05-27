# Mind Map

![Version](https://img.shields.io/badge/version-0.2.12-CC785C)

Record explicit project thinking into Obsidian-compatible Markdown nodes, layered indexes, an interactive HTML mind map, and an AI context pack. It is for preserving decisions, principles, questions, constraints, and implementation intent before future agents change project code.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/mind-map
rsync -a skills/mind-map/ ~/.codex/skills/mind-map/
```

Requires: Python 3.8+.

Optional vault configuration:

```bash
export MIND_MAP_VAULT=/path/to/obsidian-vault
```

If `MIND_MAP_VAULT` is not set, the script reads `~/.agent-mindmap/config.json` field `defaultVaultPath`, then falls back to the default path declared in `SKILL.md`.

## Usage

Create a structured entry JSON, then update the current project's mind map:

```bash
python3 skills/mind-map/scripts/update_mindmap.py \
  --entry-json /tmp/mind-map-entry.json \
  --project-path "$PWD" \
  --language auto
```

Record a plain thought directly:

```bash
python3 skills/mind-map/scripts/update_mindmap.py \
  --thought "This project should treat the evaluation protocol as immutable." \
  --project-path "$PWD" \
  --language en
```

## Outputs

| Output | Description |
|--------|-------------|
| `mindmap.json` | Machine-readable project mind map state |
| `mindmap.html` | Interactive visual review page with search, filters, focus, details, reset, and Chinese / English chrome |
| `context.md` | AI-readable implementation context pack for future work |
| `log 时间线.md` / `.json` | Project chronological log; graph-neutral timeline linked from the project page |
| root `index.md` / `index.json` | All-project directory and machine-readable status index |
| `<projectId>.md` | Project-named Obsidian directory page linking indexes, nodes, entries, and generated artifacts |
| `entries/*.md` | Append-only record of each captured project thought |
| `nodes/*.md` | Real Obsidian Markdown nodes with frontmatter and source-entry metadata |
| `indexes/source 溯源索引.md` / `.json` | Background source and provenance index; graph-neutral and not linked from the project anchor |
| `indexes/schema 判断模型索引.md` / `.json` | Project judgment model index |
| `indexes/schema/<type label>.md` / `.json` | Type-specific schema indexes, such as `decision 决策.md` |
| `indexes/relations 关系索引.md` / `.json` | Semantic relation index |
| `indexes/relations/<type label>.md` / `.json` | Relation-type indexes, such as `supports 支持.md` |
| `indexes/runtime 交接索引.md` / `.json` | Runtime and technical handoff index |
| `index.json` | Vault-level project index |

Markdown index links use project anchors: the all-project index links project
pages, each project page links visible schema / relations / runtime indexes and
their child indexes, while source stays as a background provenance layer.
`source 溯源索引.md` is rendered without Obsidian wiki links so it does not create graph
edges.
Visible index filenames are bilingual, for example `decision 决策.md`.

## Boundary

This skill records only. A `mind-map` invocation must not implement the captured idea, run the project build or tests, install dependencies, create commits, open PRs, or publish releases. Implementation should be started as a separate task after reading the generated `context.md`.

## License

MIT
