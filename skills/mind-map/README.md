# Mind Map

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

Record explicit project thinking into an Obsidian-hosted HTML mind map and an AI context pack. It is for preserving decisions, principles, questions, constraints, and implementation intent before future agents change project code.

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
| `entries/*.md` | Append-only record of each captured project thought |
| `index.json` | Vault-level project index |

## Boundary

This skill records only. A `mind-map` invocation must not implement the captured idea, run the project build or tests, install dependencies, create commits, open PRs, or publish releases. Implementation should be started as a separate task after reading the generated `context.md`.

## License

MIT
