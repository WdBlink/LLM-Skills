---
name: mind-map
description: "Actively record project thinking into an Obsidian-hosted HTML mind map and AI context pack. Use when the user says mind map, 项目思维导图, 记录项目想法, or asks to preserve a project decision/design direction for later AI implementation."
metadata:
  version: "0.1.0"
  default_vault: "/Users/echooo/SynologyDrive/Typora"
  output_root: "AI-Projects/MindMaps"
  default_language: "auto"
---

# Mind Map - Active Project Thinking

This skill records the user's latest project thinking as durable project memory.
It is not a transcript summarizer. It only records ideas the user explicitly asks
to preserve.

## Hard Boundary: Record Only

This skill is a recorder, not an implementer.

When `mind-map` is invoked, the agent must only preserve the user's stated
thought into the mind map artifacts. Do not implement the idea, change project
code, edit product files, run project build/test commands, create commits,
open PRs, publish releases, install dependencies, or perform any action that
turns the recorded thought into executed work.

This remains true even if the recorded thought contains words like "implement",
"fix", "build", "refactor", "发布", "实现", or "修复". In this skill, those words
become `task`, `decision`, `question`, or `implementationIntent` nodes. They are
not permission to execute the task.

Allowed during this skill:

- Read the existing mind map state.
- Structure the user's latest thought.
- Write `mindmap.json`, `context.md`, `mindmap.html`, and `entries/*.md`.
- Validate the generated mind map artifacts, such as JSON syntax or HTML script
  syntax.
- Report the generated paths.

Forbidden during this skill unless the user gives a separate, explicit
post-recording instruction outside the `mind-map` invocation:

- Editing the actual project implementation.
- Running the project's normal test/build/deploy/release workflow.
- Creating commits, tags, PRs, or releases.
- Installing or upgrading dependencies.
- Treating tasks captured in the mind map as immediately executable todos.

If the user asks to both record and implement in the same `mind-map` invocation,
finish the recording first and stop. Tell the user the idea has been recorded
and that implementation should be started as a separate task after reading the
generated `context.md`.

## Product Definition

`mind-map` is a project-level active thinking recorder:

1. The user states a new idea, decision, principle, question, or implementation
   intent in a Codex or Claude Code conversation.
2. The agent structures that thought into a small set of durable nodes and
   semantic relations.
3. The helper script writes the result into the default Obsidian vault as:
   - `mindmap.json` for machine-readable state.
   - `context.md` for future AI implementation agents.
   - `mindmap.html` for human visual review.
   - `entries/*.md` for append-only history.

The generated page supports Chinese and English interface labels. Chinese user
input should default to Chinese output unless the user explicitly asks for
English.

The generated page must be interactive enough for project exploration. At the
MVP level it should support search, node-type filters, click-to-focus related
nodes, expandable node details, and reset-to-full-map.

## Storage Contract

Default vault resolution:

1. `MIND_MAP_VAULT` environment variable.
2. `~/.agent-mindmap/config.json` field `defaultVaultPath`.
3. `/Users/echooo/SynologyDrive/Typora`.

Default output layout:

```text
<ObsidianVault>/AI-Projects/MindMaps/
  index.json
  <projectId>/
    mindmap.json
    mindmap.html
    context.md
    entries/
      2026-05-15T23-10-00.md
```

## Node Model

Keep node types narrow and implementation useful:

- `thesis`: the project's current core belief.
- `principle`: a product or engineering rule that should guide future work.
- `decision`: a confirmed design or implementation decision.
- `constraint`: a boundary future work must respect.
- `question`: an unresolved issue.
- `task`: an implementation action.
- `deprecated`: an idea that should not be revived unless explicitly changed.

Statuses:

- `active`: currently valid.
- `draft`: plausible but not confirmed.
- `resolved`: answered or completed.
- `deprecated`: intentionally abandoned.
- `conflict`: contradicts another active idea and needs review.

## Required Agent Workflow

When this skill is invoked:

0. Confirm the boundary internally: this invocation will record only. Do not
   implement the recorded idea.
1. Identify the current project path from the working directory unless the user
   explicitly names another project path.
2. Read the user's latest thought carefully. Do not summarize the whole chat
   unless the user explicitly asks you to include prior context.
3. Create a concise structured payload:

```json
{
  "thought": "Original user thought or a faithful paraphrase.",
  "language": "zh",
  "summary": "One sentence describing the new project insight.",
  "nodes": [
    {
      "type": "decision",
      "title": "Active recording is the source of truth",
      "summary": "The mind map should be updated when the user intentionally records a project thought, not by automatic transcript scanning.",
      "rationale": "Active recording preserves intent and avoids noisy auto-generated context.",
      "status": "active",
      "priority": "high",
      "implications": [
        "Codex and Claude Code skills become the creation entrypoint",
        "Obsidian stores and displays the durable result"
      ],
      "tags": ["product-direction"]
    }
  ],
  "edges": [
    {
      "fromTitle": "Active recording is the source of truth",
      "toTitle": "Obsidian stores project mind maps",
      "type": "implies",
      "label": "requires"
    }
  ],
  "implementationIntent": [
    "Future AI agents should read context.md before implementing project changes."
  ],
  "acceptanceCriteria": [
    "The project has an updated mindmap.html, mindmap.json, context.md, and entry markdown file."
  ]
}
```

4. Save the payload to a temporary JSON file.
5. Run the helper script from this skill directory:

```bash
python3 scripts/update_mindmap.py --entry-json /tmp/mind-map-entry.json --project-path "$PWD" --language auto
```

6. Report the generated `context.md` and `mindmap.html` paths. The response
   should explicitly say that no project implementation was performed.

## Language Mode

Use `language` in the structured payload or `--language` on the script:

- `zh`: Chinese `context.md`, entry headings, and default HTML chrome.
- `en`: English `context.md`, entry headings, and default HTML chrome.
- `auto`: detect from the payload; Chinese text defaults to Chinese.

The HTML page includes a Chinese / English toggle for interface labels. This
toggle translates page chrome, node type badges, status badges, lane titles,
and empty states. It does not machine-translate the user's actual node content;
the agent should write node titles and summaries in the user's preferred
language.

## Quality Rules

- Record only. Never convert a mind-map entry into implementation work during
  the same invocation.
- Prefer updating or connecting existing nodes over creating duplicates.
- Mark abandoned ideas as `deprecated` instead of deleting them.
- Keep `context.md` current and implementation-oriented. It is the file future
  AI agents should read before coding.
- Keep `entries/*.md` append-only. They preserve what the user meant at the
  time of recording.
- Do not automatically scan all local sessions. This skill records active user
  intent only.
- If the user critiques the visual form of the mind map, record that critique
  as a `question`, `constraint`, or `task`; do not pretend the current HTML
  card layout is the final mind-map design.
- If the user says the map should be interactive, preserve that as a product
  principle. The HTML output should not regress to a static-only page.

## Future Implementation Prompt

When handing work to another AI implementation agent, say:

```text
Before implementing, read:
<ObsidianVault>/AI-Projects/MindMaps/<projectId>/context.md
```
