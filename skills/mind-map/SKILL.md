---
name: mind-map
description: "Actively record project thinking into Obsidian-compatible Markdown nodes, layered indexes, an interactive HTML mind map, and an AI context pack. Use when the user says mind map, 项目思维导图, 记录项目想法, or asks to preserve a project decision/design direction for later AI implementation."
metadata:
  version: "0.2.12"
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
   - root `index.md` and `index.json` for all-project status.
   - `log 时间线.md` and `log 时间线.json` for each project's chronological
     record.
   - `entries/*.md` for append-only history.
   - `<projectId>.md` for a project-named Obsidian directory page.
   - `nodes/*.md` for real Obsidian Markdown nodes with wiki links.
   - `indexes/source.*` for background provenance data. It should be generated
     for traceability but omitted from the visible project graph anchor.
   - `indexes/schema.*`, `indexes/relations.*`, and `indexes/runtime.*` for
     visible project-model, relation, and handoff indexes.
   - `indexes/schema/<type>.*` for project-model type indexes such as
     `thesis`, `principle`, `decision`, `constraint`, and `question`.
   - `indexes/relations/<type>.*` for relation-type indexes such as
     `supports`, `depends_on`, `implements`, and project-specific edge types.

The product shape is intentionally hybrid: preserve the current interactive
HTML experience, while also generating Obsidian-native Markdown nodes and
links so the same thinking can be browsed through Obsidian's graph and backlink
views.

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
  index.md
  index.json
  <projectId>/
    mindmap.json
    mindmap.html
    context.md
    log 时间线.md
    log 时间线.json
    <projectId>.md
    entries/
      2026-05-15T23-10-00.md
    nodes/
      decision-active-recording-is-source-of-truth-1a2b3c.md
    indexes/
      source 溯源索引.md
      source 溯源索引.json
      schema 判断模型索引.md
      schema 判断模型索引.json
      schema/
        thesis 核心判断.md
        thesis 核心判断.json
        principle 原则.md
        principle 原则.json
        decision 决策.md
        decision 决策.json
        constraint 约束.md
        constraint 约束.json
        question 问题.md
        question 问题.json
        deprecated 废弃.md
        deprecated 废弃.json
      relations 关系索引.md
      relations 关系索引.json
      relations/
        supports 支持.md
        supports 支持.json
        depends_on 依赖.md
        depends_on 依赖.json
      runtime 交接索引.md
      runtime 交接索引.json
```

Layer meanings:

- `index.md`: all-project Obsidian directory page. It links each project
  directory page and shows current project status: updated time, latest entry,
  node count, active task count, and open question count.
- `log 时间线.md`: project-level chronological log. It is linked from the
  project directory page, but rendered as graph-neutral text without entry/node
  wiki links. It records entry time, summary, involved node IDs/titles, edge
  IDs, and source entry paths.
- `source`: background provenance data: raw entries and which source entries
  support each node. Keep it available for traceability, but do not link it
  from the project directory page by default. Render `source 溯源索引.md` as
  graph-neutral text without Obsidian wiki links; use `source 溯源索引.json`
  for machine-readable provenance.
- `schema`: the project judgment model: thesis, principles, decisions,
  constraints, questions, and deprecated ideas. It should link to type-specific
  schema indexes instead of containing every node inline.
- `indexes/schema/<type label>.md`: type-specific schema pages, such as
  `thesis 核心判断.md` or `principle 原则.md`, that list nodes for that type.
- `relations`: semantic relations between nodes. It should link to
  relation-type indexes instead of containing every edge inline.
- `indexes/relations/<type label>.md`: relation-type pages, such as
  `supports 支持.md` or `depends_on 依赖.md`, that list concrete node-to-node
  relations for that type.
- `runtime`: technical handoff: implementation intent, task nodes, acceptance
  criteria, project path, and generated artifacts.

Graph hierarchy rule:

- Use top-down links only for Obsidian Markdown indexes.
- The all-project `index.md` links project directory pages, not each project's
  layer indexes.
- Each project directory page is the project graph anchor. It links visible
  first-level indexes (`schema`, `relations`, and `runtime`) and every generated
  child index page under those visible layers, but still avoids direct links to
  nodes and entries. It links `log 时间线.md` and does not link `source` by
  default.
- First-level indexes link only their next layer. For example,
  `schema 判断模型索引.md` links schema type pages, and `relations 关系索引.md`
  links relation type pages.
- Child index pages should not link back to the project page, parent index, or
  sibling indexes. This intentionally reduces graph clutter even if high-level
  index nodes appear smaller.
- Node pages should avoid index backlinks and direct node-to-node wiki links.
  Relation visibility in Obsidian should come through `indexes/relations/*`.

Obsidian compatibility:

- The root `index.md` should be refreshed on every project update so the vault
  has one current all-project entrypoint.
- Each project should have a project-named directory page, such as
  `<projectId>.md`, that anchors its visible schema/relations/runtime indexes.
- Each structured node should become a real Markdown file under `nodes/`.
- Entries should link to the generated node files with `[[nodes/...|title]]`.
- Node files should carry source-entry metadata without adding source/index
  backlinks that clutter the visible graph.
- Visible layer index filenames should be bilingual, such as
  `schema 判断模型索引.md`, `relations 关系索引.md`, and `runtime 交接索引.md`.
  Project ownership is provided by the project directory page, not repeated in
  every layer index title.
- Schema type filenames should also be bilingual, such as
  `decision 决策.md`, and be linked from `schema 判断模型索引.md`.
- Visible layer indexes should also use bilingual Obsidian display labels,
  such as `schema 判断模型索引`, `relations 关系索引`, and `runtime 交接索引`,
  regardless of the content language.
- Relation type names should stay concise (`supports`, `depends_on`,
  `implements`, and project-specific edge types), but known relation type
  filenames should be bilingual, such as `supports 支持.md`, and be linked from
  `relations 关系索引.md`. Keep edge lists out of schema layer pages.
- Wiki links should use vault-root-relative paths when linking generated nodes
  and indexes across the project directory, so same-named layer files in
  different project folders do not collide.
- `mindmap.html` remains a first-class artifact; Obsidian nodes extend the
  viewing surface and must not replace or degrade the HTML interaction model.

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
   should also report the `nodes/` and `indexes/` paths when they are created,
   plus the root `index.md` when relevant, and explicitly say that no project
   implementation was performed.

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
- Keep `log 时间线.md` current for every project. It should summarize entries in
  chronological order without adding entry/node wiki links.
- Generate real Obsidian Markdown nodes for structured thoughts. Do not treat
  `entries/*.md` as the only Markdown surface.
- Keep source provenance current in `source 溯源索引.json` and
  `source 溯源索引.md`, but keep it out of the project directory page unless
  the user explicitly asks to expose provenance in the graph.
- Keep `source 溯源索引.md` graph-neutral: no `[[entry]]`, no `[[node]]`, and
  no project/index backlinks. It may list IDs, titles, paths, timestamps, and
  source-entry IDs as plain text.
- Keep visible indexes current: `schema` for project judgment, `relations` for
  semantic edges, and `runtime` for technical handoff.
- Keep `schema 判断模型索引.md` as a schema overview. Put type-specific
  listings in `indexes/schema/<type label>.md` instead of expanding
  `## thesis`, `## principle`, and similar sections directly in the schema
  overview.
- Keep `edges` as a machine data field in JSON, but expose it as
  `relations 关系索引.md` and `indexes/relations/<type label>.md` in Obsidian
  Markdown.
  Do not render `## edges` sections in schema layer pages.
- Keep Obsidian index links top-down with a project anchor: the root all-project
  index links project pages, each project page links every index page in that
  project's visible schema/relations/runtime hierarchy, and child pages do not
  add project/parent/sibling backlinks.
- Keep Obsidian graph labels project-scoped: generate a project directory page
  that links to bilingual `schema 判断模型索引.md`, `relations 关系索引.md`, and
  `runtime 交接索引.md` layer pages.
- Keep the root all-project index current so project status can be scanned from
  a single Obsidian page.
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
