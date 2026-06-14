# Paper Deconstruction

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

Read an ML / CV / NLP paper as a **story + engineering artifact** and produce a visual, evidence-backed deep-read note. Built around the **五阶法 (Five-Stage Method)** distilled from the ViLT paper-reading lecture: structure → story → deconstruct → evaluate → visualize.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/paper-deconstruction
rsync -a skills/paper-deconstruction/ ~/.codex/skills/paper-deconstruction/
```

Claude Code users can install to `~/.claude/skills/paper-deconstruction/` with the same `rsync` pattern.

## Requirements

- None. The skill is a pure markdown workflow + references; you only need the helper skills it calls (e.g. `excalidraw-diagram`, `lovstudio-pdf2png`, `social-cover-maker`) when you want to actually render figures.

## The Five Stages

| # | Stage | Time | Output | Figure | Helper |
|---|-------|------|--------|--------|--------|
| 1 | Structure | 2-3 min | Chapter map + atypical-section notes | 章节地图 (drawn) | `excalidraw-diagram` |
| 2 | Story | 5-10 min | Pain points + prior work + selling point | Timeline (drawn) | `excalidraw-diagram` |
| 3 | Deconstruct | 15-30 min | 6-block breakdown (input / model / loss / tricks / data / experiments) | 1 figure per block (screenshot + draw mix) | `lovstudio-pdf2png` + `excalidraw-diagram` |
| 4 | Evaluate | 5 min | 5-dimension scoring (story / method / experiments / code / whitespace) | Radar chart (drawn) | `excalidraw-diagram` |
| 5 | Visualize | 5 min | Visual-asset list + cover | Unified captions + cover | `social-cover-maker` |

Details for each stage live in `SKILL.md`. The two output formats (academic note vs. WeChat post) live in `templates/`.

## When To Trigger

- A user hands you a paper PDF / arXiv link / lecture video and asks for "精读", "讲清楚", "组会汇报", or "我应该 follow 哪个方向".
- A user asks "这篇论文到底做了什么 / 为什么 work / 有什么坑可以挖".
- A user wants a survey, a topic, or a related-work draft.

Do **not** use this skill for: skimming conclusions, running baselines, or PyTorch implementation details (route those to a code-reading skill instead).

## When To Refuse

The lecture this skill is distilled from never covered diffusion. If a user asks for a paper-deconstruction of a math proof, an HCI/system paper, or a pure benchmark, refuse with a reason and route them to a more appropriate tool.

## Usage

1. Fill `templates/task-brief.md` (5 min) — title, source URL/PDF, target output format.
2. Walk the five stages in `SKILL.md`, ticking items in `templates/checklist.md` as you go.
3. Write the final note using `templates/output-template.md` (academic) or `templates/wechat-post-template.md` (公众号 / 知乎 / 小红书).
4. Use `references/visual-assets.md` to decide when to draw vs. when to screenshot, and how many figures to include (5-9 recommended).

## Layout

```text
paper-deconstruction/
├── SKILL.md                         # Main entry, ~300 lines, 5-stage method
├── README.md                        # This file
├── references/
│   ├── procedures.md                # 5 questions per deconstruction block
│   ├── concepts.md                  # 8 mental models
│   ├── source-map.md                # ViLT video → method evidence ids
│   ├── examples.md                  # Worked ViLT example
│   ├── gotchas.md                   # 12 common pitfalls
│   ├── visual-assets.md             # Draw-vs-screenshot, both output formats
│   └── chapter-digests.md           # Per-section condensed digests
├── templates/
│   ├── checklist.md                 # 35-item reading checklist
│   ├── output-template.md           # Academic note (5 sections)
│   ├── wechat-post-template.md      # WeChat post (6 narrative beats)
│   └── task-brief.md                # First-step brief
├── evals/
│   └── evals.json                   # 8 trigger / quality test cases
└── source/
    └── full_text.md                 # ViLT lecture transcript (immutable)
```

## Design Principles

| Principle | Meaning |
|-----------|---------|
| Story + engineering | A paper is a *narrative* sold to a reviewer as much as it is an engineering artifact. Read both. |
| Five stages, not five chapters | The stages produce *artifacts* (maps, timelines, breakdowns, scores, asset lists) — not essays. |
| Visual evidence | Every figure must answer "what is this figure about?". Decoration is banned. |
| Two output formats | Academic note and WeChat post share the first four stages; only stage 5 diverges by audience. |
| Traceability to source | All mental models and templates link back to `source/full_text.md` via `references/source-map.md`. |
| Evals as first-class | `evals/evals.json` tests trigger conditions, refusal conditions, and the 30-item checklist. |

## Source

- Lecture: ViLT 论文精读 (B 站 / YouTube, 2021 ICML paper).
- Extracted: 2026-06-02.
- Extractor: `extract_docx.py` (custom, docx → md).

## License

MIT
