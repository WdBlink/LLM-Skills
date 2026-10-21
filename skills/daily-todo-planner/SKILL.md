---
name: daily-todo-planner
description: "Generate daily task plans from long free-form descriptions, classify work with the Eisenhower matrix, estimate effort, schedule time blocks, learn from actual execution, and pre-research research-heavy tasks. Use when the user says daily todo, 今日待办, 每日计划, 四象限, time blocking, execution review, task复盘, or asks to plan a day from messy notes."
metadata:
  version: "0.1.0"
  state_dir: "~/.codex/daily-todo-planner"
  default_language: "zh-first"
---

# Daily Todo Planner

Turn a messy daily brief into an executable day plan with local feedback
learning.

## Quick Start

1. Collect or infer: target date, timezone, working window, fixed commitments,
   hard deadlines, energy constraints, and the user's long task description.
2. Convert the brief into atomic tasks with observable deliverables.
3. Classify each task into Eisenhower quadrant: Q1 important+urgent, Q2
   important+not urgent, Q3 not important+urgent, Q4 not important+not urgent.
4. Estimate minutes, apply historical multipliers from the local profile, and
   schedule time blocks with buffers.
5. Detect research-heavy tasks and start pre-research immediately when tools are
   available; save or return a research pack before the scheduled block.
6. At review time, collect actual minutes/status and update the local profile.

## Planning Workflow

Use Chinese output by default for Chinese input.

1. If constraints are missing, assume today, the local timezone, a 09:00-18:00
   work window, lunch as a fixed block, and 10-20% buffer. State assumptions.
2. Read the local profile with `scripts/todo_state.py profile` when available.
3. Normalize tasks using `references/schema.md`: title, outcome, quadrant,
   importance, urgency, estimate, deadline, energy, type, tags, dependencies.
4. Before scheduling, explain the quadrant assignment for each non-trivial task.
5. Run `scripts/todo_state.py plan --tasks tasks.json --date YYYY-MM-DD` for
   deterministic time blocking, or manually follow the same ordering if script
   execution is not available.
6. Return a compact plan: timeline, task table, research prefetch queue,
   overflow/deferral list, and assumptions.

When extracting tasks from a long brief, split by deliverable instead of by
sentence. Preserve deadlines, meetings, dependency clues, and "later" ideas as
structured task fields. If a quadrant is not explicit, infer it from
importance/urgency: high+high Q1, high+low Q2, low+high Q3, low+low Q4.

## Review And Learning

1. When the user reports execution, capture status, actual minutes,
   interruption reason, energy fit, and useful notes for each task.
2. Run `scripts/todo_state.py review --actuals actuals.json --plan plan.json`.
3. Summarize what changed in the profile: systematic underestimation, task
   types that need buffers, best time windows, and recurring blockers.
4. Never invent actual execution data. If the user only gives partial feedback,
   update only those tasks.

## Research Prefetch

Treat GitHub projects, papers, market/technical research, unfamiliar tools,
decision memos, and comparison tasks as research candidates.

For each candidate, create a research brief with: question, expected decision,
sources, depth, scheduled handoff time, and deliverable format. If multi-agent,
browser, web, GitHub, paper, or local repository tools are available, start the
research as soon as the plan is accepted or when the user explicitly asks for
prefetch. If background execution is not available, produce a fast prep pack
and mark deeper research as pending.

Use `scripts/todo_state.py prefetch --plan plan.json --answers answers.json` to
create local research pack files for scheduled research tasks, then fill each
pack with findings before the task's scheduled handoff time.

Do not create recurring reminders, monitors, new threads, or automations unless
the user explicitly asks for them.

## Commands

```bash
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py profile
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py plan --tasks tasks.json --date YYYY-MM-DD --markdown plan.md
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py review --actuals actuals.json --plan plan.json
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py prefetch --plan plan.json
```

See `references/schema.md` for the task, plan, actuals, and research-pack
schemas.
