# Daily Todo Planner Schema

This skill keeps user planning data local and auditable. The agent may create
temporary JSON files in the working directory, then persist plan/review outputs
under the configured state directory.

## State Layout

```text
~/.codex/daily-todo-planner/
  profile.json
  plans/
    YYYY-MM-DD.json
  reviews/
    YYYY-MM-DD.json
  research/
    YYYY-MM-DD/
      task-id.md
```

Use `DAILY_TODO_STATE_DIR` or `--state-dir` to override the default location.

## Task Input

```json
{
  "constraints": {
    "work_start": "09:00",
    "work_end": "18:00",
    "buffer_ratio": 0.15,
    "default_lunch": true,
    "fixed_blocks": [
      {"label": "Lunch", "start": "12:00", "end": "13:00"}
    ]
  },
  "tasks": [
    {
      "id": "t1",
      "title": "Research a GitHub project",
      "outcome": "One-page evaluation with install risk and fit decision",
      "quadrant": "Q2",
      "importance": 5,
      "urgency": 2,
      "estimated_minutes": 90,
      "deadline": "YYYY-MM-DD",
      "energy": "deep",
      "type": "research",
      "tags": ["github", "research"],
      "depends_on": [],
      "fixed_start": null,
      "fixed_end": null
    }
  ]
}
```

Task rules:

- `id` must be stable within the day so review data can map back to the plan.
- `quadrant` is one of `Q1`, `Q2`, `Q3`, `Q4`.
- If `quadrant` is omitted, the helper infers it from `importance` and
  `urgency`: both >= 4 gives `Q1`, important only gives `Q2`, urgent only gives
  `Q3`, and neither gives `Q4`.
- `importance` and `urgency` use a 1-5 scale.
- `estimated_minutes` is the agent's best base estimate before profile
  adjustment.
- `type` should be one of `deep_work`, `admin`, `communication`, `research`,
  `errand`, `review`, or a concise custom type.
- Research candidates should include `research` in either `type` or `tags`.

## Plan Output

The script writes:

```json
{
  "date": "YYYY-MM-DD",
  "work_window": {"start": "09:00", "end": "18:00"},
  "scheduled": [
    {
      "id": "t1",
      "title": "Research a GitHub project",
      "start": "09:00",
      "end": "10:45",
      "adjusted_estimate_minutes": 105,
      "buffer_minutes": 15,
      "quadrant": "Q2",
      "prefetch_recommended": true
    }
  ],
  "unscheduled": [],
  "research_prefetch": []
}
```

The agent should present the plan in a human-readable timeline and explicitly
call out unscheduled overflow. Research-prefetch tasks include a
`research_pack_path` where the pre-research result should be written.

## Actuals Input

```json
{
  "actuals": [
    {
      "id": "t1",
      "status": "done",
      "actual_minutes": 120,
      "estimated_minutes": 90,
      "quadrant": "Q2",
      "type": "research",
      "tags": ["github", "research"],
      "interruption_reason": "",
      "energy_fit": "good",
      "notes": "More dependency reading than expected."
    }
  ]
}
```

Use `done`, `partial`, `blocked`, `skipped`, or `moved` for `status`. Only
entries with both `actual_minutes` and `estimated_minutes` update the profile.
Skipped tasks should be logged but not used for estimation learning.

## Long Brief Extraction

For long user descriptions, the agent should create the task JSON first. Split
the brief into deliverables, not literal sentences. Treat vague items as tasks
only when there is a clear next action; otherwise put them in assumptions or
questions. Preserve fixed meetings as `fixed_blocks`, hard deadlines as
`deadline`, research-heavy work as `type: "research"`, and lower-confidence
items in tags such as `maybe` or `needs_clarification`.

## Research Pack

For pre-researched tasks, save a concise Markdown pack:

```md
# Research Pack: task title

## Question
What must be decided or understood?

## Short Answer
The answer the user needs at task start.

## Evidence
- Source, command, or document reference.

## Risks / Unknowns
- What still needs validation.

## Next Actions
- Concrete steps for the scheduled work block.
```

Create packs with:

```bash
python3 scripts/todo_state.py prefetch --plan plan.json --answers answers.json
```

Optional `answers.json` format:

```json
{
  "answers": [
    {
      "id": "t1",
      "short_answer": "The project is promising but needs a license check.",
      "evidence": ["README install path is clear", "Recent releases exist"],
      "risks": ["API may change"],
      "next_actions": ["Clone and run the smoke test"]
    }
  ]
}
```
