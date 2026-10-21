# Daily Todo Planner

![Private](https://img.shields.io/badge/Private-blueviolet)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

Codex skill for turning long daily descriptions into Eisenhower-matrix task
plans, local time-block schedules, execution reviews, and research prefetch
queues.

## Install

```bash
mkdir -p ~/.codex/skills/daily-todo-planner
rsync -a skills/daily-todo-planner/ ~/.codex/skills/daily-todo-planner/
```

## Use

Ask the agent to plan the day from a messy brief:

```text
Use daily-todo-planner. Here is my day: ...
```

Run deterministic helpers directly:

```bash
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py profile
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py plan --tasks tasks.json --date YYYY-MM-DD --markdown plan.md
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py review --actuals actuals.json --plan plan.json
python3 ~/.codex/skills/daily-todo-planner/scripts/todo_state.py prefetch --plan plan.json --answers answers.json
```

Planning and review data defaults to `~/.codex/daily-todo-planner`. Override it
with `DAILY_TODO_STATE_DIR` or `--state-dir`.

## Self-test

```bash
python3 -m unittest discover -s skills/daily-todo-planner/tests
```
