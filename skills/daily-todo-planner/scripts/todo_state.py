#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE_DIR = "~/.codex/daily-todo-planner"
QUADRANT_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
LEARNABLE_STATUSES = {"done", "partial", "completed"}
VALID_QUADRANTS = set(QUADRANT_ORDER)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_state_dir(value: str | None) -> Path:
    raw = value or os.environ.get("DAILY_TODO_STATE_DIR") or DEFAULT_STATE_DIR
    return Path(raw).expanduser()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def default_profile() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": None,
        "task_count": 0,
        "global": {"count": 0, "mean_multiplier": 1.0},
        "quadrants": {},
        "types": {},
        "tags": {},
    }


def load_profile(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "profile.json"
    if not path.exists():
        return default_profile()
    profile = read_json(path)
    merged = default_profile()
    merged.update(profile)
    for key in ("quadrants", "types", "tags"):
        merged.setdefault(key, {})
    merged.setdefault("global", {"count": 0, "mean_multiplier": 1.0})
    return merged


def save_profile(state_dir: Path, profile: dict[str, Any]) -> None:
    profile["updated_at"] = now_iso()
    write_json(state_dir / "profile.json", profile)


def parse_time(value: str) -> int:
    try:
        hour, minute = value.split(":", 1)
        return int(hour) * 60 + int(minute)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"invalid HH:MM time: {value}") from exc


def format_time(minutes: int) -> str:
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def ceil_to_five(value: float) -> int:
    return int(math.ceil(value / 5.0) * 5)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bucket_multiplier(bucket: dict[str, Any] | None) -> float | None:
    if not bucket or int(bucket.get("count", 0)) <= 0:
        return None
    return float(bucket.get("mean_multiplier", 1.0))


def estimate_multiplier(profile: dict[str, Any], task: dict[str, Any]) -> float:
    values: list[float] = []
    for bucket in (
        profile.get("global"),
        profile.get("quadrants", {}).get(str(task.get("quadrant", "")).upper()),
        profile.get("types", {}).get(str(task.get("type", ""))),
    ):
        value = bucket_multiplier(bucket)
        if value is not None:
            values.append(value)
    for tag in task.get("tags", []) or []:
        value = bucket_multiplier(profile.get("tags", {}).get(str(tag)))
        if value is not None:
            values.append(value)
    if not values:
        return 1.0
    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        return clamp(values[midpoint], 0.5, 2.5)
    return clamp((values[midpoint - 1] + values[midpoint]) / 2, 0.5, 2.5)


def infer_quadrant(task: dict[str, Any]) -> tuple[str, str]:
    explicit = str(task.get("quadrant", "")).upper()
    if explicit in VALID_QUADRANTS:
        return explicit, "explicit"

    importance = int(task.get("importance", 1) or 1)
    urgency = int(task.get("urgency", 1) or 1)
    is_important = importance >= 4
    is_urgent = urgency >= 4
    if is_important and is_urgent:
        return "Q1", "importance>=4 and urgency>=4"
    if is_important and not is_urgent:
        return "Q2", "importance>=4 and urgency<4"
    if not is_important and is_urgent:
        return "Q3", "importance<4 and urgency>=4"
    return "Q4", "importance<4 and urgency<4"


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(task)
    quadrant, reason = infer_quadrant(normalized)
    normalized["quadrant"] = quadrant
    normalized["quadrant_reason"] = reason
    if not normalized.get("id"):
        normalized["id"] = slugify(str(normalized.get("title", "task")))
    normalized.setdefault("tags", [])
    normalized.setdefault("depends_on", [])
    normalized.setdefault("type", "deep_work")
    return normalized


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:80] or "task"


def update_bucket(container: dict[str, Any], key: str, multiplier: float) -> None:
    bucket = container.setdefault(key, {"count": 0, "mean_multiplier": 1.0})
    count = int(bucket.get("count", 0))
    mean = float(bucket.get("mean_multiplier", 1.0))
    bucket["count"] = count + 1
    bucket["mean_multiplier"] = round(((mean * count) + multiplier) / (count + 1), 3)


def is_research_task(task: dict[str, Any]) -> bool:
    task_type = str(task.get("type", "")).lower()
    tags = {str(tag).lower() for tag in task.get("tags", []) or []}
    title = str(task.get("title", "")).lower()
    keywords = ("research", "github", "paper", "论文", "调研", "研究", "竞品")
    return task_type == "research" or "research" in tags or any(k in title for k in keywords)


def research_pack_filename(task: dict[str, Any]) -> str:
    return f"{slugify(str(task.get('id') or task.get('title') or 'research-task'))}.md"


def deadline_sort_key(value: Any) -> str:
    if not value:
        return "9999-12-31"
    text = str(value)
    try:
        date.fromisoformat(text)
        return text
    except ValueError:
        return "9999-12-31"


def task_priority(task: dict[str, Any]) -> tuple[Any, ...]:
    quadrant = str(task.get("quadrant", "Q4")).upper()
    importance = int(task.get("importance", 1) or 1)
    urgency = int(task.get("urgency", 1) or 1)
    return (
        QUADRANT_ORDER.get(quadrant, 9),
        deadline_sort_key(task.get("deadline")),
        -importance,
        -urgency,
        str(task.get("title", "")),
    )


def normalize_fixed_blocks(payload: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
    constraints = payload.get("constraints", {}) or {}
    blocks = list(constraints.get("fixed_blocks", []) or [])
    if constraints.get("default_lunch", True) and start < parse_time("12:00") and end > parse_time("13:00"):
        has_lunch = any(str(block.get("label", "")).lower() == "lunch" for block in blocks)
        if not has_lunch:
            blocks.append({"label": "Lunch", "start": "12:00", "end": "13:00"})

    normalized: list[dict[str, Any]] = []
    for block in blocks:
        block_start = parse_time(str(block["start"]))
        block_end = parse_time(str(block["end"]))
        if block_end <= start or block_start >= end:
            continue
        normalized.append(
            {
                "label": str(block.get("label", "Fixed block")),
                "start": format_time(max(start, block_start)),
                "end": format_time(min(end, block_end)),
                "_start": max(start, block_start),
                "_end": min(end, block_end),
            }
        )
    return sorted(normalized, key=lambda item: item["_start"])


def free_windows(start: int, end: int, busy: list[tuple[int, int]]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    cursor = start
    for busy_start, busy_end in sorted(busy):
        if busy_end <= cursor:
            continue
        if busy_start > cursor:
            windows.append((cursor, busy_start))
        cursor = max(cursor, busy_end)
    if cursor < end:
        windows.append((cursor, end))
    return windows


def place_task(
    task: dict[str, Any],
    duration: int,
    buffer_minutes: int,
    start: int,
    end: int,
    busy: list[tuple[int, int]],
) -> tuple[int, int] | None:
    required = duration + buffer_minutes
    for window_start, window_end in free_windows(start, end, busy):
        if window_end - window_start >= required:
            task_start = window_start
            task_end = task_start + duration
            busy.append((task_start, task_end + buffer_minutes))
            return task_start, task_end
    return None


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = resolve_state_dir(args.state_dir)
    payload = read_json(Path(args.tasks))
    profile = load_profile(state_dir)
    constraints = payload.get("constraints", {}) or {}
    start = parse_time(args.start or constraints.get("work_start", "09:00"))
    end = parse_time(args.end or constraints.get("work_end", "18:00"))
    if end <= start:
        raise SystemExit("work end must be after work start")

    plan_date = args.date or date.today().isoformat()
    buffer_ratio = float(constraints.get("buffer_ratio", args.buffer_ratio))
    fixed_blocks = normalize_fixed_blocks(payload, start, end)
    busy: list[tuple[int, int]] = [(block["_start"], block["_end"]) for block in fixed_blocks]
    tasks = [normalize_task(task) for task in list(payload.get("tasks", []) or [])]
    scheduled: list[dict[str, Any]] = []
    unscheduled: list[dict[str, Any]] = []

    fixed_tasks: list[dict[str, Any]] = []
    floating_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("fixed_start") and task.get("fixed_end"):
            fixed_tasks.append(task)
        else:
            floating_tasks.append(task)

    for task in fixed_tasks:
        task_start = parse_time(str(task["fixed_start"]))
        task_end = parse_time(str(task["fixed_end"]))
        if task_start < start or task_end > end or task_end <= task_start:
            unscheduled.append({"reason": "fixed time is outside work window", **task})
            continue
        if any(task_start < busy_end and busy_start < task_end for busy_start, busy_end in busy):
            unscheduled.append({**task, "reason": "fixed time conflicts with a busy interval"})
            continue
        busy.append((task_start, task_end))
        scheduled.append(task_to_plan_item(task, task_start, task_end, 0, 1.0))

    for task in sorted(floating_tasks, key=task_priority):
        base_estimate = int(task.get("estimated_minutes", 30) or 30)
        multiplier = estimate_multiplier(profile, task)
        adjusted = max(5, ceil_to_five(base_estimate * multiplier))
        buffer_minutes = max(5, ceil_to_five(adjusted * buffer_ratio)) if buffer_ratio > 0 else 0
        slot = place_task(task, adjusted, buffer_minutes, start, end, busy)
        if slot is None:
            unscheduled.append(
                {
                    "reason": "not enough remaining work-window capacity",
                    "adjusted_estimate_minutes": adjusted,
                    "buffer_minutes": buffer_minutes,
                    **task,
                }
            )
            continue
        scheduled.append(task_to_plan_item(task, slot[0], slot[1], buffer_minutes, multiplier))

    scheduled.sort(key=lambda item: item["start"])
    research_prefetch = [item for item in scheduled if item.get("prefetch_recommended")]
    research_dir = state_dir / "research" / plan_date
    for item in research_prefetch:
        item["research_pack_path"] = str(research_dir / research_pack_filename(item))
    plan = {
        "version": 1,
        "date": plan_date,
        "generated_at": now_iso(),
        "state_dir": str(state_dir),
        "work_window": {"start": format_time(start), "end": format_time(end)},
        "fixed_blocks": [{k: v for k, v in block.items() if not k.startswith("_")} for block in fixed_blocks],
        "scheduled": scheduled,
        "unscheduled": unscheduled,
        "research_prefetch": research_prefetch,
    }

    output = Path(args.output) if args.output else state_dir / "plans" / f"{plan_date}.json"
    write_json(output, plan)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(render_markdown(plan), encoding="utf-8")
    return {"output": str(output), "plan": plan}


def task_to_plan_item(
    task: dict[str, Any],
    start: int,
    end: int,
    buffer_minutes: int,
    multiplier: float,
) -> dict[str, Any]:
    item = dict(task)
    item.update(
        {
            "start": format_time(start),
            "end": format_time(end),
            "estimated_minutes": int(task.get("estimated_minutes", end - start) or (end - start)),
            "adjusted_estimate_minutes": end - start,
            "buffer_minutes": buffer_minutes,
            "estimate_multiplier": round(multiplier, 3),
            "quadrant": str(task.get("quadrant", "Q4")).upper(),
            "quadrant_reason": task.get("quadrant_reason", "explicit"),
            "prefetch_recommended": is_research_task(task),
        }
    )
    return item


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# Daily Plan / 每日计划 {plan['date']}",
        "",
        f"Work window: {plan['work_window']['start']}-{plan['work_window']['end']}",
        "",
        "## Timeline",
        "",
        "| Time | Item | Quadrant | Estimate | Notes |",
        "|---|---|---:|---:|---|",
    ]
    timeline: list[dict[str, Any]] = []
    for block in plan.get("fixed_blocks", []):
        timeline.append({"start": block["start"], "end": block["end"], "title": block["label"], "quadrant": "-", "adjusted_estimate_minutes": "-", "notes": "fixed"})
    for item in plan.get("scheduled", []):
        timeline.append(item)
    for item in sorted(timeline, key=lambda entry: str(entry["start"])):
        notes = []
        if item.get("prefetch_recommended"):
            notes.append("prefetch")
        if item.get("buffer_minutes"):
            notes.append(f"+{item['buffer_minutes']}m buffer")
        lines.append(
            f"| {item['start']}-{item['end']} | {item.get('title', '')} | "
            f"{item.get('quadrant', '')} | {item.get('adjusted_estimate_minutes', '')}m | "
            f"{', '.join(notes) or item.get('notes', '')} |"
        )

    lines.extend(["", "## Research Prefetch", ""])
    research = plan.get("research_prefetch", [])
    if research:
        for item in research:
            lines.append(f"- {item['start']} {item['title']} (`{item.get('id', '')}`)")
    else:
        lines.append("- None")

    lines.extend(["", "## Overflow", ""])
    overflow = plan.get("unscheduled", [])
    if overflow:
        for item in overflow:
            lines.append(f"- {item.get('title', item.get('id', 'untitled'))}: {item.get('reason', 'unscheduled')}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def review_actuals(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = resolve_state_dir(args.state_dir)
    profile = load_profile(state_dir)
    payload = read_json(Path(args.actuals))
    actuals = payload if isinstance(payload, list) else payload.get("actuals", [])
    plan_lookup = load_plan_lookup(Path(args.plan)) if args.plan else {}
    updates: list[dict[str, Any]] = []

    for actual in actuals:
        task_id = str(actual.get("id", ""))
        planned = plan_lookup.get(task_id, {})
        merged = {**planned, **actual}
        status = str(merged.get("status", "")).lower()
        estimated = float(merged.get("estimated_minutes") or 0)
        actual_minutes = float(merged.get("actual_minutes") or 0)
        if status not in LEARNABLE_STATUSES or estimated <= 0 or actual_minutes <= 0:
            continue
        multiplier = clamp(actual_minutes / estimated, 0.4, 3.0)
        update_bucket(profile, "global", multiplier)
        quadrant = str(merged.get("quadrant", "")).upper()
        if quadrant:
            update_bucket(profile.setdefault("quadrants", {}), quadrant, multiplier)
        task_type = str(merged.get("type", ""))
        if task_type:
            update_bucket(profile.setdefault("types", {}), task_type, multiplier)
        for tag in merged.get("tags", []) or []:
            update_bucket(profile.setdefault("tags", {}), str(tag), multiplier)
        profile["task_count"] = int(profile.get("task_count", 0)) + 1
        updates.append(
            {
                "id": task_id,
                "title": merged.get("title"),
                "estimated_minutes": estimated,
                "actual_minutes": actual_minutes,
                "multiplier": round(multiplier, 3),
            }
        )

    save_profile(state_dir, profile)
    review_date = args.date or (read_json(Path(args.plan)).get("date") if args.plan else date.today().isoformat())
    review = {
        "version": 1,
        "date": review_date,
        "reviewed_at": now_iso(),
        "actuals": actuals,
        "updates": updates,
        "profile": profile,
    }
    output = Path(args.output) if args.output else state_dir / "reviews" / f"{review_date}.json"
    write_json(output, review)
    return {"output": str(output), "updates": updates, "profile": profile}


def create_prefetch_packs(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = resolve_state_dir(args.state_dir)
    plan = read_json(Path(args.plan))
    plan_date = str(plan.get("date") or date.today().isoformat())
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else state_dir / "research" / plan_date
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = load_research_answers(Path(args.answers)) if args.answers else {}
    created: list[dict[str, str]] = []

    for task in plan.get("research_prefetch", []):
        path = output_dir / research_pack_filename(task)
        task_id = str(task.get("id", ""))
        content = render_research_pack(task, plan_date, answers.get(task_id, {}))
        if path.exists() and not args.force:
            created.append({"id": task_id, "path": str(path), "status": "exists"})
            continue
        path.write_text(content, encoding="utf-8")
        created.append({"id": task_id, "path": str(path), "status": "created"})
    return {"output_dir": str(output_dir), "packs": created}


def load_research_answers(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return {str(item.get("id", "")): item for item in payload if item.get("id")}
    if "answers" in payload and isinstance(payload["answers"], list):
        return {str(item.get("id", "")): item for item in payload["answers"] if item.get("id")}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def render_research_pack(task: dict[str, Any], plan_date: str, answer: dict[str, Any] | None = None) -> str:
    answer = answer or {}
    title = str(task.get("title", "Research task"))
    task_id = str(task.get("id", ""))
    outcome = str(task.get("outcome", "Prepare enough context to avoid a cold start."))
    start = str(task.get("start", ""))
    short_answer = str(answer.get("short_answer") or "Pending pre-research. Replace this section with the result before the scheduled block.")
    sources = answer.get("evidence") or task.get("sources", []) or task.get("tags", []) or []
    risks = answer.get("risks") or ["Pending validation."]
    next_actions = answer.get("next_actions") or [
        "Open this pack at the scheduled handoff time.",
        "Continue from the short answer and evidence instead of starting cold.",
    ]
    source_lines = "\n".join(f"- {source}" for source in sources) or "- To be filled by the pre-research agent."
    risk_lines = "\n".join(f"- {risk}" for risk in risks)
    next_action_lines = "\n".join(f"- {action}" for action in next_actions)
    return "\n".join(
        [
            f"# Research Pack: {title}",
            "",
            f"- Task ID: `{task_id}`",
            f"- Plan date: {plan_date}",
            f"- Scheduled handoff: {start}",
            "",
            "## Question",
            outcome,
            "",
            "## Short Answer",
            short_answer,
            "",
            "## Evidence",
            source_lines,
            "",
            "## Risks / Unknowns",
            risk_lines,
            "",
            "## Next Actions",
            next_action_lines,
            "",
        ]
    )


def load_plan_lookup(path: Path) -> dict[str, dict[str, Any]]:
    plan = read_json(path)
    lookup: dict[str, dict[str, Any]] = {}
    for item in plan.get("scheduled", []):
        task_id = str(item.get("id", ""))
        if task_id:
            lookup[task_id] = item
    return lookup


def print_profile(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = resolve_state_dir(args.state_dir)
    return {"state_dir": str(state_dir), "profile": load_profile(state_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily todo planner state helper")
    parser.add_argument("--state-dir", help="override local state directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profile", help="print current execution profile")

    plan = subparsers.add_parser("plan", help="build a time-block plan")
    plan.add_argument("--tasks", required=True, help="task JSON input")
    plan.add_argument("--date", help="plan date, YYYY-MM-DD")
    plan.add_argument("--start", help="workday start, HH:MM")
    plan.add_argument("--end", help="workday end, HH:MM")
    plan.add_argument("--buffer-ratio", type=float, default=0.15)
    plan.add_argument("--output", help="plan JSON output path")
    plan.add_argument("--markdown", help="optional Markdown timeline output path")

    review = subparsers.add_parser("review", help="update profile from actual execution")
    review.add_argument("--actuals", required=True, help="actuals JSON input")
    review.add_argument("--plan", help="plan JSON path")
    review.add_argument("--date", help="review date, YYYY-MM-DD")
    review.add_argument("--output", help="review JSON output path")

    prefetch = subparsers.add_parser("prefetch", help="create Markdown research-pack files from a plan")
    prefetch.add_argument("--plan", required=True, help="plan JSON path")
    prefetch.add_argument("--answers", help="optional research answers JSON to fill packs")
    prefetch.add_argument("--output-dir", help="research-pack output directory")
    prefetch.add_argument("--force", action="store_true", help="overwrite existing packs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "profile":
        result = print_profile(args)
    elif args.command == "plan":
        result = build_plan(args)
    elif args.command == "review":
        result = review_actuals(args)
    elif args.command == "prefetch":
        result = create_prefetch_packs(args)
    else:
        parser.error("unknown command")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
