#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_TOP = ["SKILL.md", "README.md", "references/source-map.md", "evals/evals.json"]
RECOMMENDED = [
    "references/chapter-digests.md",
    "references/concepts.md",
    "references/procedures.md",
    "references/examples.md",
    "references/gotchas.md",
    "templates/task-brief.md",
    "templates/checklist.md",
    "templates/output-template.md",
]


def read_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md missing YAML frontmatter")
    end = text.find("\n---", 4)
    if end == -1:
        raise ValueError("SKILL.md frontmatter is not closed")
    fm: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"\'')
    return fm


def count_source_map_rows(path: Path) -> int:
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*-+", line) and "Extracted Knowledge" not in line:
            rows += 1
    return rows


def validate(skill_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not skill_dir.is_dir():
        return [f"not a directory: {skill_dir}"], warnings

    for rel in REQUIRED_TOP:
        if not (skill_dir / rel).is_file():
            errors.append(f"missing required file: {rel}")
    for rel in RECOMMENDED:
        if not (skill_dir / rel).is_file():
            warnings.append(f"missing recommended file: {rel}")
    if errors:
        return errors, warnings

    try:
        fm = read_frontmatter(skill_dir / "SKILL.md")
    except ValueError as exc:
        errors.append(str(exc))
        fm = {}
    if not fm.get("name"):
        errors.append("SKILL.md frontmatter missing name")
    if not fm.get("description") or len(fm.get("description", "")) < 50:
        warnings.append("SKILL.md description should be specific enough to trigger reliably")

    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if len(skill_text.split()) > 3500:
        warnings.append("SKILL.md is large; move details to references/ for progressive disclosure")
    for keyword in ["references/source-map.md", "evals", "validation"]:
        if keyword not in skill_text:
            warnings.append(f"SKILL.md does not mention {keyword}")

    source_rows = count_source_map_rows(skill_dir / "references/source-map.md")
    if source_rows < 3:
        warnings.append("references/source-map.md has fewer than 3 mapped knowledge items")

    try:
        data = json.loads((skill_dir / "evals/evals.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"evals/evals.json is invalid JSON: {exc}")
        data = {}
    evals = data.get("evals") if isinstance(data, dict) else None
    if not isinstance(evals, list) or len(evals) < 6:
        errors.append("evals/evals.json must contain at least 6 eval cases")
    else:
        required_ids = {"trigger", "non", "source"}
        joined_ids = " ".join(str(item.get("id", "")) for item in evals if isinstance(item, dict)).lower()
        for marker in required_ids:
            if marker not in joined_ids:
                warnings.append(f"eval ids should include a {marker!r}-related case")
        for idx, item in enumerate(evals, 1):
            if not isinstance(item, dict):
                errors.append(f"eval #{idx} is not an object")
                continue
            for field in ["id", "prompt", "expected", "checks"]:
                if field not in item:
                    errors.append(f"eval #{idx} missing field: {field}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated book-derived skill.")
    parser.add_argument("skill_dir")
    args = parser.parse_args()
    errors, warnings = validate(Path(args.skill_dir).expanduser().resolve())
    if warnings:
        print("Warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Generated book skill validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
