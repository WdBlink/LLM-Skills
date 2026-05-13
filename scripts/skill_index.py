#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "skills.yaml"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_manifest(path: Path = MANIFEST) -> dict[str, object]:
    version: int | None = None
    skills: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line == stripped and stripped.startswith("version:"):
            version = int(stripped.split(":", 1)[1].strip())
            continue
        if stripped == "skills:":
            continue
        if stripped.startswith("- name:"):
            current = {"name": _unquote(stripped.split(":", 1)[1])}
            skills.append(current)
            continue
        if current is not None and raw_line.startswith("    ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = _unquote(value)
            if value.lower() == "true":
                current[key] = True
            elif value.lower() == "false":
                current[key] = False
            else:
                current[key] = value

    if version is None:
        raise ValueError("skills.yaml is missing version")
    return {"version": version, "skills": skills}


def read_skill_frontmatter(skill_md: Path) -> dict[str, str]:
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{skill_md}: missing frontmatter")
    frontmatter = text.split("---", 2)[1]
    result: dict[str, str] = {}
    active_key: str | None = None
    for raw_line in frontmatter.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.startswith(" ") and active_key:
            result[active_key] = (result[active_key] + "\n" + raw_line.strip()).strip()
            continue
        if ":" in raw_line:
            key, value = raw_line.split(":", 1)
            result[key.strip()] = _unquote(value.strip().rstrip("|").strip())
            active_key = key.strip()
    return result
