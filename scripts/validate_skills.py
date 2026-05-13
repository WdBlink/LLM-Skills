#!/usr/bin/env python3
from __future__ import annotations

import sys

from skill_index import ROOT, load_manifest, read_skill_frontmatter


REQUIRED_FIELDS = {
    "name",
    "display_name",
    "name_zh",
    "category",
    "version",
    "license",
    "path",
    "description",
    "tagline_en",
    "tagline_zh",
}


def validate() -> list[str]:
    errors: list[str] = []
    manifest = load_manifest()
    skills = manifest["skills"]
    seen: set[str] = set()

    if not isinstance(skills, list) or not skills:
        return ["skills.yaml must contain at least one skill"]

    for skill in skills:
        missing = sorted(REQUIRED_FIELDS - set(skill))
        name = str(skill.get("name", "<missing>"))
        if missing:
            errors.append(f"{name}: missing manifest fields: {', '.join(missing)}")
            continue
        if name in seen:
            errors.append(f"{name}: duplicate skill name")
        seen.add(name)

        path = ROOT / str(skill["path"])
        skill_md = path / "SKILL.md"
        if not path.is_dir():
            errors.append(f"{name}: missing directory {path.relative_to(ROOT)}")
            continue
        if not skill_md.is_file():
            errors.append(f"{name}: missing {skill_md.relative_to(ROOT)}")
            continue
        try:
            frontmatter = read_skill_frontmatter(skill_md)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        actual_name = frontmatter.get("name")
        if actual_name != name:
            errors.append(f"{name}: SKILL.md frontmatter name is {actual_name!r}")
        if not frontmatter.get("description"):
            errors.append(f"{name}: SKILL.md frontmatter missing description")

    mirrored_dirs = sorted(p.name for p in (ROOT / "skills").iterdir() if p.is_dir())
    indexed = sorted(str(skill["path"]).split("/", 1)[1] for skill in skills)
    for dirname in sorted(set(mirrored_dirs) - set(indexed)):
        errors.append(f"skills/{dirname}: directory is not listed in skills.yaml")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Skill index validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Skill index validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
