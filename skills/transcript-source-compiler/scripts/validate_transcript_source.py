#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


VALID_COVERAGE_STATUS = {"extracted", "duplicate", "filler", "unclear"}
EVIDENCE_RE = re.compile(r"\bE\d{4,}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate transcript source compiler outputs.")
    parser.add_argument("run_dir", help="Run directory containing evidence-fragments.jsonl and Markdown outputs.")
    return parser.parse_args()


def read_fragment_ids(path: Path) -> set[str]:
    if not path.exists():
        raise ValueError(f"missing required file: {path.name}")
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path.name}:{line_number}: {exc}") from exc
        evidence_id = row.get("id")
        if not isinstance(evidence_id, str) or not EVIDENCE_RE.fullmatch(evidence_id):
            raise ValueError(f"invalid evidence id at {path.name}:{line_number}: {evidence_id!r}")
        ids.add(evidence_id)
    if not ids:
        raise ValueError("no evidence fragments found")
    return ids


def parse_markdown_table_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        raise ValueError(f"missing required file: {path.name}")
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if set(cells[0]) <= {"-"} or cells[0].lower() in {"evidence", "paragraph"}:
            continue
        rows.append(cells)
    return rows


def parse_coverage(source_record: Path) -> dict[str, str]:
    coverage: dict[str, str] = {}
    for cells in parse_markdown_table_rows(source_record):
        if len(cells) < 3:
            continue
        evidence_id = cells[0]
        status = cells[2]
        if EVIDENCE_RE.fullmatch(evidence_id):
            coverage[evidence_id] = status
    return coverage


def parse_article_map_evidence(article_map: Path) -> set[str]:
    used: set[str] = set()
    for cells in parse_markdown_table_rows(article_map):
        if len(cells) < 4:
            continue
        for evidence_id in EVIDENCE_RE.findall(cells[3]):
            used.add(evidence_id)
    return used


def validate(run_dir: Path) -> list[str]:
    fragment_ids = read_fragment_ids(run_dir / "evidence-fragments.jsonl")
    coverage = parse_coverage(run_dir / "source-record.md")
    article_evidence = parse_article_map_evidence(run_dir / "article-map.md")
    errors: list[str] = []

    for evidence_id in sorted(fragment_ids - set(coverage)):
        errors.append(f"missing coverage: {evidence_id}")
    for evidence_id in sorted(set(coverage) - fragment_ids):
        errors.append(f"coverage references unknown evidence: {evidence_id}")
    for evidence_id, status in sorted(coverage.items()):
        if status not in VALID_COVERAGE_STATUS:
            errors.append(f"invalid coverage status for {evidence_id}: {status}")
    for evidence_id in sorted(article_evidence - fragment_ids):
        errors.append(f"unknown article-map evidence: {evidence_id}")

    return errors


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    try:
        errors = validate(run_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("PASS transcript-source-compiler validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
