#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


def document_text(doc: Any) -> str:
    parts: list[str] = []
    parts.extend(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def table_shapes(doc: Any) -> list[list[int]]:
    return [[len(row.cells) for row in table.rows] for table in doc.tables]


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad:
        raise ValueError(f"corrupt DOCX zip member: {bad}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a filled strict-format DOCX template.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()

    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install python-docx with `python3 -m pip install python-docx`.") from exc

    validate_zip(args.output)
    template = Document(str(args.template))
    output = Document(str(args.output))

    errors: list[str] = []
    if len(template.tables) != len(output.tables):
        errors.append(f"table count changed: template={len(template.tables)} output={len(output.tables)}")
    if table_shapes(template) != table_shapes(output):
        errors.append("table row/cell shape changed")

    required_terms: list[str] = []
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        required_terms = [str(term) for term in plan.get("required_terms", [])]

    text = document_text(output)
    for term in required_terms:
        if term and term not in text:
            errors.append(f"required term missing from output: {term}")

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed.")
    print(f"Tables: {len(output.tables)}")
    print(f"Paragraphs: {len(output.paragraphs)}")
    if required_terms:
        print(f"Required terms present: {len(required_terms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
