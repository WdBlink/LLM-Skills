#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def clear_paragraph(paragraph: Any) -> None:
    for run in paragraph.runs:
        run.text = ""


def write_text_to_paragraph(paragraph: Any, text: str) -> None:
    runs = paragraph.runs
    if runs:
        first = runs[0]
        clear_paragraph(paragraph)
    else:
        first = paragraph.add_run()
    lines = str(text).splitlines() or [""]
    first.text = lines[0]
    for line in lines[1:]:
        first.add_break()
        first.add_text(line)


def write_text_to_cell(cell: Any, text: str) -> None:
    if not cell.paragraphs:
        paragraph = cell.add_paragraph()
    else:
        paragraph = cell.paragraphs[0]
    write_text_to_paragraph(paragraph, text)
    for extra in cell.paragraphs[1:]:
        clear_paragraph(extra)


def all_paragraphs(doc: Any) -> list[Any]:
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    return paragraphs


def replace_match(doc: Any, match: str, text: str) -> int:
    count = 0
    for paragraph in all_paragraphs(doc):
        for run in paragraph.runs:
            if match in run.text:
                run.text = run.text.replace(match, text)
                count += 1
    return count


def apply_replacement(doc: Any, replacement: dict[str, Any]) -> str:
    target = replacement.get("target")
    selector = replacement.get("selector")
    if selector and not target:
        target = selector.get("type")
        replacement = {**selector, **replacement}

    text = str(replacement.get("text", ""))

    if target == "paragraph":
        index = int(replacement["index"])
        write_text_to_paragraph(doc.paragraphs[index], text)
        return f"paragraph:{index}"

    if target == "paragraph_block":
        start = int(replacement["start"])
        if "end" in replacement:
            end = int(replacement["end"])
        else:
            end = start + int(replacement.get("count", 1)) - 1
        if end < start:
            raise ValueError(f"invalid paragraph_block range: {start}..{end}")
        write_text_to_paragraph(doc.paragraphs[start], text)
        for index in range(start + 1, end + 1):
            clear_paragraph(doc.paragraphs[index])
        return f"paragraph_block:{start}:{end}"

    if target == "table_cell":
        table = int(replacement["table"])
        row = int(replacement["row"])
        col = int(replacement["col"])
        write_text_to_cell(doc.tables[table].rows[row].cells[col], text)
        return f"table_cell:{table}:{row}:{col}"

    if target == "text_match":
        match = str(replacement["match"])
        count = replace_match(doc, match, text)
        if count == 0:
            raise ValueError(f"text_match not found: {match}")
        return f"text_match:{match}:{count}"

    raise ValueError(f"unsupported replacement target: {target!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a JSON report plan to a DOCX template.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install python-docx with `python3 -m pip install python-docx`.") from exc

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    replacements = plan.get("replacements", [])
    if not isinstance(replacements, list):
        raise SystemExit("report plan must contain a list named 'replacements'")

    doc = Document(str(args.template))
    applied = [apply_replacement(doc, item) for item in replacements]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(args.output))

    print(f"Saved {args.output}")
    print(f"Applied {len(applied)} replacements")
    for item in applied:
        print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
