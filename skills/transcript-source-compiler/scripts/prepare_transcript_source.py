#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


TEXT_EXTS = {".md", ".markdown", ".txt", ".srt", ".vtt"}
TIMESTAMP_RE = re.compile(
    r"^\d\d?:\d\d(?::\d\d)?[,.]\d{1,3}\s+-->\s+\d\d?:\d\d(?::\d\d)?[,.]\d{1,3}"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass(frozen=True)
class SourceUnit:
    source_file: Path
    source_label: str
    source_span: str
    timestamp: str
    text: str


@dataclass(frozen=True)
class EvidenceFragment:
    id: str
    source_file: str
    source_label: str
    source_span: str
    timestamp: str
    text: str
    type: str = "claim"
    status: str = "unreviewed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare transcript source material for evidence-backed source compilation."
    )
    parser.add_argument("inputs", nargs="+", help="Transcript file or directory paths.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated run files.")
    parser.add_argument("--title", default="演讲转写来源整理")
    return parser.parse_args()


def discover(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"input does not exist: {path}")
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS))
        elif path.suffix.lower() in TEXT_EXTS:
            files.append(path)
        else:
            raise SystemExit(f"unsupported input file: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for file in files:
        if file not in seen:
            seen.add(file)
            unique.append(file)
    if not unique:
        raise SystemExit("no transcript text files found")
    return unique


def parse_units(path: Path) -> list[SourceUnit]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    suffix = path.suffix.lower()
    if suffix in {".srt", ".vtt"}:
        return parse_caption_units(path, text)
    return parse_plain_units(path, text)


def parse_caption_units(path: Path, text: str) -> list[SourceUnit]:
    units: list[SourceUnit] = []
    current_timestamp = ""
    buffer: list[str] = []
    span_index = 1
    in_note_block = False

    def flush() -> None:
        nonlocal span_index, buffer, current_timestamp
        joined = " ".join(part.strip() for part in buffer if part.strip()).strip()
        if joined:
            units.append(
                SourceUnit(
                    source_file=path,
                    source_label=path.name,
                    source_span=f"caption {span_index}",
                    timestamp=current_timestamp,
                    text=joined,
                )
            )
            span_index += 1
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if in_note_block:
            if not line:
                in_note_block = False
            continue
        if not line:
            flush()
            continue
        if line.upper() == "WEBVTT":
            continue
        if line == "NOTE" or line.startswith("NOTE "):
            flush()
            in_note_block = True
            continue
        if line.isdigit():
            continue
        if TIMESTAMP_RE.match(line):
            flush()
            current_timestamp = line
            continue
        buffer.append(line)
    flush()
    return units


def parse_plain_units(path: Path, text: str) -> list[SourceUnit]:
    units: list[SourceUnit] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|\n", text) if part.strip()]
    for index, paragraph in enumerate(paragraphs, start=1):
        units.append(
            SourceUnit(
                source_file=path,
                source_label=path.name,
                source_span=f"paragraph {index}",
                timestamp="",
                text=paragraph,
            )
        )
    return units


def split_unit(unit: SourceUnit) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(unit.text) if part.strip()]
    if not sentences:
        return []
    fragments: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= 3:
            fragments.append("".join(current).strip())
            current = []
    if current:
        fragments.append("".join(current).strip())
    return fragments


def build_fragments(files: list[Path]) -> list[EvidenceFragment]:
    fragments: list[EvidenceFragment] = []
    next_id = 1
    for file in files:
        for unit in parse_units(file):
            for text in split_unit(unit):
                fragments.append(
                    EvidenceFragment(
                        id=f"E{next_id:04d}",
                        source_file=str(unit.source_file.resolve()),
                        source_label=unit.source_label,
                        source_span=unit.source_span,
                        timestamp=unit.timestamp,
                        text=text,
                    )
                )
                next_id += 1
    return fragments


def write_source_pack(files: list[Path], output_dir: Path, title: str) -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# {title}",
        "",
        "## Pack Metadata",
        "",
        f"- Generated: `{now}`",
        f"- Source count: `{len(files)}`",
        "",
        "## Source Files",
        "",
    ]
    for index, file in enumerate(files, start=1):
        lines.extend([f"- [{index}] `{file}`", ""])
    for index, file in enumerate(files, start=1):
        lines.extend([f"## Source {index}: {file.name}", ""])
        lines.append(file.read_text(encoding="utf-8", errors="replace").strip() or "[empty source]")
        lines.append("")
    (output_dir / "source-pack.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_fragments(fragments: list[EvidenceFragment], output_dir: Path) -> None:
    content = "\n".join(json.dumps(asdict(fragment), ensure_ascii=False) for fragment in fragments)
    (output_dir / "evidence-fragments.jsonl").write_text(content + ("\n" if content else ""), encoding="utf-8")


def write_source_record_scaffold(title: str, fragments: list[EvidenceFragment], output_dir: Path) -> None:
    lines = [
        f"# {title}",
        "",
        "## 来源信息",
        "",
        "## 内容概览",
        "",
        "## 主题脉络",
        "",
        "## 主要观点",
        "",
        "## 关键概念",
        "",
        "## 方法框架",
        "",
        "## 案例与例子",
        "",
        "## 可沉淀到 Wiki 的知识点",
        "",
        "## 矛盾与不确定项",
        "",
        "## 待核对",
        "",
        "## 证据覆盖表",
        "",
        "| Evidence | Destination | Status | Linked Items | Note |",
        "|---|---|---|---|---|",
    ]
    for fragment in fragments:
        lines.append(f"| {fragment.id} | 未归类 | unclear |  |  |")
    (output_dir / "source-record.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_article_scaffolds(title: str, output_dir: Path) -> None:
    (output_dir / "article-main.md").write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "## 导语",
                "",
                "## 正文",
                "",
                "## 补充信息",
                "",
                "## 待核对",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "article-map.md").write_text(
        "\n".join(
            [
                "# Article Map",
                "",
                "| Paragraph | Article | Items | Evidence | Coverage Note |",
                "|---|---|---|---|---|",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "uncertain-items.md").write_text(
        "\n".join(["# Uncertain Items", "", "| Item | Evidence | Reason |", "|---|---|---|", ""]),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    files = discover(args.inputs)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fragments = build_fragments(files)
    write_source_pack(files, output_dir, args.title)
    write_fragments(fragments, output_dir)
    write_source_record_scaffold(args.title, fragments, output_dir)
    write_article_scaffolds(args.title, output_dir)
    for name in [
        "source-pack.md",
        "evidence-fragments.jsonl",
        "source-record.md",
        "article-main.md",
        "article-map.md",
        "uncertain-items.md",
    ]:
        print(output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
