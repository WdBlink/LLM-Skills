#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import html.parser
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

WORDS_PER_TOKEN = 0.75
DEFAULT_CHUNK_WORDS = 900
DEFAULT_OVERLAP_WORDS = 120


def slugify(value: str, fallback: str = "book") -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value or fallback


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / WORDS_PER_TOKEN))


def run_command(args: list[str], timeout: int = 180) -> str | None:
    if not shutil.which(args[0]):
        return None
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    return None


def extract_with_docling(path: Path) -> str | None:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception:
        return None
    try:
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    except Exception:
        return None


def extract_pdf(path: Path, mode: str) -> tuple[str, str]:
    methods: list[tuple[str, callable]] = []
    if mode in {"technical", "auto"}:
        methods.append(("docling", lambda: extract_with_docling(path)))
    if mode in {"text", "auto", "technical"}:
        methods.extend([
            ("pdftotext", lambda: run_command(["pdftotext", "-layout", str(path), "-"], timeout=240)),
            ("PyPDF2", lambda: extract_pdf_pypdf2(path)),
            ("pdfminer.six", lambda: extract_pdf_pdfminer(path)),
        ])
    for name, fn in methods:
        print(f"Trying {name}...", end=" ", flush=True)
        text = fn()
        if text and text.strip():
            print("OK")
            return normalize_text(text), name
        print("not available/failed")
    raise SystemExit(
        "ERROR: Could not extract PDF text. Install one of: poppler-utils (pdftotext), PyPDF2, pdfminer.six, or docling."
    )


def extract_pdf_pypdf2(path: Path) -> str | None:
    try:
        import PyPDF2
        parts = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n\n".join(parts)
    except Exception:
        return None


def extract_pdf_pdfminer(path: Path) -> str | None:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except Exception:
        return None


class HTMLTextExtractor(html.parser.HTMLParser):
    SKIP = {"script", "style", "head"}
    BLOCK = {"p", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self.skip_depth += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return html.unescape("".join(self.parts))


def extract_epub(path: Path) -> tuple[str, str]:
    text = extract_epub_ebooklib(path)
    if text:
        return normalize_text(text), "ebooklib"
    text = extract_epub_zipfile(path)
    if text:
        return normalize_text(text), "zipfile"
    raise SystemExit("ERROR: Could not extract EPUB. Install ebooklib beautifulsoup4 or provide an unencrypted EPUB.")


def extract_epub_ebooklib(path: Path) -> str | None:
    try:
        import ebooklib
        from bs4 import BeautifulSoup
        from ebooklib import epub
        book = epub.read_epub(str(path))
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            parts.append(soup.get_text(separator="\n"))
        return "\n\n".join(parts)
    except Exception:
        return None


def extract_epub_zipfile(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            html_names = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
            parts = []
            for name in html_names:
                raw = zf.read(name).decode("utf-8", errors="replace")
                parser = HTMLTextExtractor()
                parser.feed(raw)
                parts.append(parser.text())
            return "\n\n".join(parts)
    except Exception:
        return None


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def extract_plain(path: Path) -> tuple[str, str]:
    return normalize_text(path.read_text(encoding="utf-8", errors="replace")), "plain-text"


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"ERROR: Input path not found: {path}")
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".epub", ".md", ".txt"})
    if not files:
        raise SystemExit(f"ERROR: No supported book files found under {path}")
    return files


def extract_file(path: Path, mode: str) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf" or path.read_bytes()[:4] == b"%PDF":
        return extract_pdf(path, mode)
    if suffix == ".epub" or path.read_bytes()[:2] == b"PK":
        return extract_epub(path)
    if suffix in {".md", ".txt"}:
        return extract_plain(path)
    raise SystemExit(f"ERROR: Unsupported file type: {path}")


def detect_headings(text: str, limit: int = 80) -> list[dict[str, object]]:
    patterns = [
        re.compile(r"^\s{0,3}#{1,3}\s+(.+)$"),
        re.compile(r"^\s*(chapter|part|section)\s+([0-9ivxlcdm]+)\b[:.\-\s]*(.*)$", re.I),
        re.compile(r"^\s*([0-9]{1,2})\.\s+([A-Z][^\n]{3,100})$"),
    ]
    headings = []
    offset = 0
    for line in text.splitlines(True):
        stripped = line.strip()
        for pat in patterns:
            m = pat.match(stripped)
            if m:
                headings.append({"text": stripped[:160], "char_offset": offset})
                break
        offset += len(line)
        if len(headings) >= limit:
            break
    return headings


def chunk_words(text: str, chunk_words: int, overlap_words: int) -> list[dict[str, object]]:
    words = re.findall(r"\S+", text)
    chunks = []
    if not words:
        return chunks
    start = 0
    idx = 1
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": f"E{idx:04d}",
            "word_start": start,
            "word_end": end,
            "text": chunk_text,
            "sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16],
        })
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
        idx += 1
    return chunks


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scaffold(output_dir: Path, slug: str, metadata: dict[str, object], headings: list[dict[str, object]]) -> None:
    (output_dir / "source").mkdir(parents=True, exist_ok=True)
    (output_dir / "work").mkdir(parents=True, exist_ok=True)
    target = output_dir / "output" / slug
    for sub in ["references", "templates", "evals"]:
        (target / sub).mkdir(parents=True, exist_ok=True)

    title = metadata.get("title") or metadata.get("filename") or slug
    heading_lines = "\n".join(f"- {h['text']} @ char {h['char_offset']}" for h in headings[:50]) or "- No headings detected"

    (output_dir / "work" / "chapter-digests.md").write_text(
        f"# Chapter Digests — {title}\n\n## Detected Headings\n\n{heading_lines}\n\n## Digests\n\n"
        "For each chapter/section: summarize core idea, frameworks, examples, anti-patterns, and evidence ids.\n",
        encoding="utf-8",
    )
    (output_dir / "work" / "concepts.md").write_text(
        f"# Concepts — {title}\n\n| Concept | Definition | Related Concepts | Evidence |\n|---|---|---|---|\n",
        encoding="utf-8",
    )
    (output_dir / "work" / "procedures.md").write_text(
        f"# Procedures — {title}\n\n## Workflows\n\n## Decision Rules\n\n## Checklists\n\n## Anti-patterns\n\n## Templates To Generate\n",
        encoding="utf-8",
    )
    (output_dir / "work" / "source-map.md").write_text(
        f"# Source Map — {title}\n\n| Item ID | Extracted Knowledge | Type | Chapter/Page/Location | Evidence IDs | Confidence |\n|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )

    (target / "references" / "source-map.md").write_text("# Source Map\n\nCopy the final evidence-backed source map here.\n", encoding="utf-8")
    (target / "references" / "chapter-digests.md").write_text("# Chapter Digests\n\n", encoding="utf-8")
    (target / "references" / "concepts.md").write_text("# Concepts\n\n", encoding="utf-8")
    (target / "references" / "procedures.md").write_text("# Procedures\n\n", encoding="utf-8")
    (target / "references" / "examples.md").write_text("# Examples\n\n", encoding="utf-8")
    (target / "references" / "gotchas.md").write_text("# Gotchas\n\n", encoding="utf-8")
    (target / "references" / "derived-skills.md").write_text("# Derived Skill Recommendations\n\n", encoding="utf-8")
    (target / "templates" / "task-brief.md").write_text("# Task Brief\n\n## Goal\n\n## Context\n\n## Constraints\n\n## Evidence to consult\n", encoding="utf-8")
    (target / "templates" / "checklist.md").write_text("# Checklist\n\n- [ ] Define the task\n- [ ] Select the relevant book framework\n- [ ] Check gotchas\n- [ ] Validate against source-map\n", encoding="utf-8")
    (target / "templates" / "output-template.md").write_text("# Output Template\n\n## Recommendation\n\n## Reasoning\n\n## Book-derived framework used\n\n## Caveats\n", encoding="utf-8")
    (target / "evals" / "evals.json").write_text(json.dumps({
        "skill_name": slug,
        "evals": [
            {"id": "trigger_direct", "prompt": "Apply this book's framework to a representative task.", "expected": "Uses the generated workflow and cites relevant reference files.", "checks": ["loads_skill", "uses_framework", "mentions_source"]},
            {"id": "trigger_paraphrase", "prompt": "Help me solve a problem that matches the book method without naming the book.", "expected": "Recognizes the method and applies the checklist.", "checks": ["loads_skill", "applies_checklist"]},
            {"id": "source_grounding", "prompt": "Where does this rule come from?", "expected": "Reads references/source-map.md and cites evidence ids.", "checks": ["reads_source_map", "cites_evidence"]},
            {"id": "template_use", "prompt": "Create a plan using the book's process.", "expected": "Uses a template from templates/.", "checks": ["uses_template"]},
            {"id": "non_trigger", "prompt": "Write a casual birthday message.", "expected": "Does not use the book skill.", "checks": ["does_not_trigger"]},
            {"id": "unsupported_claim", "prompt": "What does the book say about a topic not covered?", "expected": "Qualifies uncertainty and does not invent unsupported content.", "checks": ["no_hallucination", "states_limits"]},
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "SKILL.md").write_text(f"---\nname: {slug}\ndescription: TODO: generated skill from {title}.\n---\n\n# {title}\n\nTODO: replace this scaffold with the final compact skill.\n", encoding="utf-8")
    (target / "README.md").write_text(f"# {title}\n\nGenerated by book-to-skill. Complete and validate before installing.\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract books into a traceable book-to-skill work directory.")
    parser.add_argument("input", help="PDF, EPUB, Markdown, text file, or directory")
    parser.add_argument("--output-dir", required=True, help="Run output directory")
    parser.add_argument("--skill-slug", help="Target skill slug. Defaults to input filename slug.")
    parser.add_argument("--mode", choices=["auto", "technical", "text"], default="auto", help="Extraction mode for PDFs")
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    slug = slugify(args.skill_slug or input_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path)
    combined_parts = []
    methods = []
    for file_path in files:
        print(f"Extracting {file_path}")
        text, method = extract_file(file_path, args.mode)
        methods.append({"file": str(file_path), "method": method, "chars": len(text), "words": len(text.split())})
        combined_parts.append(f"\n\n<!-- SOURCE_FILE: {file_path.name} -->\n\n{text}")
    full_text = normalize_text("\n\n".join(combined_parts))
    headings = detect_headings(full_text)
    chunks = chunk_words(full_text, args.chunk_words, args.overlap_words)

    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "full_text.md").write_text(full_text, encoding="utf-8")
    write_jsonl(source_dir / "chunks.jsonl", chunks)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "filename": input_path.name,
        "skill_slug": slug,
        "files": methods,
        "chars": len(full_text),
        "words": len(full_text.split()),
        "estimated_tokens": estimate_tokens(full_text),
        "chunk_count": len(chunks),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "detected_headings": headings[:80],
    }
    (source_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_scaffold(output_dir, slug, metadata, headings)

    print("\nBook extraction prepared.")
    print(f"Run dir      : {output_dir}")
    print(f"Full text    : {source_dir / 'full_text.md'}")
    print(f"Chunks       : {source_dir / 'chunks.jsonl'} ({len(chunks)} chunks)")
    print(f"Metadata     : {source_dir / 'metadata.json'}")
    print(f"Skill scaffold: {output_dir / 'output' / slug}")
    print("\nNext: use the book-to-skill instructions to fill work/*.md, finalize output/<skill-slug>/, then validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
