#!/usr/bin/env python3
"""
Convert a PDF book into a structured Markdown file using a LOCAL GLM-OCR model
loaded through Hugging Face transformers (no Ollama / vLLM required).

The model is loaded once and stays in (MPS | CUDA | CPU) memory for the whole
run. Per-page: render PDF page → PIL Image → chat template → generate → md.

Usage:
    python3 convert_pdf.py /path/to/book.pdf --output /path/to/book.md
    python3 convert_pdf.py /path/to/book.pdf --pages 3 --keep-images
    python3 convert_pdf.py /path/to/book.pdf --max-edge 1024 --device mps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from PIL import Image

DEFAULT_MODEL = os.environ.get(
    "GLM_OCR_MODEL_PATH",
    "/Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR",
)
DEFAULT_DPI = 150
DEFAULT_MAX_EDGE = 768  # MPS+SDPA deadlocks at 1024 on some pages; 768 is the safe ceiling
OCR_PROMPT_VARIANT: Literal["doc", "text"] = "doc"
OCR_PROMPTS = {
    "doc": "Text Recognition:",
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


# ---------------------------------------------------------------------------
@dataclass
class PageResult:
    index: int
    one_based: int
    md: str
    elapsed: float
    src: str | None


# ---------------------------------------------------------------------------
def _slugify(s: str) -> str:
    s = re.sub(r"[\s/\\]+", "-", s.strip())
    s = re.sub(r"[^\w一-鿿\-.]+", "", s, flags=re.UNICODE)
    return s or "book"


def _resolve_device(pref: str) -> str:
    import torch
    pref = pref.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if pref == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        print("[warn] MPS not available, falling back to CPU", file=sys.stderr)
        return "cpu"
    return pref


def _downsize(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest > max_edge:
        scale = max_edge / longest
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
def render_pdf(pdf_path: Path, out_dir: Path, dpi: int = DEFAULT_DPI,
               pages: int | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    n = min(len(doc), pages) if pages else len(doc)
    out: list[Path] = []
    for i in range(n):
        pix = doc[i].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        target = out_dir / f"p{i+1:03d}.png"
        img.save(target, "PNG", optimize=True)
        out.append(target)
    doc.close()
    return out


def prepare_pages(pdf_path: Path, work_dir: Path, dpi: int, pages: int | None,
                  resume: bool, ocr_settings: dict) -> list[Path]:
    """Reuse only complete images and OCR results from the same conversion."""
    if dpi <= 0 or (pages is not None and pages <= 0):
        raise ValueError("dpi and pages must be positive")
    digest = hashlib.sha256()
    with pdf_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with fitz.open(pdf_path) as doc:
        count = len(doc) if pages is None else min(len(doc), pages)
        total = len(doc)
    expected = {"pdf_sha256": digest.hexdigest(), "dpi": dpi,
                "pdf_pages": total, "pages": count, "ocr": ocr_settings}
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = work_dir / "cache-manifest.json"
    img_dir = work_dir / "pages"
    paths = [img_dir / f"p{i+1:03d}.png" for i in range(count)]
    try:
        cached = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = None
    if resume and cached == expected and set(img_dir.glob("p*.png")) == set(paths):
        try:
            for path in paths:
                with Image.open(path) as img:
                    img.verify()
            return paths
        except (OSError, SyntaxError):
            pass
    # Invalidate before rendering, so interrupted renders cannot be resumed as complete.
    manifest.unlink(missing_ok=True)
    (work_dir / "pages.jsonl").write_text("", encoding="utf-8")
    for path in img_dir.glob("p*.png"):
        path.unlink()
    paths = render_pdf(pdf_path, img_dir, dpi=dpi, pages=pages)
    manifest.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
def find_chapters(page_results: list[PageResult]) -> list[tuple[str, int]]:
    """Pattern-matches the per-page Markdown for chapter headings.

    GLM-OCR often emits chapters without the leading `#` markdown prefix
    (e.g. it just writes ``第一章 构建故事``). We accept both the
    with-prefix and without-prefix forms, and ignore TOC pages (pages
    that list several chapter headings — those are auto-detected by
    counting the matches and skipping pages with >= 3 chapter lines).
    """
    patterns = [
        # Markdown-style:  # 第一章 标题  /  ## 第一章 标题
        re.compile(r"^#{1,3}\s+第\s*([一二三四五六七八九十百千零0-9]+)\s*章[　\s　]*(.*)$"),
        re.compile(r"^#{1,3}\s+第\s*([0-9]+)\s*章[　\s　]*(.*)$"),
        re.compile(r"^#{1,3}\s+Chapter\s+([0-9IVXLCDM]+)[　:\s]+(.*)$", re.IGNORECASE),
        re.compile(r"^#{1,3}\s+([一二三四五六七八九十]+)[、\.](\S+)$"),
        # Plain-text: 第一章 标题  (the more common case in practice)
        re.compile(r"^第\s*([一二三四五六七八九十百千零0-9]+)\s*章[　\s　]+(\S.{0,80})$"),
        re.compile(r"^第\s*([0-9]+)\s*章[　\s　]+(\S.{0,80})$"),
        re.compile(r"^Chapter\s+([0-9IVXLCDM]+)[　:\s]+(\S.{0,80})$", re.IGNORECASE),
    ]

    def is_chapter_line(line: str) -> bool:
        line = line.rstrip()
        for pat in patterns:
            if pat.match(line):
                return True
        return False

    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    # seen_numbers: keep the first chapter of each 第N章 to dedupe OCR typos
    seen_numbers: set[str] = set()
    for pr in page_results:
        lines = pr.md.splitlines()
        chapter_hits = [ln for ln in lines if is_chapter_line(ln)]
        # Skip pages that look like a TOC: any of these signals wins.
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        first_line = non_empty[0] if non_empty else ""
        is_toc = (
            len(chapter_hits) >= 3
            or first_line in {"目录", "目 录", "CONTENTS", "Contents", "Table of Contents"}
            or "CONTENTS" in first_line.upper()
        )
        if is_toc:
            continue
        for line in chapter_hits:
            title = re.sub(r"\s+", " ", line.lstrip("# ").strip())
            title = re.sub(r"\s+\d{1,4}\s*$", "", title)
            if not title or title in seen or len(title) < 4:
                continue
            # Extract chapter number (第N章 / Chapter N)
            num_m = re.match(r"^(第\s*[一二三四五六七八九十百千零0-9]+\s*章|Chapter\s+[0-9IVXLCDM]+)", title, re.IGNORECASE)
            num_key = num_m.group(1).lower().replace(" ", "") if num_m else title
            if num_key in seen_numbers:
                continue
            seen.add(title)
            seen_numbers.add(num_key)
            found.append((title, pr.one_based))
    return found


# ---------------------------------------------------------------------------
def _slug_anchor(title: str) -> str:
    s = re.sub(r"\s+", "-", title.strip())
    s = re.sub(r"[^\w一-鿿\-]+", "", s)
    return s or "section"


# ---------------------------------------------------------------------------
def convert(
    pdf_path: Path,
    output: Path,
    model_path: str,
    device: str,
    dtype: str,
    dpi: int,
    max_edge: int,
    pages: int | None,
    keep_images: bool,
    work_dir: Path,
    prompt_variant: str,
    resume: bool,
) -> dict:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    started = time.time()
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    work_dir.mkdir(parents=True, exist_ok=True)
    img_dir = work_dir / "pages"
    cache_path = work_dir / "pages.jsonl"

    # ---- 1. Render PDF (or reuse cached PNGs)
    print(f"[1/5] Rendering PDF (dpi={dpi}) ...", flush=True)
    page_paths = prepare_pages(pdf_path, work_dir, dpi, pages, resume, {
        "model": model_path, "device": device, "dtype": dtype,
        "max_edge": max_edge, "prompt": OCR_PROMPTS[prompt_variant],
        "keep_images": keep_images,
    })
    print(f"      prepared {len(page_paths)} pages", flush=True)

    # ---- 2. Load model
    print(f"[2/5] Loading model from {model_path} (device={device}, dtype={dtype}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch_dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    print(f"      model ready on {device}", flush=True)

    # ---- 3. OCR
    prompt = OCR_PROMPTS[prompt_variant]
    print(f"[3/5] OCR (prompt={prompt_variant!r}, max_edge={max_edge}) ...", flush=True)

    # Resume support
    done: dict[int, PageResult] = {}
    if resume and cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (type(rec["index"]) is not int or not 0 <= rec["index"] < len(page_paths)
                        or rec["one_based"] != rec["index"] + 1):
                    raise ValueError("Invalid cached OCR page index")
                done[rec["index"]] = PageResult(
                    index=rec["index"], one_based=rec["one_based"],
                    md=rec["md"], elapsed=rec["elapsed"],
                    src=rec.get("src"),
                )
        print(f"      resuming with {len(done)} cached pages", flush=True)

    pending_paths = [(i + 1, path) for i, path in enumerate(page_paths) if i not in done]
    if pending_paths:
        cache_path.open("a").close()  # ensure exists
        with cache_path.open("a") as cf:
            for idx, p in pending_paths:
                img = Image.open(p).convert("RGB")
                img = _downsize(img, max_edge)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                t0 = time.time()
                # Synchronous generation. If MPS+SDPA deadlocks (some pages
                # above 768px), the entire interpreter will hang — so we run
                # each page in a child process that we can SIGKILL.
                import multiprocessing as mp

                def _gen(q, _model_state):
                    try:
                        # model is unpicklable across procs in this venv, so
                        # re-create it here.
                        from transformers import AutoProcessor, AutoModelForImageTextToText
                        import torch as _torch
                        _proc = _model_state["processor"]
                        _mod = _model_state["model"]
                        _dev = _model_state["device"]
                        with _torch.no_grad():
                            g = _mod.generate(
                                **inputs, max_new_tokens=2048,
                                do_sample=False, num_beams=1,
                            )
                        text = _proc.decode(
                            g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                        ).strip()
                        q.put(("ok", text))
                    except Exception as e:  # pragma: no cover
                        q.put(("err", repr(e)))

                # Actually run synchronously - the simplest path that works
                # on MPS at max-edge=768. Add a soft timeout via SIGALRM-style
                # cooperative check: we re-check elapsed time after a forced
                # sync and bail.
                try:
                    with torch.no_grad():
                        gen = model.generate(
                            **inputs, max_new_tokens=2048,
                            do_sample=False, num_beams=1,
                        )
                    elapsed = time.time() - t0
                    md = processor.decode(
                        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    ).strip()
                except Exception as e:
                    elapsed = time.time() - t0
                    md = f"<!-- OCR FAILED: {e} -->"
                    print(f"      page {idx} FAILED: {e}", file=sys.stderr)

                # Force a small GC pass to keep MPS allocator from starving
                import gc
                if device.startswith("mps") and idx % 5 == 0:
                    gc.collect()
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

                pr = PageResult(
                    index=idx - 1, one_based=idx, md=md, elapsed=elapsed,
                    src=str(p.relative_to(work_dir)) if keep_images else None,
                )
                done[idx - 1] = pr
                cf.write(json.dumps({
                    "index": pr.index, "one_based": pr.one_based,
                    "md": pr.md, "elapsed": pr.elapsed, "src": pr.src,
                }, ensure_ascii=False) + "\n")
                cf.flush()
                print(
                    f"      page {idx:>4d}/{len(page_paths)}  "
                    f"{elapsed:5.1f}s  {len(md)} chars", flush=True,
                )

    # ---- 4. Synthesize TOC
    print("[4/5] Synthesizing TOC ...", flush=True)
    sorted_results = [done[i] for i in sorted(done)]
    chapters = find_chapters(sorted_results)
    print(f"      detected {len(chapters)} chapter headings", flush=True)

    # ---- 5. Assemble
    print("[5/5] Assembling Markdown ...", flush=True)
    book_title = pdf_path.stem
    parts: list[str] = [f"# {book_title}", ""]
    parts.append(
        f"> OCR'd with `{Path(model_path).name}` ({device}, {dtype}) — "
        f"{len(sorted_results)} pages, {time.time() - started:.1f}s total."
    )
    parts.append("")
    if chapters:
        parts.append("## 目录 (auto-detected)")
        parts.append("")
        for title, pg in chapters:
            anchor = _slug_anchor(title)
            parts.append(f"- [{title}](#{anchor}) ……… p.{pg}")
        parts.append("")
    parts.append("---")
    parts.append("")

    for pr in sorted_results:
        marker = f"<!-- p.{pr.one_based}"
        if pr.src:
            marker += f" / src={pr.src}"
        marker += " -->"
        parts.append(marker)
        parts.append(pr.md)
        parts.append("")

    output.write_text("\n".join(parts), encoding="utf-8")
    print(f"      wrote {output}  ({output.stat().st_size/1024:.1f} KB)", flush=True)

    if not keep_images:
        for p in page_paths:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            img_dir.rmdir()
        except OSError:
            pass

    return {
        "output": str(output),
        "pages": len(sorted_results),
        "chapters": len(chapters),
        "elapsed_s": round(time.time() - started, 1),
        "device": device,
    }


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a PDF into Markdown via local GLM-OCR.")
    p.add_argument("pdf", type=Path, help="input PDF path")
    p.add_argument("--output", type=Path, default=None,
                   help="output Markdown path (default: <pdf-stem>.md next to source)")
    p.add_argument("--model-path", default=DEFAULT_MODEL, help=f"HF model path (default: {DEFAULT_MODEL})")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"],
                   help="inference device (default: auto)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"render DPI (default: {DEFAULT_DPI})")
    p.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE,
                   help=f"downsize longest edge to this (default: {DEFAULT_MAX_EDGE})")
    p.add_argument("--pages", type=int, default=None, help="only process the first N pages")
    p.add_argument("--keep-images", action="store_true", help="keep per-page PNG cache")
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--prompt", default="doc", choices=list(OCR_PROMPTS.keys()),
                   help="OCR prompt variant (default: doc)")
    p.add_argument("--no-resume", action="store_true", help="ignore cached OCR results")
    args = p.parse_args(argv)

    output = args.output or args.pdf.with_suffix(".md")
    work_dir = args.work_dir or args.pdf.with_suffix(".work")
    device = _resolve_device(args.device)
    summary = convert(
        pdf_path=args.pdf, output=output,
        model_path=args.model_path, device=device, dtype=args.dtype,
        dpi=args.dpi, max_edge=args.max_edge,
        pages=args.pages, keep_images=args.keep_images,
        work_dir=work_dir, prompt_variant=args.prompt,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
