---
name: glm-ocr-pdf2md
description: "Convert a scanned or no-text-layer PDF book into a structured Markdown file with faithful layout, auto-detected table of contents, and explicit page markers, by running every page through a locally-served ZhipuAI/GLM-OCR model (0.9B vision-language OCR, 94.62 on OmniDocBench V1.5). The model is loaded directly via Hugging Face transformers on Apple Silicon (MPS) or CUDA — no Ollama, no vLLM, no external API. Output Markdown has page-anchored HTML comments so downstream skills (book-to-skill, minimax-docx) can recover the source map. Trigger when the user wants to digitize a scanned book, convert a PDF to Markdown, mentions 'GLM-OCR', '本地OCR', 'PDF转Markdown', 'OCR 排版', '扫描版PDF', or asks for a digital text version of a printed academic book."
argument-hint: <pdf-path> [--output md-path] [--pages N] [--max-edge 768]
---

# GLM-OCR PDF → Markdown

Turn a PDF into a clean, page-anchored Markdown book by running every page
through a locally-served `ZhipuAI/GLM-OCR` model (0.9B params, BF16).

## When this skill applies

- The PDF is a **scanned book** or has no text layer (e.g. Z-Library downloads,
  photocopied manuscripts, screenshots).
- The user wants a **digital Markdown / DOCX** version of the book.
- The user explicitly mentions **GLM-OCR / 本地OCR / PDF转Markdown / OCR 排版**.
- Layout fidelity (headings, tables, formulas, page numbers) matters more
  than raw speed.

Skip this skill if the PDF already has a clean text layer — use
`minimax-pdf` or `book-to-skill` directly.

## Contract

| Input | Output |
| --- | --- |
| Absolute path to a `.pdf` file | `<book-slug>.md` next to the source (or `--output`) |
| `--output PATH` | explicit output `.md` |
| `--pages N` | only OCR the first N pages (smoke test) |
| `--max-edge N` | downsize longest edge to this px (default **768**) |
| `--device {auto,mps,cuda,cpu}` | inference device (default `auto`) |
| `--dtype {bf16,fp16,fp32}` | weight precision (default `bf16`) |
| `--keep-images` | keep the per-page PNG cache |
| `--work-dir PATH` | scratch dir (default: `<pdf-stem>.work` next to source) |

Output format:

```markdown
# <book title>

> OCR'd with `GLM-OCR` (mps, bf16) — 153 pages, 1539.0s total.

## 目录 (auto-detected)

- [第一章 构想故事](#第一章-构想故事) ……… p.15
- [第三章 故事的组成部分：词语、句子和段落](#第三章-故事的组成部分词语句子和段落) ……… p.32
- [第十一章 应对评议](#第十一章-应对评议) ……… p.123
- ...

---

<!-- p.1 -->
如何让一篇学术成果丰硕的文章进入更多人的视野和思维？...

<!-- p.2 -->
How to Use Storytelling in Your Academic Writing
...

<!-- p.3 -->
图书在版编目（CIP）数据
...
```

`<!-- p.N -->` markers are the source of truth — `book-to-skill` and
`minimax-docx` recover the page→paragraph map from them.

## Pipeline

```
PDF ──PyMuPDF──▶ pages/p%03d.png (DPI=150, long-edge ≤ 768 px)
            │
            ▼
   for each page image:
            │
            ▼
   chat-template prompt:  "Text Recognition:"
            │
            ▼
   AutoModelForImageTextToText (BF16, SDPA, MPS / CUDA)
   attn_implementation=sdpa
            │
            ▼
   per-page Markdown ──┐
                       ▼
   TOC synthesizer (skips pages whose first line is "目录" / "CONTENTS",
                    and pages with >= 3 chapter lines in compact form)
                       │
                       ▼
   Concatenate with explicit page-marker HTML comments
                       │
                       ▼
   Final .md
```

The OCR prompt is fixed (see `references/ocr-prompt.md`); do **not**
paraphrase it — GLM-OCR is finetuned on a near-exact match.

## Install & deploy

### 1. Pull the model

```bash
python3 -c "from modelscope import snapshot_download; \
  snapshot_download('ZhipuAI/GLM-OCR', cache_dir='/Users/wdblink/Models/huggingface')"
```

The first pull downloads ~2.5 GB and lands in
`/Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR/`.

### 2. Install Python deps

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install "transformers>=5.0" torch accelerate PyMuPDF Pillow
```

### 3. Sanity check

```bash
./scripts/run_server.sh check
# => model dir: /Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR
# => weights:  2.5G
# => config:   OK
# => all deps OK
```

## Run

From the installed skill:

```bash
python3 ~/.codex/skills/glm-ocr-pdf2md/scripts/convert_pdf.py \
  /path/to/book.pdf \
  --output /path/to/book.md
```

Quick smoke test (first 3 pages only):

```bash
python3 scripts/convert_pdf.py book.pdf --pages 3 --keep-images
```

From the repo checkout:

```bash
python3 skills/glm-ocr-pdf2md/scripts/convert_pdf.py \
  /path/to/book.pdf
```

Resume is **on by default**: if the script crashes mid-book, re-running
the same command continues from the last cached page. Pass
`--no-resume` to start over.

## Wiring into other skills

- `minimax-docx` can re-typeset the produced `.md` into a Word file.
- `book-to-skill` can split the `.md` by chapter for a knowledge skill.
- `mind-map` can turn the TOC + first paragraphs into a graph.

Pass the same `--output` Markdown into all of them — the page-marker
comments are stable anchors.

## Gotchas

- **Why not Ollama / vLLM?** `ollama pull glm-ocr` does not exist on
  ollama.com's registry (the upstream blog post that mentions it is
  wrong). `vllm serve` requires CUDA which Apple Silicon doesn't have.
  Loading the model via `transformers` is the most portable path.
- **MPS deadlocks above 768px** — `attn_implementation='sdpa'` on M-series
  hangs on some pages with `max-edge > 768`. The default is 768; raise
  to 1024 only on CUDA.
- **First request is slow** (~5–10 s on M-series for the model load).
  Subsequent pages are 1–25 s each.
- **GLM-OCR doesn't always emit markdown `#` prefixes** — the TOC
  synthesizer matches `第X章` / `Chapter N` even without the prefix.
- **Sandbox / Downloads folder** — on macOS, `~/Downloads/*` requires
  Full Disk Access. If `python3` can't open the PDF, copy it:
  `cp <file> /tmp/book.pdf`.
- **Page markers are HTML comments, not plain text** — keep them when
  piping into `minimax-docx` or `book-to-skill`; they are how those tools
  recover page numbers.

## Files

- `scripts/convert_pdf.py` — main entry point.
- `scripts/run_server.sh` — `check / clean / benchmark` helpers.
- `scripts/check_env.py` — standalone env probe (no shelling out).
- `references/ocr-prompt.md` — the exact prompt sent per page.
- `references/toc-rules.md` — how the TOC synthesizer picks chapter titles.
- `tests/make_fixture.py` — generate a 3-page synthetic PDF.
- `tests/test_pipeline.py` — smoke test (synthetic PDF → Markdown).
- `tests/test_eval_book.py` — regression test on the canonical eval book.
- `evals/学术写作如何讲好故事.md` — full Markdown from the canonical book.
