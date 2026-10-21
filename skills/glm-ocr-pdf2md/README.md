# GLM-OCR PDF → Markdown

A self-contained Agent Skill that turns a **scanned / no-text-layer PDF book**
into a clean Markdown file by running every page through a local
**ZhipuAI GLM-OCR** model (the 0.9B vision-language OCR model that scored
**94.62 on OmniDocBench V1.5**).

The model is loaded directly via `transformers` on Apple Silicon (MPS) or
any CUDA box — no Ollama, no vLLM, no external API. Output Markdown is
page-anchored with explicit `<!-- p.N -->` markers so downstream skills
(`book-to-skill`, `minimax-docx`, `mind-map`) can recover the source map.

| | |
| --- | --- |
| **Category** | Multimodal |
| **Version** | 0.1.0 |
| **License** | MIT |
| **Model** | `ZhipuAI/GLM-OCR` (Hugging Face) |
| **Storage** | `/Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR` |

## What it does

```
PDF ──PyMuPDF──▶ per-page PNG (DPI=150, long-edge ≤ 1024 px)
            │
            ▼
   chat-template prompt: "Text Recognition:"
            │
            ▼
   AutoModelForImageTextToText (BF16, MPS / CUDA)
            │
            ▼
   per-page Markdown ──┐
                       ▼
   TOC synthesizer scans for 第X章 / Chapter N / 一、二
                       │
                       ▼
   Final .md with: H1 title, `> OCR'd with ...` meta,
   auto TOC, page-marker HTML comments, body.
```

## Install

```bash
# 1. Pull the model (~2.5 GB, lands in /Users/wdblink/Models/huggingface/)
python3 -c "from modelscope import snapshot_download; snapshot_download('ZhipuAI/GLM-OCR', cache_dir='/Users/wdblink/Models/huggingface')"

# 2. Install deps
pip install PyMuPDF Pillow "transformers>=5.0" torch accelerate

# 3. Install the skill
mkdir -p ~/.codex/skills/glm-ocr-pdf2md
rsync -a skills/glm-ocr-pdf2md/ ~/.codex/skills/glm-ocr-pdf2md/
```

## Usage

```bash
# Whole book
python3 ~/.codex/skills/glm-ocr-pdf2md/scripts/convert_pdf.py \
  /path/to/book.pdf \
  --output /path/to/book.md

# Smoke test (3 pages)
python3 scripts/convert_pdf.py book.pdf --pages 3 --keep-images

# Specific device
python3 scripts/convert_pdf.py book.pdf --device mps    # or cuda / cpu
```

### All flags

```
--output PATH        output .md (default: <pdf-stem>.md next to source)
--model-path PATH    HF model path (default: /Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR)
--device {auto,mps,cuda,cpu}
--dtype  {bf16,fp16,fp32}
--dpi    N           render DPI (default 150)
--max-edge N         downsize longest edge to this (default 1024)
--pages  N           only OCR the first N pages
--keep-images        keep per-page PNG cache
--work-dir PATH      scratch dir (default: <pdf-stem>.work next to source)
--prompt {doc,text,table,formula}
--no-resume          ignore cached OCR results
```

## Speed

M5 Max (BF16, MPS, `--max-edge 1024`):

| Pages | Total time | Per-page avg |
| --- | --- | --- |
| 5 (smoke) | 95 s | 19 s |
| 232 (full) | ~50 min | ~13 s |

Cold start (model load) is ~6 s; per-page is 4–10 s for most pages and
30–70 s for pages with dense tables / formulas (longer generation, no
throughput issue).

## Output format

```markdown
# 学术写作如何讲好故事

> OCR'd with `GLM-OCR` (mps, bf16) — 232 pages, 3050.4s total.

## 目录 (auto-detected)

- [第 1 章 讲故事的重要性](#第-1-章-讲故事的重要性) ……… p.13
- [第 2 章 故事讲述的要素](#第-2-章-故事讲述的要素) ……… p.31
- ...

---

<!-- p.1 -->
如何让一篇学术成果丰硕的文章进入更多人的视野和思想？...

<!-- p.2 -->
How to Use Storytelling in Your Academic Writing
...

<!-- p.3 -->
图书在版编目（CIP）数据
...
```

The `<!-- p.N -->` markers are stable anchors — `book-to-skill` and
`minimax-docx` rely on them.

## Files

```
skills/glm-ocr-pdf2md/
├── SKILL.md
├── README.md
├── scripts/
│   ├── convert_pdf.py      # main pipeline
│   └── check_env.py        # verify deps + endpoint
├── references/
│   ├── ocr-prompt.md       # exact prompt sent per page
│   └── toc-rules.md        # TOC synthesis rules
├── tests/
│   ├── make_fixture.py     # generate a 3-page synthetic PDF
│   └── test_pipeline.py    # end-to-end smoke test
└── evals/
    └── book_output.md      # full Markdown of the canonical eval book
```

## Running the tests

```bash
cd ~/.codex/skills/glm-ocr-pdf2md
python3 tests/test_pipeline.py
# => ✅ pipeline smoke test passed
```

The test:
1. Generates a 3-page synthetic PDF with `第X章` and `Chapter N` headings.
2. Runs the converter end-to-end.
3. Asserts the produced `.md` starts with `# `, has ≥3 page markers, has
   the auto-detected TOC, and contains Chinese characters.

Set `GLM_OCR_SKIP=1` to skip the test when the model is unavailable.

## Gotchas

- **First run is slow** because the model is loaded into unified memory
  on M-series (or VRAM on NVIDIA). After the first run, the same model
  reuses the OS page cache.
- **Resume is on by default** — if the script crashes mid-book, re-running
  the same command continues from the last cached page. Use `--no-resume`
  to start over.
- **Quality vs. speed** — `--max-edge 1024` is a tested sweet spot on
  M5 Max. Drop to 768 for ~2× speedup at the cost of some small-character
  accuracy; raise to 1280 if your book has dense formulas.
- **MPS kernel bugs** — `attn_implementation='sdpa'` is the default. If
  you see hangs, switch the model to CPU first to confirm the issue, then
  file a torch issue.
- **Sandboxed environments** — `~/Downloads/*` requires Full Disk Access
  on macOS. Copy the file to a non-restricted path first if you hit
  `Operation not permitted`.

## Provenance

- Model: [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (also
  on [ModelScope](https://www.modelscope.cn/models/ZhipuAI/GLM-OCR)).
- Architecture: `GlmOcrForConditionalGeneration` (CogViT vision encoder +
  GLM-0.5B language decoder, 0.9B params).
- Conversion script: `transformers.AutoModelForImageTextToText` with
  `attn_implementation='sdpa'`, BF16 weights, prompt `Text Recognition:`.

## Eval fixture

`evals/book_output.md` is the Markdown produced from the canonical
evaluation book (学术写作如何讲好故事 by Timothy G. Pollock, Z-Library
PDF, 232 pages). It is the ground truth used in this skill's regression
checks.
