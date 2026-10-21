# OCR Prompt

The `convert_pdf.py` script sends the same per-page prompt to GLM-OCR for every
page image. Do **not** paraphrase this prompt; the model is finetuned on a
near-exact match and the layout fidelity drops if you reword it.

```text
Please output the text content from the image in Markdown format.
Follow these rules strictly:
1. Preserve the original document structure, including headings, paragraphs, lists, and tables.
2. Use # for main titles, ## for chapters, ### for sections, and #### for sub-sections.
3. Convert tables to Markdown table syntax with | separators.
4. Convert mathematical formulas to LaTeX ($...$ for inline, $$...$$ for display).
5. Maintain the original reading order. Do not summarize or omit any content.
6. If the page is a chapter title page, table of contents, or section divider, transcribe it as-is.
```

The temperature is forced to `0.0` and `max_tokens` to `4096`. Increase the latter
for long pages (the 学术写作如何讲好故事 preface page is ~1.8 k tokens, well under
the cap).
