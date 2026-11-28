# Figure workflow

## 1. Brief and route

Read the actual source methods and intended caption. The main figure should
answer a question with a small set of reader takeaways. Record purpose, source
paths and hashes, reading dimensions, semantic invariants and optional caller
metadata. An edit adds the baseline path/hash and exactly what may change.

| Route | Render method | Boundary |
| --- | --- | --- |
| illustration | Reference-guided image generation | Explain only mechanisms supported by the source |
| edit | Baseline image input and focused changes | Preserve unaffected semantics, imagery and style |
| statistics | Real data and executable plotting | Never invent measurements or rely on an image model for exact marks |

A hybrid figure may use a data-rendered inset. Preserve the user's requested
illustration style. A geometric defect remains unresolved until actual pixels
match the calculation. A valid equation does not prove a valid drawing.

## 2. References and plan

Prefer the existing venue reading index, user references and previous project
figures. Use authorized public sources if needed. Rank by visual intent and
structure before subject similarity. Record each reference's path/hash, source
URL or PDF page/panel, actual visual observation and transferable design lesson.
Reference images supply composition ideas; scientific facts come from this study.
Do not redistribute copyrighted references without checking their terms.

Inspect the images themselves. If vision is unavailable, record the limitation.
No-reference runs are allowed with an explicit basis; do not invent search
results. There is no fixed top-ten quota or default reference dataset.

Write `plan.md` with the message and reading order, panels, modules, short labels,
symbols and directed edges with source, destination and meaning. Distinguish
training, inference, fallback, failure and independent comparator paths. Preserve
parallel outputs as parallel, not mutually exclusive alternatives. Do not infer
coordinate orientation from an unspecified local metric frame.

List material omitted for clarity. Move dense derivations, redundant labels and
solver settings into the caption or methods. Include an equation only when it
helps explain the mechanism. Do not impose a universal panel or word count.

## 3. Style and rendering

Refine the plan into `prompt.md` using observed reference lessons and the actual
venue or requested style. Specify spacing, typography at reading size, consistent
colors/symbols, line weights, arrowheads and image treatment. Keep captions
outside the artwork. Styling cannot introduce scientific dependencies, change
formulas or comparison conditions, or standardize every venue to NeurIPS.

Discover actual image-tool capabilities. Prefer exposed tools, then an installed
headless transport such as `scripts/magpie_image.py`, then an already configured
provider helper. Do not build a provider router, launch a browser or require a
new upstream installation. A past installation does not prove current availability.
Read schemas and record the tool, model when known, operation, actual image inputs,
requested dimensions and available receipt. Unknown usage stays unknown.

For an edit, pass the baseline as an image input; distinguish it from style
references. State the requested changes and preservation constraints. If editing
is unsupported, preserve the baseline and report the blocker. A prompt describing
the old image or a complete redraw is not a local edit.

Save each result to a new version. For asynchronous operations, retain the task
ID and retrieve it in the same MCP server process. Restarting a stdio server may
lose task state. A timeout is not permission to issue a second generation request.
Record failure/unknown usage and inspect receipts and output before recovery.
Decode the actual file to measure pixels; a requested 4K preset is not a measured
4K result. Preserve raw bytes and use lossless working masters when available.

## 4. Pixel critique and revision

Inspect the candidate itself, source, caption and baseline for edits. Record the
image hash and actual inspection evidence. Check scientific relationships, label
spelling, arrow endpoints, formula rendering and failure/fallback distinctions.
For numerical elements, independently calculate and inspect point positions,
scales, means, ellipse orientation and units. Then check line continuity, closed
frames, overlap, crossings, hierarchy and readability at the intended size.

Use `pass` only after actual inspection with no blocking finding. Use `revise` for
an actionable defect, `unreviewed` when no visual inspection happened, and
`blocked` when a required capability failed. Malformed feedback never means pass.
The CLI checks record consistency; it cannot verify the truth of an inspection
statement or replace scientific judgment.

For local correction, edit the original or best valid image, not automatically
the most recent failed revision. Compare changed and unchanged regions. If lines
or typography degrade, change the tactic or simplify within accepted scope.
Keep the rejected attempt and choose the best valid candidate explicitly. Apply
user feedback through the same conversation. Respect actual budget and stop
conditions; do not silently stop at an invented round limit.

## 5. Selection and delivery

Select only a reviewed version. Record why it was chosen, rejected attempts,
unresolved nonblocking issues and a source-supported caption. Return the original
master and requested export, with actual dimensions and limits.

The backend supports PNG/TIFF export with no resize and verifies decoded pixels.
Converting transparent pixels to RGB/L requires an explicit background. DPI is
metadata; assigning it adds no pixels. A raster is not an editable vector file.
Run `validate --require-selection` before final delivery. Changed source, image,
baseline or inspection evidence requires a renewed assessment.

For paper integration, the caller binds the image and records into its accepted
artifacts, inserts the literal selected filename in the manuscript, compiles and
inspects the affected page at reading size. A standalone preview cannot certify
manuscript readability, and production critique is not independent paper review.
