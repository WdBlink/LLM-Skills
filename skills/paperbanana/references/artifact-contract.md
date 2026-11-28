# Artifact contract

## Files

Use one root per figure. Paths below are relative to it unless explicitly noted.
The agent prepares these records; users interact only through the conversation.

```text
brief.json
plan.md
prompt.md
candidates/v1/image.png
candidates/v1/candidate.json
candidates/v1/receipt.json
candidates/v1/inspection.md
candidates/v1/critique.json
selection.json
export/main-method.png
export/export.json
```

Versions are immutable image identities. Repeating an identical archive is safe;
new bytes require a new version. Source and reference records bind paths to SHA-256.
Absolute paths are allowed for explicitly supplied input files; output paths are
relative to the figure root. `ingest` copies actual tool-input images and receipts
into each candidate directory. `critique` archives inspection text under
`inspections/<sha256>.md`, retaining earlier findings and evidence when re-reviewed.
Those archived copies are authoritative; original temporary files can be removed.
For a movable run, copy source/reference/baseline files into the root before writing
the brief and use relative paths there. An explicitly external brief source stays
a live dependency and its change invalidates the figure. Do not place credentials
in any request or receipt.

## Brief

`init` writes an empty template. Complete and freeze it before rendering. Create
source hashes from actual files, not example strings. All versions in one root
share that brief. If source meaning, baseline or allowed edits change, create a
new figure root and preserve the previous one; do not rewrite old evidence to fit
a new request. Keep optional caller metadata separately.

```json
{
  "schema_version": 1,
  "figure_id": "F-method",
  "route": "illustration",
  "purpose": "Explain the source-supported method",
  "sources": [{"path": "source.md", "sha256": "actual file digest"}],
  "references": [],
  "no_reference_reason": "No reference supplied; composition derived from source",
  "caption": "Source-supported caption",
  "placement": "main",
  "reading_size": {"width_px": 1800, "height_px": 900},
  "invariants": ["Rejected inputs never enter the summarizer"]
}
```

A reference adds `path`, `sha256`, `source` locating its origin, and `lesson`
describing the design actually observed. An edit adds `baseline` with `path` and
`sha256`, plus a nonempty `permitted_changes` array. `route` is `edit` in that case.
`statistics` hands off to a real-data renderer; it does not authorize image-model
fabrication of measured marks. Add `requirement_id` or `claim_ids` only when a
caller supplies them.

## Candidate and critique

`ingest` receives the returned image, tool identity and actual inputs. It stores
raw bytes, measured format/pixels, SHA-256 and a copied receipt if supplied.

```bash
python3 <skill>/scripts/paperbanana.py ingest --root <root> --version v2 \
  --source /absolute/edited.png --tool actual-tool --operation edit \
  --input-image /absolute/baseline.png --receipt /absolute/tool-receipt.json
```

`--input-image` is repeatable. It records the agent's declaration; the actual
provider request/receipt must support that declaration. A matching hash alone
cannot prove that a provider received an input.

After seeing the image, save the actual inspection in a Markdown/text file. For
an edit, state whether the intended change landed and the unaffected content
survived. Optional `--findings` points to a JSON array:

```json
[
  {
    "location": "right panel, bottom arrow",
    "blocking": true,
    "issue": "Arrow enters the wrong module",
    "requested_change": "Move the endpoint to the validator"
  }
]
```

Record through `critique --version v2 --verdict revise --evidence <file>
--method <actual inspection method> --findings <file> --note <baseline comparison>`.
Verdicts are `pass`, `revise`, `unreviewed`, `blocked`. Do not use `pass` without
actual pixel inspection or with a blocking finding. These are host-authored
judgments, not an independent reviewer or model self-score certificate.

`select` records a reviewed candidate, caption and reason. `--rejected v1:reason`
and `--unresolved <nonblocking issue>` are repeatable. `export` writes PNG or TIFF;
explicit positive `--dpi` is required if the source has no resolution metadata.
Targets must be new PNG/TIFF paths below `export/` within the figure root. Export
requires a passing critique, even without a selection. After switching selection,
export the selected version to a new path before delivery; validation rejects a
stale export belonging to a different candidate. Use `--width-pixels`, `--height-pixels`, `--min-dpi` and `--max-bytes` only when an
actual publication requirement supplies them. No command enlarges the image.

## Magpie transport requests

First send `{}` with `--request` and `--receipt` to discover tools. Inspect the
returned schemas before building a generation or edit request. Supply no model
unless selected from actual current capability information or a user instruction.

```json
{
  "tool": "actual discovered tool name",
  "arguments": {
    "prompt": "Actual styled prompt",
    "output_path": "/absolute/new-version/raw.png"
  },
  "output_path": "/absolute/new-version/raw.png"
}
```

Argument names above are examples, not an assumed schema. Use the actual tool's
image-input argument for references and edit baselines. Optional `executable`
selects a local Magpie binary; `timeout` and `poll_timeout` are transport deadlines,
not total task-budget policies. `poll_tool` and `poll_id_argument` can adapt a
changed schema; their defaults match Magpie's `generation_result` and `id`.
The helper checks that result-tool contract before submitting generation. It
archives one image per call, so use `n: 1` and an explicit tool output path matching
`output_path`; additional candidates are separate versioned attempts. The receipt
retains initialization, tool discovery, responses and pending task IDs. Reported
usage is preserved as reported, without summing repeated polling reports; absent
usage stays unknown.

A server error, timeout, missing file or undecodable image is a failed or unknown
attempt. Do not automatically submit generation again. Preserve receipts and
previous candidates. A task ID from a terminated server may be unrecoverable;
a replacement request is a new operation that consumes its own budget.

## What validation establishes

Checks establish file existence, decodability, measured pixels, unchanged hashes,
version identity and consistency of records. They do not establish that text
assertions are truthful, that a model was really called, or that a diagram's
science is correct. Report offline fixtures, real tool runs and visual inspection
as separate evidence.
