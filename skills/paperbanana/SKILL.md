---
name: paperbanana
description: "Create and revise publication main figures using the PaperBanana reference, planning, styling, image generation and visual critique workflow. Use for paper main figures, method illustrations, architecture or pipeline diagrams, conceptual explanations, and surgical edits to an existing figure, including 论文主图、方法图、框架图、架构图、局部修图. Interact through the agent conversation and use headless image tools. Route measured plots to a data renderer."
license: MIT
compatibility: Python 3.10+, Pillow for image processing; a configured image generation or editing tool for rendering. Optional local Magpie image MCP transport.
metadata:
  version: "0.1.0"
---

# PaperBanana

Turn source methods and the user's visual intent into a focused main figure. The
host agent performs reference selection, planning, styling and critique; an
available image tool renders or edits; the bundled backend archives and validates
artifacts. All user interaction stays in the agent conversation.

Read [workflow.md](references/workflow.md) before drawing or editing. Read
[artifact-contract.md](references/artifact-contract.md) when recording candidates,
critiques or delivery. See [provenance.md](references/provenance.md) for the method
source and the boundaries of this adaptation.

## Inputs and outputs

Accept pasted methods, a paper or project path, a figure specification, optional
reference images, or an existing image plus a change request. Extract the desired
message, supported mechanism, caption, reading size and preservation constraints.
Ask only for missing information that changes scientific meaning or an explicit
user preference. Do not ask the user to construct JSON or operate a client.

Deliver the selected original image and, when requested, a lossless PNG/TIFF
export; a caption; and records of sources, references, prompts, actual tool
receipts, critiques and version selection. The backend's file checks do not
certify scientific correctness. Unreviewed results remain drafts.

## Workflow

1. Choose `illustration`, `edit` or `statistics`. Use real data and an executable
   renderer for measured plots and exact numerical marks. For a hybrid main
   figure, use an accurately rendered inset while retaining the requested art.
2. Create a focused brief and inspect actual reference images. Reuse a paper's
   venue reading index when available. No references is allowed when recorded;
   invented retrieval or unseen references are not.
3. Plan reading order, panels, labels, directed edges and semantic invariants.
   Refine typography, palette, spacing and imagery without changing mechanisms.
   Reduce dense text and formulas when they obscure the main message.
4. Discover the current image tools and their real schemas. Generate a candidate,
   or supply the baseline as an actual input for a bounded edit. Preserve raw
   bytes. Persist asynchronous IDs and retrieve in the same server session.
5. Inspect the actual pixels against the source, caption and baseline. Record
   `pass`, `revise`, `unreviewed` or `blocked` with concrete evidence. Correct
   numbers do not excuse incorrect point geometry, arrows or branching.
6. Revise focused defects, retain the best valid version, select a reviewed
   candidate and validate delivery. A later image is not automatically better.

Use existing authorization and task budgets. Do not invent a fixed number of
candidates or revision rounds. If the user asks to discuss composition first,
complete that conversation before rendering. Respect requested drawing style;
do not replace local image editing with a programmatic redraw.

## Headless tools

Prefer image tools exposed by the host. Inspect their schemas for image inputs,
editing, model choices, sizing, asynchronous retrieval and file delivery.
Existing provider skills can supply an already configured route; read their
current instructions. Do not require a new Gemini key or install upstream
PaperBanana to use this workflow.

If local Magpie is available but its image tools are not exposed, use
`scripts/magpie_image.py`. First discover its tools without a generation call:

```bash
python3 <skill>/scripts/magpie_image.py --request tools-request.json --receipt tools-receipt.json
```

A discovery request is `{}`. Then construct a request from the returned schema,
with `tool`, `arguments`, and an absolute `output_path`. Optional `executable` or
`PAPERBANANA_MAGPIE_BIN` selects an installed binary. Keep private paths and model
IDs in runtime requests, outside the shared skill. See the contract for examples.
This transport starts no browser or GUI. It must retain the same process through
asynchronous result retrieval and must not retry generation after a timeout.

If generation, editing or visual inspection is unavailable, expose that specific
capability gap. Keep any usable baseline and partial records. Never claim that
writing a prompt, changing DPI, copying a fixture or validating hashes constitutes
successful drawing or visual review.

## Artifact backend

Resolve `<skill>` to this installed directory. Use a figure-specific directory in
the user's project, such as `figures/main-method/`; no research controller is
required. Choose a Python with Pillow installed. The agent writes the brief and
records through ordinary file tools, then uses the CLI for deterministic work.

```bash
python3 <skill>/scripts/paperbanana.py init --root figures/main-method --figure-id F-method
python3 <skill>/scripts/paperbanana.py ingest --root figures/main-method --version v1 --source /absolute/returned.png --tool actual-tool --operation generate
python3 <skill>/scripts/paperbanana.py critique --root figures/main-method --version v1 --verdict pass --evidence /absolute/inspection.md --method "actual image inspection tool"
python3 <skill>/scripts/paperbanana.py select --root figures/main-method --version v1 --reason "Correct mechanism and readable labels" --caption "Source-supported caption"
python3 <skill>/scripts/paperbanana.py export --root figures/main-method --version v1 --target export/main-method.png --dpi 300
python3 <skill>/scripts/paperbanana.py validate --root figures/main-method --require-selection
```

These commands illustrate syntax. Fill factual values from the actual run; do not
copy example verdicts or tool identities. Use `--input-image` for every actual
image sent to an editing tool. The agent's external tool receipt is the evidence
of that call; a CLI argument alone cannot prove the input reached the provider.

## Integration with autoresearch-paper

Accept its method-figure specification, source files, caption, references and
optional requirement metadata. Return selected image and records. The caller
retains ownership of its frozen Brief, budget, WorkOrder, requirement binding,
manuscript insertion, compilation and independent scientific review. Standalone
use needs none of those controller fields.
