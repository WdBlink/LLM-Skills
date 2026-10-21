---
name: karpathy-autoresearch-adapter
description: Adapt an existing GitHub project into a Karpathy-style autoresearch protocol project. Use when the user wants Codex or Claude Code to scan a repo, identify or design an immutable evaluator, produce a plan first, and only after confirmation create project-specific autoresearch protocol files. Does not run indefinite research loops by itself.
metadata:
  short-description: Plan and adapt repos into autoresearch protocol projects
---

# Karpathy Autoresearch Adapter

Use this skill to turn an existing repository into a project that Codex or
Claude Code can continue as a Karpathy-style autoresearch loop.

The skill is an adapter, not a runner. Its first job is to decide whether the
target project has a real evaluator. Never fabricate one.

## Default Mode: Plan First

By default, do not edit files. Scan the project and output an adaptation plan in
chat. Wait for explicit user confirmation before writing code or creating
`autoresearch/` files.

Only write files immediately if the user explicitly asks to apply, implement,
or modify the project in the same request.

## Scan

Inspect local project truth before proposing anything:

- README and project docs
- package manifests and lockfiles
- CI, Makefile, scripts, CLI entrypoints
- tests, benchmarks, evaluation harnesses, golden files
- datasets, fixtures, data protocols, prompts, configs
- prior experiment logs or generated artifacts

Prefer `rg` and structured parsers where available. Do not rely on repository
name or README claims alone.

## Evaluator Classification

Classify evaluator readiness before any adaptation:

- `ready`: a deterministic command, test, benchmark, or scoring harness can
  compare baseline and candidate outputs and return a keep/discard or score.
- `partial`: useful tests or metrics exist, but baseline, data split, scoring,
  or acceptance gates are incomplete.
- `missing`: no credible evaluator exists.

For `partial` or `missing`, output an evaluator construction plan and stop
unless the user confirms that plan.

## Plan Output

The plan must include:

- target project and objective
- candidate research subject
- evaluator classification and evidence
- immutable judge proposal
- baseline definition
- mutable surface
- allowed and forbidden files
- candidate invocation command
- scoring and KEEP/DISCARD rules
- runtime artifact policy
- git advance/revert discipline
- fresh-agent test
- open questions for the user

Use this skeleton:

```markdown
## Autoresearch Adaptation Plan

### Project
[path, language/runtime, detected commands]

### Evaluator Status
[ready | partial | missing]
[evidence from files/commands]

### Proposed Immutable Judge
[fixed data, fixed evaluator command, fixed score/gates]

### Baseline
[how to evaluate the current/default behavior]

### Mutable Surface
[exact files/params/prompts/code areas an agent may edit]

### Runtime Artifacts
[where logs/results go; what must be gitignored]

### Apply Plan
1. [files to create/change]
2. [tests/checks to add]
3. [verification commands]

### Stop Before Apply
[questions or evaluator decisions the user must confirm]
```

## Apply Rules

After confirmation, create project-specific files. Do not copy a generic
Karpathy `program.md`. The generated protocol must fit the target project.

Typical output layout:

```text
autoresearch/
  program.md
  evaluator_contract.md
  mutable/
  best/
  context/
  proposals/
  results.template.tsv
  latest_state.template.json
```

Add `.gitignore` rules for runtime state such as:

- `autoresearch/results.tsv`
- `autoresearch/latest_state.json`
- `autoresearch/logs/`
- generated run directories

Do not generate runtime folders if the evaluator is not ready.

## When No Evaluator Exists

If the project has no real test or benchmark system, it is acceptable to create
an evaluation protocol document after user confirmation, for example:

```text
autoresearch/evaluator_plan.md
```

In this case do not create `autoresearch/program.md`, `mutable/`, `best/`, or
runtime templates. A project without an evaluator is not yet an autoresearch
runtime.

## Program Contract Requirements

When generating `autoresearch/program.md`, it must tell a fresh Codex or Claude
Code session:

- which files to read first
- how to run the baseline
- how to propose exactly one candidate
- how to invoke the fixed evaluator
- who decides KEEP/DISCARD
- how to restore discarded candidates
- when it may stop
- which files are immutable
- where runtime artifacts are written
- what must never be committed

The program must define a session budget or stop contract. A single evaluator
call or one discarded batch must not be treated as completion unless the
program's stop contract allows it.

## Hard Boundaries

- Do not modify evaluator, frozen data, split rules, scoring, or gates during a
  research run.
- Do not use evaluator code as mutable candidate surface.
- Do not create a fake score from ad hoc reasoning.
- Do not browse the web for factors, prompts, or metrics unless the user asks.
- Do not make the protocol depend on Codex-only or Claude-only hidden features.
- Do not commit runtime experiment output.

## Verification

After applying, verify:

- generated JSON/Markdown files are valid enough to read
- runtime outputs are ignored by git
- evaluator command or smoke command works, if one exists
- a fresh-agent test is plausible: given only the project path, a new Codex or
  Claude Code session can read `autoresearch/program.md` and continue without
  inventing another process

Summarize changed files, evaluator status, and commands run.
