# Karpathy Autoresearch Adapter

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

Plan and apply a Karpathy-style autoresearch protocol to an existing GitHub
project. The skill scans the target repository, identifies whether a real
evaluator exists, proposes an immutable judge and mutable surface, and waits for
confirmation before writing project files.

Part of WdBlink LLM Skills.

## Install

Codex:

```bash
mkdir -p ~/.codex/skills/karpathy-autoresearch-adapter
rsync -a skills/karpathy-autoresearch-adapter/ ~/.codex/skills/karpathy-autoresearch-adapter/
```

Claude Code:

```bash
mkdir -p ~/.claude/skills/karpathy-autoresearch-adapter
rsync -a skills/karpathy-autoresearch-adapter/ ~/.claude/skills/karpathy-autoresearch-adapter/
```

Requires no extra runtime dependencies.

## Usage

Ask the agent to adapt a repository:

```text
Use karpathy-autoresearch-adapter on /path/to/project and produce the plan first.
```

Default behavior is plan-only. The skill writes files only after explicit user
confirmation of the evaluator design.

## Workflow

| Step | Description |
|------|-------------|
| Scan | Inspect docs, tests, CI, scripts, benchmarks, data protocols, and configs |
| Classify | Mark evaluator status as ready, partial, or missing |
| Plan | Propose immutable judge, baseline, mutable surface, runtime policy, and verification |
| Confirm | Wait for the user to approve the evaluator plan |
| Apply | Create project-specific autoresearch protocol files only when safe |
| Verify | Run available checks and confirm runtime artifacts are gitignored |

## Boundary

If a project has no evaluator, the skill may produce an evaluator protocol
document after confirmation, but it must not create a full autoresearch runtime.

## License

MIT
