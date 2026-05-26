---
name: skill-forge-checklist
description: Skill Forge-style audit checklist for local-ready and portable skill quality.
---

# Skill Forge Checklist

Use one result per row: `PASS`, `MUST-FIX`, or `SUGGESTION`.

## Checks

- Discoverable: frontmatter name matches the folder; description names image-to-YOLO, detection datasets, and manifest conversion triggers.
- Reliable: deterministic script validates inputs and exits non-zero on invalid boxes, unknown classes, or missing images.
- Efficient: SKILL.md is concise and sends detailed schema or migration content to references only when needed.
- Trustworthy: no secrets, credentials, network calls, or destructive filesystem behavior.
- Bounded: skill states that annotation quality depends on the active vision model and does not promise model training.
- Valuable: generated assets cover at least YOLO plus two additional detection dataset standards.
- Structure: all referenced files exist; references have frontmatter; scripts live under `scripts/`.
- Security: scan for private keys, API tokens, `.env`, credential files, and personal absolute paths before publishing.
- Cross-platform: Codex, Claude Code, OpenCode, and other hosts can point at the same skill source; detector services replace only the manifest-producing detection step.
