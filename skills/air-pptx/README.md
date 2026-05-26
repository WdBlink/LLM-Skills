# PV Forecast PPTX

![Private](https://img.shields.io/badge/Private-blueviolet)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

Create professional PowerPoint research presentations for photovoltaic and energy forecasting projects. The skill keeps a bundled reference deck, brand guidance, and PPTX generation resources together so agents can build project proposal or research-report slide decks with consistent structure and styling.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/air-pptx
rsync -a skills/air-pptx/ ~/.codex/skills/air-pptx/
```

## Use

Use the instructions in `SKILL.md` to map source content into the bundled research proposal slide structure:

| Resource | Purpose |
|----------|---------|
| `resource/【参考】课题需提供--20250704.pptx` | Reference presentation template |
| `resource/pptx/` | PPTX generation and validation resources |
| `resource/brand-guidelines/SKILL.md` | Brand-style guidance |
| `resource/output.pptx` | Example generated output |

Some bundled PPTX resource scripts require optional Python packages such as `python-pptx` and `Pillow`. Install those dependencies before running inventory, thumbnail, or generation helpers directly.

## Workflow

| Step | Description |
|------|-------------|
| Analyze | Identify slide count, proposal section, and content hierarchy |
| Design | Map content to title, technology, platform, statistics, or summary slides |
| Generate | Use the bundled PPTX resources to produce the deck |
| QA | Check text overflow, alignment, contrast, and slide-number consistency |

## License

Proprietary
