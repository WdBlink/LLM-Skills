# Social Cover Maker

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

Generate social media cover images for WeChat, Xiaohongshu, Zhihu, and other publishing platforms from Markdown articles or notes. The skill uses Open Design style systems, deterministic SVG layout, and PNG export checks so Chinese titles remain editable and reliable.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/social-cover-maker
rsync -a skills/social-cover-maker/ ~/.codex/skills/social-cover-maker/
```

Claude Code users can install to `~/.claude/skills/social-cover-maker/` with the same `rsync` pattern.

## Requirements

- macOS `sips`
- `xmllint`
- Optional local Open Design checkout at `/Users/wdblink/Code/my_repo/open-design`

## Usage

Ask for a platform and style:

```text
Use the kami Open Design style to generate WeChat covers for this Markdown article.
```

The skill should produce editable SVG files and rendered PNG files, for example:

```text
wechat-main.svg
wechat-main.png
wechat-square.svg
wechat-square.png
```

Render and validate an output directory:

```bash
bash skills/social-cover-maker/scripts/render_svg_covers.sh path/to/output-dir
```

## Platform Presets

| Platform | Outputs |
|---|---|
| WeChat Official Account | `1350x600` main cover, `1080x1080` square cover |
| Xiaohongshu | `1242x1660` portrait cover, optional `1080x1080` square |
| Zhihu | `1200x675` wide cover, optional `1080x1080` square |
| Generic social | `1350x600` wide, `1080x1080` square |

## Design Principles

| Principle | Meaning |
|---|---|
| SVG first | Keep Chinese titles deterministic, editable, and inspectable. |
| One metaphor | Translate each article into one dominant visual concept. |
| Platform fit | Recompose each platform version instead of cropping blindly. |
| Style systems | Use Open Design `DESIGN.md`, `tokens.css`, and component examples when available. |
| Verified export | Validate SVG and PNG dimensions before delivery. |

## License

MIT
