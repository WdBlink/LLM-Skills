---
name: social-cover-maker
description: Generate social media cover images for WeChat, Xiaohongshu, Zhihu, and other publishing platforms from Markdown articles or notes. Use when the user asks for article covers, social post posters, platform-specific cover images, or Open Design styled SVG/PNG publishing assets.
argument-hint: <article-or-topic> [platform] [style]
---

# Social Cover Maker

Create platform-ready social cover assets from article content. Prefer deterministic SVG layout and PNG export over image-model text rendering, especially for Chinese titles.

## Core Contract

- Inputs: Markdown article, pasted article content, title + brief, or a local article path.
- Outputs: editable `.svg` source plus final `.png` assets.
- Default style source: Open Design systems, then user-pasted design-system text.
- Do not modify the source article unless the user explicitly asks.
- Keep text readable at thumbnail size; recomposition beats simple scaling.

## Workflow

1. Read the source content.
2. Extract:
   - title
   - central thesis
   - strongest metaphor
   - 2-4 reusable keywords
   - target platform and requested style
3. Choose the platform preset. If unspecified, create a wide cover and a square cover.
4. Look up the design system. Read `DESIGN.md` first; inspect `tokens.css`, `components.html`, and `preview/` if available or if the style is ambiguous.
5. Translate the article into one dominant visual metaphor, not several competing illustrations.
6. Generate SVG source files with fixed dimensions.
7. Export PNGs and validate dimensions.
8. Preview both PNGs, then refine text wrapping, spacing, visual density, and platform fit.

## Platform Presets

### WeChat Official Account

Use for `公众号`, `微信公众号`, `WeChat`.

- Main cover: `1350x600`, ratio `2.25:1`
- Square/share cover: `1080x1080`, ratio `1:1`
- Naming: `wechat-main.svg/png`, `wechat-square.svg/png`

### Xiaohongshu

Use for `小红书`, `RED`, `xhs`.

- Portrait cover: `1242x1660`, ratio about `3:4`
- Square fallback: `1080x1080`
- Naming: `xhs-portrait.svg/png`, `xhs-square.svg/png`
- Design note: stronger title hierarchy, fewer tiny labels, more immediate visual impact.

### Zhihu

Use for `知乎`, `Zhihu`.

- Wide cover: `1200x675`, ratio `16:9`
- Square fallback: `1080x1080`
- Naming: `zhihu-wide.svg/png`, `zhihu-square.svg/png`
- Design note: restrained editorial tone; title clarity beats decoration.

### Generic Social

Use when the platform is unspecified.

- Wide: `1350x600`
- Square: `1080x1080`
- Optional portrait: choose platform-specific dimensions only when requested.
- Naming: `cover-wide.svg/png`, `cover-square.svg/png`, `cover-portrait.svg/png`.

## Open Design Lookup

When the user names a style, locate the design system in this order:

1. Local checkout: `/Users/wdblink/Code/my_repo/open-design/design-systems/<slug>/`
2. Official source: `https://github.com/nexu-io/open-design/tree/main/design-systems`
3. User-provided pasted design-system text

Useful files:

```text
DESIGN.md
tokens.css
components.html
manifest.json or open-design.json
preview/
assets/
```

Normalize common slugs:

```text
linear -> linear-app
kami -> kami
apple -> apple
minimax -> minimax
energetic -> energetic
```

If the local checkout is missing the requested style, browse the official GitHub directory. Treat Open Design files as style guidance, not brand authorization. Extract layout grammar, color roles, typography, component feel, and constraints.

## Design Rules

- Use SVG for deterministic text-heavy covers.
- Avoid generative-image text for Chinese titles.
- Use one strong metaphor derived from the article.
- Keep accent colors scarce and semantically meaningful.
- Make the square or portrait cover separately; do not just crop the wide cover.
- Output both editable SVG and final PNG.
- Preserve the article's actual title unless the user asks for title rewriting.

## Output Path

For local Markdown articles under a wiki/posts-like directory, place assets near the article:

```text
wiki/posts/assets/<article-slug>-<style>-<platform>/
```

For generic requests, use a local output directory under the current project:

```text
generated/social-cover-maker/<article-slug>-<style>-<platform>/
```

Use semantic names when the platform is known. Use generic names only when unknown:

```text
wechat-main.svg
wechat-main.png
wechat-square.svg
wechat-square.png
xhs-portrait.svg
xhs-portrait.png
zhihu-wide.svg
zhihu-wide.png
cover-wide.svg
cover-wide.png
cover-square.svg
cover-square.png
```

## Rendering And Validation

After creating SVG files, run the bundled renderer from an installed skill:

```bash
bash ~/.codex/skills/social-cover-maker/scripts/render_svg_covers.sh <output-dir>
```

From a repo checkout:

```bash
bash skills/social-cover-maker/scripts/render_svg_covers.sh <output-dir>
```

The script validates XML, exports each `.svg` to `.png` with `sips`, and checks that PNG dimensions match the SVG dimensions.

If rendering must be done manually:

```bash
xmllint --noout path/to/cover.svg
sips -s format png path/to/cover.svg --out path/to/cover.png
sips -g pixelWidth -g pixelHeight path/to/cover.png
```

Do not use `qlmanage` for final export; it may output square thumbnails for wide SVGs.

## Style-Specific Guidance

For `kami`, use warm parchment, ivory cards, ink-blue accents, serif-led hierarchy, fine rules, section numbers, tags, tables, and paper-like composition. Avoid pure white, pure black, strong shadows, gradients except the single low-contrast tag brush, and overuse of ink-blue.

For `linear`, use near-black native dark canvas, subtle translucent borders, Inter-like geometric typography, sparse indigo-violet accents, engineered panels, and precise product-layout rhythm. Avoid noisy gradients and excessive glow.

When a different Open Design system is requested, read its design files and follow those constraints instead of carrying over these examples.

## Quality Checklist

- The title is readable at small thumbnail size.
- The main visual metaphor matches the article thesis.
- The target platform dimensions are correct.
- SVG source and PNG export both exist.
- PNG dimensions match SVG dimensions.
- Text does not collide with panels, border rules, tags, or captions.
- The style follows the requested Open Design system rather than generic poster aesthetics.

## Common Failure Modes

- **Text model errors:** image generation creates wrong Chinese text. Fix by using SVG text.
- **Overcrowded square cover:** fix by recomposing the square version, not scaling the wide version.
- **Style drift:** reread `DESIGN.md` and reduce off-system colors, shadows, and fonts.
- **Bad export tool:** `qlmanage` creates wrong dimensions. Use `sips` or the bundled renderer.
- **Too many metaphors:** choose one visual idea and make the rest secondary labels.
