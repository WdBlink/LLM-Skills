---
name: pv-forecast-pptx
description: "Create professional research presentations for photovoltaic/energy forecasting projects. Use when: (1) creating presentations about solar power prediction, photovoltaic field forecasting, or radiation calculation, (2) building 课题汇报 or 项目申报 style research proposals, (3) making slides with technical architecture diagrams and data callouts. Applies Anthropic brand styling (colors: #141413/#faf9f5/#d97757/#6a9bcc/#788c5d, fonts: Poppins/Lora) and follows the structured research proposal format with numbered sections, large stat callouts, and hierarchical content."
license: Proprietary
---

# Photovoltaic Forecast Research Presentation Skill

Create professional PowerPoint presentations for photovoltaic forecasting research. Combines Anthropic brand aesthetics with Chinese research proposal conventions.

## Reference Template

Template: `resource/【参考】课题需提供--20250704.pptx`

**Template Structure (4 slides):**
1. 关键技术1: 光伏场站云团结构临近精准预报和地表太阳辐射快速计算
2. 服务平台建设 (with 01-04 numbered sections)
3. 示范地选择 (Shandong Province with large stat callouts)
4. 综合应用示范内容 (metrics cards: 1套/3个/≥15个/≥80万kW)

## Brand Styling (Anthropic)

### Colors
| Name | Hex | Usage |
|------|-----|-------|
| Dark | `#141413` | Primary text, dark backgrounds |
| Light | `#faf9f5` | Light backgrounds |
| Mid Gray | `#b0aea5` | Secondary text, slide numbers |
| Light Gray | `#e8e6dc` | Subtle backgrounds |
| Orange | `#d97757` | Primary accent, key numbers |
| Blue | `#6a9bcc` | Secondary accent, info |
| Green | `#788c5d` | Tertiary accent, positive |

### Typography
- **Headings (24pt+)**: Poppins (fallback: Arial)
- **Body text**: Lora (fallback: Georgia)
- **Slide numbers**: 10-12pt, Mid Gray

### RGBColor Values
```python
DARK = RGBColor(0x14, 0x14, 0x13)
LIGHT = RGBColor(0xfa, 0xf9, 0xf5)
MID_GRAY = RGBColor(0xb0, 0xae, 0xa5)
LIGHT_GRAY = RGBColor(0xe8, 0xe6, 0xdc)
ORANGE = RGBColor(0xd9, 0x77, 0x57)
BLUE = RGBColor(0x6a, 0x9b, 0xcc)
GREEN = RGBColor(0x78, 0x8c, 0x5d)
```

## Slide Templates

### T1: Title Slide (标题页)
- Dark background (`#141413`)
- Large centered title (36-44pt, Poppins Bold, Light color)
- Subtitle below (18-20pt, Lora, Mid Gray)
- Bottom right: slide number (10pt, Mid Gray)

### T2: Technology Overview (技术概述)
- Light background with left-aligned title
- Top: Title (28-32pt, Poppins Bold)
- Upper section: "难点" callout box (Orange left border, Light Gray background)
- Lower section: 3-column or icon-row layout for key components

### T3: Platform/Service Section (平台服务)
- Numbered sections in Orange circles: `01`, `02`, `03`, `04`
- Each section: title + bullet points
- Two-column or card-based layout
- Module names in Bold, details in regular Lora

### T4: Statistics Slide (数据展示)
- Large stat callouts (60-72pt, ORANGE)
- Small labels below (12-14pt, Mid Gray)
- Units in regular text (14pt)
- 2x2 or 1x3 grid layout

### T5: Summary Card Slide (汇总卡片)
- Cards with Light Gray background
- Large number + unit + label per card
- Consistent spacing between cards

## Design Patterns

### Layout Rules
- **Margins**: 0.5" minimum all sides
- **Title position**: 0.4" from top
- **Content spacing**: 0.3-0.5" between blocks
- **Numbered circles**: 0.5" diameter, ORANGE fill, white text

### Typography Scale
| Element | Size | Font | Style |
|---------|------|------|-------|
| Slide title | 32-36pt | Poppins | Bold |
| Section header | 24-28pt | Poppins | Bold |
| Card title | 18-20pt | Poppins | Bold |
| Body text | 14-16pt | Lora | Regular |
| Stat number | 60-72pt | Poppins | Bold |
| Stat label | 12-14pt | Lora | Regular |
| Slide number | 10-12pt | Lora | Regular |

### Bullet Formatting
```python
{
    "text": "bullet content",
    "bullet": True,
    "level": 0
}
# Left-align all bullets (no center alignment)

## Content Structure for Research Proposals

### Section 1: 技术难点与攻关方向
- 难点标题 + 描述
- 3-4个关键技术方向
- 每个方向: 标题 + 子项列表

### Section 2: 服务平台架构
- 平台名称 + 一句话描述
- 4个模块 (01-04)
- 每个模块: 名称 + 功能列表

### Section 3: 示范地选择
- 地点 + 入选理由
- 3-4个关键指标 (大数字展示)
- 特色说明列表

### Section 4: 应用示范规划
- 指标卡片 (规模/效能/推广)
- 具体实施内容列表

## Workflow

1. **Analyze content** → Determine slide count and template types
2. **Design layout** → Map content to slide templates
3. **Apply brand** → Use Anthropic colors/fonts consistently
4. **Generate PPTX** → Use html2pptx or pptxgenjs
5. **QA check** → Verify text cutoff, alignment, contrast

## Common Issues

- Text overflow: Reduce font size or wrap to multiple lines
- Low contrast: Ensure text is Dark on Light backgrounds
- Misaligned bullets: Left-align, consistent indentation
- Number cutoff: Ensure stat numbers fit within card bounds
