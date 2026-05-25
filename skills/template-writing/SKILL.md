---
name: template-writing
description: Draft and fill strict-format DOCX templates from reference materials such as technical proposals, contracts, technical agreements, test protocols, meeting notes, and project documents. Use when the output template format is fixed, while the input sources are contextual references rather than one-to-one form fields.
---

# Template Writing

Use this skill when the user needs a completed strict-format DOCX report or
document template, and the inputs are reference materials rather than a simple
field-value form. It is especially useful for acceptance test reports and
project acceptance materials.

The central job is controlled report drafting:

- Preserve the template structure, styles, tables, headers, footers, numbering,
  and signature areas.
- When the user identifies a file as the template, treat its existing narrative
  text as provisional template content, not as final source truth.
- Keep key commitments consistent with the sources, especially contract scope,
  acceptance criteria, test indicators, quantities, dates, deliverables, and
  party names.
- Draft missing narrative sections from the references, but mark unsupported
  facts instead of inventing them.
- Produce a companion evidence/field plan so important content can be audited.

## When To Use

Use this skill for:

- 验收测试报告
- 项目验收报告
- 合同/技术协议约定内容转验收材料
- 技术方案、测试方案、会议纪要等参考资料汇总成固定模板 DOCX

Do not treat the task as a direct label-value form unless the template is
explicitly built that way. Most acceptance reports require synthesis and
section-level drafting.

## Template Interpretation

When the user says a file is the template, the template defines form, not final
content.

Preserve:

- Section outline, numbering, table layout, table header structure, page
  geometry, fonts, headers, footers, signature areas, and other formatting.
- Template-local writing instructions such as `【要求】`, `【示例】`, notes, and
  section hints as drafting constraints.

Rewrite from reference materials:

- Existing body paragraphs, examples, default wording, instructional prose,
  non-empty placeholder-like text, `XXX` text, blank slots, and table body cells.
- Any old project/product names, old conclusions, old indicators, old dates, or
  boilerplate that appear in the template.

Do not decide editability only from blank cells or `XXX`. If the user names a
document as the template, assume all non-structural content is a rewrite
candidate unless there is a clear reason to preserve it.

## Workflow

### 1. Set Up A Run Directory

Create a working directory beside the output or under a temporary project run:

```text
acceptance-report-run/
├── sources/
├── template-scan.json
├── fact-ledger.md
├── report-plan.json
├── missing-and-uncertain.md
└── output.docx
```

Copy or reference source files without modifying them.

### 2. Scan The Template

Run:

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/scan_template.py" \
  --template /path/to/template.docx \
  --output acceptance-report-run/template-scan.json
```

If the skill is installed under Claude Code instead of Codex, set
`SKILL_ROOT="$HOME/.claude/skills/template-writing"`.

If working from the repository checkout:

```bash
python3 skills/template-writing/scripts/scan_template.py \
  --template /path/to/template.docx \
  --output acceptance-report-run/template-scan.json
```

Use the scan to identify:

- Stable replacement targets: paragraph indexes, table cell coordinates,
  existing placeholders, blank content regions, existing body text, table body
  cells, and fixed structural text that must not be changed.
- `rewrite_candidates`, which lists the likely content regions to rewrite under
  the user-declared-template rule. Treat this as a planning aid, not as a final
  automatic edit list.
- Template-local drafting guidance from `guidance_blocks`, especially
  `【要求】`, `【示例】`, `填写要求`, `填写说明`, and `注：`.

When a section contains requirements or examples, treat them as part of the
template contract. They outrank generic writing style. Remove or replace only
the instructional text that the template clearly expects the final report to
replace; do not leave `【要求】` or `【示例】` blocks in the final deliverable
unless the template explicitly asks to retain them.

### 3. Build A Fact Ledger

Read all reference materials and create `fact-ledger.md`.

Use this structure:

```markdown
# Fact Ledger

## Source Inventory
| ID | File | Type | Notes |
|---|---|---|---|

## Key Facts
| Fact ID | Category | Value | Source | Confidence | Notes |
|---|---|---|---|---|---|

## Consistency Constraints
| Constraint | Required Value | Source | Report Locations |
|---|---|---|---|

## Missing Or Unsupported
| Item | Why It Is Missing | Suggested Handling |
|---|---|---|
```

Key categories:

- 项目基本信息
- 合同约定内容
- 技术协议约定
- 验收范围
- 验收依据
- 测试环境
- 测试方法
- 验收指标
- 实测结果
- 交付物
- 结论与整改项
- 签署/日期/单位信息

Facts that affect acceptance conclusions must cite source files or exact
snippets. If sources conflict, do not silently choose one; record the conflict
and ask the user if it blocks the report.

### 4. Draft By Template Section

Write report content against the template's actual sections, not against the
source document order.

Rules:

- Start from `rewrite_candidates`, not only from blank fields or placeholders.
  The normal task is to replace all non-structural template content with content
  grounded in the reference materials.
- For each section, first read the nearby template requirements, examples, and
  notes. Match the required scope, granularity, order, and wording pattern
  before drafting from external references.
- Contract scope, acceptance indicators, quantities, deliverable names, party
  names, and dates must match the fact ledger.
- Narrative sections may be drafted from the references when the template asks
  for explanation, process, basis, test method, or conclusion.
- If a required section has no source support, use conservative wording such as
  `原始资料未提供，需补充确认` only when leaving it blank would be misleading.
- Do not add new sections unless the template has an obvious editable region for
  them.
- Keep language formal, specific, and low on generic filler.

### 5. Create `report-plan.json`

Use replacement targets from `template-scan.json`. Example:

```json
{
  "replacements": [
    {
      "target": "paragraph",
      "index": 12,
      "text": "本次验收依据合同、技术协议及项目实施过程中形成的测试记录开展。"
    },
    {
      "target": "paragraph_block",
      "start": 73,
      "end": 79,
      "text": "本报告适用于垂直起降无人机平台及配套设备验收测试。"
    },
    {
      "target": "table_cell",
      "table": 1,
      "row": 2,
      "col": 3,
      "text": "符合合同约定的验收指标"
    },
    {
      "target": "text_match",
      "match": "{{项目名称}}",
      "text": "..."
    }
  ],
  "required_terms": [
    "合同约定的关键指标或交付物名称"
  ]
}
```

Prefer stable placeholders when the template has them, but do not limit the plan
to placeholders. For templates without placeholders, use paragraph indexes,
paragraph blocks, and table cell coordinates from the scan.
If a paragraph is an example or instruction block, replace the whole block with
final report content only after preserving its intent in the drafted text.

### 6. Fill The DOCX

Run:

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/fill_docx_template.py" \
  --template /path/to/template.docx \
  --plan acceptance-report-run/report-plan.json \
  --output acceptance-report-run/output.docx
```

The script replaces only the selected paragraphs, cells, or matched placeholder
text. It does not decide report content.

### 7. Validate

Run:

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/validate_output.py" \
  --template /path/to/template.docx \
  --output acceptance-report-run/output.docx \
  --plan acceptance-report-run/report-plan.json
```

If working from the repository checkout, use the same script path under
`skills/template-writing/scripts/`.

Validation checks structural readability and obvious consistency requirements.
It does not replace manual review. After validation, review:

- Fixed headings and table structure stayed in place.
- Required terms from `report-plan.json` appear in the output.
- All key conclusions trace back to `fact-ledger.md`.
- Missing items are documented in `missing-and-uncertain.md`.

## Borrowed From `fill-form`

The LovStudio `fill-form` skill is useful for template scanning, field
normalization, table-cell targeting, CJK text handling, and JSON-driven DOCX
updates. For this skill, use those ideas only as low-level mechanics.

Do not copy its core assumption that each template label maps directly to one
input value. Acceptance reports usually require source synthesis, consistency
checking, and section-level drafting.

## Delivery

Return:

- Output DOCX path.
- `fact-ledger.md` path.
- `missing-and-uncertain.md` path, if any.
- Validation result summary.

Do not paste the full report unless the user asks.
