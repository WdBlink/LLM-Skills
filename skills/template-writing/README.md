# Template Writing

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Private](https://img.shields.io/badge/Private-blueviolet)

基于技术方案、合同、技术协议、测试记录等参考资料，撰写并填充固定格式的 DOCX 文档模板，重点支持中文验收测试报告和项目验收材料。

## 安装到 Codex

```bash
mkdir -p ~/.codex/skills/template-writing
rsync -a skills/template-writing/ ~/.codex/skills/template-writing/
python3 -m pip install python-docx
```

## 安装到 Claude Code

```bash
mkdir -p ~/.claude/skills/template-writing
rsync -a skills/template-writing/ ~/.claude/skills/template-writing/
python3 -m pip install python-docx
```

## 使用

先扫描模板：

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/scan_template.py" \
  --template /path/to/template.docx \
  --output run/template-scan.json
```

再由 Agent 读取参考资料，生成：

- `fact-ledger.md`
- `missing-and-uncertain.md`
- `report-plan.json`
- 必要时生成 `.drawio` 图源文件和导出的 PNG 图片

填充模板：

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/fill_docx_template.py" \
  --template /path/to/template.docx \
  --plan run/report-plan.json \
  --output run/output.docx
```

校验输出：

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/validate_output.py" \
  --template /path/to/template.docx \
  --output run/output.docx \
  --plan run/report-plan.json \
  --check-placeholders
```

## 工作原则

- 输入资料是参考资料，不默认是一对一字段。
- 用户指定某文件是模板时，模板定义格式和结构；模板中的正文、示例、默认文字和表格正文通常都需要按参考资料重写。
- 始终以原始模板 DOCX 为底稿。不要用上一轮生成稿当模板底稿，除非用户明确要求；不要重建短文档、签署页、页眉页脚、目录域或嵌套表格。
- 模板里已经写好的句子默认也是示例，例如 `××软件驻留/部署在××，主要功能为××，其性能指标为××。`，不能只替换空白或 `××`。
- 图题/表题保留编号和样式，但题名文字如 `图 1 测试环境拓扑图` 也要按实际参考资料判断是否重写。
- 模板要求绘制结构框图、流程图、组成图、拓扑图、架构图、部署图、示意图等时，要按参考资料生成图，而不是用文字说明替代；保留 `.drawio` 源文件，导出图片后插入 DOCX 对应位置。图形必须适配 Word 可用版心，不能横向无限延展。
- 模板小节中的 `【要求】`、`【示例】`、填写说明和注释是优先参照的写作约束。
- 合同约定、验收指标、交付物、数量、日期、单位名称等关键事实必须和来源保持一致。
- 模板格式是固定边界，脚本只替换指定段落、表格单元格、占位符，或插入指定图像。
- 交付前检查题号。不同图题/表题不能重复使用同一个编号，除非模板明确在附录或新章重新编号。
- 无来源支撑的内容要记录为缺失或不确定，不直接编造。

## 图形输出

先写图规格：

```json
{
  "title": "测试环境拓扑图",
  "layout": "horizontal",
  "max_width": 850,
  "nodes": [
    {"id": "terminal", "label": "测试终端"},
    {"id": "system", "label": "被测系统"}
  ],
  "edges": [
    {"source": "terminal", "target": "system", "label": "测试访问"}
  ]
}
```

生成 draw.io 源文件：

```bash
SKILL_ROOT="${SKILL_ROOT:-$HOME/.codex/skills/template-writing}"
python3 "$SKILL_ROOT/scripts/create_drawio_diagram.py" \
  --spec run/diagrams/test-env.json \
  --output run/diagrams/test-env.drawio
```

用 draw.io CLI 导出图片后，在 `report-plan.json` 中使用：

```json
{
  "target": "image_before_paragraph",
  "index": 42,
  "image_path": "run/diagrams/test-env.png",
  "page_width_ratio": 0.96
}
```

`fill_docx_template.py` 默认会把插入图片夹到 Word 页面正文宽度内。不要设置超过版心宽度的 `width_inches`；普通报告图也不要使用 `allow_overflow`。
