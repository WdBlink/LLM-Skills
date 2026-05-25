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
  --plan run/report-plan.json
```

## 工作原则

- 输入资料是参考资料，不默认是一对一字段。
- 用户指定某文件是模板时，模板定义格式和结构；模板中的正文、示例、默认文字和表格正文通常都需要按参考资料重写。
- 模板小节中的 `【要求】`、`【示例】`、填写说明和注释是优先参照的写作约束。
- 合同约定、验收指标、交付物、数量、日期、单位名称等关键事实必须和来源保持一致。
- 模板格式是固定边界，脚本只替换指定段落、表格单元格或占位符。
- 无来源支撑的内容要记录为缺失或不确定，不直接编造。
