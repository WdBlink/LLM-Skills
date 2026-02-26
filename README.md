# LLM Skills

本项目包含一系列 LLM 相关的技能（Skills），每个技能位于独立的文件夹中。

## 技能列表

### Claude Code (claude-code/)

Claude Code 与 OpenClaw 的集成 skill，提供：
- Claude Code 官方文档查询
- 最佳实践和工作流指导
- 故障排除指南
- 直接 CLI 调用支持

**使用方法:**
```bash
# 文档查询
claude-code query quickstart
claude-code query best-practices

# CLI 调用 (需要在系统安装 Claude Code)
claude -p "解释项目结构"
```

更多详细信息请参考 [claude-code/SKILL.md](claude-code/SKILL.md)。

### MS-Qwen-VL (yolo-dataset-generator/)

基于 ModelScope Qwen3-VL 系列模型的多模态视觉识别技能，支持图片描述、OCR、视觉问答、目标检测、图表解析，并可将检测结果按 YOLO 数据集格式保存。

更多详细信息请参考 [yolo-dataset-generator/SKILL.md](yolo-dataset-generator/SKILL.md)。
