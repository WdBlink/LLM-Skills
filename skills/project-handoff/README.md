# Project Handoff

将项目资产与可访问的历史记录交叉核对，生成可单独交给另一位协作者或 AI 阅读的中文 `PROJECT_HANDOFF.md`。

## 使用

```text
$project-handoff /path/to/project，重点讨论当前方案是否满足验收；历史导出在 /path/to/export
```

项目路径省略时使用当前工作区；讨论重点、历史导出和输出位置可选。需要宿主具备文件读取、Python 3 和独立 Agent 调用能力。

宿主先确认材料归属并脱敏，再经规划、写作、独立审查和至多一次修订生成终稿。文档覆盖目标、方案、证据、当前状态、历史转折及待决策问题；历史缺失会明确披露。只读取项目，不修改代码、不运行项目实验、不自动上传。

详细流程与恢复方式见 [SKILL.md](SKILL.md)。本地程序检查：

```bash
python3 skills/project-handoff/scripts/check.py
```

程序检查不能替代真实独立 Agent 审查；缺少该能力时只保留取证包并说明限制。
