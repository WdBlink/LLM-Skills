# Research Context · 研究任务上下文

Version: 0.2.0 · 原创技能与模板：MIT

将论文、实验、讨论和阶段认识组织成下一会话能继续使用的研究上下文。方法来自 OpenMMLab 课程课件，补充 Agent 所需的证据追溯与恢复规则；[来源和阅读限制](references/source-notes.md)。

## 使用

```text
使用 research-context，为 /绝对路径/研究项目 整理研究上下文，材料是……
使用 research-context，把这次实验结果更新到既有课题，保留失败尝试和结论限制。
使用 research-context，读取 /绝对路径/研究项目/research-context/context.md，恢复上下文，告诉我下一步。
使用 research-context，检查当前判断是否都有证据，先不要改文件。
```

建立/更新采用 ResearchStudio 式混合编排：程序冻结输入 → writer → 程序合成 → 独立 critic → 受限修订或 apply。新项目维护 `context.md`、已审查阶段记录及可恢复的运行目录；已有入口保留完整旧快照，获授权整理后原地更新。恢复和检查默认只读，不自动启动研究、创建自动任务或上传材料。

## 安装

从本仓库根目录执行：

```bash
mkdir -p ~/.codex/skills/research-context
rsync -a skills/research-context/ ~/.codex/skills/research-context/
```

入口：[SKILL.md](SKILL.md)，详细 [命令与数据合同](references/protocol.md)。只依赖 Python 3 标准库及宿主真实独立 Agent 能力，不新增 API Key 或工作流服务。程序不调用模型；宿主读 next 的动作包后创建、等待子 Agent。

```bash
python3 ~/.codex/skills/research-context/scripts/check.py
```

检查涵盖输入冻结、审查绑定、越权补丁、冲突拒绝与中断恢复。JSON 合格不等于科研结论正确；外部写者须串行协调，不提供数据库事务或 OS 沙箱。
