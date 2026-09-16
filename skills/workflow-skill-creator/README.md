# Workflow Skill Creator

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

按十步法创建或重构可执行的工作流 Skill，将程序、模型与宿主的分工落实为入口说明、脚本、阶段 prompt、文件交接和运行验证。

## 安装

已安装 Node.js 时，一条命令安装到 Codex 全局技能目录：

```bash
npx -y skills add WdBlink/LLM-Skills --skill workflow-skill-creator --agent codex --global --yes
```

安装参数见 [Skills CLI 文档](https://github.com/vercel-labs/skills#options)。

### 手动安装

在本仓库根目录执行：

```bash
mkdir -p ~/.codex/skills/workflow-skill-creator
rsync -a skills/workflow-skill-creator/ ~/.codex/skills/workflow-skill-creator/
```

## 使用

```text
用 $workflow-skill-creator 创建一个研究任务上下文管理 Skill，
将证据整理、写作、独立审查和受限修订接成可执行流程。
```

调用后默认读取十步法，再按任务规模选择程序骨架、独立模型阶段及失败恢复方式。生成的 Skill 直接说明用途和执行方法。

## 内容与要求

- `SKILL.md`：入口、设计步骤与交付要求。
- `references/`：十步法、宿主派发、prompt 加载、设计与验收说明。
- `assets/evidence-brief/`：可改造的简报样例，含程序、三个阶段 prompt 和虚构输入。
- `agents/openai.yaml`：Codex 展示与调用配置。

阅读与设计不依赖外部服务。运行随附样例需要 Python 3；样例的模型阶段需要宿主支持独立子 Agent 调用。脚本负责准备、校验、合并和导航，不会自行调用模型。

## 验证

```bash
python3 skills/workflow-skill-creator/assets/evidence-brief/scripts/check.py
```

该检查使用确定性测试数据，覆盖正常完成、受限修订、字段保护与失败终态；不代表实际模型执行或生成质量评测。真实模型阶段按 `references/host-dispatch.md` 执行和验收。
