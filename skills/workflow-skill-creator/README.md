# Workflow Skill Creator

![Version](https://img.shields.io/badge/version-0.3.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

按脚本优先、模型按需的方式创建或重构 Skill：固化重复操作、参数化任务差异，默认由当前宿主完成必要的结构化判断，并验证交付质量与实际成本。

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
用 $workflow-skill-creator 把重复的资料选段操作做成可复用 Skill，
保留原文和限制条件，提供脚本入口、必要的宿主判断与失败恢复。
```

先确定目标 Skill 的名称、路径、输入输出和完成条件，再复用已有工具与脚本。短任务可以只提供入口说明；需要语义判断时，默认采用 `prepare → 当前宿主判断 → validate/apply`。批量任务、无人值守或明确需要独立审查时，再考虑独立后端或子 Agent。

## 内容与要求

- `SKILL.md`：使用合同、设计步骤与完成条件。
- `scripts/judgment_handoff.py`：准备段落快照、校验宿主判断并按来源顺序复制原文。
- `tests/test_judgment_handoff.py`：判断绑定、输入保护、冲突拒绝与幂等检查。
- `references/`：设计、判断接口、执行边界与要求闭环审计、prompt、恢复和验收说明。
- `assets/evidence-brief/`：可选高级示例，含程序、独立角色 prompt 和明确标注的虚构输入。
- `agents/openai.yaml`：Codex 展示与调用配置。

阅读与设计不依赖外部服务。判断接口仅使用 Python 3.8+ 标准库，不主动调用模型或网络。脚本的 `needs_judgment` 将控制权交还当前宿主；可选高级示例的独立模型阶段需要宿主支持真实子 Agent 调用。

## 判断接口

输入是 UTF-8 文本，空行分段。从仓库根目录准备请求：

```bash
python3 skills/workflow-skill-creator/scripts/judgment_handoff.py prepare \
  --input /绝对路径/article.txt \
  --goal "找出明确限制结论适用范围的段落" \
  --out /绝对路径/request.json
```

当前宿主读取请求，按真实 `request_id` 与 `unit_id` 写入 `decision.json`，再验证并应用：

```bash
python3 skills/workflow-skill-creator/scripts/judgment_handoff.py apply \
  --request /绝对路径/request.json \
  --decision /绝对路径/decision.json \
  --out /绝对路径/result.json
```

判断格式、缺材料处理、字符预算与覆盖规则见 [判断接口合同](references/judgment-interface.md)。`applied` 只表示选段已导出；语义正确性仍需核对。

## 验证

```bash
python3 -B -m unittest discover -s skills/workflow-skill-creator/tests -v
python3 skills/workflow-skill-creator/assets/evidence-brief/scripts/check.py
```

这些检查使用确定性数据，覆盖正常选段、原文保护、旧判断拒绝、无效输入、冲突输出与高级示例的受限修订及失败终态。它们不代表实际模型执行或生成质量评测；真实宿主判断与实际成本按 [验收方法](references/acceptance-cases.md) 另行核验。

## 可选评测与工程宿主

[Promptfoo 接线](references/promptfoo.md) 复用既有判断合同。合成回放只检验裁判与程序，不代表模型效果；真实输出和运行配置单独留证。P-Stack 是 poteto 在 [cursor/plugins 的 pstack](https://github.com/cursor/plugins/tree/main/pstack)，按已安装的 Codex adapter 解析模型与委派能力，不把上游模型名或本机绝对路径固化进领域技能。
