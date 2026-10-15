# Good Benchmarks

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

依据 Ivan Bercovich 的 [*Good Benchmarks*, arXiv:2607.12217v1](https://arxiv.org/html/2607.12217v1)，将真实任务、可解性、评分器正确性、环境稳定性与失败诊断组织为可执行的 benchmark 设计流程。适用于新建、实现或审计评测任务、任务集和评分器。

该 Skill 交付任务契约、要求与检查的映射、参考解证据、评分器正反例、试跑及审阅记录。低通过率需要结合实际失败原因解释，不能单独证明任务有效。

## 安装

需要 Node.js，执行以下命令安装到 Codex：

```bash
npx -y skills add WdBlink/LLM-Skills --skill good-benchmarks --agent codex --global --yes
```

也可在本仓库根目录手动安装：

```bash
mkdir -p ~/.codex/skills/good-benchmarks
rsync -a skills/good-benchmarks/ ~/.codex/skills/good-benchmarks/
```

技能本身是纯文本，无额外运行依赖；具体 benchmark 的依赖和资源按项目定义。

## 使用

提供项目资料、目标能力、真实工作场景，以及已有题面、数据、评分器和日志中可获得的部分。例如：

```text
用 $good-benchmarks 为当前项目设计 benchmark。
目标是比较两个系统能否准确完成设备占用时间汇总。
请交付任务契约、可解性依据、评分器验证方案与正反例；
如果需要实现，复用当前项目的测试设施，并报告实际运行结果。
```

审计已有评测时，可以要求：

```text
用 $good-benchmarks 审计这个任务集和评分器。
检查隐藏要求、参考解特权信息、合法解误拒、错误解误收和环境故障归因，
给出可定位到文件或运行轨迹的发现与修复建议。
```

若希望跨项目默认采用，可在自己的全局 `AGENTS.md` 中加入：

> 新建、重设计或审计 benchmark、评测任务集及评分器时，默认使用 `$good-benchmarks`，按任务契约、可解性、评分正确性和实际证据完成设计与验证。用户明确的目标和约束优先；仅运行既有 benchmark 时不擅自重设协议。

安装技能文件不会自动修改全局规则。

## 内容

| 文件 | 用途 |
|---|---|
| [SKILL.md](SKILL.md) | 八步工作流程、决策边界、验收与停止条件 |
| [任务契约](references/task-contract.md) | 输入输出、权限资源、要求映射、验证矩阵和故障分类 |
| [六组案例](references/examples.md) | 数据汇总、故障定位、主观写作、时延与协调、评分隔离、随机实验的正反例 |
| [来源与适配](references/source-notes.md) | 原论文章节索引及本地扩展的适用边界 |
| [Codex 配置](agents/openai.yaml) | 展示名称与默认调用提示 |

## 验证与适用范围

本地已进行两个构造场景的行为试用。其中设备占用汇总场景返回的评分函数，经独立执行通过 2,519 项判分检查，包括合法结果、错误变体、边界及重复判分。该结果仅支持有限场景中的技能执行与评分函数行为，不代表真实业务任务的难度、系统排名或专家认证。

原文主要讨论可程序验证的 Agent 任务。统计、主观及实时评价的处理明确标为本地适配；原文经验指引中的运行次数不作为跨领域固定门槛。具体任务仍须建立可解证据、核验评分器，并如实记录未运行和待人工审阅项目。

## 许可证与来源

本 Skill 的原创工作流、文档和构造示例按 [MIT](LICENSE) 发布。论文的权利属于原作者，本仓库仅提供引用、方法提炼和本地实施设计，不附带论文全文；MIT 许可不适用于论文原文。
