# 来源、阅读覆盖与本地适配

- Ivan Bercovich，*Good Benchmarks*，arXiv:2607.12217v1，2026-07-13。
- [HTML](https://arxiv.org/html/2607.12217v1)、[PDF](https://arxiv.org/pdf/2607.12217v1)、[版本记录](https://arxiv.org/abs/2607.12217v1)。
- 2026-10-15 通读摘要、引言、§1–10 与参考文献列表，并核对 7 页 PDF。正文无图表，无附录；未将参考文献列表等同于已读全部被引论文。
- 文体是实践性方法文章，经验主要来自 Terminal-Bench；不是对不同 benchmark 设计方案的随机比较，也不是作者发布的可执行软件或正式行业标准。

## 定位索引

本技能不复制全文。以下指向原章节，方便核对原则及语境；执行格式和案例为本地设计。

| 章节 | 本技能中的关联位置 |
|---|---|
| [§1](https://arxiv.org/html/2607.12217v1#S1)、[§2](https://arxiv.org/html/2607.12217v1#S2) | 目标、契约与验收 |
| [§3](https://arxiv.org/html/2607.12217v1#S3) | 难点归因；时延与协调案例 |
| [§4](https://arxiv.org/html/2607.12217v1#S4) | 输出语义与合法变化 |
| [§5](https://arxiv.org/html/2607.12217v1#S5) | 可解证据；故障定位案例 |
| [§6](https://arxiv.org/html/2607.12217v1#S6) | 环境和评分隔离 |
| [§7](https://arxiv.org/html/2607.12217v1#S7) | 验证矩阵；库存汇总案例 |
| [§8](https://arxiv.org/html/2607.12217v1#S8) | 试跑与归因记录 |
| [§9](https://arxiv.org/html/2607.12217v1#S9) | 专家审阅及争议证据 |
| [§10](https://arxiv.org/html/2607.12217v1#S10) | 按用途选择任务集合 |

## 不能直接外推的地方

本地适配包括：需求 ID 与测试映射、错误变体矩阵、状态名、统计聚合、数据划分和版本隔离。2026-10-18 按用户要求增加设计阶段默认准备可复用数据与脚本、复测调用保存入口及文件哈希检查。这些是用于实施和审计的工程约定，不冒充原文给出的 schema、数值门槛或实验结论。

原文偏重可程序判定的 Agent 任务。本地将主观任务拆为客观底线和人工评价、将随机性纳入重复设计、将现场系统分为受控与实时部分。这些模式有自己的验收证据，不能以“适配”名义声称完全满足原文的确定性评测要求。

原文对流程约束较严格。本地允许真实业务接口、安全/合规义务或明确作为测量目标的操作过程，但必须在题面中说明且提供依据；不能用作者实现偏好限制合法解。

两个段落、每批五次试跑、数百次验证等是原文语境中的经验指引，不是所有领域的固定预算。此技能按成本、风险和用途记录实际选择；降低规模也相应限制可宣称的验证范围。

技能生成与模型试用不能代替未来任务的真人领域审阅。原文的作者经验保留为方法依据，不作为本技能已普遍提升 benchmark 质量的证据。

## 跨场景选型的补充来源

2026-10-21，结合用户提供的 Agent benchmark 对照材料、对 DFM-Bench 的本会话审查，以及下列官方文档/关键代码核对，补充 [场景选型](benchmark-forms.md)。这是本地综合：任务承载形式与成功依据分开选择、四类样本及多种切分的区分、逐指标执行映射、评分可复算/验收有效/比较公平三类证据，不归为 Bercovich 原文的正式分类。

| 来源与阅读范围 | 用于本技能的有限结论 |
|---|---|
| [GAIA 官方 scorer.py](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)，核对数值、列表与文本规范化的判分分支 | 多步 Agent 解题仍可使用标准答案匹配；规范化和列表顺序属于具体协议，不能外推为所有任务的语义规则 |
| [SWE-bench 评测指南](https://www.swebench.com/SWE-bench/guides/evaluation/) 与 [grading.py](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py)，核对补丁执行、FAIL_TO_PASS/PASS_TO_PASS 和完整解决状态 | 用可执行行为与回归测试作为成功依据，不要求补丁文本相同；执行、解析和环境故障仍需核验 |
| AgentBench `v0.2` 的 [OS](https://github.com/THUDM/AgentBench/blob/v0.2/src/server/tasks/os_interaction/task.py)、[Mind2Web](https://github.com/THUDM/AgentBench/blob/v0.2/src/server/tasks/mind2web/task.py)、[LTP](https://github.com/THUDM/AgentBench/blob/v0.2/src/server/tasks/ltp/task.py) 关键评分分支 | 统一框架内可并存答案/脚本检查、参考动作指标和模型参与的语义判断；动作对齐、整程完成及公式的标签来源分别解释 |
| [DFM-Bench 提交 224712b](https://github.com/Gen-Verse/DFM-Bench/tree/224712b3407f543a077ff8b656167fb41f441963) 的评测指南、快照、评审、聚合、校准及测试模块 | 可以冻结运行与证据后评审；输入固定、裁判校准与系统可比并非同一个验收项；同一模型裁判的缓存不证明重新调用裁判稳定 |
| [DiscoveryRSI 任务示例版本 05d6044](https://huggingface.co/datasets/Gen-Verse/DiscoveryRSI-Hypothesis-Task-Examples/tree/05d6044bd649a0e25bb3b58b0fbb582adb66b327) 的数据卡、任务表、ZIP、绑定、验证器、fixture 和来源记录 | 编译任务包与运行记录评审是不同层；有文件哈希和合成正反例仍不等于具备真实可解证据，预处理输入需要实际文件或准备流程 |

这些读取不表示已经复现 GAIA、SWE-bench 或 AgentBench 的整套实验，也不把上述项目现有状态视为未来版本承诺。本会话对 DFM-Bench 的离线工程与合成样例检查不构成真人校准、真实科研可解或系统优越性证据。任务示例中观察到的数值关系漏检仅支持对相应验证器的具体发现，不推广为所有任务都存在同一错误。

参考动作可作为被测目标时的真值；一般结果任务不要求唯一轨迹。统计判据允许合理随机性；rubric 保留主观判断属性。这些适配与原文偏重确定性验收的边界一并保留，不把任何一种形式命名为所有 Agent benchmark 的最佳形态。
