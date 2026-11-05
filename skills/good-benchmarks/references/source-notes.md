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

## 测量主张与统计解释的补充来源

2026-10-29 通读 Giovanni D'Antonio 的 [The science of benchmarking: from Zero to Hero, Part I](https://www.giovannidantonio.com/blog/science-of-benchmarking-part-1) 正文，并核对下列主源的相关部分。该文是方法入门文章；测量主张、覆盖与区分、参照及聚合解释用于补充现有 [任务契约](task-contract.md)，不替代可解性与评分器验证。

| 核对来源与范围 | 适配依据与限制 |
|---|---|
| [Measuring what Matters](https://papers.neurips.cc/paper_files/paper/2025/file/1967e0fc3aa6cbbace562f5cb8e3954e-Paper-Datasets_and_Benchmarks_Track.pdf)，摘要、引言及构念效度定义 | 明确能力、任务、指标与结论的关系；本文的综述结论不证明本技能已具备效度 |
| [BetterBench](https://arxiv.org/html/2411.12990v1)，统计报告、基线参照及复现相关部分 | 参照用于解释分数，须说明人群或方法条件；不把其全部评分条目变成每个任务的强制流程 |
| [NIST：二项比例区间](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)，Wilson 与正态近似说明 | 小样本及比例接近边界时不能机械使用 Wald 近似；具体区间仍依赖抽样与独立性假设 |
| [Agarwal 等，Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/html/2108.13264v4)，§2、§4.1 | 少量训练运行、分层重采样与差值解释的边界；不移植特定任务的种子数、聚合量或显著性阈值作为统一要求 |
| [SimpleQA](https://arxiv.org/abs/2411.04368)，摘要；[SimpleQA Verified](https://arxiv.org/html/2509.07968v1)，摘要及 §2 筛选流程 | 原任务测短答事实性；Verified 既有难度筛选，也有去重、分布调整、纠错和裁判改进，不能将升级效果全部归为加难 |

固定集合描述与总体推断的区分、基线的证据责任、配对与重复层次的登记，以及 [案例](examples.md) 中的中止、弃答和相关样本反例，均为本地综合，不是上述来源发布的统一模板。Part I 只概述统计不确定性，将更深入的比较方法留待后续；本技能的统计边界另据所列主源核对，不归为 Part I 已展开的规范。这些读取不包含相关 benchmark 的实验复现，也不授权新增运行或改变既有协议。

## 按需求检索与复用的本地约定

2026-11-05，按用户要求将现有 benchmark 的检索、获取、需求匹配和缺口分析放到方案设计之前，新增 [检索与复用](reuse-existing.md)。需求画像、一手材料追溯、逐需求覆盖判定、原样复用/适配/组合/新建决策、上游版本与比较边界，均为本地工程约定，不冒称为上述论文规定的步骤。

本技能中的公开机制例证用于解释评分方式，不是固定推荐名单。实际使用时须按当前需求检索并核实相关版本，查看真实任务、数据、评分实现及协议；不能把方法文档当成已经完成某次候选调查或真实复跑的证据。

## 四项结果验收的本地约定

2026-11-05，按用户要求增加 [结果验收与返工](result-acceptance.md)：数据集实际形成、数据对应入口实际可运行、得分可复算、相关现有方法区分度有实测；缺项或未达标返回构建步骤，不能自报完成。验收合同、主要方法对及阈值预先冻结，脚本固化可计算检查；这些是本地工程机制，不归为原论文公布的统一验收协议。

新增程序只适配固定集合、逐单位配对且能用 mean/sum 聚合的指标，描述性分差不等于总体效应或稳定排名。构造测试验证完整链路、缺陷拒绝和局部修复，不构成真实方法区分证据；评分/比较语义与人类专家审阅仍需真实材料，不因程序通过升级。
