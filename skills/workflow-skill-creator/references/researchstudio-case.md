# ResearchStudio：方法来源与迁移边界

需要独立审查或复杂编排时选读，或在核对 ResearchStudio 的来源与实现边界时查阅。少量语义判断默认采用 [当前宿主判断接力](judgment-interface.md)，普通顺序和少量分支用现有工具与普通代码；本页的多 Agent、导航与恢复机制均按实际需求选择。

证据版本：microsoft/ResearchStudio，commit `0597891df1a153b8e4cbdc8c1c685f43a0a6abcf`（2026-09-06），核验日 2026-09-12。本文件是原创概括，不包含上游实现的复制。使用更新版本时需核对变化，不能把本文件日期当成最新保证。

需要借用这些机制时，具体脚本、派发和 prompt 接线见 [实现拆解](implementation-walkthrough.md)、[宿主派发](host-dispatch.md)、[prompt 加载](prompt-loading.md)。本页保留系统范围与反例，不替代对应实现说明，也不是所有 Skill 的必读步骤。

## 两种架构，不是一个万能模板

**IdeaSpark**：研究方向 → 文献 → 瓶颈 → K=1 候选 → coherence/collision/critique → 必要修订重试 → idea card。
它的 `next_step.py` 读磁盘，向宿主输出命令或 LLM 阶段说明；宿主执行、写回文件、再次导航。
**Reel**：论文 PDF → paper2assets → poster/video/blog → paper2reel。各技能按合同消费共享包，没有统一套用 IdeaSpark 的 navigator。
中间的实验与论文写作没有被两个套件贯通。Agent runtime 在 Claude Code / Codex。

[根 README](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/README.md)、[IdeaSpark](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/SKILL.md)、[Reel](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Reel/README.md)

## 知识为什么有用

IdeaSpark 消费语料归纳出的 15 主模式、31 子模式及成功/拒稿教训；这些是在线判断的知识资产，在线运行不重新做语料聚类。模式频率不作为成功概率；主/子模式引用关系可机械检查，战术是否真的成立仍需语义审查。

迁移时寻找“这个领域如何判断好坏”的真实依据，不照抄论文 novelty、科研预算或分类学。ResearchStudio 的 workflow engineering 和领域知识共同构成价值；单有状态机不能补上评估标准。

[Idea README](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/README.md)、[design notes](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/references/design-notes.md)

## 机制 → 解决的问题 → 有条件地借用

| 机制 | 原问题 | 迁移条件 |
| --- | --- | --- |
| 用户原话落盘、named papers 解析 | 用户意图在摘要中丢失 | 多阶段任务持续依赖原始限制 |
| 相关性分区、host recall 验证、全文缓存 | 召回噪声与关键词盲区 | 检索是决策依据；保留未覆盖状态 |
| fresh-context coherence | 作者会给流畅的逻辑错误放行 | 能做执行检查或存在明显自评偏差 |
| naive baseline 独立构造 | 用稻草人证明贡献 | 新方案的价值取决于简单方案是否已足够 |
| signature/alias 双通道 | 相同机制在另一社区换名字 | 术语差异会影响检索；时间窗口按领域选 |
| critic 与 reviser 分开 | 修改者篡改裁决或重写承诺 | 存在实质修改权边界 |
| patch-only + merger | 大 JSON 重述、字段漂移 | 大结构或受保护字段；小文案不必 |
| skeleton/fill/assemble | LLM 重抄机械字段 | 大量内容是 join/group/lookup 等机械工作 |
| lesson-driven bounded retry | 无信息的重生成 | 确有再次生成价值；上限按成本设计 |
| paper2assets 共享语义包 | 多输出重复解析和事实分叉 | 多消费者使用共同事实或昂贵材料 |
| 测量—修订—复测 | 视觉产物存在但质量不足 | 有具体可测缺陷；仍需实际视觉检查 |
| 单次解析公共参数并检查交接 | 子技能默认值相互冲突 | 多组件共同决定一个用户可见行为 |

## 必须知道的反例与限制

1. **navigator 不是强终态验证器。** 上游遇到三个卡片文件存在就 DONE；2026-09-11 以三个空文件且无前序证据复现了这一行为。最后的 validate 还允许修复两次后带 caveat 渲染。迁移时按自己的交付要求定义终态。
2. **文档隔离不等于写权限沙箱。** merger/assembler 可以拒绝某次操作的字段变更，不能证明宿主无法绕过它直接写文件。
3. **实际协议是文本，不是统一 JSON NextAction。** 上游 next 接受 --dir/--query，terminal rc=0；--input/--run-id、rc=20、stage-terminal.json 是其他设计，不能当上游接口。
4. **Python 也有 LLM 命令桥接。** 辅助 intent/pattern-summary 支持 NOVELTY_LLM_CLASSIFY_FAST_CMD；主要推理仍由宿主承接。不要写成“从来不调用 LLM”。
5. **kill-switch 有受限例外。** compute_budget 保持一致；falsification_prediction 有审计授权的单次 rewrite 与重新审计，之后新承诺保持一致。
6. **科研语义判断仍有不确定性。** JSON、引用和字段检查不能证明新颖性或实证有效；小实例执行不能替代真实实验。
7. **Reel 的包会刷新。** build_package 逐文件原子替换 JSON、最后写带 generation hashes 的 manifest；不是整个目录事务、永久不可变 ledger 或所有文件都被哈希。
8. **源码与旧说明可能漂移。** 当前 retry 分支是 lesson-driven、有界候选循环与一次瓶颈重诊断，不能抄旧短句“只重试一次”。
9. **不同技能有不同工具依赖。** IdeaSpark 自带四个 connector；独立 Paper-Search 默认六个 API 来源与模型知识补充；Scoop-Check 委托 Paper-Search。IdeaSpark 的内部 collision 不是调用独立 Scoop-Check。
10. **Reel 验收强度不同。** Paper2Video 包装层显式要求子 CLI strict QA，并统一 ppt_trigger；此合同不能证明外部依赖已经安装、版本相同或真实运行通过。

[导航器](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py)、[桥接](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/intent.py)、[kill-switch](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/validators/kill_switch_integrity.py)、[资产构建](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Reel/skills/paper2assets/scripts/build_package.py)、[视频集成](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Reel/skills/paper2video/SKILL.md)

## 何时加更强治理

哈希绑定、强 terminal receipt、事务、lease、跨 worker fencing、独立控制器可以解决真实产品问题，但不是 ResearchStudio 全局现成能力。只有“旧结果误配、并发所有权、不可逆提交、跨进程恢复”等需求出现时再选择对应机制。

同样，不应把研究系统的 RTG、人类审批或特定宿主的后台通知机制带进无关 Skill。专业判断、执行控制和授权的边界都应从新任务本身确定。

## 验证证据的强度

2026-09-11 执行 IdeaSpark routing selftest 37/37、helper/validator selftest 60/60，并执行空卡片终态反例。未运行真实模型、完整媒体流水线或外部子技能实现；没有重新复现论文的离线数据与评估成绩。

最重要的迁移检查是：新 Skill 的一个具体保证，能否由实际执行路径和反例测试支撑，而不只是由 prompt 声明。
