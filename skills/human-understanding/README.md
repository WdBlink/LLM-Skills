# Human Understanding

`v0.3.0` · Proprietary

以脚本准备材料、宿主按需判断、程序组装并绑定检查结果的方式工作。简单解释可直接回答，图解和定制教学 HTML 继续支持。

围绕读者的理解障碍，按 15 条具体规则写说明：明确执行者与动作、固定术语与词义、消除指代歧义、拆开条件与结果、展示机制的中间步骤，并保留数值与不确定性。英文进一步检查动词形式、冠词、句长和名词串；描述段最多 6 句，完整定义/引用可独立展示。需要时制作图解或交互 HTML。

已核对 Karpathy 2026-10-02 的原帖正文及附图，方法见 [解释方法](references/explanation-methods.md)，原帖主张与工程扩展分见 [来源记录](references/provenance.md)。当前覆盖文字、图解和 HTML；视频保留为后续方向。

```text
$human-understanding 给第一次接触 PPO 的工程师解释 clip 为什么限制更新，用一个具体例子和图说明。

$human-understanding 用一个本地 HTML 解释二分查找，让我逐步推进并看见候选区间怎么变化。

$human-understanding 阅读 results/ 和本轮失败回执，面向项目负责人回答本轮是否值得继续投入。保留退化与缺失证据，选择最简单的表达形式。
```

在仓库根目录安装到 Codex（目录已有内容时先检查再同步）：

```bash
mkdir -p ~/.codex/skills/human-understanding
rsync -a skills/human-understanding/ ~/.codex/skills/human-understanding/
```

Claude Code 可将同一目录安装到 `~/.claude/skills/human-understanding`，通过 `/human-understanding` 调用。纯文字流程只需要读取材料；程序检查需要 Python 3 标准库；交互页面验证需要可用浏览器，不依赖特定 Agent 框架。

多项来源数字或计算需要结构化核对时使用 [证据合同](references/evidence-contract.md)。小型概念解释或教学 HTML 无需强制生成证据 JSON。

```bash
python3 skills/human-understanding/scripts/run.py --help
python3 skills/human-understanding/scripts/check_evidence.py evidence.json
python3 -m unittest discover -s skills/human-understanding/tests -v
```

检查器核对记录与本地来源的一致性，不判断来源是否真实、结论是否成立或最终报告是否忠实；这些必须按 [SKILL.md](SKILL.md) 单独复核。来源不足时会保留缺口。示例均为虚构数据。

可复用入口、结构化判断、定制产物接入、检查绑定与恢复规则见 [执行合同](references/workflow-contract.md)。常规流程不调用独立模型后端；运行数据写入任务目录。
