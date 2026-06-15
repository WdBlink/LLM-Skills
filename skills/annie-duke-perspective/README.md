# Annie Duke Perspective

![Version](https://img.shields.io/badge/version-0.2.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

> 「对赌思维始于我认识到只有两件事在决定着我们的生活：我们的决策质量和运气。」—— Annie Duke

决策顾问思维模型，蒸馏自 Annie Duke《对赌》（*Thinking in Bets*）。把决策质量与结果质量分开、用概率式叙述、强制写下责任性预测、做预先检查与反向回顾 —— 适用于"信息不完整时如何做决策"——商业投资、职业选择、长期战略、风险评估、产品方向、团队决策复盘。

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/annie-duke-perspective
rsync -a skills/annie-duke-perspective/ ~/.codex/skills/annie-duke-perspective/
```

Claude Code users can install to `~/.claude/skills/annie-duke-perspective/` with the same `rsync` pattern.

## Requirements

- None. Pure markdown workflow + distilled mental-model library; no external services required.

## When to Trigger

| Layer | Trigger phrases |
|-------|-----------------|
| **Hard triggers** | 「用 Annie Duke 的视角」「Annie 会怎么看」「对赌思维」「决策质量 vs 结果质量」「以结果为导向」「Resulting」「所有决策都是对赌」「All in 决策」「切换到 Annie Duke」「Annie 模式」「thinking in bets」 |
| **Soft triggers (scenario)** | 「我做了个决策，帮我复盘」「我这个决定到底对不对」「怎么知道是运气还是技能」「我敢为这个判断下注吗」「结果不好但我决策没错吧」「该不该辞职 / 换城市 / 买房」「为什么我老公 / 老婆不听我的」「团队总达成共识但结果不好」 |
| **Invocation** | 「@annie-duke 帮看看这个」「让 Annie 评估一下这个方案」 |
| **Do NOT trigger** | 纯学术知识问答（"卡尼曼是谁"）；编程/技术选型（"React 还是 Vue"）；已知确定性的数学/逻辑问题 |

## What's Inside

| File | Purpose |
|---|---|
| `SKILL.md` | The skill — **7 mental models (M1-M7)**、**11 heuristics (H1-H11)**、7-dimension expression DNA + 4-piece kit、Annie Duke voice + Agentic Protocol |
| `references/research/01-writings.md` | 著作研究（《对赌》正文蒸馏） |
| `references/research/02-conversations.md` | 对话与即兴反应样本 |
| `references/research/03-expression-dna.md` | 7 维度表达 DNA + 必背 5 句 / 必避 5 词 / 必用 5 隐喻 |
| `references/research/04-external-views.md` | 他者视角与第三方评价 |
| `references/research/05-decisions.md` | Annie 自述决策记录 |
| `references/research/06-timeline.md` | 人物时间线 |
| `sources/books/对赌-正文.txt` | 《对赌》中文版正文（154K 字符，纯文本蒸馏源） |
| `scripts/azw3_to_text.py` | azw3 → 纯文本转换器（用于重新蒸馏） |
| `SKILL-v1-legacy.md` | v1 备份（2026-06-08 蒸馏的初版，10 启发式）—— 备审计 |

> 注：原 azw3 文件（5.4 MB）未纳入仓库；如需重新蒸馏，可下载《对赌》中信版 2026 后用 `scripts/azw3_to_text.py` 转换。

## Skill 机制层（v2 独有）

| 机制 | 用途 |
|---|---|
| **4 层触发词** | 硬触发 / 场景化软触发 / 被引入型 / 不触发场景 |
| **角色规则 5 段** | 基础规则 + 问题路由 + 频率约束 + 跑偏自救 + 退出完整协议 |
| **加载即执行协议** | Skill 加载 30 秒内必做的动作（首启免责声明 + 主动开场） |
| **防跑偏自检 prompt** | 每次回答前内部 4 项检查（角色 / 百分比 / 故事密度 / 跑偏气味） |
| **诚实机制** | 4 类触发场景（全新议题 / 原话反驳 / 跑偏反馈 / 真实身份质疑）+ 必做动作 |
| **决策脏话罐** | 复盘时收集"早知道" + 后悔，用具体后悔量化「以结果为导向」偏差 |

## Core Mental Models（7 条）

| # | Model | 一句话 |
|---|-------|--------|
| M1 | Resulting（以结果为导向） | 把决策质量等同于结果质量是人类最顽固的思维模式 |
| M2 | All Decisions Are Bets | 任何决策都不是二元选择，而是在不确定下的概率性下注 |
| M3 | Result = Skill + Luck | 任何结果都是技能和运气的混合，几乎从不是 100% 来自任一方 |
| M4 | Belief Gullibility | 人类默认"听到就相信"，必须主动制造"挑战自己信念"的机制 |
| M5 | Mental Time Travel | 调出"未来的我"和"过去的我"来制衡"现在的我" |
| M6 | CUDOS Group Decision | 遵守默顿四规范（共有性 / 普遍性 / 无私利性 / 有条理的怀疑性）的团体比个人更准 |
| M7 | Accountable Predictions | 写下 + 公开 + 设定时间窗口 = 对抗自利性偏差的最强武器 |

## Architecture

```
Trigger phrase → Skill("annie-duke-perspective", task)
                       ↓
                Step 1: classify (fact / framework / mixed)
                       ↓
                Step 2: research (if classified as fact or mixed)
                  - 4 dimensions (decision quality / process vs result / accountability / group)
                       ↓
                Step 3: answer
                  - filter 7 mental models
                  - compose in Annie Duke voice
                  - end with question or probability
```

## Layout

```text
annie-duke-perspective/
├── SKILL.md                              # Main entry, ~1063 lines, 7 models + 11 heuristics + agentic protocol
├── README.md                             # This file
├── LICENSE                               # MIT
├── SKILL-v1-legacy.md                    # v1 backup (517 lines)
├── references/
│   └── research/
│       ├── 01-writings.md                # 《对赌》正文蒸馏
│       ├── 02-conversations.md           # 对话样本
│       ├── 03-expression-dna.md          # 表达 DNA
│       ├── 04-external-views.md          # 他者视角
│       ├── 05-decisions.md               # 决策记录
│       └── 06-timeline.md                # 人物时间线
├── sources/
│   └── books/
│       └── 对赌-正文.txt                 # 154K 字符蒸馏源（纯文本）
└── scripts/
    └── azw3_to_text.py                   # azw3 → txt 转换器
```

## Design Principles

| Principle | Meaning |
|-----------|---------|
| Decision quality ≠ result quality | 把过程与结果分开评估是整个框架的元命题 |
| Probabilistic narration | 每个判断都给百分比（X%），绝不说 100% / 显然 / 标准答案 |
| Accountability by writing | 重要决策必须写下 + 公开 + 设定时间窗口（30/60/90 天） |
| Pre-mortem + Backcasting | 风险决策前先做预先检查：假设 6 个月后失败 / 成功，最可能的原因是什么 |
| Group diversity over consensus | 团体奖励"准确"而不是"和谐"；默顿 CUDOS 四规范是基础设施 |
| Story-anchored teaching | 抽象命题必须有具体案例锚点（皮特·卡罗尔 / 巴特曼 / 希腊人尼克 / 菲尔·赫尔穆特 / 得梅因之赌） |

## Source

- 主体结构：《Thinking in Bets: Making Smarter Decisions When You Don't Have All the Facts》(2018) 中文版《对赌》（中信出版社 2026）
- 二手框架：Daniel Kahneman, Daniel Gilbert, Robert Merton, Nassim Taleb, Phil Tetlock, Jonathan Haidt
- Skill created by: [花叔](https://x.com/AlchainHust) via [女娲 · Skill造人术](https://github.com/alchaincyf/nuwa-skill)
- v1 distilled 2026-06-08（10 启发式）
- **v2 fully re-distilled 2026-06-15**（11 启发式 + 4 机制层 + 数据纠正：$30K 得梅因赌局 / 20 年职业扑克）

## License

MIT — see `LICENSE`.
