---
name: evidence-brief
description: 可选高级示例：用三条明确标注的虚构证据演示程序骨架、独立写作审查、受限修订和再审闭环。仅在需要独立审查或复杂编排时学习与改造。
---

# Evidence Brief

本例保留真实子 Agent 与独立审查合同，供需要独立审查或复杂编排时选读和运行。普通任务默认使用 [当前宿主判断接力](../../references/judgment-interface.md)，不必采用本例的多阶段结构。选用本例后，下文 `llm_subagent` 派发与隔离要求仍完整生效，不能在主上下文模拟角色后宣称通过。

交付一份中文 Markdown 证据简报。样例只使用 [虚构输入](fixtures/input.json)，不联网、不涉及真实研究结论。所有程序只使用 Python 标准库；`scripts/run.py` 不调用或模拟 LLM。

## 启动与推进

先将本 Skill 目录和一个全新的运行目录解析为绝对路径，再运行：

```bash
python3 "$SKILL_DIR/scripts/run.py" prepare --input "$SKILL_DIR/fixtures/input.json" --dir "$RUN_DIR"
python3 "$SKILL_DIR/scripts/run.py" next --dir "$RUN_DIR"
```

`next` 只读，每次输出一个 JSON action。这是本样例原创的机器可读协议，**不是 ResearchStudio 上游的文本协议或兼容接口**。action 中所有文件路径和程序 argv 都是绝对路径：

- `kind=command`：逐项执行 `argv`，不拼接 shell 字符串；再调用 next。
- `kind=llm_subagent`：宿主实际创建一个全新、无历史上下文的子 Agent。只传此 action 的 `prompt`、`inputs`、`output` 和只写 output 的约束；不传主会话、其他角色输出或期望结论。让它读取提示文件并原子写入指定 JSON；等待完成后调用 next。
- `kind=done`：读取并交付 output 指向的 `brief.md`。
- `kind=blocked`：报告 notes 及保留的运行目录，停止依赖步骤；不可通过删评审、改 verdict 或反复开新目录隐藏失败。

使用原生子 Agent 工具时设置 `fork_turns="none"`（或宿主等价选项）。writer、critic-0、reviser、critic-1 必须是各自新建的 Agent；同一 critic 提示可复用，但实例和上下文不可复用。子 Agent 只负责 action.output，不自行推进其他阶段。宿主没有真实子 Agent 能力时明确阻塞，不在主上下文模拟角色。该隔离是宿主执行约束，Python 脚本不能证明 Agent 身份。

## 文件与权限边界

| 顺序 | 执行者 | 读取 | 唯一新增输出 |
| --- | --- | --- | --- |
| prepare | 程序 | 输入文件 | input.json 快照、skeleton.json |
| writer | 新子 Agent | skeleton.json | fill_map.json |
| assemble | 程序 | skeleton.json、fill_map.json | draft.json |
| critic-0 | 新子 Agent | draft.json | critique-0.json |
| reviser（仅 revise） | 新子 Agent | draft.json、critique-0.json | patch.json |
| merge | 程序 | draft.json、critique-0.json、patch.json | revised.json |
| critic-1 | 新子 Agent | revised.json | critique-1.json |
| render | 程序 | 当前候选与通过的评审 | brief.md |

`fill_map` 恰好填写 answer、conflict、limits 三个文本字段，每字段最多 700 字符。来源、source_id、synthetic、预算和 revision 由程序保留。`patch` 的键恰好等于 critic findings 中指出的字段，不能自扩权限；合并器不回写 patch、原稿或评审。每个 JSON 文件完整写入临时文件并校验后原子发布，禁止覆盖已有产物。

critic 的非空理由、合法 verdict、finding scope、来源编号、原文引用和候选 review_token 都由程序检查；这只保证评审结构与对象正确，**不能证明语义判断正确**。证据充分性由真实独立 critic 负责。所有阶段在读取时校验；损坏 JSON、空文本、越权 patch 或旧评审都会阻塞。

最多一次修订。初审 abandon 立即停止；再审 revise/abandon 也停止。done 必须同时具备与当前候选逐字一致的非空终稿，以及绑定该候选的最新 advance 评审。发现损坏产物时先报告；确需重新实验则用新目录并保留失败原因。修改输入后必须使用新运行目录，prepare 会拒绝在已有目录装入不同输入；只读源文件后来变化不会改写已经冻结的 input.json 快照。

`next` 和其他命令都输出 JSON；blocked 的进程退出码为 2，其他为 0。可直接运行的检查：

```bash
python3 "$SKILL_DIR/scripts/check.py"
```

检查使用固定测试数据验证状态和权限，不假称执行了 LLM。真实角色运行记录应由实际宿主试跑产生。
