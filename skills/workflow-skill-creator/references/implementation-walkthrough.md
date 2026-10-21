# ResearchStudio：从源码学会写工作流脚本

需要独立审查或复杂编排时选读。这里把 ResearchStudio 的实现拆成可按需借用的方法；普通线性流程先用现有工具和普通代码，少量语义判断默认走 [当前宿主判断接力](judgment-interface.md)，不要求照搬导航器或多个子 Agent。配套的 [evidence-brief](../assets/evidence-brief/SKILL.md) 是保留独立审查保证的可选高级示例，不是新 Skill 的默认骨架，也不是 IdeaSpark 的替代实现。

证据固定为 Microsoft ResearchStudio `0597891df1a153b8e4cbdc8c1c685f43a0a6abcf`，核验日 2026-09-12。下文代码示意与样例为原创；源码用固定链接定位。历史测试和整体边界见 [案例参考](researchstudio-case.md)。

## 1. 先追一次调用，再决定脚本怎么分

IdeaSpark 的一次典型交接：

```text
宿主执行 scripts/run.py next --dir RUN
  → next_step.py 读取阶段文件与判定字段
  → 输出一包文本：TYPE / PROMPT / INPUT / OUTPUT / RUN / NOTES / THEN
  → 宿主执行准备命令，创建独立子 Agent
  → 子 Agent 读 prompt + 指定文件，写阶段 JSON
  → 宿主再次 next
  → run.py 调用 merger / assembler / validator / renderer
  → 产物成为下一阶段的输入
```

这条链路有三个不同的编程面：

| 面 | 应写什么 | 不承担什么 |
| --- | --- | --- |
| `run.py` 与专用 helper | 确定的命令入口、参数、计算、I/O 和错误 | 凭模型常识判定研究新颖性 |
| `next_step.py` | 读取当前产物，选动作，输出完整交接包 | 自己创建宿主 Agent 或悄悄执行打印的命令 |
| SKILL + 阶段 prompt | 宿主操作规则、领域判断、读写合同 | 单靠文字实现文件权限或字段保护 |

源码入口：[run.py](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/run.py)、[next_step.py:1–71](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py#L1)。

新 Skill 先把一个样本走通：定义最终文件 → 列出需要的语义内容 → 把机械字段拿回程序 → 确定判断输出 → 接上当前宿主判断与程序消费。确需独立调用时再接派发，分支或恢复复杂时再补导航。不要先造一套通用 workflow 类。

## 2. CLI 写成确定入口，减少宿主临场选择

`run.py` 根据自身 `__file__` 找 Skill 根目录；connector 子进程用参数数组、固定 `cwd`、绝对输出路径、超时与结果检查。这解决三个具体错误：从其他工作目录调用时找不到模块；子进程把相对路径写到另一个目录；失败的 connector 被当成空文献列表。

迁移后的组织方式可以很小：

```text
scripts/run.py
  parse_args()           参数和合法子命令
  prepare(input, run)    原始请求、证据与 skeleton
  next_action(run)       只读路由
  assemble(run)          应用 fill_map
  merge(run)             应用审查授权范围内的 patch
  render(run)            从通过审查的对象渲染最终文件
```

样例把这些放在一个文件，因为没有独立复用的 helper；IdeaSpark 较大，才把 connector、merger、validator 分开。拆文件的依据是责任与复用，不是每个阶段必须一个模块。

需要借用这套独立审查结构时，可以按这个顺序读随附 [run.py](../assets/evidence-brief/scripts/run.py)，再改造成自己的实现；日常执行已有入口无需重新读完源码：

| 函数 | 真正负责的事 | 迁移时要改什么 |
| --- | --- | --- |
| `source_input` | 校验输入形状、来源标识、虚构标记 | 换成新任务原始材料合同，不照搬三条虚构证据限制 |
| `skeleton` | 复制事实，填固定约束，建立待写字段 | 确定哪些字段归程序、哪些归模型 |
| `text_fields` | 模型输出的键、类型、长度 | 输出 schema 与实质约束 |
| `candidate` | 合成对象并绑定评审对象内容 | 有旧审查误用风险才保留该绑定 |
| `critique` | 合法 verdict、findings、来源原文与对象绑定 | 让程序结构检查对应领域 prompt 的判断条件 |
| `inspect_run` | 只读检查当前阶段，选择一个 action | 成功、修订、失败的实际分支 |
| `atomic_new` | 发布完整文件并拒绝覆盖已有产物 | 根据宿主文件能力保留防数据丢失处理 |
| `main` | 命令入口，只执行当前允许动作 | 对接真实公开 CLI |

下面摘自本地原创样例，是已运行实现的骨架构造，不是上游源码：

```python
def skeleton(value):
    return {**copy.deepcopy(value),
            "constraints": {"max_revisions": 1, "max_chars_per_field": 700},
            "text": {field: None for field in sorted(FIELDS)}}
```

`FIELDS`、导入和消费者在完整脚本中；片段用于说明职责，单独复制它不是完整程序。


有外部 CLI 时用其真实参数数组调用，检查退出状态，再解析实际输出。业务输入不能拼成 shell 命令。上游辅助 LLM bridge 使用可信环境变量中的 shell command，这是它的既有配置合同，不应随便复制成接收任意用户命令的接口。

源码：[connector 子进程入口](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/run.py#L474)。

## 3. 下一步是一包可执行交接信息

上游 `_emit()` 的价值不只是告诉宿主“现在 Phase 2”：它同时提供 **执行类型、阶段 prompt、完整输入列表、精确输出位置、前置/并行动作说明、需要返回的 routing signal、再次导航命令**。没有这些，宿主仍需把全部流程放在上下文里自己补全。

注意上游用文本协议：多个 INPUT/RUN 的续行没有重复标签；只抓 `PROMPT:` 或第一行 INPUT 会丢证据。`RUN` 也不永远等于串行前置操作：Phase 2 是先生成输入切片，Phase 2.3 的 NOTES 则要求并行启动 collision。必须连同 NOTES 原样理解。

新样例选择 JSON action，便于教学与检查，**这不是上游现有接口**：

```json
{
  "kind": "llm_subagent",
  "step": "writer",
  "prompt": "/ABS/evidence-brief/references/writer.md",
  "inputs": ["/ABS/run/skeleton.json"],
  "output": "/ABS/run/fill_map.json",
  "notes": "Read the prompt and listed inputs; write the output file."
}
```

上例展示协议形状，路径须以样例实际 `next` 输出为准；`command` action 给出 `argv`，`done/blocked` 是终态。生成新 Skill 时选择与宿主匹配的一种协议，把生产者、消费者一起写完，不能只交一个无人消费的 schema。

源码：[Phase 2 的准备与合并派发](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py#L492)。

## 4. 程序造骨架，模型只写语义增量

上游 Phase 4 的机械部分包括：原样保留承诺字段、文献 ID→venue/year 查表、按模式分组、把 critique findings 与修订字段关联、从用户资源约束推导预算等级。LLM 输出这些字段的完整副本，既没有新信息，又容易改错。

所以工作拆为：

```text
候选 + 审查 + 文献表 → Python build_skeleton
                            ↓
             已填机械字段 + 标注路径的 TODO
                            ↓
                       LLM fill_map
                            ↓
                  Python assemble → 验证 → 渲染
```

以证据简报为例，以下是数据合同示意，真实字段见样例：

```json
{
  "sources": [{"source_id": "E1", "title": "虚构观察", "fact": "限定条件下的测量记录", "scope": "单个场地"}],
  "text": {"answer": null, "conflict": null, "limits": null}
}
```

writer 只返回 `answer/conflict/limits` 的值；不用重抄 sources。样例用 `null` 标识三个待填字段，上游大树则用带路径与提示的 TODO 字符串。assembler 只接受声明的可写键和合法值，复制机械字段。检查“TODO 已填”与“结论有证据”分别由程序和 critic 完成。

上游 `assemble_expansion()` 硬拒绝 kill-switch 根字段，但没有对所有非 TODO 字段建立严格白名单；“只填 TODO”还部分依赖 prompt。迁移样例刻意采用小而完整的允许字段集合。不要把源码的局部保护描述成全字段写保护。

另一个精细分工是 **technical fill → partial assemble → plain derive → final assemble → method_view → implementability audit**。翻译和通俗化从已完成技术文本派生，不与核心推理一起重做；审查者只读 method_view。`expand.txt` 开头仍称它是 Phase 4 唯一模型调用，与当前导航器不一致；以实际路由为当前实现。

源码：[骨架与 assembler](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/phase4_skeleton.py#L736)、[fill/derive/view 路由](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py#L1055)。

## 5. 审查给问题，修订给 patch，程序拥有合并

上游 Phase 3.2 审查用文献、子模式 Reject lessons 和 anti-patterns 给出 findings 与 `revision_targets`。Phase 3.3 是新上下文，只处理这些 targets：不重新审判 verdict、不新增攻击、不重检索、不输出完整 candidate。

把模型输出从“大 JSON 重写”缩到“操作列表”：

```json
{
  "applied_revisions": [
    {"op": "replace", "field": "core_mechanism", "value": "修正后的完整字段值"},
    {"op": "append_sentence", "field": "hook", "value": "补充限定条件。"}
  ]
}
```

这是上游字段形状的原创示意，不是允许这两项在任意审查中修改。实际字段必须来自对应审查和阶段范围。

`merge_revisions.apply_patch()` 深复制原对象，验证 op/field/value，再执行操作；普通操作无法写 `compute_budget/falsification_prediction`。唯一特别通道是审查报告明确授权 `scope=falsification` 后的一次 `rewrite_falsification`，且之后重审。授权由 merger 读取 critique 得出，不能由 reviser 在自己 patch 内写“已授权”。

迁移到新领域时，先问“哪些字段是用户承诺/来源事实，哪些只是可修订叙述”。小流程用两三个字段白名单即可；不要为了像上游而支持通用点路径解析器和五种操作。需要例外时，把授权来源、次数和重验写进程序。

源码：[revise prompt](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/references/system-prompts/revise.txt)、[merge 与授权](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/merge_revisions.py#L216)。

## 6. 用消费视图缩小上下文，保留关键负面证据

只写“少读文件”不够。上游做了实际输入裁剪：

| 谁消费 | 给它什么 | 为什么 |
| --- | --- | --- |
| candidate generator | `closest_abstracts.json` + 选中的 C## 卡片 | 不把全部检索 dump 带进生成阶段 |
| reviser | `revise_brief.json`，必要时回查完整审查 | targets 足以定位修改；原始 Reject 引文保留可追溯入口 |
| implementability auditor | `method_view.json` | 按实现步骤挑错，不承担全部领域与文献任务 |
| 下一次 generator | 以前的阻断 findings、structural requirements、威胁论文 | 把失败变成约束，避免只换措辞重试 |

切片的标准是下游判断是否充分。负面证据、原始用户限制不能因为“精简”而漏掉。上游 Phase 2 显式传 `user_query.txt`，因为 intake 摘要不能替代用户原话。

原始数据→程序选出本阶段视图→模型按需读取关联原文，比在 prompt 中要求模型自己从巨量文件筛选更可控；仍不能宣称操作系统阻止模型读取其他文件。

## 7. 并行依据依赖字段，完成后要核对它们

上游把 coherence 与 collision 并行，是因为 collision 只读 `signature_terms/alias_terms`，coherence 大多改方法叙述。但“大多不改”不是一致性保证：检索记录 `.collision_terms.json`，导航器在合并修订后比较当前 canonical candidate 的词集；变了就重新检索，未变就复用。

由此得到可直接使用的判断顺序：

1. 列出两个动作各读哪些字段、各写哪个文件；不能共享写目标。
2. 共享输入会不会被其中一方改变？不会则直接并行。
3. 会改变但代价值得提前启动？记录消费的相关字段，join 时比较；不一致重跑受影响消费者。
4. 没有这种校验，也无可靠不可变输入，则先串行。

样例 writer→critic 有真实依赖，不能为了展示多 Agent 而并行。独立报告分支可以并行，各写自己的文件后汇总。

源码：[并行启动与词集比对](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py#L575)。

## 8. 另一条模型入口：sentinel 与命令桥接

大推理阶段没有一个通用 `run.py phaseN` 自动调 LLM；宿主读阶段 prompt 并执行。辅助意图/分类/签名提取还有两条路径：

| 路径 | 实际实现 | 宿主应怎样承接 |
| --- | --- | --- |
| 配置 LLM command | `intent.call_llm()` 从环境变量取命令，stdin 给文本，stdout 解析 JSON | 先有可用 wrapper；包装器自行实现消息角色映射和模型调用 |
| 未配置 command | `emit_host_llm_sentinel()` 写 `.STEP_pending`，rc=10/11 | 读 rubric、输入、期望输出、re_invocation，完成后续跑 |

sentinel 字段是 `step_name/rubric_file/inputs/expected_outputs/instruction/re_invocation`；它是一次工具→宿主的控制权交还。识别约定的 rc 和 sentinel 才能处理，普通非零退出不能一律解释成“请模型继续”。`intent.call_llm()` 自身未配置时抛异常，写 sentinel 的是调用它的 orchestrator 路径。

`<<SYSTEM>>` 和 `<<USER>>` 只是 bridge 文本标记；prompt 文件夹叫 `system-prompts` 也不自动产生 API system role。真正如何加载与约束见 [prompt 方法](prompt-loading.md)。

源码：[sentinel](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/run.py#L385)、[LLM bridge](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/intent.py#L47)。

## 9. 组合已有 Skill：只实现交接缺口

Reel 的 Paper2Video 已有制片与视频工具，父技能不再实现一份。它单次解析共享的 `ppt_trigger`，传给两个子技能，把交接写到文件，最终解析 PPTX 原生 timing 验证实际动画行为；视频 CLI 明确传 `--qa-mode strict`。

这一方法迁移到“报告→幻灯片→视频”时：父技能负责共同材料、语言/格式/触发选项、路径与验收；子技能负责自己擅长的生成。**既要传参数，也要检查下游实际交付**，不能假设子组件默认值一致。先查真实已安装入口，不由名称编造 `python -m child_skill`。

源码：[Paper2Video 合同](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Reel/skills/paper2video/SKILL.md)。该合同不证明当前宿主已安装并跑通那些外部依赖。

## 10. 把方法变成自己的脚本时要留下什么

- 一条从任意工作目录可调用的真实入口，以及最小真实输入。
- 每个必要判断的规则、输入视图、输出 schema 和宿主接力步骤；采用独立调用时另给真实派发步骤。
- 一条确定的内容流：生成→合并/校验→消费，含失败时的下一动作。
- 针对关键不变量的可运行反例，及一次真实模型阶段运行记录。

不是每个 Skill 都要以上全部模块。若任务主要是一次性理解或创作，可只用指令和已有工具；稳定重复的转换应复用脚本。若声称提供类似 IdeaSpark 的多阶段编排，就不能缺失宿主如何执行模型阶段这一环。宿主判断产物被程序真实消费即可验证默认接力；只有声明独立派发时才要求对应的真实 child 与调用证据。
