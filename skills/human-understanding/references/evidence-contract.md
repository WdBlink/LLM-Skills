# 证据记录与检查覆盖

当多项来源数字或计算值得结构化核对时使用。先复用现有记录；是否持久化、是否使用 HTML 本身不触发 JSON 要求。简单文字解释可直接在正文保留同等证据信息；概念图或公式演示用相应来源/计算检查，无需填实验字段。

## 文件与字段

`evidence.json` 是解释层记录，不取代原始材料。相对来源路径以此文件所在目录为基准。可参考 [虚构记录](../examples/evidence.json) 及其 [虚构原始结果](../examples/results.json)。不得把示例当成用户的结果。

| 字段 | 合同 |
|---|---|
| `version` | 固定整数 `1` |
| `task` | `question` 审阅问题、`reader` 读者背景、`scope` 材料范围/截止点 |
| `sources` | 每项有唯一 `id`、`kind`（`primary`/`summary`/`derived`）、`status`（`available`/`missing`/`unread`）、`path` 或 `url`；缺失/未读时有 `reason`。用 `version` 记录来源版本、日期或运行 ID |
| `evidence` | 每项有唯一 `id`、`source_id`、`kind`；形式见下 |
| `claims` | 每项有唯一 `id`、`text`、`scope`、`kind`、`evidence_ids`、`limitations` 字符串数组；空限制数组仅表示暂未记录限制，不表示无限适用 |

`claims.kind`：`observation`、`inference`、`missing`、`conflicting`。后三类必须有 `reason`；推断的 reason 说明推理与假设；缺失说明影响；冲突说明分歧。观察至少连接一项已核对的原始证据；冲突至少列两项不同证据。脚本只校验连接存在，不判断它们是否支持结论。

建议写在解释中，关联结论 ID，并写清决策标准、选择与改变建议的条件。不强制虚构概率、负责人、截止时间或量化阈值。

### 可程序检查的证据

**结构化指标**：`kind: metric`，`pointer` 为指向原始 JSON 对象的 RFC 6901 路径，`metric` 逐字段抄录：

- `name`、`experiment`、`conditions`、`baseline_name`：指标、实验、条件、基线标识。
- `unit`：原记录单位，比例 `fraction` 与百分数 `%` 不得混写；单位转换在解释中另列公式并复核。
- `direction`：`higher`、`lower` 或 `neutral`，与评价协议一致。
- `value`、`baseline`：有限数值。
- `n`、`baseline_n`：正整数样本数。样本单位、配对关系、种子数/重复数和数据切分写入 `conditions` 或结论限制，不把运行次数当样本数。

上述字段均需出现；原始来源没有某字段时填 `null`，在同项的 `missing` 对象用同名键解释缺口。不能把已有数据改为 null。缺少关键指标或基线时，解释不得继续声称提升。

可选 `computed`：

- `delta = value - baseline`，沿用原单位。百分数单位的差值在正文写“百分点”，不是相对百分比。
- `relative_change_pct = (value - baseline) / baseline * 100`，仅接受正基线；零或负基线用原单位差值，并解释原因。
- `trend`：`improved`、`worsened`、`unchanged`、`undetermined`，仅表示该指标数值方向，不表示显著性、因果或整体成功。`neutral` 的非零变化为 `undetermined`。

保留计算全精度，展示时可舍入；脚本用 `1e-9` 容差比较计算结果。源值抄录必须一致。趋势直接比较源值，不依赖差值；相对变化采用等价运算顺序避免中间减法溢出。结果超出有限数值范围时明确 fail，不用无穷大代替。

**原文摘录**：`kind: text`，`lines: [起始行, 结束行]`（从 1 开始，含结束行），`excerpt` 必须与这些行完全一致（不含末尾换行）。脚本核对摘录，模型再核对语义、数字和上下文。

格式无法直接匹配的 CSV、PDF、日志、网页不必为适配脚本改写原始文件。使用可定位的文本摘录或手工证据卡；需要提取成 JSON 时，将派生文件列为 `derived`，另列原始来源及转换方法，逐项复核，不称为已自动验证的原始指标。JSON 原记录字段不齐全时，也可以用带缺口的 metric 并人工补读协议，不能为了通过脚本补造源数据。

网络来源须由宿主先读取并保存允许保留的本地内容，附 URL 和版本/抓取时间，检查器不联网。只有 URL 无本地快照时标记 `unread` 或改用手工核验并说明程序未覆盖。将来源列为 primary 是执行者的声明，不是脚本的认证。

## 执行与解释

```bash
python3 <skill-dir>/scripts/check_evidence.py /absolute/path/evidence.json
```

输出 JSON；`fail` 返回退出码 1，`pass` 和 `pass_with_gaps` 返回 0。调用者必须读 `status`、`errors`、`gaps`，不能只看退出码。`pass_with_gaps` 允许交付有边界的解释，不允许将缺口包装成已证实。

`semantic_review: not_checked` 始终提醒：程序通过仅说明已提供字段与文件一致，不能证明来源真实性、来源清单完整、样本口径正确、条件可比或结论成立。程序不读取最终 Markdown/HTML；最终呈现中的数字、单位、图例、引用与结论必须再次对照记录核对。没有执行检查时不能写“已通过”。

## 表达中的可追溯性

文字、表格、图和 HTML 引用同一组结论/证据 ID。读者应能从结论找到实验、条件、基线、指标和来源。最终文字不必暴露 JSON 字段名；证据卡可放在相邻段落、附表或 HTML details 中。

所有会改变决策的缺口、不可比条件和失败结果在首层可见。多组数据筛选只改变可见记录，不能静默重算总体成功；需要聚合时明确权重、分母、排除条件和方法，并额外验证计算。
