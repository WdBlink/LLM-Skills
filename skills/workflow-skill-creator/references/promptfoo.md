# 用 Promptfoo 回放选段判断

这份可选接线检查 [选段判断接口](judgment-interface.md) 的保存结果。它不替换原有 Python 测试，不负责创作整个 Skill，也不默认增加模型调用。`adapter.py` 只读取本地文件；Promptfoo 未安装时仍可运行相同评分入口。

## 输入与评分边界

评价单位是一个绑定冻结请求的 `decision.json`。当前只有两个公开的合成开发样例：`scope-limits` 含两段明确的结论适用限制，`no-match` 的材料充分但没有符合项。它们用于检查评分器和接线，不是隐藏测试集，也不支持模型排名、真实业务准确率或泛化结论。

| 要求 | 检查及依据 | 合法变化 | 应拒绝的反例 |
| --- | --- | --- | --- |
| R1 请求与判断匹配 | 复用 `validate_request`、`validate_decision`；核对源文件 SHA-256 | JSON 空白变化 | 陈旧 request_id、未知或重复 ID、额外字段 |
| R2 完整选择符合目标的段落 | 与 `grader/references.json` 的冻结段落集合比较 | 返回顺序和非空理由措辞变化 | 字段合法但选错、漏选、多选 |
| R3 充分材料中的无匹配 | 空集合与参考一致，且有非空 no_match_reason | 无匹配说明的措辞变化 | 有符合项却返回空、空集合缺理由 |

评分仅为 `0` 或 `1`，R1、R2、R3 的失败不能用平均分抵消。理由只检查非空，不评估其语义质量；新领域、新材料的标签仍需独立审阅。当前标签由 Agent 根据合成原文编写，`human_review_pending`。材料不足属于 `needs_input`，不在这两个“材料充分”的样例中，不得用无匹配替代。

`inputs/` 是求解侧材料。`grader/` 和 `replays/synthetic/` 是评分侧标签与人工构造的回放示例。`create_prompt` 只读取请求与输入，输出目标、原文、ID 和 JSON 合同，不附参考答案。公开仓库内的目录划分不是安全隔离；盲测时只把导出的 prompt 提供给求解器，不给它整个仓库、评分上下文或回放目录。

## 不安装 Promptfoo 也能检查

把 `SKILL_DIR` 设为此 Skill 的实际绝对路径，`RUN_DIR` 设为本次任务目录。从任意目录运行：

```sh
python3 -B -m unittest discover -s "$SKILL_DIR/evals" -p 'test_*.py' -v
python3 -B "$SKILL_DIR/evals/adapter.py" grade scope-limits \
  "$SKILL_DIR/evals/replays/synthetic/scope-limits.json"
```

这些检查会覆盖合法替代、结构合法但选错段、遗漏、多选、过期绑定、重复键、缺失文件与路径越界。回放正例是合成产物，不是实际模型执行记录。

## 接入当前宿主的真实输出

导出不含答案的输入：

```sh
mkdir -p "$RUN_DIR"
python3 -B "$SKILL_DIR/evals/adapter.py" prompt scope-limits > "$RUN_DIR/scope-limits.prompt.txt"
python3 -B "$SKILL_DIR/evals/adapter.py" prompt no-match > "$RUN_DIR/no-match.prompt.txt"
```

让被测宿主分别读取这两个输入，将原始 JSON 判断保存为 `$RUN_DIR/scope-limits.json` 和 `$RUN_DIR/no-match.json`。另记宿主、模型、输入来源、实际调用时间和可取得的成本；不要把 bundled 回放文件复制后标成真实输出。没有重新执行的保存产物只能证明这次回放的得分。

```sh
python3 -B "$SKILL_DIR/evals/adapter.py" grade scope-limits "$RUN_DIR/scope-limits.json"
```

退出码 `0` 表示结构与本样例的选择集合通过，`1` 表示被测判断不满足合同，`2` 表示评测输入、文件或参考绑定出错。评测错误单独排查，不能作为模型能力失败，也不能算通过。所有预定案例都应报告，不能忽略缺失输出；跨运行比较需固定同一输入、标签、代码、宿主权限和重试口径。

## 使用已安装的 Promptfoo

先用实际安装位置的 `promptfoo --version` 确认可执行；不要因示例命令存在就声称已经运行。

```sh
# 默认读取两个合成回放；仅验证评测接线。
promptfoo eval -c "$SKILL_DIR/evals/promptfooconfig.yaml" --no-cache

# 改为回放指定目录里的已保存判断；仍不会调用模型。
WORKFLOW_EVAL_RESPONSES_DIR="$RUN_DIR" \
  promptfoo eval -c "$SKILL_DIR/evals/promptfooconfig.yaml" --no-cache
```

Promptfoo 的 Python provider 调用 `call_api`，动态 prompt 调用 `create_prompt`，Python assertion 调用 `get_assert`。回放保留原始返回，坏 JSON 不自动修复，缺文件返回 provider error。配置中的相对文件路径由 Promptfoo 解析；adapter 内的输入和默认回放路径相对脚本定位，指定目录可使用绝对路径。

新增材料时先使用现有 `judgment_handoff.py prepare` 生成请求，再独立审核选择标签并绑定新 request_id，同时增加正反例。不要为了通过旧结果直接改标签。保存输入、配置、评分器及原脚本的 Git revision 或内容哈希，变更后单独报告新版本成绩。Promptfoo CLI、真实宿主调用、理由质量、人工校准和成本应分别记录为实际运行状态，Python 回调测试通过不能替它们背书。

接口依据核对于 2026-10-23：[Python provider](https://www.promptfoo.dev/docs/providers/python/)、[动态 prompt](https://www.promptfoo.dev/docs/configuration/prompts/)、[Python assertion](https://www.promptfoo.dev/docs/configuration/expected-outputs/python/)。
