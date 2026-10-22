# 可复用执行合同

## 步骤与责任

| 阶段 | 输入 → 动作 → 输出 | 消费者与失败 |
| --- | --- | --- |
| prepare | 问题/读者/范围/本地材料/可选 evidence → 编号、快照、原有证据检查、绑定规则 → request.json | 当前宿主；缺文件或记录不一致退出 2；不静默跳过 |
| judge | request 与表达规则 → 选择形式、解释关系、形成来源 ID 和语义增量 → decision.json | apply；当前宿主完成，不调用额外模型 |
| apply | 请求/判断/可选自定义图或 HTML → 校验、复制证据、拼装产物 → explanation.md、evidence.md、可选 HTML/图、review-request.json、receipt.json | 当前宿主实际复核；成功状态 needs_review |
| finish | 当前请求/产物/宿主检查 → 核对文件哈希与检查记录 → completion.json | 交付；complete / complete_with_limits / needs_revision |

来源与规则绑定用于防误配，不是签名或事实认证。自定义 HTML 不会被脚本执行/清洗，需由可信宿主创作并实际检查；不得直接拿不可信来源网页当自定义脚本执行。

## 参数

`prepare`：`--question` 必需；`--source` 可多次；`--evidence` 为既有 bundle；`--reader` 默认“用户，背景未另说明”；`--scope` 默认“仅本次列出的材料与已披露缺口”；`--format` 默认 auto，可选 text/diagram/html；`--missing` 可重复；`--max-chars` 默认 12000、合法范围 1–50000；`--out` 为请求文件。

语义参数进入 request_id。材料通过工具获取时，来源 URL、版本/抓取时间可写入所附材料或 evidence 来源字段。prepare/应用时只读取本地文件，不刷新网络。无外部材料的概念演示可不传 source，但须在 scope 中说明依据，不能假装读过来源。

`apply`：必需 `--request --decision --out`，out 是目录。选 text 时生成 Markdown；html 无 artifact 时生成证据比较页。`--artifact` 接受 html 的 `.html` 或 diagram 的 `.svg/.png/.jpg/.webp/.mmd`；diagram 必须提供 artifact。仅复制不解释执行代码，所有自定义产物须遵守主入口的源文本转义、限制可见与实际检查要求。Mermaid 文件需用可用工具渲染/查看，文件写出不代表视觉检查。

原始资料读取上限按字符检查，总量不静默裁剪；字符不是 token。自定义产物最大 10 MiB。段落正文保留内部原文和换行，不保留分隔空行和段末换行。JSON 原材料同样作为文本快照；指标提取复用 evidence 合同，不猜 CSV/PDF 字段。

## 判断文件

宿主从实际 request 取得 request_id、`S1:P1` 等段落 ID，以及可选 bundle 的 evidence ID。合法完整形状如下；示例是合同演示，不是默认结论：

```json
{
  "request_id": "取当前 request 的真实 ID",
  "status": "ready",
  "format": "text",
  "format_reason": "这次一个结论与一个具体例子即可理解",
  "title": "这轮结果支持什么",
  "answer": "按本次材料写直接答案，保留范围。",
  "limitations": ["会影响判断的全局限制"],
  "sections": [{
    "id": "C1",
    "title": "一个具体关系",
    "kind": "explanation",
    "text": "按15条规则写解释，不重抄证据。",
    "scope": "此解释的适用条件",
    "unit_ids": [],
    "evidence_ids": [],
    "limitations": [],
    "example": {
      "kind": "demonstration",
      "input": "明确标注的演示输入",
      "rule": "使用的规则和假设",
      "intermediate": "读者可跟随的中间一步",
      "output": "该输入得到的结果"
    }
  }]
}
```

`ready` 表示可组装，不表示已交付。section 的 kind 是 observation/inference/missing/conflicting/recommendation/explanation/demonstration。每段必需 id/title/kind/text/scope/unit_ids/evidence_ids/limitations；example 可省略，存在时五字段都必需，kind 为 source 或 demonstration。来源例子和观察至少有一个引用，冲突至少两个引用；存在性校验不证明支持关系。推断说明假设，建议说明约束与改变建议的证据，缺失说明影响。术语和例子都按规则解释，不强制所有解释塞进实验模板。

引用使用 ID，程序按原材料复制。文字正文保留紧邻结论的限制和证据链接；完整原文、指标和来源声明集中到 evidence.md，同一引用只保存一次。附件也进入审核哈希绑定。原 evidence 已记录的限制和检查缺口会自动保留，不由模型静默删掉。材料本身为二手总结或虚构样例时，在答案/范围明确身份，声明 source 类型不是真实性认证。

关键材料不足且无法给有用解释时，仅返回：

```json
{"request_id":"当前真实ID","status":"needs_input","missing":["缺什么、影响哪个判断"]}
```

apply 保存 pending.json，返回 needs_input，不生成完成产物。可以有边界地回答时，使用 ready 并写 missing 段和首层限制，不把缺资料当失败实验。

## 成品检查文件

实际查看成品后，从 output/review-request.json 复制 request_id 和整个 artifact_hashes；不要自己假造哈希。review.json 形状：

```json
{
  "request_id": "当前真实ID",
  "artifact_hashes": {"explanation.md":"实际哈希", "evidence.md":"实际哈希"},
  "checks": {
    "facts": {"status":"passed","note":"对照哪些材料核对了哪些数值、限制与引用"},
    "language": {"status":"passed","note":"按15条规则复核，说明适用例外"},
    "understanding": {"status":"passed","note":"用哪个例子推演了哪一步或预测了什么变化"},
    "visual": {"status":"not_applicable","note":"本次纯文字"},
    "interaction": {"status":"not_applicable","note":"本次纯文字"}
  }
}
```

所有检查需要非空说明，status 仅 passed/failed/not_run/not_applicable。事实、表达与理解始终适用；图解适用 visual；HTML 适用 visual 和 interaction，不能填 not_applicable。浏览器 note 指明正常、边界/空结果、重置、键盘、小屏及适用的暂停检查与证据位置；未执行填 not_run，不写通过。

finish 仅核对内容绑定与记录结构，不认证检查是否执行。全部适用项 passed 才是 complete；有 failed 为 needs_revision；无 failed 但有 not_run 为 complete_with_limits，交付必须披露具体未验证项，不能称全验收通过。人为自检不等于独立审查，也不等于实际读者已经理解。

## 错误与恢复

CLI 正常返回 JSON 和退出码 0；结构/文件/冲突失败返回 stderr JSON、status=failed、退出码 2。状态与命令成功范围分开判断。needs_input 和 needs_review 均不是终态完成。

输出永不强制覆盖；同内容幂等。apply 在写入前检查所有目标是否冲突，最后发布 receipt；单文件原子写不是多文件事务。本地单写者流程中，中断后用相同参数恢复；内容变化用新输出目录，防止破坏旧结果。多个并发写者不在保证范围。

输出不能与当前输入、来源或其符号/硬链接别名相同。原数据或表达规则/实现发生变化时，旧请求被拒，重建请求再判断；外部来源刷新由宿主决定。修订判断一次仍失败时保留错误，先检查接口；没有新材料不无限重试模型。
