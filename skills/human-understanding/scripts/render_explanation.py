"""Render validated explanation data without executing model or source markup."""

from html import escape
import json
from pathlib import Path
import re


KINDS = {
    "observation": "观察", "inference": "推断", "missing": "缺失",
    "conflicting": "冲突", "recommendation": "建议", "explanation": "解释",
    "demonstration": "演示",
}
EXAMPLE_FIELDS = {"input": "输入", "rule": "规则", "intermediate": "中间状态", "output": "结果"}
CHECK_NOTES = {
    "pass": "已核对记录与本地文件的一致性（仍需语义复核）。",
    "pass_with_gaps": "已核对记录与本地文件的一致性（仍需语义复核）；存在缺口，详见上方。",
    "fail": "记录与本地文件的一致性检查未通过；仍需修复并复核语义。",
}


def display(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def markdown_text(value):
    return re.sub(r"([\\`*_{}\[\]()#+.!|~\-])", r"\\\1", escape(display(value), quote=False))


def fenced(text):
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}"


def global_limits(request, decision):
    limits = list(decision["limitations"]) + list(request["gaps"])
    for section in decision["sections"]:
        limits.extend(f"{section['id']}：{limit}" for limit in section["limitations"])
    if request["evidence"]:
        for claim in request["evidence"]["bundle"]["claims"]:
            limits.extend(f"{claim['id']}：{limit}" for limit in claim["limitations"])
        limits.extend(request["evidence"]["report"]["gaps"])
    return list(dict.fromkeys(limits)) or ["未记录额外限制；不代表结论无限适用。"]


def references(request, section):
    units = {unit["id"]: (source, unit) for source in request["sources"] for unit in source["units"]}
    for uid in section["unit_ids"]:
        source, unit = units[uid]
        yield {"key": ("unit", uid), "id": uid, "path": source["path"], "location": f"段落 {uid}",
               "text": unit["text"], "tables": {}, "declaration": ""}
    if section["evidence_ids"]:
        evidence = request["evidence"]
        sources = {source["id"]: source for source in evidence["bundle"]["sources"]}
        items = {item["id"]: item for item in evidence["bundle"]["evidence"]}
        for eid in section["evidence_ids"]:
            item = items[eid]
            source = sources[item["source_id"]]
            path = str(Path(evidence["path"]).parent / source["path"]) if source.get("path") else source["url"]
            metric = item["kind"] == "metric"
            yield {"key": ("evidence", eid), "id": eid, "path": path,
                   "location": item["pointer"] if metric else f"行 {item['lines'][0]}–{item['lines'][1]}",
                   "text": None if metric else item["excerpt"],
                   "declaration": "；".join(f"{label}：{display(source[key])}" for key, label in
                                          (("kind", "类型"), ("status", "状态"), ("version", "版本")) if key in source),
                   "tables": {name: item[name] for name in ("metric", "computed", "missing") if name in item} if metric else {}}


def reference_catalog(request, decision):
    catalog = {}
    for section in decision["sections"]:
        for reference in references(request, section):
            if reference["key"] not in catalog:
                catalog[reference["key"]] = {**reference, "anchor": f"ref-{len(catalog) + 1}"}
    return catalog


def render_markdown(request, decision):
    md = markdown_text
    task = request["task"]
    catalog = reference_catalog(request, decision)
    parts = [f"# {md(decision['title'])}", md(decision["answer"]),
             f"**材料范围：** {md(task['scope'])}",
             "## 限制与缺口", *[f"- {md(limit)}" for limit in global_limits(request, decision)]]
    if request["evidence"]:
        parts.append(CHECK_NOTES[request["evidence"]["report"]["status"]])
    for section in decision["sections"]:
        parts.extend([f"## {md(section['id'])} · {md(section['title'])}",
                      f"**{KINDS[section['kind']]}**", md(section["text"]),
                      f"**适用范围：** {md(section['scope'])}"])
        parts.extend(f"- **本节限制：** {md(limit)}" for limit in section["limitations"])
        if section.get("example"):
            example = section["example"]
            parts.append("### 源材料中的示例" if example["kind"] == "source" else "### 演示示例（不是真实观测）")
            parts.extend(f"**{label}：** {md(example[key])}" for key, label in EXAMPLE_FIELDS.items())
        refs = [catalog[reference["key"]] for reference in references(request, section)]
        if refs:
            parts.append("证据：" + "、".join(f"[{md(ref['id'])}](evidence.md#{ref['anchor']})" for ref in refs))
    return "\n\n".join(parts) + "\n"


def render_evidence_markdown(request, decision):
    md = markdown_text
    parts = ["# 来源材料", f"**材料范围：** {md(request['task']['scope'])}"]
    catalog = reference_catalog(request, decision)
    if not catalog:
        parts.append("本次解释未引用来源段落或结构化证据。")
    for reference in catalog.values():
        parts.extend([f'<a id="{reference["anchor"]}"></a>',
                      f"## 来源材料 {md(reference['id'])}",
                      f"来源：{md(reference['path'])}；位置：{md(reference['location'])}"])
        if reference["declaration"]:
            parts.append(f"来源声明（非认证）：{md(reference['declaration'])}")
        if reference["text"] is not None:
            parts.append(fenced(reference["text"]))
        for name, fields in reference["tables"].items():
            parts.append(f"**{md(name)}**\n\n| 字段 | 记录值 |\n| --- | --- |\n" +
                         "\n".join(f"| {md(key)} | {md(value).replace(chr(10), '<br>')} |" for key, value in fields.items()))
    return "\n\n".join(parts) + "\n"


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f6f7f9;color:#17212b;font:17px/1.65 system-ui,sans-serif}
main{max-width:960px;margin:auto;padding:32px 20px 64px}h1,h2,h3{line-height:1.3}h1{font-size:2rem}
header,aside,article,.controls{background:white;border:1px solid #d5dce3;border-radius:12px;padding:24px;margin-bottom:20px}
aside{border-left:5px solid #955600}.text{white-space:pre-wrap;overflow-wrap:anywhere}h1,h2,h3,li,td,th{overflow-wrap:anywhere}
.meta{color:#425466}.scope,.section-limits{border-left:3px solid #697e91;padding-left:12px}
label{font-weight:650}select,button{font:inherit;padding:8px 12px;border:1px solid #6c7a88;border-radius:6px;background:white;color:inherit}
button{cursor:pointer}select:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #0b67bc;outline-offset:3px}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}.controls p{margin:0;width:100%}
summary{cursor:pointer;font-weight:650;padding:8px 0}details{border-top:1px solid #d5dce3;margin-top:20px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f6;padding:16px;border-radius:6px;font-size:.9rem}
table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:.95rem}th,td{text-align:left;vertical-align:top;border:1px solid #d5dce3;padding:8px;white-space:pre-wrap}th:first-child{width:30%}
dt{font-weight:650}dd{margin:0 0 12px}code{overflow-wrap:anywhere}[hidden]{display:none!important}
@media(max-width:600px){main{padding:16px 10px 40px}header,aside,article,.controls{padding:16px}h1{font-size:1.6rem}select{max-width:100%}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
"""

FILTER_SCRIPT = """
const filter = document.getElementById('kind-filter');
const cards = Array.from(document.querySelectorAll('article[data-kind]'));
function updateFilter() {
  let visible = 0;
  for (const card of cards) {
    card.hidden = filter.value !== 'all' && card.dataset.kind !== filter.value;
    if (!card.hidden) visible += 1;
  }
  document.getElementById('visible-count').textContent = `显示 ${visible} / ${cards.length} 项`;
  document.getElementById('empty-result').hidden = visible !== 0;
}
filter.addEventListener('change', updateFilter);
document.getElementById('reset-filter').addEventListener('click', () => {
  filter.value = 'all';
  updateFilter();
  filter.focus();
});
updateFilter();
"""


def render_html(request, decision):
    def text(value):
        return escape(display(value), quote=True)

    task = request["task"]
    parts = ["<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>{text(decision['title'])}</title><style>{STYLE}</style></head><body><main>",
             f"<header><h1>{text(decision['title'])}</h1><p class=\"text\">{text(decision['answer'])}</p>",
             f"<p class=\"text\"><strong>问题：</strong>{text(task['question'])}</p>",
             f"<p class=\"text meta\">读者：{text(task['reader'])}</p>",
             f"<p class=\"text\"><strong>材料范围：</strong>{text(task['scope'])}</p></header>",
             '<aside id="global-limits" aria-labelledby="limits-title"><h2 id="limits-title">限制与缺口</h2><ul>',
             *[f'<li class="text">{text(limit)}</li>' for limit in global_limits(request, decision)], "</ul>"]
    if request["evidence"]:
        parts.append(f"<p>{CHECK_NOTES[request['evidence']['report']['status']]}</p>")
    parts.extend(['</aside><div class="controls"><label for="kind-filter">按内容类别筛选</label>',
                  '<select id="kind-filter"><option value="all">全部类别</option>',
                  *[f'<option value="{kind}">{label}</option>' for kind, label in KINDS.items()],
                  '</select><button id="reset-filter" type="button">重置筛选</button>',
                  f'<p id="visible-count" role="status" aria-live="polite">显示 {len(decision["sections"])} / {len(decision["sections"])} 项</p></div>',
                  '<p id="empty-result" role="status" hidden>没有匹配的内容，请选择其他类别或重置筛选。</p>'])
    for section in decision["sections"]:
        parts.extend([f'<article data-kind="{text(section["kind"])}">',
                      f"<h2>{text(section['id'])} · {text(section['title'])}</h2>",
                      f"<p class=\"meta\">{KINDS[section['kind']]}</p><p class=\"text\">{text(section['text'])}</p>",
                      f"<p class=\"text scope\"><strong>适用范围：</strong>{text(section['scope'])}</p>"])
        if section["limitations"]:
            parts.extend(['<div class="section-limits"><strong>本节限制</strong><ul>',
                          *[f'<li class="text">{text(limit)}</li>' for limit in section["limitations"]], "</ul></div>"])
        if section.get("example"):
            example = section["example"]
            label = "源材料中的示例" if example["kind"] == "source" else "演示示例（不是真实观测）"
            parts.extend([f"<h3>{label}</h3><dl>",
                          *[f'<dt>{label}</dt><dd class="text">{text(example[key])}</dd>' for key, label in EXAMPLE_FIELDS.items()], "</dl>"])
        refs = list(references(request, section))
        if refs:
            parts.append(f"<details><summary>来源材料（{len(refs)}）</summary>")
            for reference in refs:
                parts.extend([f"<h3>{text(reference['id'])}</h3>",
                              f"<p class=\"text meta\">来源：{text(reference['path'])}；位置：{text(reference['location'])}</p>"])
                if reference["declaration"]:
                    parts.append(f"<p class=\"text meta\">来源声明（非认证）：{text(reference['declaration'])}</p>")
                if reference["text"] is not None:
                    parts.append(f"<pre>{text(reference['text'])}</pre>")
                for name, fields in reference["tables"].items():
                    parts.append(f"<h3>{text(name)}</h3><table><thead><tr><th scope=\"col\">字段</th><th scope=\"col\">记录值</th></tr></thead><tbody>")
                    parts.extend(f"<tr><th scope=\"row\">{text(key)}</th><td>{text(value)}</td></tr>" for key, value in fields.items())
                    parts.append("</tbody></table>")
            parts.append("</details>")
        parts.append("</article>")
    parts.append(f"</main><script>{FILTER_SCRIPT}</script></body></html>")
    return "\n".join(parts) + "\n"
