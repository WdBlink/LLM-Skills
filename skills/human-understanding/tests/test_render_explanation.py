import copy
from html import escape
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import unittest


SKILL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("render_explanation", SKILL / "scripts/render_explanation.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags, self.data, self.contents, self.active = [], [], [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag in ("pre", "script", "td", "th"):
            self.active.append((tag, []))

    def handle_data(self, data):
        self.data.append(data)
        for _, chunks in self.active:
            chunks.append(data)

    def handle_endtag(self, tag):
        if self.active and self.active[-1][0] == tag:
            _, chunks = self.active.pop()
            self.contents.append((tag, "".join(chunks)))


class RenderTests(unittest.TestCase):
    def setUp(self):
        bundle = json.loads((SKILL / "examples/evidence.json").read_text())
        bundle["evidence"][0]["metric"]["n"] = None
        bundle["evidence"][0]["missing"] = {"n": "原始记录未给样本数"}
        bundle["evidence"].append({"id": "E2", "source_id": "S1", "kind": "text",
                                    "lines": [2, 3], "excerpt": "逐字摘录。\n包括第二行与  空格。"})
        self.raw = "  原文 <script>alert('source')</script> & 引号\"\n`````\n[链接](javascript:alert(1))\n保留尾部空格  "
        self.request = {
            "task": {"question": "这些结果说明什么？", "reader": "工程师", "scope": "本轮一次实验", "format": "html"},
            "sources": [{"id": "S1", "path": '/tmp/source <draft> "one".txt', "sha256": "0" * 64,
                         "units": [{"id": "S1:P1", "text": self.raw}]}],
            "gaps": ["原始材料缺少成本结果"],
            "evidence": {"bundle": bundle, "path": "/tmp/task/evidence.json",
                         "report": {"status": "pass_with_gaps", "gaps": ["E1.n：样本数缺失"], "semantic_review": "not_checked"}},
        }
        self.decision = {
            "request_id": "sha256:" + "1" * 64, "status": "ready", "format": "html",
            "format_reason": "比较证据与限制", "title": "本轮结果解释", "answer": "延迟降低，整体效果仍未知。",
            "limitations": ["仅能解释本轮材料"],
            "sections": [{"id": "C1", "title": "数值变化", "kind": "observation", "text": "延迟由 100 ms 降至 90 ms。",
                          "scope": "本轮所列硬件与并发", "unit_ids": ["S1:P1"], "evidence_ids": ["E1", "E2"],
                          "limitations": ["尚未验证重复运行"],
                          "example": {"kind": "source", "input": "100 ms 与 90 ms", "rule": "相减", "intermediate": "90 − 100", "output": "−10 ms"}},
                         {"id": "C2", "title": "演示", "kind": "demonstration", "text": "用虚构数字解释比较方法。",
                          "scope": "只解释公式", "unit_ids": [], "evidence_ids": [], "limitations": ["演示不是观测"],
                          "example": {"kind": "demonstration", "input": "10", "rule": "减去 3", "intermediate": "10 − 3", "output": "7"}}],
        }

    def test_original_evidence_metrics_and_locations_are_preserved(self):
        page = Page(renderer.render_html(self.request, self.decision))
        markdown = renderer.render_evidence_markdown(self.request, self.decision)
        self.assertIn(("pre", self.raw), page.contents)
        excerpt = self.request["evidence"]["bundle"]["evidence"][1]["excerpt"]
        self.assertIn(("pre", excerpt), page.contents)
        self.assertIn(self.raw, markdown)
        self.assertIn(excerpt, markdown)
        for name in ("metric", "computed", "missing"):
            for field, value in self.request["evidence"]["bundle"]["evidence"][0][name].items():
                with self.subTest(group=name, field=field):
                    self.assertIn(("th", field), page.contents)
                    self.assertIn(("td", renderer.display(value)), page.contents)
                    self.assertIn(renderer.markdown_text(field), markdown)
                    self.assertIn(renderer.markdown_text(value), markdown)
        for location in ("/tmp/task/results.json", "/runs/0", "行 2–3", "S1:P1"):
            self.assertIn(location, "".join(page.data))
            self.assertIn(renderer.markdown_text(location), markdown)

    def test_all_limits_and_gaps_remain_in_unfiltered_first_level(self):
        html = renderer.render_html(self.request, self.decision)
        markdown = renderer.render_markdown(self.request, self.decision)
        aside = html.split('<aside id="global-limits"', 1)[1].split("</aside>", 1)[0]
        limits = (self.decision["limitations"] + self.request["gaps"] +
                  self.request["evidence"]["report"]["gaps"] +
                  self.request["evidence"]["bundle"]["claims"][0]["limitations"] +
                  [limit for section in self.decision["sections"] for limit in section["limitations"]])
        for limit in limits:
            with self.subTest(limit=limit):
                self.assertIn(escape(limit), aside)
                self.assertIn(renderer.markdown_text(limit), markdown)
        self.assertNotIn("<details", aside)
        self.assertNotIn(" hidden", aside)
        for section in self.decision["sections"]:
            card = html.split(f"<h2>{section['id']} ·", 1)[1].split("</article>", 1)[0]
            visible = card.split("<details>", 1)[0]
            self.assertIn(escape(section["scope"]), visible)
            self.assertIn(escape(section["limitations"][0]), visible)
        self.assertIn("仍需语义复核", aside)
        self.assertIn("存在缺口", aside)

    def test_source_declarations_do_not_certify_secondary_material_as_primary(self):
        source = self.request["evidence"]["bundle"]["sources"][0]
        source.update(kind="summary", status="available", version="v2 <review>")
        for render in (renderer.render_evidence_markdown, renderer.render_html):
            result = render(self.request, self.decision)
            self.assertIn("来源材料", result)
            self.assertNotIn("原始证据", result)
            self.assertIn("来源声明（非认证）：类型：summary；状态：available；版本：v2 &lt;review&gt;", result)
        source["kind"] = "primary"
        self.assertIn("来源声明（非认证）：类型：primary", renderer.render_html(self.request, self.decision))

    def test_reader_output_omits_host_format_reason_and_technical_check_status(self):
        self.decision["format_reason"] = "host-only choice explanation"
        for render in (renderer.render_markdown, renderer.render_html):
            result = render(self.request, self.decision)
            self.assertNotIn("host-only choice explanation", result)
            self.assertNotIn("表达方式", result)
            self.assertNotIn("not_checked", result)
            self.assertNotIn("not\\_checked", result)
            self.assertNotIn("pass_with_gaps", result)
            self.assertNotIn("pass\\_with\\_gaps", result)
            self.assertIn("已核对记录与本地文件的一致性（仍需语义复核）", result)

    def test_dynamic_html_and_markdown_are_inert_text(self):
        attack = '\"><img src=x onerror="alert(1)"><script>alert(2)</script> & [attack](javascript:alert(3))'
        self.decision.update(title=attack, answer=attack, format_reason=attack, limitations=[attack])
        self.request["task"].update(question=attack, reader=attack, scope=attack)
        self.request["sources"][0]["path"] = attack
        section = self.decision["sections"][0]
        section.update(title=attack, text=attack, scope=attack, limitations=[attack])
        section["example"]["input"] = attack
        self.request["evidence"]["bundle"]["evidence"][0]["metric"]["conditions"] = attack
        html = renderer.render_html(self.request, self.decision)
        page = Page(html)
        scripts = [content for tag, content in page.contents if tag == "script"]
        self.assertEqual(scripts, [renderer.FILTER_SCRIPT])
        self.assertFalse(any(tag in ("img", "iframe", "svg", "a") for tag, _ in page.tags))
        self.assertFalse(any(name.startswith("on") for _, attrs in page.tags for name in attrs))
        self.assertIn(escape(attack), html)
        self.assertIn(("td", attack), page.contents)
        for render in (renderer.render_markdown, renderer.render_evidence_markdown):
            markdown = render(self.request, self.decision)
            self.assertIn("&lt;script&gt;alert\\(2\\)&lt;/script&gt;", markdown)
            self.assertIn("\\[attack\\]\\(javascript:alert\\(3\\)\\)", markdown)
            self.assertNotIn('<img src=x', markdown)

    def test_markdown_fence_cannot_be_closed_by_source_content(self):
        markdown = renderer.render_evidence_markdown(self.request, self.decision)
        self.assertIn(f"``````text\n{self.raw}\n``````", markdown)
        self.assertNotIn(f"\n```text\n{self.raw}\n```\n", markdown)

    def test_examples_keep_each_step_and_distinguish_demonstrations(self):
        html = renderer.render_html(self.request, self.decision)
        markdown = renderer.render_markdown(self.request, self.decision)
        for label in ("源材料中的示例", "演示示例（不是真实观测）", "输入", "规则", "中间状态", "结果"):
            self.assertIn(label, html)
            self.assertIn(label, markdown)
        for section in self.decision["sections"]:
            for key in ("input", "rule", "intermediate", "output"):
                self.assertIn(escape(section["example"][key]), html)
                self.assertIn(renderer.markdown_text(section["example"][key]), markdown)

    def test_filter_controls_have_labels_status_reset_and_empty_state(self):
        page = Page(renderer.render_html(self.request, self.decision))
        by_id = {attrs["id"]: (tag, attrs) for tag, attrs in page.tags if "id" in attrs}
        self.assertEqual(by_id["kind-filter"][0], "select")
        self.assertTrue(any(tag == "label" and attrs.get("for") == "kind-filter" for tag, attrs in page.tags))
        self.assertEqual(by_id["reset-filter"], ("button", {"id": "reset-filter", "type": "button"}))
        self.assertEqual(by_id["visible-count"][1]["aria-live"], "polite")
        self.assertIn("hidden", by_id["empty-result"][1])
        self.assertIn("显示 2 / 2 项", page.data)
        self.assertEqual([attrs["data-kind"] for tag, attrs in page.tags if tag == "article"], ["observation", "demonstration"])
        self.assertTrue(any(tag == "option" and attrs.get("value") == "missing" for tag, attrs in page.tags))

    def test_rendering_is_deterministic_and_does_not_mutate_inputs(self):
        original = copy.deepcopy((self.request, self.decision))
        for render in (renderer.render_markdown, renderer.render_evidence_markdown, renderer.render_html):
            self.assertEqual(render(self.request, self.decision), render(self.request, self.decision))
            self.assertEqual((self.request, self.decision), original)

    def test_optional_evidence_bundle_is_not_required(self):
        self.request["evidence"] = None
        self.request["gaps"] = []
        self.decision["limitations"] = []
        for section in self.decision["sections"]:
            section["evidence_ids"] = []
            section["limitations"] = []
        for render in (renderer.render_markdown, renderer.render_html):
            result = render(self.request, self.decision)
            self.assertIn("未记录额外限制；不代表结论无限适用。", result)
            self.assertIn("S1:P1", result)
        for section in self.decision["sections"]:
            section["unit_ids"] = []
        appendix = renderer.render_evidence_markdown(self.request, self.decision)
        self.assertIn("本次解释未引用来源段落或结构化证据。", appendix)
        self.assertNotIn('<a id=', appendix)
        self.assertNotIn("evidence.md#", renderer.render_markdown(self.request, self.decision))

    def test_markdown_keeps_compact_links_and_deduplicates_evidence_attachment(self):
        self.decision["sections"][1]["unit_ids"] = ["S1:P1"]
        self.decision["sections"][1]["evidence_ids"] = ["E1", "E2"]
        body = renderer.render_markdown(self.request, self.decision)
        appendix = renderer.render_evidence_markdown(self.request, self.decision)
        self.assertNotIn(self.raw, body)
        self.assertNotIn("| 字段 |", body)
        self.assertNotIn("**问题：**", body)
        self.assertNotIn("**读者：**", body)
        self.assertIn(self.decision["answer"], body)
        self.assertIn(self.request["task"]["scope"], body)
        self.assertEqual(appendix.count(self.raw), 1)
        for index, identifier in enumerate(("S1:P1", "E1", "E2"), 1):
            self.assertEqual(body.count(f"[{identifier}](evidence.md#ref-{index})"), 2)
            self.assertEqual(appendix.count(f'<a id="ref-{index}"></a>'), 1)
            self.assertEqual(appendix.count(f"## 来源材料 {identifier}\n"), 1)
        anchors = re.findall(r'<a id="(ref-\d+)"></a>', appendix)
        self.assertEqual(set(re.findall(r"evidence\.md#(ref-\d+)", body)), set(anchors))

    def test_markdown_anchors_do_not_embed_untrusted_identifiers(self):
        identifier = '\"><img src=x onerror=alert(1)>[odd](javascript:bad)'
        self.request["sources"][0]["units"][0]["id"] = identifier
        self.decision["sections"][0]["unit_ids"] = [identifier]
        body = renderer.render_markdown(self.request, self.decision)
        appendix = renderer.render_evidence_markdown(self.request, self.decision)
        self.assertIn(f"[{renderer.markdown_text(identifier)}](evidence.md#ref-1)", body)
        self.assertEqual(re.findall(r'<a id="([^"]+)"></a>', appendix), ["ref-1", "ref-2", "ref-3"])
        self.assertNotIn("<img", body)
        self.assertNotIn("<img", appendix)

    def test_global_limits_deduplicate_only_identical_content(self):
        self.decision["limitations"] = ["仅有一次运行", "仅有一次运行"]
        self.request["gaps"] = ["仅有一次运行", "仅有两次运行"]
        body = renderer.render_markdown(self.request, self.decision)
        first_level = body.split("## 限制与缺口", 1)[1].split("## C1", 1)[0]
        self.assertEqual(first_level.count("仅有一次运行"), 1)
        self.assertEqual(first_level.count("仅有两次运行"), 1)


if __name__ == "__main__":
    unittest.main()
