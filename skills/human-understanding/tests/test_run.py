"""CLI contracts with synthetic judgments; these tests do not certify semantics."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


SKILL = Path(__file__).resolve().parents[1]
SCRIPT = SKILL / "scripts/run.py"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.txt"
        self.original = "  来源记录：延迟 100 ms。\n原文 <script>不能执行</script>。\n\n尚未测试并发负载。\n"
        self.source.write_text(self.original, encoding="utf-8")
        self.request = self.root / "request.json"
        self.decision = self.root / "decision.json"
        self.output = self.root / "output"
        self.prepare()
        self.decision_data = self.make_decision()
        self.write_json(self.decision, self.decision_data)

    def write_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def cli(self, *arguments, error=None, script=SCRIPT):
        process = subprocess.run([sys.executable, "-B", str(script), *map(str, arguments)],
                                 cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 2 if error else 0, process.stderr)
        if error:
            self.assertEqual(process.stdout, "")
            report = json.loads(process.stderr)
            self.assertEqual(report["status"], "failed")
            self.assertIn(error, report["error"])
            return report
        self.assertEqual(process.stderr, "")
        return json.loads(process.stdout)

    def prepare(self, *, question="这份虚构记录支持什么？", output_format="text", extra=(), script=SCRIPT):
        return self.cli("prepare", "--source", self.source, "--question", question,
                        "--format", output_format, "--out", self.request, *extra, script=script)

    def make_decision(self, output_format="text"):
        return {
            "request_id": self.read_json(self.request)["request_id"], "status": "ready",
            "format": output_format, "format_reason": "仅用于验证程序合同", "title": "虚构记录说明",
            "answer": "记录仅支持限定范围内的观察。", "limitations": [],
            "sections": [{"id": "C1", "title": "观察范围", "kind": "observation",
                          "text": "本段是测试语义增量，不是来源正文。", "scope": "仅该虚构记录",
                          "unit_ids": ["S1:P1"], "evidence_ids": [], "limitations": []}],
        }

    def apply(self, *, artifact=None, output=None, error=None, script=SCRIPT):
        arguments = ["apply", "--request", self.request, "--decision", self.decision,
                     "--out", self.output if output is None else output]
        if artifact is not None:
            arguments += ["--artifact", artifact]
        return self.cli(*arguments, error=error, script=script)

    def review_data(self):
        pending = self.read_json(self.output / "review-request.json")
        checks = {name: {"status": "passed", "note": "固定测试记录，不代表实际语义或浏览器验收"}
                  for name in pending["required_checks"]}
        if pending["format"] == "text":
            for name in ("visual", "interaction"):
                checks[name]["status"] = "not_applicable"
        elif pending["format"] == "diagram":
            checks["interaction"]["status"] = "not_applicable"
        return {"request_id": pending["request_id"], "artifact_hashes": pending["artifact_hashes"], "checks": checks}

    def finish(self, review, *, name="completion.json", error=None):
        path = self.root / "review.json"
        self.write_json(path, review)
        return self.cli("finish", "--request", self.request, "--dir", self.output,
                        "--review", path, "--out", self.root / name, error=error)

    def use_format(self, output_format):
        self.request = self.root / (output_format + "-request.json")
        self.prepare(output_format=output_format)
        self.decision_data = self.make_decision(output_format)
        self.write_json(self.decision, self.decision_data)

    def use_evidence(self):
        bundle = {
            "version": 1, "task": {"question": "虚构测试", "reader": "测试者", "scope": "仅此记录"},
            "sources": [{"id": "S1", "path": "source.txt", "kind": "primary", "status": "available"}],
            "evidence": [{"id": "E1", "source_id": "S1", "kind": "text", "lines": [1, 2],
                          "excerpt": self.original.split("\n\n")[0]}],
            "claims": [{"id": "EC1", "text": "虚构观察", "scope": "本次测试", "kind": "observation",
                        "evidence_ids": ["E1"], "limitations": ["原证据限制必须保留"]}],
        }
        path = self.root / "evidence.json"
        self.write_json(path, bundle)
        self.request = self.root / "evidence-request.json"
        self.prepare(extra=("--evidence", path, "--missing", "尚未取得另一负载记录"))
        self.decision_data = self.make_decision()
        self.decision_data["sections"][0].update(unit_ids=[], evidence_ids=["E1"])
        self.write_json(self.decision, self.decision_data)
        return path

    def test_complete_flow_copies_source_and_resumes_idempotently(self):
        self.assertEqual(self.prepare()["write"], "unchanged")
        report = self.apply()
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["semantic_review"], "not_checked")
        markdown = self.output / "explanation.md"
        evidence = (self.output / "evidence.md").read_text()
        self.assertIn(self.original.split("\n\n")[0], evidence)
        self.assertNotIn(self.original.split("\n\n")[0], markdown.read_text())
        anchors = re.findall(r"\]\(evidence\.md#([^)]+)\)", markdown.read_text())
        self.assertTrue(anchors)
        for anchor in anchors:
            self.assertIn(f'<a id="{anchor}"></a>', evidence)
        self.assertNotIn(self.original.split("\n\n")[0], json.dumps(self.decision_data, ensure_ascii=False))
        self.assertIn("evidence.md", report["artifact_hashes"])
        for name, expected_hash in report["artifact_hashes"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), expected_hash)
        before = (markdown.read_bytes(), markdown.stat().st_mtime_ns)
        (self.output / "receipt.json").unlink()
        (self.output / "review-request.json").unlink()
        self.assertEqual(self.apply()["status"], "needs_review")
        self.assertEqual((markdown.read_bytes(), markdown.stat().st_mtime_ns), before)
        unchanged = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        self.apply()
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}, unchanged)
        review = self.review_data()
        self.assertEqual(self.finish(review)["status"], "complete")
        completion = self.root / "completion.json"
        before = (completion.read_bytes(), completion.stat().st_mtime_ns)
        self.finish(review)
        self.assertEqual((completion.read_bytes(), completion.stat().st_mtime_ns), before)
        self.assertEqual(self.source.read_text(), self.original)

    def test_evidence_and_its_limits_are_copied_without_model_repetition(self):
        evidence = self.use_evidence()
        original = evidence.read_bytes()
        duplicate = copy.deepcopy(self.decision_data["sections"][0])
        duplicate["id"] = "C2"
        self.decision_data["sections"].append(duplicate)
        self.write_json(self.decision, self.decision_data)
        self.apply()
        markdown = (self.output / "explanation.md").read_text()
        appendix = (self.output / "evidence.md").read_text()
        excerpt = self.original.split("\n\n")[0]
        self.assertEqual(appendix.count(excerpt), 1)
        self.assertNotIn(excerpt, markdown)
        anchors = re.findall(r"\[E1\]\(evidence\.md#([^)]+)\)", markdown)
        self.assertEqual(len(anchors), 2)
        self.assertEqual(len(set(anchors)), 1)
        self.assertIn(f'<a id="{anchors[0]}"></a>', appendix)
        for value in ("原证据限制必须保留", "尚未取得另一负载记录", "E1"):
            self.assertIn(value, markdown)
        self.assertEqual(self.read_json(self.output / "receipt.json")["evidence_check"]["status"], "pass")
        self.assertEqual(evidence.read_bytes(), original)

    def test_changed_source_or_question_rejects_old_judgment(self):
        self.source.write_text(self.original + "来源新增内容。\n", encoding="utf-8")
        self.apply(error="Source changed")
        self.assertFalse(self.output.exists())
        self.source.write_text(self.original, encoding="utf-8")
        self.request = self.root / "new-question.json"
        self.prepare(question="现在询问另一个关系。")
        self.apply(error="Stale or unrelated decision")
        self.assertFalse(self.output.exists())

    def test_changed_rules_in_isolated_skill_reject_old_request(self):
        copied = self.root / "isolated-skill"
        shutil.copytree(SKILL, copied, ignore=shutil.ignore_patterns("__pycache__"))
        script = copied / "scripts/run.py"
        self.request = self.root / "isolated-request.json"
        self.prepare(script=script)
        self.write_json(self.decision, self.make_decision())
        rules = copied / "references/explanation-methods.md"
        rules.write_text(rules.read_text() + "\n测试规则版本发生变化。\n")
        self.apply(script=script, error="implementation/rules changed")
        self.assertFalse(self.output.exists())

    def test_unknown_duplicate_references_and_extra_fields_are_rejected(self):
        self.use_evidence()
        cases = [("unit_ids", ["S9:P9"]), ("unit_ids", ["S1:P1", "S1:P1"]),
                 ("evidence_ids", ["E9"]), ("evidence_ids", ["E1", "E1"]),
                 ("extra_section_field", "not allowed"), ("extra_decision_field", "not allowed")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                decision = copy.deepcopy(self.decision_data)
                target = decision if field == "extra_decision_field" else decision["sections"][0]
                target[field] = value
                self.write_json(self.decision, decision)
                self.apply(error="Expected fields" if field.startswith("extra") else "Duplicate or unknown")
                self.assertFalse(self.output.exists())

    def test_needs_input_publishes_only_pending_record(self):
        pending = {"request_id": self.decision_data["request_id"], "status": "needs_input",
                   "missing": ["缺少比较基线，不能判断提升"]}
        self.write_json(self.decision, pending)
        self.assertEqual(self.apply()["status"], "needs_input")
        self.assertEqual({p.name for p in self.output.iterdir()}, {"pending.json"})
        self.assertEqual(self.read_json(self.output / "pending.json"), pending)
        artifact = self.root / "unused.html"
        artifact.write_text("<p>不能交付</p>")
        self.apply(artifact=artifact, error="needs_input cannot publish")
        self.assertEqual({p.name for p in self.output.iterdir()}, {"pending.json"})

    def test_output_conflict_is_checked_before_any_publication(self):
        self.output.mkdir()
        existing = self.output / "receipt.json"
        existing.write_bytes(b"user-owned previous result\n")
        self.apply(error="Output conflict")
        self.assertEqual({p.name for p in self.output.iterdir()}, {"receipt.json"})
        self.assertEqual(existing.read_bytes(), b"user-owned previous result\n")
        self.assertEqual(self.source.read_text(), self.original)

    def test_input_aliases_cannot_be_used_as_outputs(self):
        self.apply()
        review_path = self.root / "review.json"
        self.write_json(review_path, self.review_data())
        for phase, source in (("prepare", self.source), ("finish", review_path)):
            for alias in ("direct", "symlink", "hardlink"):
                with self.subTest(phase=phase, alias=alias):
                    target = source if alias == "direct" else self.root / (phase + "-" + alias)
                    if alias == "symlink":
                        target.symlink_to(source)
                    elif alias == "hardlink":
                        os.link(source, target)
                    original = source.read_bytes()
                    args = (["prepare", "--source", self.source, "--question", "测试", "--out", target]
                            if phase == "prepare" else
                            ["finish", "--request", self.request, "--dir", self.output,
                             "--review", review_path, "--out", target])
                    self.cli(*args, error="Output must differ from input")
                    self.assertEqual(source.read_bytes(), original)
        for alias in ("symlink", "hardlink"):
            with self.subTest(phase="apply", alias=alias):
                output = self.root / ("alias-output-" + alias)
                output.mkdir()
                target = output / "explanation.md"
                target.symlink_to(self.source) if alias == "symlink" else os.link(self.source, target)
                self.apply(output=output, error="Output must differ from input")
                self.assertEqual(self.source.read_text(), self.original)
                self.assertEqual({p.name for p in output.iterdir()}, {"explanation.md"})

    def test_requested_format_and_custom_artifact_contract(self):
        for output_format, suffix, content in (("diagram", ".svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'),
                                               ("html", ".html", b'<html><body>custom teaching steps</body></html>')):
            with self.subTest(format=output_format):
                self.use_format(output_format)
                self.output = self.root / (output_format + "-output")
                wrong = copy.deepcopy(self.decision_data)
                wrong["format"] = "text"
                self.write_json(self.decision, wrong)
                self.apply(error="Requested format must be preserved")
                self.write_json(self.decision, self.decision_data)
                if output_format == "diagram":
                    self.apply(error="diagram needs --artifact")
                artifact = self.root / ("custom" + suffix)
                artifact.write_bytes(content)
                wrong_suffix = self.root / "custom.txt"
                wrong_suffix.write_bytes(content)
                self.apply(artifact=wrong_suffix, error="Artifact type must match")
                artifact.write_bytes(b"")
                self.apply(artifact=artifact, error="Artifact is empty")
                self.assertFalse(self.output.exists())
                artifact.write_bytes(content)
                report = self.apply(artifact=artifact)
                self.assertEqual(report["status"], "needs_review")
                self.assertEqual((self.output / ("artifact" + suffix)).read_bytes(), content)
                self.assertIn("artifact" + suffix, report["artifact_hashes"])
                self.assertEqual(artifact.read_bytes(), content)
                self.assertEqual(self.finish(self.review_data(), name=output_format + "-completion.json")["status"], "complete")

    def test_generated_html_requires_visual_and_interaction_checks(self):
        self.use_format("html")
        self.apply()
        self.assertTrue((self.output / "explanation.html").is_file())
        review = self.review_data()
        for name in ("visual", "interaction"):
            with self.subTest(check=name):
                invalid = copy.deepcopy(review)
                invalid["checks"][name]["status"] = "not_applicable"
                self.finish(invalid, error=name + " is required")
                self.assertFalse((self.root / "completion.json").exists())
        self.assertEqual(self.finish(review)["status"], "complete")

    def test_review_format_cannot_downgrade_html_to_skip_required_checks(self):
        for requested in ("html", "auto"):
            with self.subTest(requested=requested):
                self.request = self.root / (requested + "-request.json")
                self.prepare(output_format=requested)
                self.write_json(self.decision, self.make_decision("html"))
                self.output = self.root / (requested + "-output")
                self.apply()
                path = self.output / "review-request.json"
                pending = self.read_json(path)
                pending["format"] = "text"
                self.write_json(path, pending)
                review = self.review_data()
                self.assertEqual(review["checks"]["interaction"]["status"], "not_applicable")
                error = ("Review format must preserve requested format" if requested == "html"
                         else "Review request differs from apply receipt")
                self.finish(review, error=error)
                self.assertFalse((self.root / "completion.json").exists())

    def test_review_manifest_cannot_omit_evidence_even_with_matching_receipt(self):
        self.apply()
        pending_path = self.output / "review-request.json"
        receipt_path = self.output / "receipt.json"
        pending = self.read_json(pending_path)
        receipt = self.read_json(receipt_path)
        del pending["artifact_hashes"]["evidence.md"]
        self.write_json(pending_path, pending)
        self.finish(self.review_data(), error="Review request differs from apply receipt")
        del receipt["artifact_hashes"]["evidence.md"]
        self.write_json(receipt_path, receipt)
        self.finish(self.review_data(), error="Missing required explanation/evidence artifacts")
        self.assertFalse((self.root / "completion.json").exists())

    def test_review_cannot_reuse_old_hashes_or_changed_artifacts(self):
        self.apply()
        review = self.review_data()
        stale = copy.deepcopy(review)
        stale["artifact_hashes"][next(iter(stale["artifact_hashes"]))] = "0" * 64
        self.finish(stale, error="Stale review binding")
        for name in review["artifact_hashes"]:
            with self.subTest(artifact=name):
                artifact = self.output / name
                original = artifact.read_bytes()
                artifact.write_bytes(original + b"\nchanged after review\n")
                self.finish(review, error="Artifact changed")
                self.assertFalse((self.root / "completion.json").exists())
                artifact.write_bytes(original)

    def test_failed_and_unperformed_checks_are_not_complete(self):
        self.apply()
        for failure, unperformed, expected in ((True, False, "needs_revision"),
                                               (False, True, "complete_with_limits"),
                                               (True, True, "needs_revision")):
            with self.subTest(failed=failure, not_run=unperformed):
                review = self.review_data()
                if failure:
                    review["checks"]["facts"]["status"] = "failed"
                if unperformed:
                    review["checks"]["understanding"]["status"] = "not_run"
                result = self.finish(review, name=f"completion-{failure}-{unperformed}.json")
                self.assertEqual(result["status"], expected)


if __name__ == "__main__":
    unittest.main()
