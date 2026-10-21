"""CLI contract checks; no network, model, or nonstandard dependencies required."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "judgment_handoff.py"


class JudgmentHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.article = self.directory / "article.txt"
        self.request = self.directory / "request.json"
        self.decision = self.directory / "decision.json"
        self.output = self.directory / "result.json"
        self.paragraphs = ["  第一段：原文  \r\n第二行。", "Second ‘paragraph’.  ", "第三段。"]
        self.raw = ("\ufeff" + "\r\n\r\n".join(self.paragraphs) + "\r\n").encode("utf-8")
        self.article.write_bytes(self.raw)
        self.prepared = self.run_cli(*self.prepare_args())
        self.request_data = json.loads(self.request.read_text(encoding="utf-8"))
        self.decision_data = {
            "request_id": self.request_data["request_id"],
            "selected": [{"unit_id": "u0003", "reason": "第三段相关"},
                         {"unit_id": "u0001", "reason": "  第一段相关  "}],
        }
        self.write_json(self.decision, self.decision_data)

    def write_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def prepare_args(self, output=None, goal="选取有关方法的段落"):
        return ["prepare", "--input", self.article, "--goal", goal,
                "--out", output if output is not None else self.request]

    def apply_args(self, output=None, request=None):
        return ["apply", "--request", request if request is not None else self.request,
                "--decision", self.decision, "--out", output if output is not None else self.output]

    def run_cli(self, *arguments, error=None):
        process = subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 2 if error else 0, process.stderr)
        if error:
            self.assertEqual(process.stdout, "")
            self.assertIn(error, process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            return None
        self.assertEqual(process.stderr, "")
        return json.loads(process.stdout)

    def test_prepare_and_apply_preserve_text_and_source_order(self):
        self.assertEqual(self.prepared["status"], "needs_judgment")
        self.assertEqual(self.prepared["unit_count"], 3)
        self.assertEqual(self.prepared["input_characters"], len(self.raw.decode("utf-8-sig")))
        self.assertEqual(self.request_data["input_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(self.request_data["units"], [
            {"unit_id": "u0001", "text": self.paragraphs[0]},
            {"unit_id": "u0002", "text": self.paragraphs[1]},
            {"unit_id": "u0003", "text": self.paragraphs[2]},
        ])
        report = self.run_cli(*self.apply_args())
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(result["selected"], [
            {"unit_id": "u0001", "text": self.paragraphs[0], "reason": "第一段相关"},
            {"unit_id": "u0003", "text": self.paragraphs[2], "reason": "第三段相关"},
        ])
        self.assertEqual(result["request_id"], self.request_data["request_id"])
        self.assertEqual(result["validation"]["semantic"], "not_verified")
        self.assertEqual(self.article.read_bytes(), self.raw)

    def test_identical_prepare_and_apply_are_idempotent(self):
        self.assertEqual(self.prepared["write"], "written")
        self.assertEqual(self.run_cli(*self.apply_args())["write"], "written")
        for path, arguments in [(self.request, self.prepare_args()), (self.output, self.apply_args())]:
            with self.subTest(command=arguments[0]):
                before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                self.assertEqual(self.run_cli(*arguments)["write"], "unchanged")
                self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), before)

    def test_changed_input_or_goal_rejects_old_decision(self):
        for change in ("input", "goal"):
            with self.subTest(change=change):
                self.article.write_bytes(self.raw + ("\r\n新增段落。".encode() if change == "input" else b""))
                new_request = self.directory / (change + "-request.json")
                report = self.run_cli(*self.prepare_args(new_request, "不同目标" if change == "goal" else "选取有关方法的段落"))
                self.assertNotEqual(report["request_id"], self.request_data["request_id"])
                self.run_cli(*self.apply_args(request=new_request), error="Decision request_id mismatch")
                self.assertFalse(self.output.exists())

    def test_corrupt_requests_are_rejected(self):
        cases = [
            ({**self.request_data, "request_id": "sha256:" + "0" * 64}, "Request hash mismatch"),
            ({**self.request_data, "goal": "修改但未重新 prepare"}, "Request hash mismatch"),
            ({**self.request_data, "input_sha256": "invalid"}, "Invalid input_sha256"),
            ({**self.request_data, "extra": True}, "Request must contain exactly"),
            ({**self.request_data, "units": [{"unit_id": "u0001", "text": "改写原文"}]}, "Request hash mismatch"),
            ({**self.request_data, "units": [{"unit_id": "u0002", "text": "乱序"}]}, "unique and sequential"),
            ({**self.request_data, "units": [{"unit_id": "u0001", "text": "原文", "extra": True}]}, "exactly unit_id and text"),
        ]
        for value, message in cases:
            with self.subTest(error=message, request=value):
                self.write_json(self.request, value)
                self.run_cli(*self.apply_args(), error=message)
                self.assertFalse(self.output.exists())
        self.request.write_text("{broken", encoding="utf-8")
        self.run_cli(*self.apply_args(), error="invalid UTF-8 JSON")

    def test_invalid_decisions_are_rejected(self):
        first = {"unit_id": "u0001", "reason": "理由"}
        cases = [
            ({"selected": [{"unit_id": "unknown", "reason": "理由"}]}, "Unknown unit_id"),
            ({"selected": [first, first]}, "Duplicate selected unit_id"),
            ({"extra": True}, "only no_match_reason is optional"),
            ({"selected": [{**first, "text": "模型改写"}]}, "exactly unit_id and reason"),
            ({"selected": [{"unit_id": "u0001", "reason": "  "}]}, "must be a nonempty string"),
            ({"selected": "u0001"}, "selected must be an array"),
            ({"no_match_reason": "与已选段落矛盾"}, "must be absent or empty"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                self.write_json(self.decision, {**self.decision_data, **changes})
                self.run_cli(*self.apply_args(), error=message)
                self.assertFalse(self.output.exists())

    def test_duplicate_json_keys_and_nonstandard_numbers_are_rejected(self):
        for path, data in [(self.request, self.request_data), (self.decision, self.decision_data)]:
            for suffix, message in [(', "request_id": "duplicate"}', "JSON contains duplicate key"),
                                    (', "extra": NaN}', "JSON contains nonstandard number: NaN")]:
                with self.subTest(file=path.name, suffix=suffix):
                    path.write_text(json.dumps(data)[:-1] + suffix, encoding="utf-8")
                    self.run_cli(*self.apply_args(), error=message)
                    self.assertFalse(self.output.exists())
                    self.write_json(path, data)

    def test_empty_selection_requires_a_reason(self):
        empty = {"request_id": self.request_data["request_id"], "selected": []}
        for value in [empty, {**empty, "no_match_reason": " \n "}]:
            with self.subTest(decision=value):
                self.write_json(self.decision, value)
                self.run_cli(*self.apply_args(), error="no_match_reason when selected is empty")
                self.assertFalse(self.output.exists())
        self.write_json(self.decision, {**empty, "no_match_reason": "  没有相关内容  "})
        self.assertEqual(self.run_cli(*self.apply_args())["selected_count"], 0)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["no_match_reason"], "没有相关内容")

    def test_oversized_input_is_not_silently_truncated(self):
        for text, extra_args, message in [
            ("界" * 12_001, [], "above --max-chars=12000"),
            ("x" * 50_001, ["--max-chars", "50000"], "above --max-chars=50000"),
            ("x" * 48_004, [], "input exceeds the 48003-byte limit"),
            ("有效内容", ["--max-chars", "50001"], "--max-chars must be between"),
        ]:
            with self.subTest(length=len(text), options=extra_args):
                self.article.write_text(text, encoding="utf-8")
                self.run_cli(*self.prepare_args(self.output), *extra_args, error=message)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.article.read_text(encoding="utf-8"), text)
        self.article.write_text("界" * 12_001, encoding="utf-8")
        self.run_cli(*self.prepare_args(self.output), "--max-chars", "12001")
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["units"][0]["text"], "界" * 12_001)

    def test_conflicting_output_is_preserved_without_force(self):
        for arguments in [self.prepare_args(self.output), self.apply_args()]:
            with self.subTest(command=arguments[0]):
                sentinel = b"user-owned existing output\n"
                self.output.write_bytes(sentinel)
                self.run_cli(*arguments, error="already exists with different content")
                self.assertEqual(self.output.read_bytes(), sentinel)
                self.assertEqual(self.run_cli(*arguments, "--force")["write"], "written")
                self.assertNotEqual(self.output.read_bytes(), sentinel)
                self.assertEqual(list(self.directory.glob(".result.json.*")), [])

    def test_force_cannot_overwrite_inputs_directly_or_through_links(self):
        for command, source in [("prepare", self.article), ("apply", self.request), ("apply", self.decision)]:
            for alias in ("direct", "symlink", "hardlink"):
                with self.subTest(command=command, source=source.name, alias=alias):
                    before = source.read_bytes()
                    target = source if alias == "direct" else self.directory / (source.name + "." + alias)
                    if alias == "symlink":
                        target.symlink_to(source)
                    elif alias == "hardlink":
                        os.link(source, target)
                    arguments = self.prepare_args(target) if command == "prepare" else self.apply_args(target)
                    self.run_cli(*arguments, "--force", error="Output must differ from input")
                    self.assertEqual(source.read_bytes(), before)
                    self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
