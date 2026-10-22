"""Grader regressions, not model capability tests. No network or paid calls."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eval_adapter", ROOT / "adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.context = {"vars": {"case_id": "scope-limits"}}
        self.good = json.loads((ROOT / "replays/synthetic/scope-limits.json").read_text())

    def grade(self, decision, name="scope-limits"):
        return adapter.get_assert(json.dumps(decision, ensure_ascii=False), {"vars": {"case_id": name}})

    def test_positive_and_order_reason_alternative(self):
        self.assertTrue(self.grade(self.good)["pass"])
        alternative = copy.deepcopy(self.good)
        alternative["selected"].reverse()
        alternative["selected"][0]["reason"] = "雨夜及移动场景尚无支持证据。"
        self.assertTrue(self.grade(alternative)["pass"])

    def test_structurally_valid_but_wrong_selection_fails(self):
        self.good["selected"] = [{"unit_id": "u0001", "reason": "错误地把实验结果当成适用范围。"}]
        adapter.handoff.validate_decision(self.good, adapter.load_request("scope-limits"))
        result = self.grade(self.good)
        self.assertFalse(result["pass"])
        self.assertIn("selection_mismatch", result["reason"])

    def test_missing_or_extra_paragraph_fails(self):
        missing = copy.deepcopy(self.good)
        missing["selected"].pop()
        extra = copy.deepcopy(self.good)
        extra["selected"].append({"unit_id": "u0004", "reason": "无关的目录整理。"})
        for decision in (missing, extra):
            with self.subTest(decision=decision):
                self.assertFalse(self.grade(decision)["pass"])

    def test_stale_unknown_and_duplicate_ids_fail(self):
        stale = copy.deepcopy(self.good)
        stale["request_id"] = "sha256:" + "0" * 64
        unknown = copy.deepcopy(self.good)
        unknown["selected"][0]["unit_id"] = "u9999"
        duplicate = copy.deepcopy(self.good)
        duplicate["selected"].append(copy.deepcopy(duplicate["selected"][0]))
        for decision in (stale, unknown, duplicate):
            with self.subTest(decision=decision):
                self.assertIn("decision_invalid", self.grade(decision)["reason"])

    def test_no_match_requires_reason_and_actual_absence(self):
        empty = json.loads((ROOT / "replays/synthetic/no-match.json").read_text())
        self.assertTrue(self.grade(empty, "no-match")["pass"])
        del empty["no_match_reason"]
        self.assertFalse(self.grade(empty, "no-match")["pass"])
        self.good["selected"] = []
        self.good["no_match_reason"] = "错误地声称没有限制条件。"
        self.assertFalse(self.grade(self.good)["pass"])

    def test_malformed_output_and_duplicate_json_keys_fail(self):
        for output in ("", "not-json", "[]", '{"selected":[],"selected":[]}', '{"x":NaN}', None):
            with self.subTest(output=output):
                self.assertFalse(adapter.get_assert(output, self.context)["pass"])

    def test_invalid_request_is_evaluation_error(self):
        request = adapter.load_request("scope-limits")
        request["units"][0]["text"] += " changed"
        with patch.object(adapter.handoff, "read_json", return_value=request):
            with self.assertRaisesRegex(adapter.handoff.HandoffError, "hash mismatch"):
                adapter.get_assert(json.dumps(self.good), self.context)

    def test_stale_grader_reference_is_evaluation_error(self):
        request = adapter.load_request("scope-limits")
        stale = {"version": 1, "cases": {"scope-limits": {"request_id": "old", "selected_ids": []}}}
        with patch.object(adapter, "load_request", return_value=request):
            with patch.object(adapter.handoff, "read_json", return_value=stale):
                with self.assertRaisesRegex(adapter.handoff.HandoffError, "Grader reference"):
                    adapter.get_assert(json.dumps(self.good), self.context)

    def test_prompt_reads_only_solver_request(self):
        with patch.object(adapter.handoff, "read_json", wraps=adapter.handoff.read_json) as read:
            prompt = adapter.create_prompt(self.context)
        self.assertEqual([call.args[0] for call in read.call_args_list],
                         [ROOT / "inputs/scope-limits.request.json"])
        self.assertNotIn("selected_ids", prompt)
        self.assertNotIn("rationale", prompt)

    def test_replay_does_not_repair_wrong_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "scope-limits.json").write_text("not-json")
            with patch.dict(os.environ, {"WORKFLOW_EVAL_RESPONSES_DIR": temporary}):
                result = adapter.call_api(adapter.create_prompt(self.context), {}, self.context)
        self.assertEqual(result, {"output": "not-json"})
        self.assertFalse(adapter.get_assert(result["output"], self.context)["pass"])

    def test_bundled_prompt_provider_assertion_callbacks(self):
        with patch.dict(os.environ, {}, clear=True):
            for name in ("scope-limits", "no-match"):
                context = {"vars": {"case_id": name}}
                prompt = adapter.create_prompt(context)
                result = adapter.call_api(prompt, {"config": {"replayDir": "replays/synthetic"}}, context)
                self.assertNotIn("error", result)
                self.assertTrue(adapter.get_assert(result["output"], context)["pass"])

    def test_missing_replay_and_path_traversal_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"WORKFLOW_EVAL_RESPONSES_DIR": temporary}):
                result = adapter.call_api(adapter.create_prompt(self.context), {}, self.context)
        self.assertIn("error", result)
        self.assertNotIn("output", result)
        with self.assertRaises(adapter.handoff.HandoffError):
            adapter.create_prompt({"vars": {"case_id": "../grader/references"}})

    def test_cli_works_from_other_directory_and_has_distinct_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "wrong.json"
            wrong.write_text("not-json")
            for decision, code in ((ROOT / "replays/synthetic/scope-limits.json", 0),
                                   (wrong, 1), (Path(temporary) / "missing.json", 2)):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "adapter.py"), "grade", "scope-limits", str(decision)],
                    cwd=temporary, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
