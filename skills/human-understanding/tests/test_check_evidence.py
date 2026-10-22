import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SKILL = Path(__file__).resolve().parents[1]
SCRIPT = SKILL / "scripts/check_evidence.py"
spec = importlib.util.spec_from_file_location("check_evidence", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = json.loads((SKILL / "examples/evidence.json").read_text())
        self.raw = json.loads((SKILL / "examples/results.json").read_text())

    def check(self):
        (self.root / "results.json").write_text(json.dumps(self.raw))
        return checker.validate(self.bundle, self.root)

    def test_valid_source_and_computation(self):
        result = self.check()
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["semantic_review"], "not_checked")

    def test_mistranscribed_values_metadata_and_counts(self):
        original = copy.deepcopy(self.bundle)
        for key, value in {"value": 9, "unit": "s", "direction": "higher", "n": 99, "baseline_n": 98, "conditions": "另一批请求"}.items():
            with self.subTest(key=key):
                self.bundle = copy.deepcopy(original)
                self.bundle["evidence"][0]["metric"][key] = value
                self.assertEqual(self.check()["status"], "fail")

    def test_inverted_trend(self):
        self.bundle["evidence"][0]["computed"]["trend"] = "worsened"
        self.assertEqual(self.check()["status"], "fail")

    def test_wrong_delta_and_relative_percent(self):
        for key in ("delta", "relative_change_pct"):
            with self.subTest(key=key):
                self.bundle["evidence"][0]["computed"] = {key: 10}
                self.assertEqual(self.check()["status"], "fail")

    def test_zero_and_negative_baseline(self):
        for baseline in (0, -1):
            with self.subTest(baseline=baseline):
                self.raw["runs"][0]["baseline"] = baseline
                self.bundle["evidence"][0]["metric"]["baseline"] = baseline
                self.bundle["evidence"][0]["computed"] = {"relative_change_pct": 100}
                self.assertEqual(self.check()["status"], "fail")
                self.bundle["evidence"][0]["computed"] = {"delta": 90 - baseline}
                self.assertEqual(self.check()["status"], "pass")

    def test_relative_percent_avoids_overflowing_intermediate_difference(self):
        for value, baseline in ((-1e308, 1e308), (-10**308, 10**308)):
            with self.subTest(value=value, baseline=baseline):
                self.set_metric("value", value)
                self.set_metric("baseline", baseline)
                self.bundle["evidence"][0]["computed"] = {"relative_change_pct": -200}
                result = self.check()
                self.assertEqual(result["status"], "pass", result)

    def test_unrepresentable_computations_fail_with_json_report(self):
        for value, baseline, key in ((10**308, -10**308, "delta"),
                                     (1e308, -1e308, "delta"),
                                     (1e308, 1e-308, "relative_change_pct")):
            with self.subTest(value=value, baseline=baseline, computation=key):
                self.set_metric("value", value)
                self.set_metric("baseline", baseline)
                self.bundle["evidence"][0]["computed"] = {key: 0}
                result = self.check()
                self.assertEqual(result["status"], "fail", result)
                self.assertTrue(any(f"E1.{key}: result exceeds finite numeric range" in error
                                    for error in result["errors"]), result)
                path = self.root / "evidence.json"
                path.write_text(json.dumps(self.bundle))
                run = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                     capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertEqual(run.stderr, "")
                self.assertEqual(json.loads(run.stdout), result)

    def test_trend_compares_extreme_values_without_needing_a_finite_delta(self):
        for value, baseline, direction, trend in (
            (10**308, -10**308, "higher", "improved"),
            (-1e308, 1e308, "lower", "improved"),
            (1e308, -1e308, "lower", "worsened"),
            (-1e308, 1e308, "neutral", "undetermined"),
            (1e308, 1e308, "neutral", "unchanged"),
        ):
            with self.subTest(value=value, baseline=baseline, direction=direction):
                for field, entry in (("value", value), ("baseline", baseline), ("direction", direction)):
                    self.set_metric(field, entry)
                self.bundle["evidence"][0]["computed"] = {"trend": trend}
                result = self.check()
                self.assertEqual(result["status"], "pass", result)

    def test_nonfinite_and_invalid_counts(self):
        original_bundle, original_raw = copy.deepcopy(self.bundle), copy.deepcopy(self.raw)
        for field, value in (("value", float("nan")), ("value", float("inf")), ("value", 10**400), ("n", 0), ("n", True), ("n", 1.5)):
            with self.subTest(field=field, value=value):
                self.bundle, self.raw = copy.deepcopy(original_bundle), copy.deepcopy(original_raw)
                self.set_metric(field, value)
                self.bundle["evidence"][0]["computed"] = {}
                self.assertEqual(self.check()["status"], "fail")

    def set_metric(self, key, value):
        self.raw["runs"][0][key] = value
        self.bundle["evidence"][0]["metric"][key] = value

    def test_missing_metadata_remains_visible(self):
        self.set_metric("n", None)
        self.bundle["evidence"][0]["missing"] = {"n": "原始记录未提供样本数"}
        result = self.check()
        self.assertEqual(result["status"], "pass_with_gaps", result)
        self.assertTrue(result["gaps"])

    def test_cannot_hide_present_source_value(self):
        self.bundle["evidence"][0]["metric"]["n"] = None
        self.bundle["evidence"][0]["missing"] = {"n": "伪造的缺失理由"}
        self.assertEqual(self.check()["status"], "fail")

    def test_summary_cannot_support_observation(self):
        self.bundle["sources"][0]["kind"] = "summary"
        self.assertEqual(self.check()["status"], "fail")

    def test_missing_source_can_be_reported_without_invented_evidence(self):
        self.bundle["sources"][0].update(status="missing", reason="没有收到原始回执")
        self.bundle["evidence"] = []
        self.bundle["claims"][0].update(kind="missing", evidence_ids=[], reason="不能确认提升")
        self.assertEqual(self.check()["status"], "pass_with_gaps")

    def test_broken_reference_and_duplicate_id(self):
        self.bundle["claims"][0]["evidence_ids"] = ["absent"]
        self.assertEqual(self.check()["status"], "fail")
        self.bundle["claims"][0]["evidence_ids"] = ["E1"]
        self.bundle["evidence"].append(copy.deepcopy(self.bundle["evidence"][0]))
        self.assertEqual(self.check()["status"], "fail")

    def test_conflict_requires_two_distinct_entries(self):
        self.bundle["claims"][0].update(kind="conflicting", evidence_ids=["E1", "E1"], reason="来源冲突")
        self.assertEqual(self.check()["status"], "fail")

    def test_text_excerpt_is_checked_against_actual_lines(self):
        (self.root / "notes.txt").write_text("前言\n第 2 组实验失败。\n尾注\n")
        self.bundle["sources"][0]["path"] = "notes.txt"
        self.bundle["evidence"] = [{"id": "E1", "kind": "text", "source_id": "S1", "lines": [2, 2], "excerpt": "第 2 组实验失败。"}]
        self.assertEqual(self.check()["status"], "pass")
        self.bundle["evidence"][0]["excerpt"] = "第 2 组实验成功。"
        self.assertEqual(self.check()["status"], "fail")
        self.bundle["evidence"][0]["lines"] = [0, 0]
        self.assertEqual(self.check()["status"], "fail")

    def test_semantic_overclaim_is_not_certified(self):
        self.bundle["claims"][0]["text"] = "所有场景都已证明稳定成功。"
        result = self.check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["semantic_review"], "not_checked")

    def test_json_pointer_escaping_and_negative_index(self):
        self.assertEqual(checker.pointer({"a/b": {"~": [5]}}, "/a~1b/~0/0"), 5)
        for location in ("/runs/-1", "/runs/01", "/runs/4", "/runs/~2"):
            self.bundle["evidence"][0]["pointer"] = location
            self.assertEqual(self.check()["status"], "fail")

    def test_malformed_top_level_and_cli(self):
        self.assertEqual(checker.validate([], self.root)["status"], "fail")
        path = self.root / "bad.json"
        path.write_text("not json")
        run = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertEqual(json.loads(run.stdout)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
