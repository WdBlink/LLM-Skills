"""Controlled execution fixtures, not evidence of real method discrimination."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from accept_benchmark import accept


SCORER = '''import json, sys
from pathlib import Path
dataset, predictions = map(Path, sys.argv[1:])
predictions = json.loads(predictions.read_text())
scores = []
for raw in dataset.read_text().splitlines():
    row = json.loads(raw)
    task = json.loads((dataset.parent / row["files"][0]).read_text())
    scores.append({"unit_id": row["id"], "metrics": {
        "accuracy": int(predictions[row["id"]] == task["answer"])}})
print(json.dumps({"scores": scores}))
'''

BASELINE = '''import json, sys
from pathlib import Path
dataset = Path(sys.argv[1])
predictions = {}
for i, raw in enumerate(dataset.read_text().splitlines()):
    row = json.loads(raw)
    task = json.loads((dataset.parent / row["files"][0]).read_text())
    predictions[row["id"]] = task["x"] * 2 if sys.argv[2] == "double" or i % 2 else task["x"]
print(json.dumps(predictions))
'''


class AcceptanceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "bundle"
        self.root.mkdir()
        self.contract = self.root / "acceptance.json"
        self.config = {
            "schema_version": 1, "registration": "registration.json",
            "data_role": "evaluation", "dataset_manifest": "tasks.jsonl", "expected_units": 4,
            "scorer": "score.py", "timeout_seconds": 5,
            "score_argv": [sys.executable, "{scorer}", "{dataset}", "{predictions}"],
            "metrics": [{"id": "accuracy", "aggregation": "mean", "direction": "higher",
                         "min": 0, "max": 1, "score_basis": "exact agreement with independent answer"}],
            "methods": [{"id": name, "kind": "existing_method", "description": f"controlled {name} baseline",
                         "predictions": name + ".json", "run_evidence": name + ".run.json"}
                        for name in ("double", "alternating")],
            "discrimination": {"scope": "fixed_set_descriptive", "metric": "accuracy",
                               "pairs": [["double", "alternating"]],
                               "min_gap": 0.25, "unit_tie_tolerance": 0,
                               "require_separation": True, "rationale": "controlled comparison"},
        }
        self.write("score.py", SCORER)
        self.write("baseline.py", BASELINE)
        rows = []
        for i in range(4):
            self.write(f"task-{i}.json", {"x": i + 1, "answer": (i + 1) * 2})
            rows.append({"id": f"u{i}", "files": [f"task-{i}.json"]})
        self.write("tasks.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")
        self.run_baselines()
        self.freeze()

    def write(self, name, data):
        (self.root / name).write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    def run_baselines(self):
        for method in self.config["methods"]:
            argv = [sys.executable, str(self.root / "baseline.py"), str(self.root / "tasks.jsonl"), method["id"]]
            result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=True)
            self.write(method["predictions"], result.stdout)
            self.write(method["run_evidence"], {"argv": argv, "returncode": result.returncode,
                                              "stdout": result.stdout, "stderr": result.stderr})

    def freeze(self):
        self.write("acceptance.json", self.config)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()
                  if p.is_file() and p.name != "registration.json"}
        self.write("registration.json", {"benchmark_id": "CONTROLLED-TEST", "version": "1",
                                         "status": "passed", "sha256": hashes})

    def failed(self, stage, fragment):
        report = accept(self.contract)
        self.assertEqual(report["status"], "needs_work")
        self.assertEqual(report["checks"][stage]["status"], "fail")
        self.assertIn(fragment, report["checks"][stage]["error"])
        self.assertTrue(report["checks"][stage]["next_action"])
        return report

    def cli(self, *args):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("accept_benchmark.py")),
                               str(self.contract), *map(str, args)], cwd=self.base,
                              capture_output=True, text=True, timeout=10)

    def test_real_cli_chain_from_other_directory(self):
        report_path = self.base / "reports" / "result.json"
        result = self.cli("--report", report_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["status"], "execution_checks_passed")
        self.assertEqual(report["semantic_validity"], "not_assessed")
        self.assertEqual({c["status"] for c in report["checks"].values()}, {"pass"})
        self.assertEqual(report["scores"], {"double": {"accuracy": 1}, "alternating": {"accuracy": 0.5}})
        comparison = report["comparison"]["metrics"]["accuracy"]
        self.assertEqual(comparison["score_range"], 0.5)
        self.assertEqual(comparison["pairs"][0]["first_wins"], 2)
        self.assertEqual(comparison["pairs"][0]["ties"], 2)
        self.assertEqual(len(report["executions"]), 2)
        self.assertEqual(report["executions"][0]["returncode"], 0)

    def test_empty_missing_duplicate_and_wrong_count_dataset(self):
        original = (self.root / "tasks.jsonl").read_text()
        for text, fragment in [("", "0 units"), (original + original, "duplicate"),
                               (original.splitlines()[0] + "\n", "1 units")]:
            with self.subTest(fragment=fragment):
                self.write("tasks.jsonl", text)
                self.freeze()
                self.failed("dataset", fragment)
        self.write("tasks.jsonl", original)
        (self.root / "task-0.json").unlink()
        self.freeze()
        self.failed("dataset", "unregistered")

    def test_fixture_data_and_methods_cannot_be_substituted(self):
        self.config["data_role"] = "judge_fixture"
        self.freeze()
        self.failed("dataset", "evaluation")
        self.config["data_role"] = "evaluation"
        self.config["methods"][0]["kind"] = "judge_fixture"
        self.freeze()
        self.failed("execution", "existing_method")

    def test_at_least_two_methods_and_run_evidence(self):
        saved = self.config["methods"].copy()
        self.config["methods"] = saved[:1]
        self.freeze()
        self.failed("execution", "two existing methods")
        self.config["methods"] = saved
        (self.root / "double.run.json").unlink()
        self.freeze()
        self.failed("execution", "unregistered")

    def test_scorer_missing_or_command_does_not_consume_inputs(self):
        self.config["score_argv"] = [sys.executable, "{scorer}"]
        self.freeze()
        self.failed("execution", "{dataset}")
        (self.root / "score.py").unlink()
        self.freeze()
        self.failed("execution", "unregistered")

    def test_nonzero_command_and_timeout(self):
        self.write("score.py", 'import sys; print("failure", file=sys.stderr); sys.exit(7)')
        self.freeze()
        report = self.failed("execution", "commands failed")
        self.assertEqual(report["executions"][0]["returncode"], 7)
        self.assertIn("failure", report["executions"][0]["stderr"])
        self.write("score.py", 'import time; time.sleep(2)')
        self.config["timeout_seconds"] = 0.05
        self.freeze()
        report = self.failed("execution", "commands failed")
        self.assertTrue(report["executions"][0]["timed_out"])

    def test_summary_or_empty_scores_are_not_a_score_trace(self):
        for payload in [{"status": "pass", "accuracy": 1}, {"scores": []}]:
            self.write("score.py", "print(" + repr(json.dumps(payload)) + ")")
            self.freeze()
            self.failed("scores", "scores list" if "scores" not in payload else "coverage")

    def test_invalid_per_unit_scores(self):
        cases = [
            ([{"unit_id": "u0", "metrics": {"accuracy": True}}], "finite number"),
            ([{"unit_id": "u0", "metrics": {"accuracy": 2}}], "bounds"),
            ([{"unit_id": "u0", "metrics": {"accuracy": 1}}] * 2, "duplicate"),
            ([{"unit_id": "foreign", "metrics": {"accuracy": 1}}], "unknown"),
            ([{"unit_id": "u0", "metrics": {"unrelated": 1}}], "metric coverage"),
        ]
        for rows, fragment in cases:
            with self.subTest(fragment=fragment):
                self.write("score.py", "print(" + repr(json.dumps({"scores": rows})) + ")")
                self.freeze()
                self.failed("scores", fragment)

    def test_nonfinite_and_duplicate_json_keys(self):
        for raw, fragment in [('{"scores": NaN}', "non-finite"),
                              ('{"scores": [], "scores": []}', "duplicate")]:
            self.write("score.py", "print(" + repr(raw) + ")")
            self.freeze()
            self.failed("scores", fragment)

    def test_no_separation_blocks_ranking_but_retains_numbers(self):
        self.write("alternating.json", (self.root / "double.json").read_text())
        self.freeze()
        report = self.failed("discrimination", "minimum gap")
        self.assertEqual(report["comparison"]["metrics"]["accuracy"]["score_range"], 0)
        self.assertEqual(report["comparison"]["metrics"]["accuracy"]["pairs"][0]["ties"], 4)
        result = self.cli()
        self.assertEqual(result.returncode, 1)

    def test_predeclared_nonranking_purpose_still_measures_zero(self):
        self.write("alternating.json", (self.root / "double.json").read_text())
        self.config["discrimination"].update(require_separation=False, rationale="coverage regression only")
        self.freeze()
        report = accept(self.contract)
        self.assertEqual(report["status"], "execution_checks_passed")
        self.assertFalse(report["comparison"]["criterion_met"])
        self.assertEqual(report["comparison"]["metrics"]["accuracy"]["score_range"], 0)

    def test_extra_weak_method_cannot_mask_tied_primary_pair(self):
        self.write("alternating.json", (self.root / "double.json").read_text())
        self.write("extra.json", {f"u{i}": -1 for i in range(4)})
        self.config["methods"].append(dict(self.config["methods"][0], id="extra", predictions="extra.json"))
        self.freeze()
        report = self.failed("discrimination", "primary method pair")
        self.assertEqual(report["comparison"]["metrics"]["accuracy"]["score_range"], 1)
        self.assertEqual(report["comparison"]["primary_pairs"][0]["absolute_gap"], 0)

    def test_stale_artifacts_do_not_reuse_old_pass(self):
        self.assertEqual(accept(self.contract)["status"], "execution_checks_passed")
        self.write("double.json", {})
        report = self.failed("dataset", "verification failed")
        self.assertEqual(report["executions"], [])

    def test_scorer_cannot_silently_mutate_registered_inputs(self):
        self.write("score.py", SCORER + '\n(dataset.parent / "task-0.json").write_text("changed")\n')
        self.freeze()
        self.failed("execution", "inputs changed")

    def test_failure_repair_preserves_expensive_method_outputs(self):
        paths = [self.root / name for name in ("double.json", "alternating.json", "double.run.json", "alternating.run.json")]
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
        self.write("score.py", "raise RuntimeError('broken scorer')")
        self.freeze()
        self.assertEqual(self.cli().returncode, 1)
        self.write("score.py", SCORER)
        self.freeze()
        self.assertEqual(self.cli().returncode, 0)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])

    def test_report_conflict_never_overwrites_inputs(self):
        original = self.contract.read_bytes()
        result = self.cli("--report", self.contract)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(original, self.contract.read_bytes())

    def test_second_dataset_reuses_entry_without_code_change(self):
        scorer_hash = hashlib.sha256((self.root / "score.py").read_bytes()).hexdigest()
        rows = (self.root / "tasks.jsonl").read_text().splitlines()
        self.write("tasks.jsonl", "\n".join(rows[:2]) + "\n")
        self.config["expected_units"] = 2
        self.run_baselines()
        self.freeze()
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["checks"]["dataset"]["units"], 2)
        self.assertEqual(scorer_hash, hashlib.sha256((self.root / "score.py").read_bytes()).hexdigest())

    def test_sum_lower_direction_and_no_invented_inference(self):
        self.config["metrics"][0].update(aggregation="sum", direction="lower")
        self.freeze()
        report = accept(self.contract)
        self.assertEqual(report["scores"]["double"]["accuracy"], 4)
        self.assertEqual(report["comparison"]["metrics"]["accuracy"]["pairs"][0]["second_wins"], 2)
        self.assertEqual(report["comparison"]["uncertainty"], "not_assessed")
        self.config["discrimination"]["scope"] = "population"
        self.freeze()
        self.failed("discrimination", "project validator")

    def test_bad_criterion_cannot_default_to_acceptance(self):
        original = dict(self.config["discrimination"])
        for key, value, fragment in [("min_gap", 0, "positive"), ("metric", [], "unknown"),
                                     ("require_separation", "false", "boolean"), ("rationale", "", "rationale")]:
            self.config["discrimination"] = dict(original, **{key: value})
            self.freeze()
            report = self.failed("discrimination", fragment)
            self.assertEqual(report["executions"], [])

    def test_invalid_metric_contract_does_not_launch_scorer(self):
        self.config["metrics"][0]["aggregation"] = "invented"
        self.freeze()
        report = self.failed("scores", "project validator")
        self.assertEqual(report["executions"], [])


if __name__ == "__main__":
    unittest.main()
