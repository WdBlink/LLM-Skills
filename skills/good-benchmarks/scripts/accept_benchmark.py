#!/usr/bin/env python3
"""Run four acceptance checks for a fixed-set benchmark with decomposable scores.

Calls the saved domain scorer; does not generate method outputs or prove validity.
See references/result-acceptance.md for the input and completion contract.
"""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import time

from verify_artifacts import sha256_file, unique_object, verify


CHECKS = {
    "dataset": "Build or acquire the evaluation instances and fix their manifest.",
    "execution": "Supply registered method outputs, run evidence and a working scorer.",
    "scores": "Fix score coverage, numeric values or the declared aggregation.",
    "discrimination": "Complete the comparison or investigate insufficient separation.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def number(value, label):
    require(type(value) in (int, float), f"{label} must be a finite number")
    require(math.isfinite(value), f"{label} must be a finite number")
    return value


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def registered_file(root, hashes, name):
    require(nonempty(name) and name in hashes, f"unregistered artifact: {name}")
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), f"invalid artifact path: {name}")
    return path


def load_dataset(config, root, hashes):
    require(config.get("data_role") == "evaluation",
            "data_role must be evaluation; judge fixtures are not an evaluation set")
    expected = config.get("expected_units")
    require(type(expected) is int and expected > 0, "expected_units must be a positive integer")
    path = registered_file(root, hashes, config.get("dataset_manifest"))
    units = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(bool(raw.strip()), f"empty dataset row at line {line_number}")
        row = read_json(raw)
        require(isinstance(row, dict) and nonempty(row.get("id")),
                f"dataset row {line_number} needs a nonempty id")
        unit_id = row["id"]
        require(unit_id not in units, f"duplicate dataset id: {unit_id}")
        files = row.get("files")
        require(isinstance(files, list) and files and all(nonempty(p) for p in files),
                f"{unit_id} needs actual instance files")
        for name in files:
            registered_file(root, hashes, name)
        units[unit_id] = row
    require(len(units) == expected,
            f"dataset has {len(units)} units; expected {expected}")
    return path, units


def load_methods(config, root, hashes):
    methods = config.get("methods")
    require(isinstance(methods, list) and len(methods) >= 2,
            "at least two existing methods are required")
    seen = set()
    for method in methods:
        require(isinstance(method, dict) and nonempty(method.get("id")), "method needs an id")
        require(method["id"] not in seen, f"duplicate method id: {method['id']}")
        seen.add(method["id"])
        require(method.get("kind") == "existing_method" and nonempty(method.get("description")),
                f"{method['id']} needs an existing_method description, not judge fixtures")
        for key in ("predictions", "run_evidence"):
            registered_file(root, hashes, method.get(key))
    return methods


def load_metrics(config):
    metrics = config.get("metrics")
    require(isinstance(metrics, list) and metrics, "metrics must be a nonempty list")
    seen = set()
    for metric in metrics:
        require(isinstance(metric, dict) and nonempty(metric.get("id")), "metric needs an id")
        metric_id = metric["id"]
        require(metric_id not in seen, f"duplicate metric id: {metric_id}")
        seen.add(metric_id)
        require(metric.get("aggregation") in ("mean", "sum"),
                f"{metric_id}: use a project validator for non-decomposable aggregation")
        require(metric.get("direction") in ("higher", "lower"), f"{metric_id}: invalid direction")
        require(nonempty(metric.get("score_basis")), f"{metric_id}: score_basis is required")
        for bound in ("min", "max"):
            if bound in metric:
                number(metric[bound], f"{metric_id}.{bound}")
        if "min" in metric and "max" in metric:
            require(metric["min"] < metric["max"], f"{metric_id}: min must be below max")
    return metrics


def execute(config, root, hashes, dataset, methods):
    scorer = registered_file(root, hashes, config.get("scorer"))
    template = config.get("score_argv")
    require(isinstance(template, list) and template and all(nonempty(p) for p in template),
            "score_argv must be a nonempty argument array")
    for placeholder in ("{scorer}", "{dataset}", "{predictions}"):
        require(any(placeholder in part for part in template),
                f"score_argv must consume {placeholder}")
    timeout = number(config.get("timeout_seconds", 60), "timeout_seconds")
    require(timeout > 0, "timeout_seconds must be positive")
    runs = []
    for method in methods:
        values = dict(root=str(root), scorer=str(scorer), dataset=str(dataset),
                      predictions=str(registered_file(root, hashes, method["predictions"])))
        try:
            argv = [part.format_map(values) for part in template]
        except (KeyError, ValueError, AttributeError, IndexError) as exc:
            raise ValueError(f"invalid score_argv placeholder: {exc}") from exc
        run = {"method_id": method["id"], "argv": argv, "cwd": str(root),
               "returncode": None, "timed_out": False, "stdout": "", "stderr": ""}
        started = time.monotonic()
        try:
            completed = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=timeout,
                                       check=False)
            run.update(returncode=completed.returncode, stdout=completed.stdout,
                       stderr=completed.stderr)
        except subprocess.TimeoutExpired as exc:
            def text_output(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            run.update(timed_out=True, stdout=text_output(exc.stdout), stderr=text_output(exc.stderr))
        except OSError as exc:
            run["stderr"] = str(exc)
        run["elapsed_seconds"] = time.monotonic() - started
        runs.append(run)
    return runs


def collect_scores(runs, units, metrics):
    by_method = {}
    metric_ids = {metric["id"] for metric in metrics}
    for run in runs:
        try:
            result = read_json(run["stdout"])
        except ValueError as exc:
            raise ValueError(f"{run['method_id']}: invalid scorer JSON: {exc}") from exc
        require(isinstance(result, dict) and isinstance(result.get("scores"), list),
                f"{run['method_id']}: scorer must return a scores list")
        rows = {}
        for row in result["scores"]:
            require(isinstance(row, dict) and nonempty(row.get("unit_id")),
                    f"{run['method_id']}: score row needs unit_id")
            unit_id = row["unit_id"]
            require(unit_id in units and unit_id not in rows,
                    f"{run['method_id']}: unknown or duplicate score id: {unit_id}")
            values = row.get("metrics")
            require(isinstance(values, dict) and set(values) == metric_ids,
                    f"{run['method_id']}/{unit_id}: metric coverage mismatch")
            for metric in metrics:
                label = f"{run['method_id']}/{unit_id}/{metric['id']}"
                value = number(values[metric["id"]], label)
                require(("min" not in metric or value >= metric["min"])
                        and ("max" not in metric or value <= metric["max"]),
                        f"{label}: score outside declared bounds")
            rows[unit_id] = values
        require(set(rows) == set(units), f"{run['method_id']}: incomplete score coverage")
        by_method[run["method_id"]] = rows
    summary = {}
    for method_id, rows in by_method.items():
        summary[method_id] = {}
        for metric in metrics:
            value = math.fsum(rows[unit_id][metric["id"]] for unit_id in units)
            if metric["aggregation"] == "mean":
                value /= len(units)
            summary[method_id][metric["id"]] = number(value, "aggregate score")
    return by_method, summary


def load_criterion(config, metrics, method_ids):
    criterion = config.get("discrimination")
    require(isinstance(criterion, dict), "discrimination criterion is required")
    require(criterion.get("scope") == "fixed_set_descriptive",
            "use a project validator for population inference or stochastic ranking")
    primary = criterion.get("metric")
    require(nonempty(primary) and primary in {m["id"] for m in metrics},
            "unknown discrimination metric")
    gap = number(criterion.get("min_gap"), "min_gap")
    require(gap > 0, "min_gap must be positive and declared before comparing")
    tolerance = number(criterion.get("unit_tie_tolerance", 0), "unit_tie_tolerance")
    require(tolerance >= 0, "unit_tie_tolerance must be nonnegative")
    separate = criterion.get("require_separation")
    require(type(separate) is bool, "require_separation must be boolean")
    require(nonempty(criterion.get("rationale")), "discrimination rationale is required")
    declared_pairs = criterion.get("pairs")
    require(isinstance(declared_pairs, list) and declared_pairs,
            "declare the primary method pairs before comparing")
    seen = set()
    for pair in declared_pairs:
        require(isinstance(pair, list) and len(pair) == 2 and all(nonempty(p) for p in pair),
                "each primary pair needs two method ids")
        require(pair[0] != pair[1] and all(p in method_ids for p in pair),
                "unknown or identical pair methods")
        key = frozenset(pair)
        require(key not in seen, "duplicate primary pair")
        seen.add(key)
    return criterion


def compare(criterion, units, metrics, by_method, summary):
    primary = criterion["metric"]
    tolerance = criterion.get("unit_tie_tolerance", 0)
    results = {}
    for metric in metrics:
        key = metric["id"]
        values = [row[key] for row in summary.values()]
        spread = number(max(values) - min(values), "score range")
        pairs = []
        for first, second in combinations(by_method, 2):
            differences = [(by_method[first][u][key] - by_method[second][u][key])
                           * (1 if metric["direction"] == "higher" else -1) for u in units]
            # Tie tolerance is for the primary metric's per-unit scale only.
            tie = tolerance if key == primary else 0
            pairs.append({"first": first, "second": second,
                          "score_difference": number(summary[first][key] - summary[second][key], "score difference"),
                          "first_wins": sum(d > tie for d in differences),
                          "ties": sum(abs(d) <= tie for d in differences),
                          "second_wins": sum(d < -tie for d in differences)})
        item = {"score_range": spread, "paired_units": len(units), "pairs": pairs}
        for bound in ("min", "max"):
            if bound in metric:
                item[f"at_{bound}_fraction"] = {
                    method: sum(row[key] == metric[bound] for row in rows.values()) / len(units)
                    for method, rows in by_method.items()}
        results[key] = item
    primary_pairs = [{"first": first, "second": second,
                      "absolute_gap": abs(summary[first][primary] - summary[second][primary])}
                     for first, second in criterion["pairs"]]
    criterion_met = all(pair["absolute_gap"] >= criterion["min_gap"] for pair in primary_pairs)
    return {"scope": "fixed_set_descriptive", "uncertainty": "not_assessed",
            "criterion": criterion, "metrics": results, "primary_pairs": primary_pairs,
            "criterion_met": criterion_met,
            "accepted_for_declared_purpose": not criterion["require_separation"] or criterion_met}


def accept(contract, root=None):
    report = {"status": "needs_work", "scope": "four_execution_checks",
              "semantic_validity": "not_assessed", "checks": {}, "executions": []}
    for key, action in CHECKS.items():
        report["checks"][key] = {"status": "not_run", "next_action": action}
    stage = "dataset"
    try:
        contract = Path(contract).resolve(strict=True)
        root = Path(root).resolve(strict=True) if root is not None else contract.parent
        config = read_json(contract.read_text(encoding="utf-8"))
        require(isinstance(config, dict) and type(config.get("schema_version")) is int
                and config["schema_version"] == 1,
                "contract needs schema_version: 1")
        registration_name = config.get("registration")
        require(nonempty(registration_name), "registration is required")
        registration = (root / registration_name).resolve(strict=True)
        require(registration.is_relative_to(root), "registration escapes artifact root")
        integrity = verify(registration, root)
        report["integrity"] = integrity
        require(integrity["status"] == "artifacts_verified", "artifact verification failed")
        data = read_json(registration.read_text(encoding="utf-8"))
        hashes = data["sha256"]
        require(contract.is_relative_to(root), "contract escapes artifact root")
        registered_file(root, hashes, contract.relative_to(root).as_posix())
        before = sha256_file(registration)
        report["binding"] = {"benchmark_id": integrity["benchmark_id"],
                             "version": integrity["version"], "registration_sha256": before,
                             "contract_sha256": sha256_file(contract), "artifacts": hashes}
        dataset, units = load_dataset(config, root, hashes)
        report["checks"][stage].update(status="pass", units=len(units),
                                       manifest=str(dataset), next_action=None)
        stage = "execution"
        methods = load_methods(config, root, hashes)
        # Check the entire score/comparison contract before running external code.
        stage = "scores"
        metrics = load_metrics(config)
        stage = "discrimination"
        criterion = load_criterion(config, metrics, {method["id"] for method in methods})
        stage = "execution"
        runs = execute(config, root, hashes, dataset, methods)
        report["executions"] = runs
        require(sha256_file(registration) == before
                and verify(registration, root)["status"] == "artifacts_verified",
                "registered inputs changed during scoring; invalidate these results")
        require(all(run["returncode"] == 0 and not run["timed_out"] for run in runs),
                "one or more scorer commands failed; see executions")
        report["checks"][stage].update(status="pass", methods=len(methods), next_action=None)
        stage = "scores"
        by_method, summary = collect_scores(runs, units, metrics)
        report["scores"] = summary
        report["checks"][stage].update(status="pass", metrics=len(metrics), next_action=None)
        stage = "discrimination"
        comparison = compare(criterion, units, metrics, by_method, summary)
        report["comparison"] = comparison
        require(comparison["accepted_for_declared_purpose"],
                "a primary method pair is below the predeclared minimum gap")
        report["checks"][stage].update(status="pass", next_action=None)
        report["status"] = "execution_checks_passed"
    except (OSError, ValueError, RuntimeError, OverflowError) as exc:
        report["checks"][stage].update(status="fail", error=str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--root", type=Path, help="artifact root; default: contract parent")
    parser.add_argument("--report", type=Path, help="save evidence to a new file; never overwrite")
    args = parser.parse_args()
    # Reject an existing report before running any command.
    if args.report is not None and (args.report.exists() or args.report.is_symlink()):
        parser.error("--report already exists; use a new path")
    report = accept(args.contract, args.root)
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(output)
        except OSError as exc:
            parser.error(f"cannot save report: {exc}")
    print(output, end="")
    return 0 if report["status"] == "execution_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
