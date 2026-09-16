#!/usr/bin/env python3
"""Executable contract checks. JSON fixtures here are test doubles, never LLM runs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run.py"
FIXTURE = ROOT / "fixtures/input.json"
FILL = {
    "answer": "现有虚构观察不足以支持全面更换。A 馆入口的询问比例下降，但这只是两周观察。[E1]",
    "conflict": "入口找路与安静区使用目标不同；B 馆有亮度干扰反馈，也有更新便利反馈，不能直接比较两馆效果。[E1][E2]",
    "limits": "断电时纸牌作为备用。[E3] 样本有限，无法估计全年可靠性和总成本；观察与自愿反馈不构成因果证据。[E1][E2][E3]",
}


def put(directory, name, value):
    # Test-only replacement deliberately creates corrupt/stale states for negative checks.
    (directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def call(directory, operation="next", *args):
    result = subprocess.run([sys.executable, str(SCRIPT), operation, "--dir", str(directory), *map(str, args)],
                            capture_output=True, text=True)
    action = json.loads(result.stdout)
    assert result.returncode == (2 if action["kind"] == "blocked" else 0), result
    assert not result.stderr, result.stderr
    if action["kind"] == "command":
        assert all(Path(action["argv"][index]).is_absolute() for index in (0, 1, -1))
    if action["kind"] == "llm_subagent":
        assert Path(action["prompt"]).is_file()
        assert Path(action["output"]).is_absolute()
        assert all(Path(path).is_absolute() and Path(path).is_file() for path in action["inputs"])
    return action


def expect(directory, kind, operation="next", *args):
    action = call(directory, operation, *args)
    assert action["kind"] == kind, action
    return action


def fresh(parent, name):
    directory = parent / name
    assert expect(directory, "command")["step"] == "prepare"
    assert not directory.exists(), "next must not create the run directory"
    assert expect(directory, "llm_subagent", "prepare", "--input", FIXTURE)["step"] == "writer"
    put(directory, "fill_map.json", FILL)
    assert expect(directory, "command")["step"] == "assemble"
    assert expect(directory, "llm_subagent", "assemble")["step"] == "critic-0"
    return directory


def review(directory, verdict, revision=0):
    doc = read(directory, "draft.json" if revision == 0 else "revised.json")
    return {"review_token": doc["review_token"], "verdict": verdict,
            "reason": "结构检查测试数据；不代表真实模型评审。",
            "findings": [] if verdict == "advance" else [{
                "scope": "limits", "reason": "测试修订门控：需要说明记录不足以估计长期可靠性。",
                "evidence": [{"source_id": "E3", "quote": "没有全年停机率"}]}]}


def snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix="evidence-brief-check-") as temporary:
        parent = Path(temporary)
        d = fresh(parent, "happy")
        before = snapshot(d)
        call(d)
        assert snapshot(d) == before
        checks.append("next is read-only; actions use absolute paths")
        put(d, "critique-0.json", review(d, "advance"))
        assert expect(d, "command")["step"] == "render"
        expect(d, "done", "render")
        done = snapshot(d)
        expect(d, "done")
        assert snapshot(d) == done
        expect(d, "blocked", "assemble")
        assert snapshot(d) == done, "out-of-order command must not overwrite"
        checks.append("advance renders; done validates content; no overwrite")
        (d / "brief.md").write_text("", encoding="utf-8")
        expect(d, "blocked")
        checks.append("empty final never counts as done")

        d = fresh(parent, "revision")
        put(d, "critique-0.json", review(d, "revise"))
        assert expect(d, "llm_subagent")["step"] == "reviser"
        prior = snapshot(d)
        for patch in ({"sources": []}, {"constraints": "changed"}, {"verdict": "advance"},
                      {"answer": "unrequested"}, {"limits": " "}, {"limits": FILL["limits"]}):
            put(d, "patch.json", patch)
            expect(d, "blocked", "merge")
            assert not (d / "revised.json").exists()
        put(d, "patch.json", {"limits": FILL["limits"] + " 单次事件不能预测未来停机频率。[E3]"})
        patch_before = (d / "patch.json").read_bytes()
        assert expect(d, "llm_subagent", "merge")["step"] == "critic-1"
        for name, state in prior.items():
            assert (d / name).read_bytes() == state[0], name
        assert (d / "patch.json").read_bytes() == patch_before
        original = read(d, "draft.json")
        revised = read(d, "revised.json")
        for key in ("sources", "constraints", "topic", "synthetic"):
            assert original[key] == revised[key]
        assert original["review_token"] != revised["review_token"]
        checks.append("patch whitelist, no-op rejection, immutable inputs, fresh review token")
        for artifact in ("patch.json", "revised.json", "critique-1.json"):
            saved = (d / artifact).read_bytes() if (d / artifact).exists() else None
            (d / artifact).write_text('{"broken":', encoding="utf-8")
            expect(d, "blocked")
            if saved is None:
                (d / artifact).unlink()
            else:
                (d / artifact).write_bytes(saved)
        stale_review = review(d, "advance")
        put(d, "critique-1.json", stale_review)
        expect(d, "blocked")
        checks.append("damaged revision-stage JSON and stale initial review rejected")
        put(d, "critique-1.json", review(d, "revise", 1))
        expect(d, "blocked")
        expect(d, "blocked", "merge")
        expect(d, "blocked", "render")
        assert not (d / "brief.md").exists()
        put(d, "critique-1.json", review(d, "abandon", 1))
        expect(d, "blocked")
        checks.append("one revision only; failed re-review cannot render")
        put(d, "critique-1.json", review(d, "advance", 1))
        expect(d, "done", "render")
        checks.append("successful re-review renders revised content")

        d = fresh(parent, "bad-reviews")
        good = review(d, "revise")
        invalid = []
        for field, value in (("verdict", "pass"), ("reason", " "), ("review_token", "old"),
                             ("findings", []), ("verdict", "advance")):
            item = copy.deepcopy(good)
            item[field] = value
            invalid.append(item)
        for field, value in (("scope", "sources"), ("scope", "unknown"), ("reason", ""), ("evidence", [])):
            item = copy.deepcopy(good)
            item["findings"][0][field] = value
            invalid.append(item)
        for field, value in (("source_id", "E99"), ("quote", "not in source"), ("quote", "")):
            item = copy.deepcopy(good)
            item["findings"][0]["evidence"][0][field] = value
            invalid.append(item)
        for item in invalid:
            put(d, "critique-0.json", item)
            expect(d, "blocked")
        put(d, "critique-0.json", review(d, "abandon"))
        expect(d, "blocked")
        checks.append("invalid verdict, thin review, wrong scope/evidence, stale review and abandon rejected")

        for artifact in ("input.json", "skeleton.json", "fill_map.json", "draft.json", "critique-0.json"):
            d = fresh(parent, "corrupt-" + artifact)
            (d / artifact).write_text('{"broken":', encoding="utf-8")
            expect(d, "blocked")
        checks.append("damaged JSON blocked at each initial stage")
        d = fresh(parent, "json-values")
        for payload in ('{"answer":"a","answer":"b","conflict":"c","limits":"l"}', 'NaN', '[]'):
            (d / "fill_map.json").write_text(payload, encoding="utf-8")
            expect(d, "blocked")
        for field in FILL:
            put(d, "fill_map.json", {**FILL, field: " "})
            expect(d, "blocked")
        put(d, "fill_map.json", {**FILL, "answer": "a" * 701})
        expect(d, "blocked")
        checks.append("duplicate keys, nonstandard JSON, empty fields and budget excess rejected")

        d = fresh(parent, "changed")
        before = snapshot(d)
        changed = read(d, "input.json")
        changed["topic"] += "（另一问题）"
        alternate = parent / "alternate.json"
        alternate.write_text(json.dumps(changed), encoding="utf-8")
        expect(d, "blocked", "prepare", "--input", alternate)
        assert snapshot(d) == before
        base = read(d, "skeleton.json")
        base["constraints"]["max_revisions"] = True
        put(d, "skeleton.json", base)
        expect(d, "blocked")
        checks.append("changed input rejected without writes; mechanical fields checked strictly")
        d = fresh(parent, "orphan")
        (d / "draft.json").unlink()
        put(d, "critique-0.json", {})
        expect(d, "blocked")
        checks.append("orphaned downstream output cannot skip a predecessor")
    print(json.dumps({"status": "passed", "checks": checks,
                      "llm_execution": "not tested; these are deterministic test doubles"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
