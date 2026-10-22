#!/usr/bin/env python3
"""Small executable contract checks. Fixtures are synthetic, not model reviews."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import run


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def cli(directory, operation, *extra):
    result = subprocess.run([sys.executable, str(Path(run.__file__).resolve()), operation,
                             "--dir", str(directory), *map(str, extra)], capture_output=True, text=True)
    action = json.loads(result.stdout)
    assert result.returncode == (2 if action["kind"] == "blocked" else 0), result.stderr
    return action


def fixture(root, name):
    base = root / name
    base.mkdir()
    context = base / "context.md"
    context.write_bytes(b"# Legacy\r\nKeep uncertainty.\r\n")
    source = base / "source.txt"
    source.write_bytes(b"Unused line\r\nSynthetic observation: one sample.\r\nNo causal claim.\r\n")
    value = {"topic": "Synthetic check", "request": "Update context. Do not run experiments.",
             "constraints": ["Do not run experiments."], "context_path": str(context),
             "sources": [{"id": "E1", "path": str(source), "start_line": 2, "end_line": 3}]}
    write(base / "intake.json", value)
    directory = base / "run"
    assert cli(directory, "prepare", "--input", base / "intake.json")["step"] == "writer"
    frozen = run.read(directory / "frozen.json")
    assert frozen["before"] == "# Legacy\r\nKeep uncertainty.\r\n"
    assert frozen["sources"][0]["quote"].endswith("claim.\r\n")
    write(directory / "fill.json", {field: "Synthetic narrative E1: one sample; uncertainty retained." for field in run.FIELDS})
    assert cli(directory, "assemble")["step"] == "critic-0"
    return directory, context, source


def verdict(directory, result="advance", revision=0, field="understanding"):
    doc = run.read(directory / ("revised.json" if revision else "draft.json"))
    value = {"review_token": doc["review_token"], "verdict": result, "reason": "Synthetic test judgement",
             "findings": [] if result == "advance" else [{"field": field, "reason": "Retain limits.",
                         "evidence": [{"source_id": "E1", "quote": "one sample."}]}]}
    write(directory / f"review-{revision}.json", value)
    return value


def blocked(directory, operation="next"):
    action = cli(directory, operation)
    assert action["kind"] == "blocked", action


def check():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        blocked(root / "missing")
        directory, context, source = fixture(root, "success")
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}
        assert cli(directory, "next")["step"] == "critic-0"
        assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}
        blocked(directory, "apply")
        assert context.read_bytes().startswith(b"# Legacy\r\n")
        verdict(directory)
        assert cli(directory, "apply")["kind"] == "done"
        snapshot = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.parent.rglob("*") if p.is_file()}
        assert cli(directory, "apply")["kind"] == "done"
        assert snapshot == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.parent.rglob("*") if p.is_file()}
        assert len(list((context.parent / "records").iterdir())) == 1
        assert b"Do not run experiments." in context.read_bytes()

        for name, invalid in [("bad-json", "{"), ("duplicates", '{"evidence":"x","evidence":"y"}'),
                              ("nan", '{"evidence":NaN}'), ("unknown", '{"sources":"changed"}')]:
            directory, context, _ = fixture(root, name)
            (directory / "fill.json").write_text(invalid, encoding="utf-8")
            blocked(directory)
            assert context.read_bytes().startswith(b"# Legacy\r\n")

        directory, _, source = fixture(root, "source-change")
        verdict(directory)
        source.write_bytes(source.read_bytes().replace(b"Unused line", b"Changed outside selected lines"))
        blocked(directory)
        blocked(directory, "apply")

        directory, _, _ = fixture(root, "stale-review")
        value = verdict(directory)
        value["review_token"] = "wrong"
        write(directory / "review-0.json", value)
        blocked(directory)

        directory, _, _ = fixture(root, "unknown-reference")
        fill = run.read(directory / "fill.json")
        fill["evidence"] = "Unsupported [@MISSING]"
        write(directory / "fill.json", fill)
        blocked(directory)
        frozen = run.load_frozen(directory)
        fill["evidence"] = "[@E1] [@BEFORE] [@REQUEST] [@CONSTRAINTS]"
        assert run.draft(frozen, fill, 0)["text"] == fill

        directory, context, _ = fixture(root, "base-change")
        verdict(directory)
        context.write_text("Colleague update", encoding="utf-8")
        blocked(directory, "apply")
        assert context.read_text() == "Colleague update"
        assert not (context.parent / "records").exists()

        for name, patch in [("unauthorized-patch", {"request": "Alter constraints"}),
                             ("unknown-reference-patch", {"understanding": "Unsupported [@MISSING]"}),
                             ("unchanged-patch", {"understanding": "Synthetic narrative E1: one sample; uncertainty retained."})]:
            directory, _, _ = fixture(root, name)
            verdict(directory, "revise")
            assert cli(directory, "next")["step"] == "reviser"
            write(directory / "patch.json", patch)
            blocked(directory, "merge")

        for final in ("advance", "revise", "abandon"):
            directory, context, _ = fixture(root, "revision-" + final)
            verdict(directory, "revise")
            write(directory / "patch.json", {"understanding": "E1: one sample, no causal inference; legacy uncertainty retained."})
            assert cli(directory, "merge")["step"] == "critic-1"
            blocked(directory, "apply")
            verdict(directory, final, revision=1)
            if final == "advance":
                assert cli(directory, "apply")["kind"] == "done"
            else:
                blocked(directory)
                blocked(directory, "apply")
                assert context.read_bytes().startswith(b"# Legacy\r\n")

        directory, _, _ = fixture(root, "abandon")
        verdict(directory, "abandon", field="sources")
        blocked(directory)
        blocked(directory, "apply")

        for interruption in ("record", "context"):
            directory, context, _ = fixture(root, "interrupted-" + interruption)
            verdict(directory)
            action, doc = run.inspect(directory)
            assert action["step"] == "apply"
            real_publish = run.publish

            def interrupt(path, data, replace=False):
                real_publish(path, data, replace)
                if (interruption == "record" and path.parent.name == "records") or path == context:
                    raise OSError("Simulated process interruption after publish")

            run.publish = interrupt
            try:
                try:
                    run.apply(directory, doc)
                    assert False, "Interruption was not triggered"
                except OSError:
                    pass
            finally:
                run.publish = real_publish
            assert not (directory / "applied.json").exists()
            if interruption == "record":
                assert context.read_bytes().startswith(b"# Legacy\r\n")
            assert cli(directory, "next")["step"] == "apply"
            assert cli(directory, "apply")["kind"] == "done"
            assert len(list((context.parent / "records").iterdir())) == 1

        directory, context, _ = fixture(root, "record-collision")
        verdict(directory)
        _, doc = run.inspect(directory)
        _, record, _, _ = run.rendered(directory, run.load_frozen(directory), doc)
        record.parent.mkdir()
        record.write_text("Do not overwrite existing record", encoding="utf-8")
        blocked(directory, "apply")
        assert record.read_text() == "Do not overwrite existing record"
        assert context.read_bytes().startswith(b"# Legacy\r\n")

        directory, context, source = fixture(root, "input-contract")
        value = run.read(directory.parent / "intake.json")
        for update in ({"context_path": str(source)}, {"context_path": str(directory.parent / "changed.json")},
                       {"constraints": ["Invented restriction"]}, {"extra": True}):
            write(directory.parent / "changed.json", {**value, **update})
            assert cli(directory.parent / "new-run", "prepare", "--input", directory.parent / "changed.json")["kind"] == "blocked"
        value["sources"][0]["id"] = "REQUEST"
        write(directory.parent / "changed.json", value)
        assert cli(directory.parent / "new-run", "prepare", "--input", directory.parent / "changed.json")["kind"] == "blocked"
    print("PASS: routing, strict JSON, frozen sources/boundaries, stale review, patch limits, base conflict, terminal states, idempotence and interruption recovery")


if __name__ == "__main__":
    check()
