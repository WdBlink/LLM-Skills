#!/usr/bin/env python3
"""Original, stdlib-only evidence-brief example; it never calls an LLM."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"answer", "conflict", "limits"}
ARTIFACTS = ["input.json", "skeleton.json", "fill_map.json", "draft.json",
             "critique-0.json", "patch.json", "revised.json", "critique-1.json", "brief.md"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant,
                          object_pairs_hook=unique_object)
    except (OSError, ValueError) as error:
        raise ValueError(f"无法读取有效 JSON：{path}：{error}") from None


def invalid_constant(value):
    raise ValueError(f"不允许非标准 JSON 常量 {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON 字段重复：{key}")
        result[key] = value
    return result


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            f"{label} 必须恰好包含字段 {sorted(expected)}")


def nonempty(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label} 必须是非空文本")


def source_input(value):
    keys(value, {"topic", "synthetic", "sources"}, "input")
    nonempty(value["topic"], "topic")
    require(value["synthetic"] is True, "本教学样例只接受 synthetic=true 的虚构证据")
    require(isinstance(value["sources"], list) and len(value["sources"]) == 3,
            "本样例恰好使用 3 条来源")
    ids = set()
    for source in value["sources"]:
        keys(source, {"source_id", "title", "fact", "scope"}, "source")
        for field, text in source.items():
            nonempty(text, f"source.{field}")
        require(re.fullmatch(r"E[1-9][0-9]*", source["source_id"]) is not None,
                "source_id 应形如 E1")
        require(source["source_id"] not in ids, "source_id 不可重复")
        ids.add(source["source_id"])
    return value


def skeleton(value):
    return {**copy.deepcopy(value), "constraints": {"max_revisions": 1, "max_chars_per_field": 700},
            "text": {field: None for field in sorted(FIELDS)}}


def text_fields(value, expected=FIELDS):
    keys(value, expected, "文本映射")
    for field, text in value.items():
        nonempty(text, field)
        require(len(text) <= 700, f"{field} 超过固定的 700 字符预算")
    return value


def candidate(base, text, revision):
    result = copy.deepcopy(base)
    result.pop("review_token", None)
    result["text"] = text_fields(text)
    result["revision"] = revision
    # Content binding prevents a review of an older draft from approving a new one.
    result["review_token"] = hashlib.sha256(
        encoded(result).encode("utf-8")).hexdigest()
    return result


def critique(value, doc):
    keys(value, {"review_token", "verdict", "reason", "findings"}, "critique")
    require(value["review_token"] == doc["review_token"], "评审未绑定当前稿件 review_token")
    require(value["verdict"] in ("advance", "revise", "abandon"), "无效 verdict")
    nonempty(value["reason"], "critique.reason")
    findings = value["findings"]
    require(isinstance(findings, list), "findings 必须是列表")
    require((value["verdict"] == "advance") == (len(findings) == 0),
            "advance 必须无 findings；revise/abandon 必须有可定位的问题")
    sources = {item["source_id"]: item for item in doc["sources"]}
    for finding in findings:
        keys(finding, {"scope", "reason", "evidence"}, "finding")
        require(finding["scope"] in FIELDS | {"sources"}, "finding.scope 不在允许范围")
        require(value["verdict"] != "revise" or finding["scope"] in FIELDS,
                "来源问题无法通过文本 patch 修复，应 abandon")
        nonempty(finding["reason"], "finding.reason")
        require(isinstance(finding["evidence"], list) and finding["evidence"],
                "finding.evidence 需要至少一条来源原文")
        for evidence in finding["evidence"]:
            keys(evidence, {"source_id", "quote"}, "evidence")
            require(isinstance(evidence["source_id"], str) and evidence["source_id"] in sources,
                    "evidence 引用了未知来源")
            nonempty(evidence["quote"], "evidence.quote")
            source = sources[evidence["source_id"]]
            require(any(evidence["quote"] in source[field] for field in ("fact", "scope")),
                    "evidence.quote 必须是该来源 fact 或 scope 中的连续原文")
    return value


def atomic_new(path, text):
    """Publish a complete file without replacing any existing artifact."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".pending-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_new(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def command(step, directory, *args):
    return {"kind": "command", "step": step,
            "argv": [str(Path(sys.executable).resolve()), str(ROOT / "scripts/run.py"),
                     step, *map(str, args), "--dir", str(directory)],
            "notes": "执行 argv 后，再调用 next；不要把整条 argv 当作一个 shell 字符串。"}


def llm(step, directory, inputs, output):
    role = "critic" if step.startswith("critic") else step
    return {"kind": "llm_subagent", "step": step,
            "prompt": str(ROOT / "references" / f"{role}.md"),
            "inputs": [str(directory / name) for name in inputs],
            "output": str(directory / output),
            "notes": "由宿主派发无会话历史的新子 Agent；仅传本 action 的 prompt、inputs、output。"
                     "只写 output，完成 JSON 校验后原子提交；不要修改输入或代替其他角色。"}


def markdown(doc):
    paragraphs = [f"# {doc['topic']}", "**教学演示：以下来源和观察全部虚构，不代表真实研究。**"]
    for field, title in (("answer", "证据支持的回答"), ("conflict", "冲突与解释"), ("limits", "适用范围与限制")):
        paragraphs.extend([f"## {title}", doc["text"][field]])
    paragraphs.append("## 来源快照（虚构）")
    for source in doc["sources"]:
        paragraphs.append(f"- [{source['source_id']}] {source['title']}：{source['fact']} 范围：{source['scope']}")
    return "\n\n".join(paragraphs) + "\n"


def inspect_run(directory):
    """Read-only navigator: return one JSON action and the current candidate."""
    path = lambda name: directory / name
    exists = lambda name: path(name).exists()
    if not exists("input.json"):
        require(not directory.exists() or not any(directory.iterdir()),
                "目录非空但缺少 input.json，请使用新 --dir")
        return command("prepare", directory, "--input", ROOT / "fixtures/input.json"), None
    original = source_input(read_json(path("input.json")))
    # Reject orphaned outputs instead of silently routing around damaged state.
    for later, earlier in zip(ARTIFACTS[2:8], ARTIFACTS[1:7]):
        require(not exists(later) or exists(earlier), f"{later} 缺少前置产物 {earlier}")
    require(not exists("brief.md") or exists("critique-0.json"), "brief.md 缺少评审")
    if not exists("skeleton.json"):
        return command("prepare", directory, "--input", path("input.json")), None
    base = read_json(path("skeleton.json"))
    require(encoded(base) == encoded(skeleton(original)), "skeleton 与输入或固定机械字段不一致")
    if not exists("fill_map.json"):
        return llm("writer", directory, ["skeleton.json"], "fill_map.json"), None
    fill = text_fields(read_json(path("fill_map.json")))
    expected = candidate(base, fill, 0)
    if not exists("draft.json"):
        return command("assemble", directory), expected
    draft = read_json(path("draft.json"))
    require(encoded(draft) == encoded(expected), "draft 不等于 skeleton + fill_map，可能被改写或已过期")
    if not exists("critique-0.json"):
        return llm("critic-0", directory, ["draft.json"], "critique-0.json"), None
    review = critique(read_json(path("critique-0.json")), draft)
    doc = draft
    if review["verdict"] == "revise":
        if not exists("patch.json"):
            require(not exists("brief.md"), "修订尚未完成，不能保留终态文件")
            return llm("reviser", directory, ["draft.json", "critique-0.json"], "patch.json"), None
        allowed = {finding["scope"] for finding in review["findings"]}
        patch = text_fields(read_json(path("patch.json")), allowed)
        require(any(patch[field] != draft["text"][field] for field in allowed), "patch 没有实际变化")
        expected = candidate(draft, {**draft["text"], **patch}, 1)
        if not exists("revised.json"):
            return command("merge", directory), expected
        doc = read_json(path("revised.json"))
        require(encoded(doc) == encoded(expected), "revised 不等于通过白名单的 patch 合并结果")
        if not exists("critique-1.json"):
            require(not exists("brief.md"), "修订稿尚未再审，不能保留终态文件")
            return llm("critic-1", directory, ["revised.json"], "critique-1.json"), None
        review = critique(read_json(path("critique-1.json")), doc)
        require(review["verdict"] == "advance", "一次修订预算已用完，再审未通过；请报告阻塞，不可强行完成")
    else:
        require(not any(exists(name) for name in ("patch.json", "revised.json", "critique-1.json")),
                "当前 verdict 不允许修订产物，状态发生冲突")
    require(review["verdict"] == "advance", f"评审为 {review['verdict']}，不能生成终稿")
    if not exists("brief.md"):
        return command("render", directory), doc
    actual = path("brief.md").read_text(encoding="utf-8")
    require(actual.strip() and actual == markdown(doc), "终稿为空、过期或与已通过评审的内容不一致")
    return {"kind": "done", "step": "done", "output": str(path("brief.md")),
            "notes": "最终内容与当前稿件一致，最新独立评审为 advance；向用户交付此文件。"}, doc


def prepare(input_path, directory):
    value = source_input(read_json(input_path))
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "input.json"
    if snapshot.exists():
        require(read_json(snapshot) == value, "输入与已有运行不同，请使用新的 --dir；旧运行不可复用")
    else:
        require(not any(directory.iterdir()), "运行目录非空，请使用新的 --dir")
        write_json(snapshot, value)
    base_path = directory / "skeleton.json"
    if base_path.exists():
        require(encoded(read_json(base_path)) == encoded(skeleton(value)), "已有 skeleton 损坏或被改写，请使用新 --dir")
    else:
        write_json(base_path, skeleton(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("prepare", "next", "assemble", "merge", "render"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dir", required=True, type=Path)
        if name == "prepare":
            sub.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    directory = args.dir.resolve()
    try:
        if args.operation == "prepare":
            prepare(args.input.resolve(), directory)
        elif args.operation != "next":
            action, doc = inspect_run(directory)
            require(action["kind"] == "command" and action["step"] == args.operation,
                    f"当前应执行 {action['step']}，不可执行 {args.operation}")
            if args.operation in ("assemble", "merge"):
                write_json(directory / ("draft.json" if args.operation == "assemble" else "revised.json"), doc)
            else:
                atomic_new(directory / "brief.md", markdown(doc))
        action, _ = inspect_run(directory)
    except (OSError, ValueError, TypeError, KeyError) as error:
        action = {"kind": "blocked", "step": "validate", "notes": str(error)}
    print(json.dumps(action, ensure_ascii=False, indent=2))
    return 2 if action["kind"] == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
