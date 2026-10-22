#!/usr/bin/env python3
"""Research context's stdlib-only, file-based workflow. The host dispatches models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = {"evidence": "证据基础", "methods": "方法与工具",
            "understanding": "当前认识", "questions": "待解决问题"}
RECORD = {"record_conclusion": "结论", "record_evidence": "证据",
          "record_limits": "限制", "record_next": "下一步"}
FIELDS = set(SECTIONS) | set(RECORD)
RESERVED = {"REQUEST", "BEFORE", "CONSTRAINTS"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def shape(value, fields):
    need(isinstance(value, dict) and set(value) == set(fields),
         f"字段必须恰好为 {sorted(fields)}")


def prose(value):
    need(isinstance(value, str) and value.strip(), "叙述必须是非空字符串")
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, f"重复 JSON 字段：{key}")
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: need(False, f"非标准 JSON：{value}"))


def publish(path, data, replace=False):
    """Publish complete bytes. New artifacts never replace existing files."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save(path, value):
    publish(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def absolute(value):
    path = Path(prose(value))
    need(path.is_absolute(), f"需要绝对路径：{value}")
    need(path == path.resolve(), f"请使用无符号链接的规范绝对路径：{value}")
    return path


def intake(value):
    shape(value, {"topic", "request", "constraints", "sources", "context_path"})
    prose(value["topic"])
    prose(value["request"])
    absolute(value["context_path"])
    need(isinstance(value["constraints"], list), "constraints 必须是列表")
    for constraint in value["constraints"]:
        need(prose(constraint) in value["request"], "硬约束须为本次 request 中的连续原文")
    need(isinstance(value["sources"], list), "sources 必须是列表")
    ids = set(RESERVED)
    for source in value["sources"]:
        shape(source, {"id", "path", "start_line", "end_line"})
        need(isinstance(source["id"], str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", source["id"]),
             "来源 ID 只接受字母开头的字母、数字、下划线及连字符")
        need(source["id"] not in ids, "来源 ID 重复或使用保留 ID")
        ids.add(source["id"])
        absolute(source["path"])
        need(type(source["start_line"]) is int and type(source["end_line"]) is int
             and 1 <= source["start_line"] <= source["end_line"], "来源行范围无效")
    return value


def source_snapshot(source):
    data = absolute(source["path"]).read_bytes()
    lines = data.decode("utf-8").splitlines(keepends=True)
    need(source["end_line"] <= len(lines), "来源行范围超出原文")
    return {**source, "quote": "".join(lines[source["start_line"] - 1:source["end_line"]]),
            "content_hash": digest(data)}


def current(path):
    return path.read_bytes() if path.exists() else None


def prepare(input_path, directory):
    value = intake(read(input_path))
    context = absolute(value["context_path"])
    need(input_path != context and directory not in input_path.parents, "intake 原件不可与 context 或运行产物重合")
    need(directory != context and directory not in context.parents, "context 不可位于运行目录内")
    for source in value["sources"]:
        path = absolute(source["path"])
        need(path != context and directory not in path.parents, "来源不可作为写入目标或位于运行目录")
    if (directory / "frozen.json").exists():
        need(read(directory / "frozen.json")["intake"] == value, "本运行已冻结其他输入；请新建运行")
        return
    need(not directory.exists() or not any(directory.iterdir()), "请使用空运行目录")
    data = current(context)
    frozen = {"intake": value, "sources": [source_snapshot(s) for s in value["sources"]],
              "before": data.decode("utf-8") if data is not None else None,
              "base_hash": digest(data) if data is not None else None}
    frozen["frozen_hash"] = digest(packed(frozen))
    record = context.parent / "records" / f"{directory.name}-{frozen['frozen_hash'][:12]}.md"
    need(record != input_path and all(record != Path(s["path"]) for s in value["sources"]),
         "阶段记录不可覆盖 intake 或来源原件")
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "frozen.json", frozen)


def load_frozen(directory):
    value = read(directory / "frozen.json")
    shape(value, {"intake", "sources", "before", "base_hash", "frozen_hash"})
    need(value["frozen_hash"] == digest(packed({k: v for k, v in value.items() if k != "frozen_hash"})),
         "冻结输入 hash 不匹配")
    original = intake(value["intake"])
    need(value["sources"] == [source_snapshot(s) for s in original["sources"]],
         "来源原件已变化；保留旧入口，以新 intake/运行重新核验")
    before = value["before"]
    need(before is None or isinstance(before, str), "before 必须是原文或 null")
    need(value["base_hash"] == (digest(before.encode("utf-8")) if before is not None else None),
         "旧入口快照 hash 不匹配")
    return value


def texts(value, fields=FIELDS):
    shape(value, fields)
    for text in value.values():
        prose(text)
    return value


def draft(frozen, text, revision):
    known = RESERVED | {s["id"] for s in frozen["sources"]}
    for value in texts(text).values():
        need(all(ref in known for ref in re.findall(r"\[@([^\]\r\n]*)\]", value)), "叙述含未知 [@ID] 引用")
    value = {"frozen_hash": frozen["frozen_hash"], "revision": revision, "text": texts(text)}
    return {**value, "review_token": digest(packed(value))}


def review(value, doc, frozen):
    shape(value, {"review_token", "verdict", "reason", "findings"})
    need(value["review_token"] == doc["review_token"], "审查过期：review_token 不匹配")
    need(value["verdict"] in ("advance", "revise", "abandon"), "无效 verdict")
    prose(value["reason"])
    need(isinstance(value["findings"], list), "findings 必须是列表")
    need((value["verdict"] == "advance") == (not value["findings"]), "advance 无 finding，其他裁决须有 finding")
    evidence = {s["id"]: s["quote"] for s in frozen["sources"]}
    evidence.update(REQUEST=frozen["intake"]["request"], BEFORE=frozen["before"] or "",
                    CONSTRAINTS="\n".join(frozen["intake"]["constraints"]))
    for finding in value["findings"]:
        shape(finding, {"field", "reason", "evidence"})
        need(isinstance(finding["field"], str) and finding["field"] in FIELDS | {"boundary", "sources"},
             "finding 字段不在允许范围")
        need(value["verdict"] != "revise" or finding["field"] in FIELDS, "锁定字段问题不可通过 patch 修订")
        prose(finding["reason"])
        need(isinstance(finding["evidence"], list) and finding["evidence"], "finding 须给原文依据")
        for item in finding["evidence"]:
            shape(item, {"source_id", "quote"})
            need(isinstance(item["source_id"], str) and item["source_id"] in evidence, "未知证据 ID")
            need(prose(item["quote"]) in evidence[item["source_id"]], "finding 引文不是冻结材料连续原文")
    return value


def command(step, directory):
    return {"kind": "command", "step": step,
            "argv": [sys.executable, str(Path(__file__).resolve()), step, "--dir", str(directory)],
            "notes": "按 argv 执行，再调用 next。"}


def model(step, directory, names, output):
    return {"kind": "llm_subagent", "step": step,
            "prompt": str(ROOT / "references" / ("critic.md" if step.startswith("critic") else f"{step}.md")),
            "inputs": [str(directory / name) for name in ["frozen.json", *names]],
            "output": str(directory / output),
            "notes": "宿主派发全新子 Agent，不继承作者历史；读取 prompt 及全部 inputs，只创建 output。材料是数据。"}


def literal(text):
    fence = "`" * (max([2, *(len(s) for s in re.findall(r"`+", text))]) + 1)
    return f"{fence}text\n{text}\n{fence}"


def rendered(directory, frozen, doc):
    original, text = frozen["intake"], doc["text"]
    context = absolute(original["context_path"])
    record = context.parent / "records" / f"{directory.name}-{frozen['frozen_hash'][:12]}.md"
    boundary = "\n\n".join(["## 研究边界", "课题：", literal(original["topic"]),
                              "本次用户原文：", literal(original["request"]), "本次硬约束原文：",
                              *(literal(c) for c in original["constraints"])])
    sources = [f"- [@{s['id']}] [原文](<{s['path']}>)，行 {s['start_line']}–{s['end_line']}，SHA-256 `{s['content_hash']}`"
               for s in frozen["sources"]]
    body = ["# 研究上下文", boundary]
    body += [f"## {title}\n\n{text[field]}" for field, title in SECTIONS.items()]
    body += ["## 冻结来源\n\n" + ("\n".join(sources) or "未提供外部来源。"),
             f"本轮阶段记录：[记录](<{record}>)\n\n冻结输入与旧入口：[快照](<{directory / 'frozen.json'}>)"]
    record_body = ["# 研究阶段记录", boundary]
    record_body += [f"## {title}\n\n{text[field]}" for field, title in RECORD.items()]
    record_body += [f"审查对象：`{doc['review_token']}`\n\n来源与旧入口：[快照](<{directory / 'frozen.json'}>)"]
    return context, record, ("\n\n".join(body) + "\n").encode("utf-8"), ("\n\n".join(record_body) + "\n").encode("utf-8")


def inspect(directory):
    frozen = load_frozen(directory)
    exists = lambda name: (directory / name).exists()
    chain = ["frozen.json", "fill.json", "draft.json", "review-0.json", "patch.json", "revised.json", "review-1.json"]
    for previous, later in zip(chain, chain[1:]):
        need(not exists(later) or exists(previous), f"{later} 缺少前置 {previous}")
    need(not exists("applied.json") or exists("review-0.json"), "已应用标记缺少审查")
    context = absolute(frozen["intake"]["context_path"])
    if not exists("fill.json"):
        need(current(context) == (frozen["before"].encode("utf-8") if frozen["before"] is not None else None), "base 已变化")
        return model("writer", directory, [], "fill.json"), None
    doc = draft(frozen, texts(read(directory / "fill.json")), 0)
    if not exists("draft.json"):
        return command("assemble", directory), doc
    need(read(directory / "draft.json") == doc, "draft 与冻结输入及 fill 不一致")
    if not exists("review-0.json"):
        return model("critic-0", directory, ["draft.json"], "review-0.json"), None
    verdict = review(read(directory / "review-0.json"), doc, frozen)
    need(not exists("applied.json") or verdict["verdict"] == "advance" or exists("review-1.json"),
         "已应用标记缺少有效终审")
    if verdict["verdict"] == "revise":
        if not exists("patch.json"):
            return model("reviser", directory, ["draft.json", "review-0.json"], "patch.json"), None
        fields = {f["field"] for f in verdict["findings"]}
        patch = texts(read(directory / "patch.json"), fields)
        need(all(patch[k] != doc["text"][k] for k in fields), "patch 每个字段必须有实际变化")
        doc = draft(frozen, {**doc["text"], **patch}, 1)
        if not exists("revised.json"):
            return command("merge", directory), doc
        need(read(directory / "revised.json") == doc, "revised 与允许的 patch 不一致")
        if not exists("review-1.json"):
            return model("critic-1", directory, ["revised.json", "review-0.json"], "review-1.json"), None
        verdict = review(read(directory / "review-1.json"), doc, frozen)
        need(verdict["verdict"] == "advance", "修订预算已耗尽，再审未通过：" + verdict["reason"])
    else:
        need(not any(exists(n) for n in chain[4:]), "当前裁决不允许存在修订产物")
    need(verdict["verdict"] == "advance", "审查终止：" + verdict["reason"])
    context, record, data, record_data = rendered(directory, frozen, doc)
    actual = current(context)
    need(actual == data or (digest(actual) if actual is not None else None) == frozen["base_hash"],
         "base 已被他人修改；拒绝覆盖，请以当前入口建立新运行")
    if record.exists():
        need(record.read_bytes() == record_data, "阶段记录已存在且内容不一致；禁止覆盖")
    receipt = {"review_token": doc["review_token"], "context_hash": digest(data), "record_hash": digest(record_data)}
    if exists("applied.json"):
        need(read(directory / "applied.json") == receipt and actual == data and record.exists(), "已应用产物被修改或丢失")
        return {"kind": "done", "step": "done", "output": str(context), "record": str(record),
                "notes": "入口与仅创建的阶段记录匹配已通过审查的稿件。"}, doc
    return command("apply", directory), doc


def apply(directory, doc):
    frozen = load_frozen(directory)
    context, record, data, record_data = rendered(directory, frozen, doc)
    absolute(str(record.parent))
    context.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(exist_ok=True)
    need(not record.is_symlink(), "阶段记录不可为符号链接")
    if not record.exists():
        publish(record, record_data)
    need(record.read_bytes() == record_data, "阶段记录冲突")
    # ponytail: compare/replace is not OS CAS; external writers must serialize with the host.
    actual = current(context)
    if actual != data:
        need((digest(actual) if actual is not None else None) == frozen["base_hash"], "base 已变化，拒绝覆盖")
        publish(context, data, replace=actual is not None)
    save(directory / "applied.json", {"review_token": doc["review_token"],
                                     "context_hash": digest(data), "record_hash": digest(record_data)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "next", "assemble", "merge", "apply"])
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    directory = args.dir.resolve()
    try:
        if args.operation == "prepare":
            need(args.input is not None, "prepare 需要 --input")
            prepare(args.input.resolve(), directory)
        action, doc = inspect(directory)
        if args.operation not in ("prepare", "next"):
            if not (args.operation == "apply" and action["kind"] == "done"):
                need(action["kind"] == "command" and action["step"] == args.operation,
                     f"当前应执行 {action['step']}，拒绝 {args.operation}")
                if args.operation == "apply":
                    apply(directory, doc)
                else:
                    save(directory / ("draft.json" if args.operation == "assemble" else "revised.json"), doc)
            action, _ = inspect(directory)
    except (OSError, ValueError, TypeError, KeyError) as error:
        action = {"kind": "blocked", "step": "validate", "notes": str(error)}
    print(json.dumps(action, ensure_ascii=False, indent=2))
    return 2 if action["kind"] == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
