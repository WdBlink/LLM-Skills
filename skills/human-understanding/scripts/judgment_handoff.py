#!/usr/bin/env python3
"""Prepare evidence, accept a host's judgment, then apply it without rewriting evidence.

Only Python's standard library is used. This script never calls an LLM or the
network. A request hash binds a decision to its evidence and goal; it is not an
authenticity signature, and structural checks do not verify semantic accuracy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


PROTOCOL_VERSION = 1
DEFAULT_MAX_CHARS = 12_000
HARD_MAX_CHARS = 50_000
MAX_UNITS = 1_000
MAX_JSON_BYTES = 1_048_576
CORE_KEYS = ("protocol_version", "input_sha256", "goal", "units")
REQUEST_KEYS = set(CORE_KEYS) | {"request_id", "status"}


class HandoffError(Exception):
    """An actionable input or protocol error, displayed without a traceback."""


def require(condition, message):
    if not condition:
        raise HandoffError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def request_id(core):
    return "sha256:" + hashlib.sha256(canonical(core)).hexdigest()


def nonempty_string(value, name, limit=4_000):
    require(isinstance(value, str) and bool(value.strip()),
            f"{name} must be a nonempty string")
    require(len(value) <= limit, f"{name} exceeds {limit} characters")
    return value.strip()


def read_bounded(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, f"{path}: input exceeds the {limit}-byte limit")
    return data


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise HandoffError(f"JSON contains nonstandard number: {value}")


def read_json(path):
    try:
        return json.loads(read_bounded(path, MAX_JSON_BYTES).decode("utf-8-sig"),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except ValueError as exc:
        # Includes decoding/JSON errors and Python's oversized-integer guard.
        raise HandoffError(f"{path}: invalid UTF-8 JSON: {exc}") from exc


def protect_inputs(output, inputs):
    for source in inputs:
        same_path = output.resolve() == source.resolve()
        same_file = output.exists() and source.exists() and output.samefile(source)
        require(not (same_path or same_file),
                f"Output must differ from input: {source} (even with --force)")


def same_content(path, data):
    try:
        return path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data
    except FileNotFoundError:
        return False


def write_json_atomic(path, value, force=False):
    """Publish a complete file; a same-directory hard link prevents clobber races.

    No-overwrite mode needs a filesystem supporting hard links. If it does not,
    the operation fails instead of falling back to a partial or unsafe write.
    """
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")
    if same_content(path, data):
        return "unchanged"
    require(force or not path.exists(),
            f"{path} already exists with different content; choose another path or use --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                if same_content(path, data):
                    return "unchanged"
                raise HandoffError(f"{path} appeared with different content; output was not overwritten") from exc
        return "written"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def split_units(text):
    # Blank lines delimit paragraphs; internal text and line endings are retained.
    # Paragraph-terminal newlines are excluded from each evidence unit.
    paragraphs, buffer = [], []
    for line in text.splitlines(keepends=True):
        if line.strip():
            buffer.append(line)
        elif buffer:
            paragraphs.append("".join(buffer).rstrip("\r\n"))
            buffer = []
    if buffer:
        paragraphs.append("".join(buffer).rstrip("\r\n"))
    require(bool(paragraphs), "Article has no nonempty paragraphs")
    require(len(paragraphs) <= MAX_UNITS, f"Article exceeds {MAX_UNITS} paragraphs")
    return [{"unit_id": f"u{i:04d}", "text": text}
            for i, text in enumerate(paragraphs, 1)]


def prepare(args):
    require(1 <= args.max_chars <= HARD_MAX_CHARS,
            f"--max-chars must be between 1 and {HARD_MAX_CHARS}")
    protect_inputs(args.out, [args.input])
    goal = nonempty_string(args.goal, "goal", 2_000)
    raw = read_bounded(args.input, 4 * args.max_chars + 3)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HandoffError(f"{args.input}: article must be UTF-8 text") from exc
    require(len(text) <= args.max_chars,
            f"Article has {len(text)} characters, above --max-chars={args.max_chars}; "
            "choose a relevant excerpt, or explicitly raise the limit after checking the host's context budget")
    core = {"protocol_version": PROTOCOL_VERSION,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "goal": goal, "units": split_units(text)}
    request = {**core, "request_id": request_id(core), "status": "needs_judgment"}
    write = write_json_atomic(args.out, request, args.force)
    return {"status": request["status"], "request_id": request["request_id"],
            "unit_count": len(request["units"]), "input_characters": len(text),
            "output": str(args.out), "write": write}


def validate_request(request):
    require(isinstance(request, dict) and set(request) == REQUEST_KEYS,
            "Request must contain exactly: " + ", ".join(sorted(REQUEST_KEYS)))
    require(type(request["protocol_version"]) is int and
            request["protocol_version"] == PROTOCOL_VERSION, "Unsupported protocol_version")
    require(request["status"] == "needs_judgment", "Request status must be needs_judgment")
    require(isinstance(request["input_sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", request["input_sha256"]) is not None,
            "Invalid input_sha256")
    nonempty_string(request["goal"], "goal", 2_000)
    units = request["units"]
    require(isinstance(units, list) and 1 <= len(units) <= MAX_UNITS,
            f"units must contain 1..{MAX_UNITS} paragraphs")
    for i, unit in enumerate(units, 1):
        require(isinstance(unit, dict) and set(unit) == {"unit_id", "text"},
                f"units[{i - 1}] must contain exactly unit_id and text")
        require(unit["unit_id"] == f"u{i:04d}", "Request unit IDs must be unique and sequential")
        nonempty_string(unit["text"], f"{unit['unit_id']}.text", HARD_MAX_CHARS)
    require(sum(len(unit["text"]) for unit in units) <= HARD_MAX_CHARS,
            f"Request evidence exceeds {HARD_MAX_CHARS} characters")
    core = {key: request[key] for key in CORE_KEYS}
    require(request["request_id"] == request_id(core),
            "Request hash mismatch: evidence or goal changed; run prepare again")
    return request


def validate_decision(decision, request):
    allowed = {"request_id", "selected", "no_match_reason"}
    require(isinstance(decision, dict) and {"request_id", "selected"} <= set(decision)
            and set(decision) <= allowed,
            "Decision needs request_id and selected; only no_match_reason is optional")
    require(decision["request_id"] == request["request_id"],
            "Decision request_id mismatch: stale or unrelated judgment; ask the host to judge this request")
    selected = decision["selected"]
    require(isinstance(selected, list), "selected must be an array")
    known = {unit["unit_id"] for unit in request["units"]}
    require(len(selected) <= len(known), "selected has more entries than evidence units")
    reasons = {}
    for item in selected:
        require(isinstance(item, dict) and set(item) == {"unit_id", "reason"},
                "Each selected item must contain exactly unit_id and reason")
        unit_id = item["unit_id"]
        require(isinstance(unit_id, str) and unit_id in known, f"Unknown unit_id: {unit_id!r}")
        require(unit_id not in reasons, f"Duplicate selected unit_id: {unit_id}")
        reasons[unit_id] = nonempty_string(item["reason"], f"{unit_id}.reason")
    no_match = decision.get("no_match_reason", "")
    require(isinstance(no_match, str), "no_match_reason must be a string")
    if not reasons:
        no_match = nonempty_string(no_match, "no_match_reason when selected is empty")
    else:
        require(not no_match.strip(), "no_match_reason must be absent or empty when units are selected")
    return reasons, no_match


def apply(args):
    protect_inputs(args.out, [args.request, args.decision])
    request = validate_request(read_json(args.request))
    reasons, no_match = validate_decision(read_json(args.decision), request)
    result = {"protocol_version": PROTOCOL_VERSION, "status": "applied",
              "request_id": request["request_id"], "input_sha256": request["input_sha256"],
              "goal": request["goal"],
              # Copy evidence from the request, preserving source order, not model-written text.
              "selected": [{**unit, "reason": reasons[unit["unit_id"]]}
                           for unit in request["units"] if unit["unit_id"] in reasons],
              "validation": {"structure": "passed", "semantic": "not_verified",
                             "note": "结构校验成功不等于语义已验证；相关性、理由和遗漏仍需评估。"}}
    if not reasons:
        result["no_match_reason"] = no_match
    write = write_json_atomic(args.out, result, args.force)
    return {"status": "applied", "selected_count": len(reasons),
            "validation": result["validation"], "output": str(args.out), "write": write}


def parser():
    root = argparse.ArgumentParser(
        description="Prepare evidence -> host supplies JSON judgment -> validate and apply. No model/network calls.",
        epilog="Python 3.8+. A character cap is not a token/context guarantee. Judgment is supplied by the current host, not this script.")
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Split a UTF-8 article into paragraphs and emit needs_judgment")
    prep.add_argument("--input", type=Path, required=True, help="UTF-8 article; blank lines separate paragraphs")
    prep.add_argument("--goal", required=True, help="Selection criterion for the host")
    prep.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                      help=f"Explicit character budget (default {DEFAULT_MAX_CHARS}, ceiling {HARD_MAX_CHARS})")
    use = commands.add_parser("apply", help="Validate a host's decision and copy selected original evidence",
                              epilog='Decision: {"request_id":"<from request>","selected":[{"unit_id":"u0001","reason":"..."}]}. Empty selected requires a nonempty no_match_reason.')
    use.add_argument("--request", type=Path, required=True)
    use.add_argument("--decision", type=Path, required=True)
    for command, handler in ((prep, prepare), (use, apply)):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--force", action="store_true",
                             help="Replace a different existing output; identical content is always idempotent")
        command.set_defaults(handler=handler)
    return root


def main():
    args = parser().parse_args()
    try:
        result = args.handler(args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (HandoffError, OSError, UnicodeError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
