#!/usr/bin/env python3
"""Archive and validate host-authored figure artifacts without judging their science.

Image export adapts MIT-licensed autoresearch/paper/lib/production.py image_bytes.
No research controller, image provider or visual reviewer is invoked here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
VERDICTS = {"pass", "revise", "unreviewed", "blocked"}
ROUTES = {"illustration", "edit", "statistics"}
IMAGE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "TIFF": ".tif"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


class Refusal(ValueError):
    """A request that would create an inconsistent artifact."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def obj(value, label):
    require(isinstance(value, dict), label + " must be an object")
    return value


def text(value, label):
    require(isinstance(value, str) and value.strip(), label + " must be nonempty text")
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest_bytes(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise Refusal("Nonfinite JSON number: " + value)

    path = Path(path)
    require(path.is_file(), "Missing JSON file: " + str(path))
    require(path.stat().st_size <= 16 * 1024 * 1024, "JSON exceeds the 16 MiB input limit")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise Refusal("Invalid JSON in " + str(path) + ": " + str(error)) from error


def safe_output(root, value):
    text(value, "Output path")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts and relative.parts,
            "Output must be a run-relative path inside the figure root: " + value)
    base = Path(root).resolve()
    current = base
    for index, part in enumerate(relative.parts):
        current = current / part
        require(not current.is_symlink(), "Symlink paths are not artifacts: " + value)
        if current.exists():
            require(current.is_dir() or (current.is_file() and current.stat().st_nlink == 1),
                    "Hardlink or special file is not an artifact: " + value)
            if index < len(relative.parts) - 1:
                require(current.is_dir(), "Output parent is not a directory: " + value)
    require(current.resolve().is_relative_to(base), "Path escapes the figure root: " + value)
    return current


def resolve_input(root, value):
    path = Path(text(value, "Input path"))
    if path.is_absolute():
        return path
    require(".." not in path.parts, "Relative input must stay in the figure root: " + value)
    return Path(root) / path


def commit(root, writes, immutable=(), inputs=()):
    """Preflight every artifact before any write, including identical retries."""
    protected = {Path(path).resolve() for path in inputs}
    pending = []
    for name, data in writes.items():
        path = safe_output(root, name)
        require(not path.exists() or path.is_file(), "Output is not a regular file: " + name)
        if path.exists():
            previous = path.read_bytes()
            if name in immutable:
                require(previous == data, "Output differs; use a new versioned path: " + name)
            if previous == data:
                continue
        require(path.resolve() not in protected, "Output would overwrite an input: " + name)
        pending.append((path, data))
    for path, data in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The CLI serializes one local operation. Reject links again at the write boundary.
        safe_output(root, path.relative_to(root).as_posix())
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
        flags |= os.O_EXCL if not path.exists() else os.O_TRUNC
        with os.fdopen(os.open(path, flags, 0o644), "wb") as stream:
            stream.write(data)


def probe_image(path):
    try:
        from PIL import Image
    except ImportError as error:
        raise Refusal("Image handling needs installed Pillow") from error
    try:
        with Image.open(path) as image:
            image.load()
            info = {"format": image.format, "mode": image.mode, "pixels": list(image.size),
                    "bytes": Path(path).stat().st_size}
            dpi = image.info.get("dpi")
            if dpi is not None:
                require(isinstance(dpi, (tuple, list)) and len(dpi) == 2,
                        "Invalid source resolution metadata")
                info["dpi"] = [float(value) for value in dpi]
            return info
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise Refusal("Cannot decode image " + str(path) + ": " + str(error)) from error


def candidate_name(version, filename="candidate.json"):
    require(isinstance(version, str) and IDENTIFIER.fullmatch(version),
            "Version must be a short identifier such as v1")
    return "candidates/" + version + "/" + filename


def load_brief(root, complete=True):
    brief = obj(read_json(safe_output(root, "brief.json")), "brief.json")
    require(brief.get("schema_version") == SCHEMA_VERSION, "Invalid brief schema_version")
    text(brief.get("figure_id"), "figure_id")
    require(isinstance(brief.get("route"), str) and brief["route"] in ROUTES, "Invalid brief route")
    if complete:
        sources = brief.get("sources")
        require(isinstance(sources, list) and sources, "brief.json needs at least one source")
        for record in sources:
            verified_digest(root, record, "source")
        references = brief.get("references", [])
        require(isinstance(references, list), "brief references must be an array")
        for record in references:
            obj(record, "reference")
            text(record.get("source"), "reference source")
            text(record.get("lesson"), "reference lesson")
            verified_digest(root, record, "reference")
        if brief.get("baseline") is not None or brief["route"] == "edit":
            verified_digest(root, brief.get("baseline"), "baseline")
        if brief["route"] == "edit":
            changes = brief.get("permitted_changes")
            require(isinstance(changes, list) and changes, "An edit brief needs permitted_changes")
            for change in changes:
                text(change, "permitted change")
    return brief


def verified_digest(root, record, label, output=False):
    record = obj(record, label)
    path_value = text(record.get("path"), label + " path")
    expected = record.get("sha256")
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            label + " needs a SHA-256 digest")
    path = safe_output(root, path_value) if output else resolve_input(root, path_value)
    require(path.is_file(), label + " file is missing: " + path_value)
    require(file_hash(path) == expected, label + " bytes changed: " + path_value)
    return path


def protected_inputs(root, brief, candidate=None, critique=None):
    records = list(brief.get("sources", [])) + list(brief.get("references", []))
    if isinstance(brief.get("baseline"), dict):
        records.append(brief["baseline"])
    if candidate:
        records.extend(candidate.get("inputs", []))
        if candidate["tool"].get("receipt") is not None:
            records.append(candidate["tool"]["receipt"])
        records.append(candidate["image"])
    paths = [resolve_input(root, item["path"]) for item in records]
    if critique:
        paths.append(resolve_input(root, critique["inspection"]["evidence"]))
    return paths


def identity(record, brief, version, label):
    obj(record, label)
    require(record.get("schema_version") == SCHEMA_VERSION, label + " schema_version differs")
    require(record.get("figure_id") == brief["figure_id"], label + " figure_id differs")
    require(record.get("version") == version, label + " version differs from its directory")


def validate_candidate(root, version, brief):
    name = candidate_name(version)
    candidate = obj(read_json(safe_output(root, name)), "candidate")
    identity(candidate, brief, version, "candidate")
    require(candidate.get("route") == brief["route"], "candidate route differs from brief")
    require(candidate.get("brief_sha256") == file_hash(safe_output(root, "brief.json")),
            "candidate brief changed; restore the frozen brief or start a new figure root")
    image = obj(candidate.get("image"), "candidate image")
    require(isinstance(image.get("format"), str) and image["format"] in IMAGE_FORMATS,
            "Unsupported candidate format")
    require(image.get("path") == candidate_name(version, "image" + IMAGE_FORMATS[image["format"]]),
            "candidate image path differs from its version")
    path = verified_digest(root, image, "candidate image", output=True)
    info = probe_image(path)
    require(all(image.get(key) == value for key, value in info.items()),
            "candidate measured image metadata differs")
    tool = obj(candidate.get("tool"), "candidate tool")
    require(isinstance(tool.get("operation"), str) and tool["operation"] in ("generate", "edit"),
            "Invalid candidate operation")
    inputs = candidate.get("inputs")
    require(isinstance(inputs, list), "candidate inputs must be an array")
    for index, record in enumerate(inputs, 1):
        path = verified_digest(root, record, "input image", output=True)
        info = probe_image(path)
        require(record["path"] == candidate_name(version, "inputs/" + str(index) + IMAGE_FORMATS.get(info["format"], ".invalid")),
                "input image path differs from its version")
    if brief["route"] == "edit" or tool["operation"] == "edit":
        require(tool["operation"] == "edit", "An edit brief requires an edit operation")
        baseline = brief.get("baseline")
        verified_digest(root, baseline, "baseline")
        require(any(item["sha256"] == baseline["sha256"] for item in inputs),
                "The recorded baseline was not passed to the tool for this edit")
    receipt = tool.get("receipt")
    if receipt is not None:
        obj(receipt, "receipt")
        require(receipt.get("path") == candidate_name(version, "receipt.json"),
                "receipt path differs from its version")
        read_json(verified_digest(root, receipt, "receipt", output=True))
    return candidate


def validate_findings(findings):
    require(isinstance(findings, list), "Findings must contain an array")
    for item in findings:
        obj(item, "finding")
        text(item.get("issue"), "finding issue")
        require(type(item.get("blocking")) is bool, "finding blocking must be a boolean")
    return findings


def validate_critique(root, version, brief, candidate, passing=False):
    name = candidate_name(version, "critique.json")
    critique = obj(read_json(safe_output(root, name)), "critique")
    identity(critique, brief, version, "critique")
    require(critique.get("candidate_sha256") == file_hash(safe_output(root, candidate_name(version))),
            "critique candidate record changed")
    require(critique.get("brief_sha256") == file_hash(safe_output(root, "brief.json")),
            "critique brief changed")
    require(critique.get("image") == {"path": candidate["image"]["path"],
                                       "sha256": candidate["image"]["sha256"]},
            "critique image path or hash differs from candidate")
    inspection = obj(critique.get("inspection"), "inspection")
    text(inspection.get("method"), "inspection method")
    evidence = {"path": inspection.get("evidence"), "sha256": inspection.get("evidence_sha256")}
    evidence_path = verified_digest(root, evidence, "inspection evidence", output=True)
    require(evidence["path"] == candidate_name(version, "inspections/" + evidence["sha256"] + ".md"),
            "inspection evidence path differs from its version")
    text(evidence_path.read_text(encoding="utf-8"), "inspection evidence")
    verdict = critique.get("verdict")
    require(isinstance(verdict, str) and verdict in VERDICTS, "Unknown critique verdict")
    findings = validate_findings(critique.get("findings"))
    require(verdict != "pass" or not any(item["blocking"] for item in findings),
            "A pass critique cannot contain a blocking finding")
    if passing:
        require(verdict == "pass", "Only a reviewed passing candidate can be selected")
    return critique


def validate_selection(root, brief):
    selection = obj(read_json(safe_output(root, "selection.json")), "selection")
    require(selection.get("schema_version") == SCHEMA_VERSION, "Invalid selection schema_version")
    require(selection.get("figure_id") == brief["figure_id"], "selection figure_id differs")
    text(selection.get("reason"), "selection reason")
    selected = obj(selection.get("selected"), "selected")
    version = selected.get("version")
    candidate = validate_candidate(root, version, brief)
    validate_critique(root, version, brief, candidate, passing=True)
    require(selected.get("image") == {"path": candidate["image"]["path"],
                                      "sha256": candidate["image"]["sha256"]},
            "selected image path or hash differs")
    require(selected.get("critique") == candidate_name(version, "critique.json"),
            "selected critique path differs")
    require(selected.get("critique_sha256") == file_hash(safe_output(root, selected["critique"])),
            "selected critique changed; select again")
    if selection.get("export") is not None:
        validate_export(root, brief, candidate, selection["export"])
    return selection


def export_target(root, target):
    path = safe_output(root, target)
    require(Path(target).parts[0] == "export" and path.suffix.lower() in {".png", ".tif", ".tiff"},
            "Export target must be a PNG or TIFF below export/")
    return path


def validate_export(root, brief, candidate, selected_export=None):
    record = obj(read_json(safe_output(root, "export/export.json")), "export")
    identity(record, brief, candidate["version"], "export")
    target = text(record.get("target"), "export target")
    export_target(root, target)
    verified_digest(root, {"path": target, "sha256": record.get("sha256")}, "export", output=True)
    require(record.get("source_sha256") == candidate["image"]["sha256"], "export source hash differs")
    require(record.get("candidate_sha256") == file_hash(safe_output(root, candidate_name(candidate["version"]))),
            "export candidate record changed")
    options = obj(record.get("options"), "export options")
    data, measured, transforms = export_image(safe_output(root, candidate["image"]["path"]), target, options)
    require(record.get("observed") == measured and record.get("transforms") == transforms,
            "export metadata differs from its policy")
    require(digest_bytes(data) == record["sha256"], "export bytes differ from declared transform")
    if selected_export is not None:
        require(selected_export == {"path": target, "sha256": record["sha256"], "observed": measured},
                "selected export differs from export record")
    return record


def command_init(args):
    root = Path(args.root).resolve()
    text(args.figure_id, "figure_id")
    path = safe_output(root, "brief.json")
    candidates = safe_output(root, "candidates")
    require(not candidates.exists() or candidates.is_dir(), "candidates must be a directory")
    if path.exists():
        brief = load_brief(root, complete=False)
        require(brief["figure_id"] == args.figure_id, "A different figure already owns this root")
        return {"ok": True, "root": str(root), "brief": "brief.json", "created": False}
    brief = {"schema_version": SCHEMA_VERSION, "figure_id": args.figure_id, "route": args.route,
             "purpose": "", "sources": [], "references": [], "caption": "", "placement": "main",
             "reading_size": {"width_px": None, "height_px": None}, "invariants": []}
    if args.route == "edit":
        brief.update(baseline=None, permitted_changes=[])
    commit(root, {"brief.json": json_bytes(brief)}, immutable={"brief.json"})
    (root / "candidates").mkdir(exist_ok=True)
    return {"ok": True, "root": str(root), "brief": "brief.json", "created": True}


def command_ingest(args):
    root = Path(args.root).resolve()
    brief = load_brief(root)
    name = candidate_name(args.version)
    safe_output(root, name)
    source = resolve_input(root, args.source)
    require(source.is_file(), "No such image: " + str(source))
    data = source.read_bytes()
    info = probe_image(source)
    require(info["format"] in IMAGE_FORMATS, "Unsupported candidate image format")
    inputs, input_paths, writes = [], [], {}
    for index, value in enumerate(args.input_image, 1):
        path = resolve_input(root, value)
        input_info = probe_image(path)
        require(input_info["format"] in IMAGE_FORMATS, "Unsupported input image format")
        input_name = candidate_name(args.version, "inputs/" + str(index) + IMAGE_FORMATS[input_info["format"]])
        input_data = path.read_bytes()
        writes[input_name] = input_data
        input_paths.append(path)
        inputs.append({"path": input_name, "sha256": digest_bytes(input_data)})
    if brief["route"] == "edit" or args.operation == "edit":
        require(args.operation == "edit", "An edit brief requires an edit operation")
        baseline = brief.get("baseline")
        verified_digest(root, baseline, "baseline")
        require(any(item["sha256"] == baseline["sha256"] for item in inputs),
                "The recorded baseline was not passed to the tool for this edit")
    image_name = candidate_name(args.version, "image" + IMAGE_FORMATS[info["format"]])
    writes[image_name] = data
    tool = {"name": args.tool, "operation": args.operation, "model": args.model,
            "task_id": args.task_id, "receipt": None}
    if args.receipt:
        receipt = resolve_input(root, args.receipt)
        read_json(receipt)
        receipt_data = receipt.read_bytes()
        receipt_name = candidate_name(args.version, "receipt.json")
        writes[receipt_name] = receipt_data
        tool["receipt"] = {"path": receipt_name, "sha256": digest_bytes(receipt_data)}
    record = {"schema_version": SCHEMA_VERSION, "figure_id": brief["figure_id"],
              "version": args.version, "route": brief["route"],
              "brief_sha256": file_hash(safe_output(root, "brief.json")),
              "image": dict(path=image_name, sha256=digest_bytes(data), **info),
              "tool": tool, "inputs": inputs, "recorded_at": now()}
    existing = safe_output(root, name)
    if existing.exists():
        previous = obj(read_json(existing), "candidate")
        record["recorded_at"] = previous.get("recorded_at")
    writes[name] = json_bytes(record)
    extra_inputs = [source] + input_paths
    if args.receipt:
        extra_inputs.append(receipt)
    commit(root, writes, immutable=set(writes), inputs=protected_inputs(root, brief) + extra_inputs)
    return {"ok": True, "version": args.version, "image": image_name,
            "sha256": record["image"]["sha256"], "pixels": info["pixels"], "format": info["format"]}


def command_critique(args):
    root = Path(args.root).resolve()
    brief = load_brief(root)
    candidate = validate_candidate(root, args.version, brief)
    evidence = resolve_input(root, args.evidence)
    require(evidence.is_file(), "No inspection evidence file: " + str(evidence))
    text(evidence.read_text(encoding="utf-8"), "inspection evidence")
    text(args.method, "inspection method")
    findings = validate_findings(read_json(resolve_input(root, args.findings))) if args.findings else []
    require(args.verdict != "pass" or not any(item["blocking"] for item in findings),
            "A pass critique cannot contain a blocking finding")
    evidence_data = evidence.read_bytes()
    evidence_hash = digest_bytes(evidence_data)
    evidence_name = candidate_name(args.version, "inspections/" + evidence_hash + ".md")
    record = {"schema_version": SCHEMA_VERSION, "figure_id": brief["figure_id"], "version": args.version,
              "image": {"path": candidate["image"]["path"], "sha256": candidate["image"]["sha256"]},
              "brief_sha256": file_hash(safe_output(root, "brief.json")),
              "candidate_sha256": file_hash(safe_output(root, candidate_name(args.version))),
              "inspection": {"method": args.method, "evidence": evidence_name,
                             "evidence_sha256": evidence_hash},
              "verdict": args.verdict, "findings": findings, "baseline_comparison": args.note,
              "recorded_at": now()}
    name = candidate_name(args.version, "critique.json")
    history = []
    path = safe_output(root, name)
    if path.exists():
        previous = obj(read_json(path), "critique")
        history = previous.get("history", [])
        require(isinstance(history, list), "critique history must be an array")
        old = {key: value for key, value in previous.items() if key not in {"recorded_at", "history"}}
        new = {key: value for key, value in record.items() if key != "recorded_at"}
        if old == new:
            record["recorded_at"] = previous.get("recorded_at")
        else:
            history = history + [{key: value for key, value in previous.items() if key != "history"}]
    record["history"] = history
    protected = protected_inputs(root, brief, candidate) + [evidence]
    if args.findings:
        protected.append(resolve_input(root, args.findings))
    commit(root, {evidence_name: evidence_data, name: json_bytes(record)},
           immutable={evidence_name}, inputs=protected)
    return {"ok": True, "version": args.version, "verdict": args.verdict, "findings": len(findings)}


def command_select(args):
    root = Path(args.root).resolve()
    brief = load_brief(root)
    candidate = validate_candidate(root, args.version, brief)
    critique = validate_critique(root, args.version, brief, candidate, passing=True)
    text(args.reason, "selection reason")
    rejected = []
    for item in args.rejected:
        version, separator, reason = item.partition(":")
        candidate_name(version)
        require(separator and reason.strip() and version != args.version, "Rejected needs another version:reason")
        rejected.append({"version": version, "reason": reason})
    for issue in args.unresolved:
        text(issue, "unresolved issue")
    critique_name = candidate_name(args.version, "critique.json")
    record = {"schema_version": SCHEMA_VERSION, "figure_id": brief["figure_id"],
              "selected": {"version": args.version, "image": {"path": candidate["image"]["path"],
                           "sha256": candidate["image"]["sha256"]}, "critique": critique_name,
                           "critique_sha256": file_hash(safe_output(root, critique_name))},
              "reason": args.reason, "rejected": rejected, "unresolved": args.unresolved,
              "caption": args.caption, "export": None, "recorded_at": now()}
    commit(root, {"selection.json": json_bytes(record)}, inputs=protected_inputs(root, brief, candidate, critique))
    return {"ok": True, "version": args.version, "sha256": candidate["image"]["sha256"]}


def export_image(source, target, options):
    from PIL import Image
    require(set(options) <= {"mode", "dpi", "min_dpi", "max_bytes", "width_pixels", "height_pixels", "background"},
            "Unknown export options")
    for key in ("dpi", "min_dpi", "max_bytes", "width_pixels", "height_pixels"):
        if key in options:
            value = options[key]
            require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                    key + " must be positive and finite")
            if key in {"max_bytes", "width_pixels", "height_pixels"}:
                require(type(value) is int, key + " must be a positive integer")
    suffix = Path(target).suffix.lower()
    require(suffix in {".png", ".tif", ".tiff"}, "Export images as PNG or TIFF")
    mode = options.get("mode", "preserve")
    require(mode in ("preserve", "RGB", "L"), "Unknown output color mode")
    if "background" in options:
        text(options["background"], "background")
    with Image.open(source) as original:
        original.load()
        image = original.copy()
        original_dpi = original.info.get("dpi")
    transforms = []
    if mode != "preserve" and ("A" in image.getbands() or "transparency" in image.info):
        rgba = image.convert("RGBA")
        if rgba.getchannel("A").getextrema() != (255, 255):
            require("background" in options, "A nonopaque image needs an explicit opaque background")
            background = Image.new("RGBA", image.size, options["background"])
            require(background.getchannel("A").getextrema() == (255, 255), "background must be opaque")
            background.alpha_composite(rgba)
            image = background
            transforms.append("composite on the declared background")
        else:
            image = rgba
            transforms.append("remove all-opaque alpha")
    if mode != "preserve" and mode != image.mode:
        image = image.convert(mode)
        transforms.append("convert to " + mode)
    if suffix != ".png" and mode == "preserve" and "transparency" in image.info:
        image = image.convert("RGBA")
        transforms.append("expand transparency to RGBA for TIFF")
    dpi = options.get("dpi", original_dpi)
    if type(dpi) in (float, int):
        dpi = [dpi, dpi]
    require(isinstance(dpi, (list, tuple)) and len(dpi) == 2,
            "Declare a positive DPI when the source has no resolution metadata")
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in dpi),
            "Invalid source resolution metadata")
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG" if suffix == ".png" else "TIFF", dpi=tuple(dpi),
                   **({} if suffix == ".png" else {"compression": "tiff_lzw"}))
    except (OSError, ValueError, TypeError, OverflowError, struct.error) as error:
        raise Refusal("Cannot encode export with the declared DPI and mode: " + str(error)) from error
    data = buffer.getvalue()
    with Image.open(io.BytesIO(data)) as encoded:
        encoded.load()
        require(encoded.size == image.size and encoded.tobytes() == image.tobytes(),
                "Lossless export changed pixel values")
        require(encoded.convert("RGBA").tobytes() == image.convert("RGBA").tobytes(),
                "Lossless export changed colors or transparency")
        measured = {"pixels": list(encoded.size), "mode": encoded.mode,
                    "dpi": [float(value) for value in encoded.info["dpi"]], "bytes": len(data)}
    require(all(math.isfinite(value) and value > 0 for value in measured["dpi"]),
            "Export DPI cannot be represented as a positive resolution")
    for key, index in (("width_pixels", 0), ("height_pixels", 1)):
        if key in options:
            require(measured["pixels"][index] == options[key], key + " differs from the declared policy; no resizing is performed")
    if "min_dpi" in options:
        # PNG rounds pixels/metre to an integer: half a step is 0.0127 DPI.
        tolerance = 0.0127 + 1e-9 if suffix == ".png" else 1e-9
        require(min(measured["dpi"]) + tolerance >= options["min_dpi"], "Export resolution is below the declared minimum")
    if "max_bytes" in options:
        require(len(data) <= options["max_bytes"], "Export exceeds the declared byte limit")
    return data, measured, transforms


def command_export(args):
    root = Path(args.root).resolve()
    brief = load_brief(root)
    candidate = validate_candidate(root, args.version, brief)
    critique = validate_critique(root, args.version, brief, candidate, passing=True)
    options = {key: getattr(args, key) for key in ("mode", "dpi", "min_dpi", "background", "max_bytes",
               "width_pixels", "height_pixels") if getattr(args, key) is not None}
    export_target(root, args.target)
    safe_output(root, "export/export.json")
    selection_path = safe_output(root, "selection.json")
    selection = validate_selection(root, brief) if selection_path.exists() else None
    if selection:
        require(selection["selected"]["version"] == args.version, "Export version differs from selected version")
    source = safe_output(root, candidate["image"]["path"])
    data, measured, transforms = export_image(source, args.target, options)
    digest = digest_bytes(data)
    record = {"schema_version": SCHEMA_VERSION, "figure_id": brief["figure_id"], "version": args.version,
              "target": args.target, "sha256": digest, "source_sha256": candidate["image"]["sha256"],
              "candidate_sha256": file_hash(safe_output(root, candidate_name(args.version))),
              "observed": measured, "transforms": transforms, "options": options, "exported_at": now()}
    writes = {args.target: data, "export/export.json": json_bytes(record)}
    if selection:
        selection["export"] = {"path": args.target, "sha256": digest, "observed": measured}
        writes["selection.json"] = json_bytes(selection)
    commit(root, writes, immutable={args.target}, inputs=protected_inputs(root, brief, candidate, critique))
    return {"ok": True, "target": args.target, "sha256": digest, "observed": measured, "transforms": transforms}


def command_validate(args):
    root = Path(args.root).resolve()
    errors, candidates = [], []
    selected = None
    try:
        brief = load_brief(root)
        candidates_root = safe_output(root, "candidates")
        if candidates_root.is_dir():
            for directory in sorted(candidates_root.iterdir()):
                try:
                    safe_output(root, directory.relative_to(root).as_posix())
                    require(directory.is_dir(), "Unexpected file in candidates: " + directory.name)
                    candidate = validate_candidate(root, directory.name, brief)
                    candidates.append(candidate)
                    critique = safe_output(root, candidate_name(directory.name, "critique.json"))
                    if critique.exists():
                        validate_critique(root, directory.name, brief, candidate)
                except (Refusal, OSError, ValueError) as error:
                    errors.append("candidate " + directory.name + ": " + str(error))
        selection_path = safe_output(root, "selection.json")
        if selection_path.exists():
            selection = validate_selection(root, brief)
            selected = selection["selected"]["version"]
        elif args.require_selection:
            errors.append("no selection.json; select a reviewed candidate before delivery")
        export_path = safe_output(root, "export/export.json")
        if export_path.exists():
            record = obj(read_json(export_path), "export")
            candidate = validate_candidate(root, record.get("version"), brief)
            validate_critique(root, candidate["version"], brief, candidate, passing=True)
            validate_export(root, brief, candidate)
            if selected is not None:
                require(record["version"] == selected, "Export version differs from selected version; export the selected candidate")
    except (Refusal, OSError, ValueError) as error:
        errors.append(str(error))
    return {"ok": not errors, "errors": errors, "candidates": len(candidates), "selected": selected}


def build_parser():
    parser = argparse.ArgumentParser(prog="paperbanana", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create a brief template")
    init.add_argument("--root", required=True)
    init.add_argument("--figure-id", required=True)
    init.add_argument("--route", default="illustration", choices=sorted(ROUTES))
    ingest = sub.add_parser("ingest", help="Archive a candidate without replacing its identity")
    ingest.add_argument("--root", required=True)
    ingest.add_argument("--version", required=True)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--tool", default="")
    ingest.add_argument("--operation", default="generate", choices=["generate", "edit"])
    ingest.add_argument("--model", default="")
    ingest.add_argument("--task-id", default="")
    ingest.add_argument("--receipt", default="")
    ingest.add_argument("--input-image", action="append", default=[])
    critique = sub.add_parser("critique", help="Record host-authored visual inspection")
    critique.add_argument("--root", required=True)
    critique.add_argument("--version", required=True)
    critique.add_argument("--verdict", required=True, choices=sorted(VERDICTS))
    critique.add_argument("--evidence", required=True)
    critique.add_argument("--method", required=True)
    critique.add_argument("--findings", default="")
    critique.add_argument("--note", default="")
    select = sub.add_parser("select", help="Select a passing candidate")
    select.add_argument("--root", required=True)
    select.add_argument("--version", required=True)
    select.add_argument("--reason", required=True)
    select.add_argument("--caption", default="")
    select.add_argument("--rejected", action="append", default=[])
    select.add_argument("--unresolved", action="append", default=[])
    export = sub.add_parser("export", help="Export PNG or TIFF without resizing")
    export.add_argument("--root", required=True)
    export.add_argument("--version", required=True)
    export.add_argument("--target", default="export/figure.png")
    export.add_argument("--mode", default=None, choices=["preserve", "RGB", "L"])
    export.add_argument("--dpi", type=float, default=None)
    export.add_argument("--min-dpi", type=float, default=None)
    export.add_argument("--background", default=None)
    export.add_argument("--max-bytes", type=int, default=None)
    export.add_argument("--width-pixels", type=int, default=None)
    export.add_argument("--height-pixels", type=int, default=None)
    validate = sub.add_parser("validate", help="Check artifact consistency")
    validate.add_argument("--root", required=True)
    validate.add_argument("--require-selection", action="store_true")
    return parser


COMMANDS = {"init": command_init, "ingest": command_ingest, "critique": command_critique,
            "select": command_select, "export": command_export, "validate": command_validate}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = COMMANDS[args.command](args)
    except (Refusal, OSError, ValueError, OverflowError) as error:
        print("paperbanana: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
