#!/usr/bin/env python3
"""Check local evidence consistency, not source truth or semantic entailment."""

import argparse
import json
import math
from pathlib import Path


METRIC_FIELDS = (
    "name", "experiment", "conditions", "baseline_name", "unit", "direction",
    "value", "baseline", "n", "baseline_n",
)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def pointer(document, location):
    """Resolve an RFC 6901 JSON Pointer, rejecting negative array indices."""
    if not isinstance(location, str) or (location and not location.startswith("/")):
        raise ValueError("pointer must be empty or start with /")
    for part in location.split("/")[1:] if location else []:
        # Only ~0 and ~1 are valid escapes.
        rest = part.replace("~1", "").replace("~0", "")
        if "~" in rest:
            raise ValueError("invalid pointer escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(document, list):
            if not part.isascii() or not part.isdigit() or (len(part) > 1 and part[0] == "0"):
                raise ValueError("invalid array index")
            document = document[int(part)]
        elif isinstance(document, dict):
            document = document[part]
        else:
            raise ValueError("pointer traverses a scalar")
    return document


def validate(bundle, root):
    errors, gaps = [], []

    def require(condition, message):
        if not condition:
            errors.append(message)
        return condition

    def index(name):
        rows = bundle.get(name)
        if not require(isinstance(rows, list), f"{name}: expected list"):
            return {}
        out = {}
        for row in rows:
            if not require(isinstance(row, dict) and nonempty(row.get("id")), f"{name}: missing object/id"):
                continue
            if require(row["id"] not in out, f"{name}: duplicate id {row['id']}"):
                out[row["id"]] = row
        return out

    if not isinstance(bundle, dict):
        return {"status": "fail", "errors": ["bundle must be an object"], "gaps": [], "semantic_review": "not_checked"}
    require(bundle.get("version") == 1 and type(bundle.get("version")) is int, "version must be 1")
    task = bundle.get("task", {})
    if require(isinstance(task, dict), "task: expected object"):
        for field in ("question", "reader", "scope"):
            require(nonempty(task.get(field)), f"task.{field}: required")
    sources, evidence, claims = index("sources"), index("evidence"), index("claims")
    require(bool(claims), "claims: at least one claim required")
    loaded = {}
    for sid, source in sources.items():
        require(source.get("kind") in ("primary", "summary", "derived"), f"{sid}: invalid source kind")
        status = source.get("status")
        if not require(status in ("available", "missing", "unread"), f"{sid}: invalid source status"):
            continue
        require(nonempty(source.get("path")) or nonempty(source.get("url")), f"{sid}: path or URL required")
        if status != "available":
            require(nonempty(source.get("reason")), f"{sid}: reason required")
            gaps.append(f"{sid}: {status}: {source.get('reason', '')}")
            continue
        if not require(nonempty(source.get("path")), f"{sid}: available source requires local path"):
            continue
        try:
            loaded[sid] = (root / source["path"]).read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{sid}: cannot read source: {exc}")
    checked_primary = set()
    for eid, item in evidence.items():
        before = len(errors)
        sid = item.get("source_id")
        if not require(isinstance(sid, str) and sid in sources, f"{eid}: unknown source_id"):
            continue
        if not require(sid in loaded, f"{eid}: evidence source not available"):
            continue
        if item.get("kind") == "text":
            lines = item.get("lines")
            if not require(isinstance(lines, list) and len(lines) == 2 and all(type(x) is int for x in lines), f"{eid}: lines must be [start, end]"):
                continue
            text = loaded[sid].splitlines()
            if require(1 <= lines[0] <= lines[1] <= len(text), f"{eid}: line range out of bounds"):
                excerpt = item.get("excerpt")
                require(nonempty(excerpt) and excerpt == "\n".join(text[lines[0] - 1:lines[1]]), f"{eid}: excerpt differs from source lines")
        elif item.get("kind") == "metric":
            metric, missing = item.get("metric"), item.get("missing", {})
            if not require(isinstance(metric, dict) and isinstance(missing, dict), f"{eid}: metric/missing must be objects"):
                continue
            try:
                raw = pointer(json.loads(loaded[sid]), item.get("pointer"))
                if not isinstance(raw, dict):
                    raise ValueError("metric pointer must select an object")
            except (ValueError, KeyError, IndexError, TypeError, RecursionError) as exc:
                errors.append(f"{eid}: cannot resolve JSON source: {exc}")
                continue
            for field in METRIC_FIELDS:
                require(field in metric, f"{eid}.{field}: field required (null for explicit gap)")
                value = metric.get(field)
                if value is None:
                    require(nonempty(missing.get(field)), f"{eid}.{field}: missing reason required")
                    require(raw.get(field) is None, f"{eid}.{field}: source has value omitted by bundle")
                    gaps.append(f"{eid}.{field}: {missing.get(field, 'missing')}")
                    continue
                require(field not in missing, f"{eid}.{field}: value and missing reason conflict")
                if field in ("value", "baseline"):
                    require(number(value), f"{eid}.{field}: finite number required")
                elif field in ("n", "baseline_n"):
                    require(type(value) is int and value > 0, f"{eid}.{field}: positive integer required")
                elif field == "direction":
                    require(value in ("higher", "lower", "neutral"), f"{eid}.direction: higher/lower/neutral required")
                else:
                    require(nonempty(value), f"{eid}.{field}: nonempty string required")
                actual = raw.get(field)
                # bool must never compare equal to a numeric observation.
                same_type = type(value) is type(actual) or (number(value) and number(actual))
                require(same_type and value == actual, f"{eid}.{field}: differs from source")
            for field in missing:
                require(field in METRIC_FIELDS, f"{eid}: unknown missing field {field}")
            value, baseline = metric.get("value"), metric.get("baseline")
            computed = item.get("computed", {})
            if not require(isinstance(computed, dict), f"{eid}: computed must be object"):
                continue
            for key, reported in computed.items():
                if not require(key in ("delta", "relative_change_pct", "trend"), f"{eid}: unknown computation {key}"):
                    continue
                if not require(number(value) and number(baseline), f"{eid}.{key}: numeric value and baseline required"):
                    continue
                if key == "trend":
                    direction = metric.get("direction")
                    expected = "unchanged" if value == baseline else "undetermined" if direction == "neutral" else "improved" if (value > baseline) == (direction == "higher") else "worsened"
                    require(direction in ("higher", "lower", "neutral"), f"{eid}: trend requires known direction")
                    require(reported == expected, f"{eid}.trend: expected {expected}")
                else:
                    if key == "relative_change_pct" and not require(baseline > 0, f"{eid}: relative percent requires positive baseline; use delta otherwise"):
                        continue
                    # Divide first so a representable percentage survives an overflowing difference.
                    expected = value - baseline if key == "delta" else (value / baseline - 1) * 100
                    if not require(number(expected), f"{eid}.{key}: result exceeds finite numeric range"):
                        continue
                    require(number(reported) and math.isclose(reported, expected, rel_tol=1e-9, abs_tol=1e-9), f"{eid}.{key}: expected {expected}")
        else:
            errors.append(f"{eid}: kind must be text or metric")
        if len(errors) == before and sources[sid].get("kind") == "primary":
            checked_primary.add(eid)
        if sources[sid].get("kind") != "primary":
            gaps.append(f"{eid}: {sources[sid].get('kind')} source requires upstream verification")
    for cid, claim in claims.items():
        for field in ("text", "scope"):
            require(nonempty(claim.get(field)), f"{cid}.{field}: required")
        kind = claim.get("kind")
        require(kind in ("observation", "inference", "missing", "conflicting"), f"{cid}: invalid claim kind")
        refs = claim.get("evidence_ids")
        valid_refs = isinstance(refs, list) and all(isinstance(x, str) for x in refs)
        require(valid_refs, f"{cid}: evidence_ids must be a string list")
        if valid_refs:
            for ref in refs:
                require(ref in evidence, f"{cid}: unknown evidence {ref}")
            if kind in ("observation", "inference", "conflicting"):
                require(bool(refs), f"{cid}: evidence required")
            if kind == "observation":
                require(bool(set(refs) & checked_primary), f"{cid}: observation requires checked primary evidence")
            if kind == "conflicting":
                require(len(set(refs)) >= 2, f"{cid}: conflicting claim requires at least two evidence entries")
        limits = claim.get("limitations")
        require(isinstance(limits, list) and all(nonempty(x) for x in limits), f"{cid}: limitations must be string list")
        if kind in ("inference", "missing", "conflicting"):
            require(nonempty(claim.get("reason")), f"{cid}: reason required")
        if kind in ("missing", "conflicting"):
            gaps.append(f"{cid}: {kind}: {claim.get('reason', '')}")
    return {"status": "fail" if errors else "pass_with_gaps" if gaps else "pass",
            "errors": errors, "gaps": gaps, "semantic_review": "not_checked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        report = validate(bundle, args.bundle.resolve().parent)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        report = {"status": "fail", "errors": [str(exc)], "gaps": [], "semantic_review": "not_checked"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
