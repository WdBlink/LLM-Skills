#!/usr/bin/env python3
"""Read-only verification of declared artifact hashes, not benchmark validity."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(registration, root=None):
    report = {
        "status": "artifact_verification_failed",
        "scope": "artifact_integrity_only",
        "benchmark_validation": "not_assessed",
        "verified_files": [],
        "errors": [],
    }
    try:
        registration = Path(registration).resolve(strict=True)
        data = json.loads(registration.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object)
        if not isinstance(data, dict):
            raise ValueError("registration must be a JSON object")
        if not isinstance(data.get("benchmark_id"), str) or not data["benchmark_id"].strip():
            raise ValueError("benchmark_id must be a nonempty string")
        versions = [data[key] for key in ("version", "design_version") if key in data]
        if not versions or any(not isinstance(v, str) or not v.strip() for v in versions):
            raise ValueError("version or design_version must be a nonempty string")
        if len(set(versions)) != 1:
            raise ValueError("version and design_version disagree")
        hashes = data.get("sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("sha256 must be a nonempty object")
        root = Path(root).resolve(strict=True) if root is not None else registration.parent
        if not root.is_dir():
            raise ValueError("artifact root must be a directory")
        report.update(benchmark_id=data["benchmark_id"], version=versions[0],
                      declared_status=data.get("status"), declared_files=len(hashes))
        for relative, expected in hashes.items():
            try:
                parts = PurePosixPath(relative)
                if (not relative or "\\" in relative or parts.is_absolute()
                        or PureWindowsPath(relative).is_absolute() or ".." in parts.parts):
                    raise ValueError("path must be relative and cannot traverse parents")
                if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                    raise ValueError("expected digest must be 64 lowercase hexadecimal characters")
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError("resolved path escapes artifact root")
                if not path.is_file():
                    raise ValueError("artifact must be a regular file")
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
                report["verified_files"].append(relative)
            except (OSError, ValueError, RuntimeError) as exc:
                report["errors"].append({"path": relative, "error": str(exc)})
    except (OSError, ValueError, RuntimeError) as exc:
        report["errors"].append({"error": str(exc)})
    if not report["errors"]:
        report["status"] = "artifacts_verified"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--root", type=Path, help="artifact directory; default: registration parent")
    args = parser.parse_args()
    report = verify(args.registration, args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "artifacts_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
