#!/usr/bin/env python3
"""Generate an auto-yolo-dataset manifest through an explicit HTTP detector."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_DETECTOR_URL = "http://127.0.0.1:11500/detect"
SUPPORTED_BBOX_FORMATS = {"xyxy_pixel", "xyxy_1000", "xyxy_normalized"}


class DetectorError(RuntimeError):
    pass


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DetectorError(f"detector request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DetectorError(f"detector request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DetectorError(f"detector returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DetectorError("detector response must be a JSON object")
    return parsed


def get_json(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise DetectorError(f"detector health check failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DetectorError(f"detector health check failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DetectorError(f"detector health check returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DetectorError("detector health check response must be a JSON object")
    return parsed


def health_url_for(detector_url: str) -> str:
    parsed = urllib.parse.urlsplit(detector_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/detect"):
        path = path[: -len("/detect")] or "/"
    health_path = (path.rstrip("/") + "/health") if path != "/" else "/health"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def image_size(path: Path) -> Optional[tuple[int, int]]:
    with path.open("rb") as handle:
        head = handle.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
            return struct.unpack(">II", head[16:24])
        if head[:2] == b"\xff\xd8":
            handle.seek(2)
            while True:
                marker_start = handle.read(1)
                if not marker_start:
                    return None
                if marker_start != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc2"}:
                    block = handle.read(7)
                    if len(block) < 7:
                        return None
                    height, width = struct.unpack(">HH", block[3:7])
                    return width, height
                size_bytes = handle.read(2)
                if len(size_bytes) < 2:
                    return None
                size = struct.unpack(">H", size_bytes)[0]
                if size < 2:
                    return None
                handle.seek(size - 2, 1)
    return None


def split_for_index(index: int, total: int) -> str:
    if total < 3:
        return "train"
    if index == total - 2:
        return "val"
    if index == total - 1:
        return "test"
    return "train"


def parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise SystemExit("--classes must contain at least one class")
    if len(set(classes)) != len(classes):
        raise SystemExit("--classes must not contain duplicates")
    return classes


def build_request(
    image_path: Path,
    relative_file: str,
    classes: list[str],
    image_transfer: str,
    confidence_threshold: Optional[float],
    max_new_tokens: Optional[int],
    instance_policy: str,
    min_object_area_ratio: float,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    payload: dict[str, Any] = {
        "file": relative_file,
        "classes": classes,
        "mode": "open-vocabulary",
        "mime_type": mime_type,
        "bbox_format": "xyxy_pixel",
        "accepted_bbox_formats": sorted(SUPPORTED_BBOX_FORMATS),
        "instance_policy": instance_policy,
        "min_object_area_ratio": min_object_area_ratio,
    }
    if image_transfer == "base64":
        payload["image_base64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    elif image_transfer == "path":
        payload["image_path"] = str(image_path)
    else:
        raise ValueError(f"unsupported image transfer mode: {image_transfer}")
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    return payload


def normalize_detection(
    det: dict[str, Any],
    classes: list[str],
    width: int,
    height: int,
    confidence_threshold: Optional[float],
    bbox_format: str,
) -> Optional[dict[str, Any]]:
    class_name = det.get("class")
    if class_name not in classes:
        raise DetectorError(f"detector returned unknown class {class_name!r}")
    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise DetectorError(f"detector returned invalid bbox for class {class_name!r}: {bbox!r}")
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise DetectorError(f"detector returned non-numeric bbox for class {class_name!r}: {bbox!r}") from exc
    if bbox_format == "xyxy_1000":
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    elif bbox_format == "xyxy_normalized":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif bbox_format != "xyxy_pixel":
        raise DetectorError(f"unsupported detector bbox_format: {bbox_format!r}")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise DetectorError(
            f"detector returned out-of-bounds bbox for class {class_name!r}: {bbox!r}; "
            f"expected 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}"
        )
    confidence = det.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        raise DetectorError(f"detector returned non-numeric confidence for class {class_name!r}")
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence_threshold is not None and confidence < confidence_threshold:
        return None
    normalized = {
        "class": class_name,
        "bbox": [x1, y1, x2, y2],
        "confidence": confidence,
        "source": str(det.get("source") or "http-detector"),
    }
    notes = det.get("notes")
    if isinstance(notes, str) and notes:
        normalized["notes"] = notes
    return normalized


def detect_image(
    detector_url: str,
    image_path: Path,
    relative_file: str,
    classes: list[str],
    timeout: int,
    image_transfer: str,
    confidence_threshold: Optional[float],
    max_new_tokens: Optional[int],
    instance_policy: str,
    min_object_area_ratio: float,
) -> dict[str, Any]:
    payload = build_request(
        image_path,
        relative_file,
        classes,
        image_transfer,
        confidence_threshold,
        max_new_tokens,
        instance_policy,
        min_object_area_ratio,
    )
    response = post_json(detector_url, payload, timeout)
    if response.get("ok") is False:
        raise DetectorError(f"detector failed for {relative_file}: {response.get('error')}")
    local_size = image_size(image_path)
    width = response.get("width")
    height = response.get("height")
    if width is None or height is None:
        if local_size is None:
            raise DetectorError(f"detector omitted width/height and local image size is unsupported: {relative_file}")
        width, height = local_size
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise DetectorError(f"detector returned invalid image size for {relative_file}: {(width, height)!r}")
    if local_size is not None and local_size != (width, height):
        raise DetectorError(f"detector size mismatch for {relative_file}: detector {(width, height)}, local {local_size}")
    detections = response.get("detections", [])
    if not isinstance(detections, list):
        raise DetectorError(f"detector response detections must be a list for {relative_file}")
    bbox_format = response.get("bbox_format") or "xyxy_pixel"
    if not isinstance(bbox_format, str):
        raise DetectorError(f"detector response bbox_format must be a string for {relative_file}")
    normalized = [
        item
        for item in (
            normalize_detection(det, classes, width, height, confidence_threshold, bbox_format)
            for det in detections
            if isinstance(det, dict)
        )
        if item is not None
    ]
    return {"width": width, "height": height, "detections": normalized, "instances": response.get("instances", [])}


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def quality_flags(
    file_value: str,
    width: int,
    height: int,
    detections: list[dict[str, Any]],
    instances: Any,
    min_object_area_ratio: float,
) -> dict[str, Any]:
    flags = []
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(width * height, 1)
        if min_object_area_ratio > 0 and area_ratio < min_object_area_ratio:
            flags.append({"type": "small_box", "detection_index": index, "area_ratio": area_ratio})
        if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
            flags.append({"type": "edge_touching_box", "detection_index": index})
    for left_index, left in enumerate(detections):
        for right_index in range(left_index + 1, len(detections)):
            right = detections[right_index]
            if left["class"] == right["class"] and box_iou(left["bbox"], right["bbox"]) >= 0.85:
                flags.append(
                    {
                        "type": "possible_duplicate",
                        "detection_indices": [left_index + 1, right_index + 1],
                        "iou": box_iou(left["bbox"], right["bbox"]),
                    }
                )
    if isinstance(instances, list) and instances and abs(len(instances) - len(detections)) >= 2:
        flags.append({"type": "instance_detection_count_mismatch", "instance_count": len(instances), "detection_count": len(detections)})
    return {"file": file_value, "flag_count": len(flags), "flags": flags}


def main(
    argv: Optional[list[str]] = None,
    *,
    default_detector_url: Optional[str] = None,
    default_dataset_source: str = "http-detector",
    default_image_transfer: str = "base64",
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--classes", required=True, help="Comma-separated class names, e.g. truck,car.")
    parser.add_argument("--manifest", required=True, help="Output detection manifest JSON path.")
    parser.add_argument(
        "--detector-url",
        default=default_detector_url or os.environ.get("AUTO_YOLO_DETECTOR_URL") or DEFAULT_DETECTOR_URL,
        help="HTTP detector /detect endpoint. Defaults to AUTO_YOLO_DETECTOR_URL or localhost:11500.",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--instance-policy", choices=("all-visible", "primary-subject"), default="all-visible")
    parser.add_argument("--min-object-area-ratio", type=float, default=0.0)
    parser.add_argument("--quality-report", help="Optional JSON report of suspicious boxes; never drops boxes by itself.")
    parser.add_argument("--max-detections", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--image-transfer", choices=("base64", "path"), default=default_image_transfer)
    parser.add_argument("--skip-health-check", action="store_true")
    args = parser.parse_args(argv)

    if args.confidence_threshold is not None and not (0.0 <= args.confidence_threshold <= 1.0):
        raise SystemExit("--confidence-threshold must be in [0,1]")
    if args.min_object_area_ratio < 0:
        raise SystemExit("--min-object-area-ratio must be >= 0")

    image_root = Path(args.image_dir).expanduser().resolve()
    classes = parse_classes(args.classes)
    image_paths = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"no images found under {image_root}")

    health: dict[str, Any] = {}
    if not args.skip_health_check:
        try:
            health = get_json(health_url_for(args.detector_url), args.timeout)
        except DetectorError as exc:
            raise SystemExit(str(exc)) from exc
        if health.get("ok") is False:
            raise SystemExit(f"detector health check failed: {health}")

    images = []
    quality_images = []
    total_detections = 0
    for index, image_path in enumerate(image_paths):
        relative_file = image_path.relative_to(image_root).as_posix()
        try:
            result = detect_image(
                args.detector_url,
                image_path,
                relative_file,
                classes,
                args.timeout,
                args.image_transfer,
                args.confidence_threshold,
                args.max_new_tokens,
                args.instance_policy,
                args.min_object_area_ratio,
            )
        except DetectorError as exc:
            raise SystemExit(str(exc)) from exc
        detections = result["detections"]
        total_detections += len(detections)
        quality_images.append(
            quality_flags(
                relative_file,
                result["width"],
                result["height"],
                detections,
                result.get("instances"),
                args.min_object_area_ratio,
            )
        )
        image_record = {
            "file": relative_file,
            "width": result["width"],
            "height": result["height"],
            "split": split_for_index(index, len(image_paths)),
            "detections": detections,
        }
        if isinstance(result.get("instances"), list):
            image_record["instances"] = result["instances"]
        images.append(image_record)

    manifest = {
        "dataset": {
            "name": image_root.name,
            "source": default_dataset_source,
            "detector_url": args.detector_url,
            "detector_backend": health.get("backend"),
            "detector_bbox_formats": health.get("bbox_formats"),
            "instance_policy": args.instance_policy,
            "image_root": str(image_root),
        },
        "classes": classes,
        "images": images,
    }
    output = Path(args.manifest).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.quality_report:
        report = {
            "ok": True,
            "image_count": len(images),
            "annotation_count": total_detections,
            "flag_count": sum(item["flag_count"] for item in quality_images),
            "images": quality_images,
        }
        report_path = Path(args.quality_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} with {len(images)} images and {total_detections} detections")


if __name__ == "__main__":
    main()
