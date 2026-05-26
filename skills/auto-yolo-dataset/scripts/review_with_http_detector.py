#!/usr/bin/env python3
"""Review and correct a detection manifest through an explicit HTTP reviewer."""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional


DEFAULT_REVIEWER_URL = "http://127.0.0.1:11500/review"
SUPPORTED_BBOX_FORMATS = {"xyxy_pixel", "xyxy_1000", "xyxy_normalized"}


class ReviewError(RuntimeError):
    pass


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ReviewError(f"reviewer request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ReviewError(f"reviewer request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"reviewer returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ReviewError("reviewer response must be a JSON object")
    return parsed


def health_url_for(reviewer_url: str) -> str:
    parsed = urllib.parse.urlsplit(reviewer_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/review"):
        path = path[: -len("/review")] or "/"
    health_path = (path.rstrip("/") + "/health") if path != "/" else "/health"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def get_json(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ReviewError(f"reviewer health check failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ReviewError(f"reviewer health check failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"reviewer health check returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ReviewError("reviewer health check response must be a JSON object")
    return parsed


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewError("manifest root must be a JSON object")
    return data


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


def normalize_box(box: Any, width: int, height: int, bbox_format: str) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ReviewError(f"bbox must be a list of four numbers: {box!r}")
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ReviewError(f"bbox contains non-numeric values: {box!r}") from exc
    if bbox_format == "xyxy_1000":
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    elif bbox_format == "xyxy_normalized":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif bbox_format != "xyxy_pixel":
        raise ReviewError(f"unsupported reviewer bbox_format: {bbox_format!r}")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ReviewError(f"reviewer returned out-of-bounds bbox {box!r} for image size {(width, height)}")
    return [x1, y1, x2, y2]


def safe_image_path(root: Path, file_value: str) -> Path:
    rel = Path(file_value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ReviewError(f"image file must be relative under image root: {file_value}")
    path = (root / rel).resolve()
    if not str(path).startswith(str(root.resolve())):
        raise ReviewError(f"image file escapes image root: {file_value}")
    return path


def render_overlay_png(image_path: Path, detections: list[dict[str, Any]]) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ReviewError("Pillow is required for review overlays; install pillow in the reviewer environment") from exc

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, det in enumerate(detections, start=1):
        box = det["bbox"]
        label = f"{index}: {det['class']}"
        if det.get("confidence") is not None:
            label += f" {float(det['confidence']):.2f}"
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 115, 22), width=max(3, int(min(image.size) / 250)))
        text_box = draw.textbbox((x1 + 3, max(0, y1 - 16)), label, font=font)
        draw.rectangle(text_box, fill=(255, 115, 22))
        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=(255, 255, 255), font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def review_image(
    reviewer_url: str,
    image_path: Path,
    file_value: str,
    width: int,
    height: int,
    classes: list[str],
    detections: list[dict[str, Any]],
    timeout: int,
    max_new_tokens: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overlay_png = render_overlay_png(image_path, detections)
    payload: dict[str, Any] = {
        "file": file_value,
        "classes": classes,
        "mode": "annotation-review",
        "width": width,
        "height": height,
        "mime_type": mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
        "overlay_mime_type": "image/png",
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "overlay_image_base64": base64.b64encode(overlay_png).decode("ascii"),
        "bbox_format": "xyxy_pixel",
        "accepted_bbox_formats": sorted(SUPPORTED_BBOX_FORMATS),
        "detections": detections,
    }
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    response = post_json(reviewer_url, payload, timeout)
    if response.get("ok") is False:
        raise ReviewError(f"reviewer failed for {file_value}: {response.get('error')}")
    response_format = response.get("bbox_format") or "xyxy_pixel"
    if not isinstance(response_format, str):
        raise ReviewError(f"reviewer bbox_format must be a string for {file_value}")
    reviewed = response.get("detections", detections)
    if not isinstance(reviewed, list):
        raise ReviewError(f"reviewer detections must be a list for {file_value}")
    out: list[dict[str, Any]] = []
    for item in reviewed:
        if not isinstance(item, dict):
            continue
        class_name = item.get("class")
        if class_name not in classes:
            raise ReviewError(f"reviewer returned unknown class {class_name!r} for {file_value}")
        confidence = item.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        det = {
            "class": class_name,
            "bbox": normalize_box(item.get("bbox"), width, height, response_format),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "source": str(item.get("source") or "http-reviewer"),
        }
        notes = item.get("notes")
        if isinstance(notes, str) and notes:
            det["notes"] = notes
        out.append(det)
    review_info = response.get("review", {})
    if not isinstance(review_info, dict):
        review_info = {}
    return out, review_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input manifest JSON.")
    parser.add_argument("--output-manifest", required=True, help="Corrected output manifest JSON.")
    parser.add_argument("--image-root", help="Source image root. Defaults to manifest dataset.image_root or manifest dir.")
    parser.add_argument(
        "--reviewer-url",
        default=os.environ.get("AUTO_YOLO_REVIEWER_URL") or DEFAULT_REVIEWER_URL,
        help="HTTP reviewer /review endpoint. Defaults to AUTO_YOLO_REVIEWER_URL or localhost:11500.",
    )
    parser.add_argument("--report", help="Optional review report JSON path.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--skip-health-check", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    data = load_json(manifest_path)
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(item, str) for item in classes):
        raise SystemExit("manifest classes must be a non-empty string list")
    images = data.get("images")
    if not isinstance(images, list):
        raise SystemExit("manifest images must be a list")
    root = Path(args.image_root or data.get("dataset", {}).get("image_root") or manifest_path.parent).expanduser().resolve()

    health: dict[str, Any] = {}
    if not args.skip_health_check:
        try:
            health = get_json(health_url_for(args.reviewer_url), args.timeout)
        except ReviewError as exc:
            raise SystemExit(str(exc)) from exc
        if health.get("ok") is False:
            raise SystemExit(f"reviewer health check failed: {health}")

    reviewed_images = []
    report_images = []
    total_before = 0
    total_after = 0
    for image in images:
        if not isinstance(image, dict):
            raise SystemExit("manifest images entries must be objects")
        file_value = image.get("file")
        if not isinstance(file_value, str) or not file_value:
            raise SystemExit("manifest image file must be a non-empty string")
        image_path = safe_image_path(root, file_value)
        if not image_path.exists():
            raise SystemExit(f"image not found: {file_value}")
        size = image_size(image_path)
        width = image.get("width")
        height = image.get("height")
        if width is None or height is None:
            if size is None:
                raise SystemExit(f"width/height missing and image size unsupported: {file_value}")
            width, height = size
        if not isinstance(width, int) or not isinstance(height, int):
            raise SystemExit(f"width/height must be integers for {file_value}")
        detections = image.get("detections", [])
        if not isinstance(detections, list):
            raise SystemExit(f"detections must be a list for {file_value}")
        total_before += len(detections)
        try:
            corrected, review_info = review_image(
                args.reviewer_url,
                image_path,
                file_value,
                width,
                height,
                classes,
                detections,
                args.timeout,
                args.max_new_tokens,
            )
        except ReviewError as exc:
            raise SystemExit(str(exc)) from exc
        total_after += len(corrected)
        reviewed_images.append({**image, "width": width, "height": height, "detections": corrected})
        report_images.append(
            {
                "file": file_value,
                "before_count": len(detections),
                "after_count": len(corrected),
                "review": review_info,
            }
        )

    output = {
        **data,
        "dataset": {
            **data.get("dataset", {}),
            "reviewer_url": args.reviewer_url,
            "reviewer_backend": health.get("backend"),
            "reviewer_capabilities": health.get("capabilities"),
        },
        "images": reviewed_images,
    }
    output_path = Path(args.output_manifest).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "ok": True,
        "reviewer_url": args.reviewer_url,
        "reviewer_backend": health.get("backend"),
        "image_count": len(reviewed_images),
        "annotation_count_before": total_before,
        "annotation_count_after": total_after,
        "images": report_images,
    }
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
