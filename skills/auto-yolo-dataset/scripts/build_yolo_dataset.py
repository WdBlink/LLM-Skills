#!/usr/bin/env python3
"""Build YOLO, COCO, VOC, and Label Studio assets from a detection manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


SPLITS = ("train", "val", "test")


class ManifestError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
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


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_image_path(root: Path, file_value: str) -> Path:
    rel = Path(file_value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ManifestError(f"image file must be relative under image root: {file_value}")
    source_path = (root / rel).resolve()
    if not is_relative_to(source_path, root):
        raise ManifestError(f"image file escapes image root: {file_value}")
    return source_path


def clean_output_safely(output: Path, protected_paths: list[Path]) -> None:
    output = output.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    if output in {Path(output.anchor).resolve(), cwd, home}:
        raise ManifestError(f"refusing to delete unsafe output directory: {output}")
    for protected in protected_paths:
        protected = protected.resolve()
        if protected == output or is_relative_to(protected, output):
            raise ManifestError(f"output directory would delete source image: {protected}")
    if output.exists() and not output.is_dir():
        raise ManifestError(f"output path exists and is not a directory: {output}")
    if output.exists():
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "annotations" / "voc" / split).mkdir(parents=True, exist_ok=True)


def rel_to(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def rel_between(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, start=base)).as_posix()


def normalized_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def as_num_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ManifestError(f"{field} must be a list of four numbers")
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ManifestError(f"{field} contains non-numeric value {item!r}")
        out.append(float(item))
    return out


def validate_manifest(data: dict[str, Any], manifest_path: Path, image_root: Optional[Path]) -> dict[str, Any]:
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ManifestError("classes must be a non-empty list")
    if any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ManifestError("classes must contain non-empty strings")
    if len(set(classes)) != len(classes):
        raise ManifestError("classes must be unique")

    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise ManifestError("images must be a non-empty list")

    class_ids = {name: idx for idx, name in enumerate(classes)}
    root = (image_root if image_root is not None else manifest_path.parent).resolve()
    normalized_images: list[dict[str, Any]] = []

    for image_index, item in enumerate(images, start=1):
        if not isinstance(item, dict):
            raise ManifestError(f"images[{image_index}] must be an object")
        file_value = item.get("file")
        if not isinstance(file_value, str) or not file_value:
            raise ManifestError(f"images[{image_index}].file must be a non-empty string")
        source_path = safe_image_path(root, file_value)
        if not source_path.exists():
            raise ManifestError(f"image file not found: {file_value}")
        actual_size = image_size(source_path)
        width = item.get("width")
        height = item.get("height")
        if width is None or height is None:
            if actual_size is None:
                raise ManifestError(f"width/height missing and image size unsupported: {file_value}")
            width, height = actual_size
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ManifestError(f"{file_value}: width and height must be positive integers")
        if actual_size is not None and actual_size != (width, height):
            raise ManifestError(f"{file_value}: manifest size {(width, height)} does not match actual {actual_size}")
        split = item.get("split", "train")
        if split not in SPLITS:
            raise ManifestError(f"{file_value}: split must be one of {', '.join(SPLITS)}")
        detections = item.get("detections", [])
        if not isinstance(detections, list):
            raise ManifestError(f"{file_value}: detections must be a list")

        normalized_detections: list[dict[str, Any]] = []
        for det_index, det in enumerate(detections, start=1):
            if not isinstance(det, dict):
                raise ManifestError(f"{file_value}: detections[{det_index}] must be an object")
            class_name = det.get("class")
            if class_name not in class_ids:
                raise ManifestError(f"{file_value}: unknown class {class_name!r}")
            box = as_num_list(det.get("bbox"), f"{file_value}: detections[{det_index}].bbox")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ManifestError(
                    f"{file_value}: invalid bbox {box}; expected 0 <= x1 < x2 <= {width} "
                    f"and 0 <= y1 < y2 <= {height}"
                )
            confidence = det.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1
            ):
                raise ManifestError(f"{file_value}: confidence must be in [0,1]")
            normalized_detections.append(
                {
                    "class": class_name,
                    "class_id": class_ids[class_name],
                    "bbox": box,
                    "confidence": confidence,
                    "source": det.get("source"),
                    "notes": det.get("notes"),
                }
            )
        normalized_images.append(
            {
                "id": image_index,
                "file": file_value,
                "source_path": source_path,
                "width": width,
                "height": height,
                "split": split,
                "detections": normalized_detections,
            }
        )

    return {
        "dataset": data.get("dataset", {}),
        "classes": classes,
        "images": normalized_images,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_yolo_and_copy_images(output: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    used_names: set[str] = set()
    for image in manifest["images"]:
        src: Path = image["source_path"]
        suffix = src.suffix.lower() or ".jpg"
        stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in src.stem).strip("-") or "image"
        dest_name = f"{image['id']:06d}-{stem}{suffix}"
        while dest_name in used_names:
            dest_name = f"{image['id']:06d}-{stem}-{len(used_names)}{suffix}"
        used_names.add(dest_name)
        split = image["split"]
        dest_img = output / "images" / split / dest_name
        shutil.copy2(src, dest_img)

        label_lines = []
        for det in image["detections"]:
            xc, yc, bw, bh = normalized_box(det["bbox"], image["width"], image["height"])
            label_lines.append(f"{det['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        label_path = output / "labels" / split / f"{Path(dest_name).stem}.txt"
        write_text(label_path, "\n".join(label_lines) + ("\n" if label_lines else ""))
        records.append({**image, "dest_name": dest_name, "dest_img": dest_img, "label_path": label_path})
    return records


def write_data_yaml(output: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {idx}: {json.dumps(name)}" for idx, name in enumerate(classes))
    text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names}\n"
    )
    write_text(output / "data.yaml", text)
    write_text(output / "classes.txt", "\n".join(classes) + "\n")


def write_coco(output: Path, classes: list[str], records: list[dict[str, Any]]) -> None:
    images = []
    annotations = []
    ann_id = 1
    for record in records:
        images.append(
            {
                "id": record["id"],
                "file_name": rel_to(record["dest_img"], output),
                "width": record["width"],
                "height": record["height"],
            }
        )
        for det in record["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": record["id"],
                    "category_id": det["class_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "info": {"description": "Generated by auto-yolo-dataset", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(classes)],
        "images": images,
        "annotations": annotations,
    }
    write_text(output / "annotations" / "coco.json", json.dumps(coco, indent=2, ensure_ascii=False) + "\n")


def write_voc(output: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = record["split"]
        ET.SubElement(root, "filename").text = record["dest_name"]
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(record["width"])
        ET.SubElement(size, "height").text = str(record["height"])
        ET.SubElement(size, "depth").text = "3"
        for det in record["detections"]:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = det["class"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bnd = ET.SubElement(obj, "bndbox")
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            ET.SubElement(bnd, "xmin").text = str(x1)
            ET.SubElement(bnd, "ymin").text = str(y1)
            ET.SubElement(bnd, "xmax").text = str(x2)
            ET.SubElement(bnd, "ymax").text = str(y2)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        path = output / "annotations" / "voc" / record["split"] / f"{Path(record['dest_name']).stem}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(path, encoding="utf-8", xml_declaration=True)


def write_label_studio(output: Path, records: list[dict[str, Any]], local_prefix: str) -> None:
    tasks = []
    for record in records:
        results = []
        for idx, det in enumerate(record["detections"], start=1):
            x1, y1, x2, y2 = det["bbox"]
            results.append(
                {
                    "id": f"{record['id']}-{idx}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": record["width"],
                    "original_height": record["height"],
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": x1 / record["width"] * 100,
                        "y": y1 / record["height"] * 100,
                        "width": (x2 - x1) / record["width"] * 100,
                        "height": (y2 - y1) / record["height"] * 100,
                        "rectanglelabels": [det["class"]],
                    },
                }
            )
        tasks.append(
            {
                "data": {"image": f"{local_prefix}{rel_to(record['dest_img'], output)}"},
                "predictions": [{"model_version": "auto-yolo-dataset", "score": 1.0, "result": results}],
            }
        )
    write_text(
        output / "annotations" / "label-studio-tasks.json",
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
    )


def color_for_class(class_id: int) -> str:
    colors = ("#f97316", "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#ca8a04", "#db2777")
    return colors[class_id % len(colors)]


def write_visualizations(output: Path, records: list[dict[str, Any]]) -> list[Path]:
    visualization_paths: list[Path] = []
    for record in records:
        split = record["split"]
        svg_path = output / "visualizations" / split / f"{Path(record['dest_name']).stem}.svg"
        image_href = escape(rel_between(record["dest_img"], svg_path.parent))
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{record["width"]}" height="{record["height"]}" viewBox="0 0 {record["width"]} {record["height"]}">',
            f'  <image href="{image_href}" x="0" y="0" width="{record["width"]}" height="{record["height"]}" preserveAspectRatio="none"/>',
        ]
        stroke_width = max(2, round(max(record["width"], record["height"]) / 400))
        font_size = max(14, round(max(record["width"], record["height"]) / 45))
        for det in record["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            color = color_for_class(det["class_id"])
            label = det["class"]
            if det.get("confidence") is not None:
                label = f"{label} {det['confidence']:.2f}"
            label = escape(label)
            label_y = max(font_size + 2, y1 - 4)
            bg_width = max(60, len(label) * font_size * 0.62)
            bg_height = font_size + 6
            lines.extend(
                [
                    f'  <rect x="{x1:.2f}" y="{y1:.2f}" width="{width:.2f}" height="{height:.2f}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>',
                    f'  <rect x="{x1:.2f}" y="{label_y - bg_height:.2f}" width="{bg_width:.2f}" height="{bg_height:.2f}" fill="{color}" opacity="0.82"/>',
                    f'  <text x="{x1 + 3:.2f}" y="{label_y - 5:.2f}" fill="#ffffff" font-family="Arial, sans-serif" font-size="{font_size}" font-weight="700">{label}</text>',
                ]
            )
        lines.append("</svg>")
        write_text(svg_path, "\n".join(lines) + "\n")
        visualization_paths.append(svg_path)
    write_visualization_index(output, visualization_paths)
    return visualization_paths


def write_visualization_index(output: Path, visualization_paths: list[Path]) -> None:
    index_path = output / "visualizations" / "index.html"
    items = []
    for path in visualization_paths:
        href = escape(rel_between(path, index_path.parent))
        items.append(f'      <li><a href="{href}">{escape(href)}</a></li>')
    html = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Annotation Visualizations</title>",
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}li{margin:6px 0}</style>",
        "</head>",
        "<body>",
        "  <h1>Annotation Visualizations</h1>",
        "  <ul>",
        *items,
        "  </ul>",
        "</body>",
        "</html>",
    ]
    write_text(index_path, "\n".join(html) + "\n")


def write_metadata(output: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    split_counts = {split: 0 for split in SPLITS}
    annotation_counts = {split: 0 for split in SPLITS}
    normalized_images = []
    for record in records:
        split_counts[record["split"]] += 1
        annotation_counts[record["split"]] += len(record["detections"])
        normalized_images.append(
            {
                "file": rel_to(record["dest_img"], output),
                "label": rel_to(record["label_path"], output),
                "width": record["width"],
                "height": record["height"],
                "split": record["split"],
                "detections": [
                    {key: value for key, value in det.items() if key in {"class", "class_id", "bbox", "confidence", "source", "notes"}}
                    for det in record["detections"]
                ],
            }
        )
    validation = {
        "ok": True,
        "generator": "auto-yolo-dataset",
        "generator_version": "0.1.0",
        "image_count": len(records),
        "annotation_count": sum(annotation_counts.values()),
        "split_counts": split_counts,
        "annotation_counts": annotation_counts,
    }
    normalized = {
        "dataset": manifest["dataset"],
        "classes": manifest["classes"],
        "images": normalized_images,
    }
    write_text(output / "validation.json", json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    write_text(output / "manifest.normalized.json", json.dumps(normalized, indent=2, ensure_ascii=False) + "\n")
    card = [
        "# Dataset Card",
        "",
        f"- Name: {manifest['dataset'].get('name', 'unnamed')}",
        f"- Version: {manifest['dataset'].get('version', 'unspecified')}",
        f"- Images: {len(records)}",
        f"- Annotations: {sum(annotation_counts.values())}",
        f"- Classes: {', '.join(manifest['classes'])}",
        f"- Splits: {split_counts}",
        "",
        "Generated by `auto-yolo-dataset/scripts/build_yolo_dataset.py` from a model-produced detection manifest.",
    ]
    write_text(output / "dataset-card.md", "\n".join(card) + "\n")


def build(
    manifest_path: Path,
    output: Path,
    image_root: Optional[Path],
    label_studio_local_prefix: str,
    visualize: bool,
) -> dict[str, Any]:
    data = load_json(manifest_path)
    manifest = validate_manifest(data, manifest_path.resolve(), image_root.resolve() if image_root else None)
    output = output.resolve()
    clean_output_safely(output, [image["source_path"] for image in manifest["images"]])
    records = write_yolo_and_copy_images(output, manifest)
    write_data_yaml(output, manifest["classes"])
    write_coco(output, manifest["classes"], records)
    write_voc(output, records)
    write_label_studio(output, records, label_studio_local_prefix)
    visualization_paths = write_visualizations(output, records) if visualize else []
    write_metadata(output, manifest, records)
    return {
        "ok": True,
        "output": rel_between(output, Path.cwd().resolve()),
        "image_count": len(records),
        "annotation_count": sum(len(record["detections"]) for record in records),
        "visualization_count": len(visualization_paths),
        "classes": manifest["classes"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build YOLO/COCO/VOC dataset assets from a detection manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to annotation-contract JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset directory; overwritten if it exists.")
    parser.add_argument("--image-root", type=Path, help="Root directory for manifest image paths. Defaults to manifest directory.")
    parser.add_argument(
        "--label-studio-local-prefix",
        default="/data/local-files/?d=",
        help="Prefix for Label Studio local-file task URLs.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write SVG overlays under visualizations/{train,val,test}/ for manual annotation review.",
    )
    args = parser.parse_args(argv)

    try:
        result = build(args.manifest, args.output, args.image_root, args.label_studio_local_prefix, args.visualize)
    except (ManifestError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
