#!/usr/bin/env python3
"""Local Gemma 4 vision detector service for auto-yolo-dataset."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Union


MODEL = None
PROCESSOR = None
LOAD_ERROR: Optional[str] = None
LOCK = threading.Lock()


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def load_model(model_dir: str) -> tuple[Any, Any]:
    global MODEL, PROCESSOR, LOAD_ERROR
    with LOCK:
        if MODEL is not None and PROCESSOR is not None:
            return MODEL, PROCESSOR
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            dtype_name = os.environ.get("GEMMA4_DTYPE", "bfloat16").lower()
            dtype = {
                "auto": "auto",
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(dtype_name)
            if dtype is None:
                raise ValueError(f"unsupported GEMMA4_DTYPE: {dtype_name}")
            device = os.environ.get("GEMMA4_DEVICE")
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            PROCESSOR = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            load_kwargs = {
                "local_files_only": True,
                "trust_remote_code": True,
                "dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            if device == "cuda":
                load_kwargs["device_map"] = "auto"
            MODEL = AutoModelForImageTextToText.from_pretrained(model_dir, **load_kwargs)
            if device != "cuda":
                MODEL.to(device)
            MODEL.eval()
            LOAD_ERROR = None
            return MODEL, PROCESSOR
        except Exception as exc:  # pragma: no cover - operational diagnostics
            LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            raise


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed, _ = decoder.raw_decode(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"model did not return parseable JSON: {text[:1000]}")


def normalize_detection(det: dict[str, Any], classes: list[str], width: int, height: int) -> Optional[dict[str, Any]]:
    cls = det.get("class")
    if cls not in classes:
        return None
    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0):
        return None
    confidence = det.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))
    return {"class": cls, "bbox": [x1, y1, x2, y2], "confidence": confidence, "source": "gemma4-local"}


def generate_from_image(model: Any, processor: Any, image: Any, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(output[0][input_len:], skip_special_tokens=True)


def count_instances(
    model: Any,
    processor: Any,
    image: Any,
    classes: list[str],
    instance_policy: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    if instance_policy == "primary-subject":
        policy_text = (
            "List only the main subject instances of the target classes. Ignore background, tiny, or incidental objects."
        )
    else:
        policy_text = (
            "List every visible, complete, localizable instance of the target classes. "
            "Do not list object parts such as wheels, cabs, trailers alone, shadows, labels, or ambiguous fragments."
        )
    prompt = (
        "You are preparing an object-detection dataset. Target classes: "
        + ", ".join(classes)
        + ". "
        + policy_text
        + " Each physical object must appear at most once. Use short location-based descriptions, no more than six words each. "
        + "If many similar objects are visible, distinguish them by position such as left, center, right, top row, or bottom row. "
        + " Return only JSON in this exact schema: "
        + '{"instances":[{"id":1,"class":"CLASS_NAME","description":"short visual description","visibility":"clear|partial|uncertain"}]}. '
        + "If no target object is visible, return {\"instances\":[]}."
    )
    text = generate_from_image(model, processor, image, prompt, max_new_tokens)
    parsed = extract_json(text)
    raw_instances = parsed.get("instances", [])
    if not isinstance(raw_instances, list):
        raw_instances = []
    instances: list[dict[str, Any]] = []
    for item in raw_instances:
        if not isinstance(item, dict):
            continue
        cls = item.get("class")
        if cls not in classes:
            continue
        description = item.get("description")
        visibility = item.get("visibility")
        instances.append(
            {
                "id": len(instances) + 1,
                "class": cls,
                "description": str(description or f"{cls} instance {len(instances) + 1}"),
                "visibility": str(visibility or "clear"),
            }
        )
    return instances, text


def detect(
    model: Any,
    processor: Any,
    image_input: Union[Path, bytes],
    classes: list[str],
    max_new_tokens: int,
    instance_policy: str,
    min_object_area_ratio: float,
) -> dict[str, Any]:
    from PIL import Image

    if isinstance(image_input, Path):
        image = Image.open(image_input).convert("RGB")
    else:
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    width, height = image.size
    instances, count_raw = count_instances(model, processor, image, classes, instance_policy, max_new_tokens)
    prompt = (
        "You are an object detection annotator. Target classes: "
        + ", ".join(classes)
        + ". Candidate object instances are: "
        + json.dumps(instances, ensure_ascii=False, separators=(",", ":"))
        + ". Return one tight full-object box for each listed instance that is visible and localizable. "
        + "Do not add boxes for object parts, duplicate views of the same instance, labels, shadows, or background clutter. "
        + "If an instance cannot be localized, omit it. Return only JSON in this exact schema: "
        + '{"detections":[{"instance_id":1,"class":"CLASS_NAME","bbox":[x1,y1,x2,y2],"confidence":0.0,"notes":"optional"}]} '
        + "Coordinates must be xyxy values on a 0 to 1000 scale relative to the full image: "
        + "x=0 is the left edge, x=1000 is the right edge, y=0 is the top edge, y=1000 is the bottom edge. "
        + "Keep the JSON compact. If no target object is visible, return {\"detections\":[]}."
    )
    text = generate_from_image(model, processor, image, prompt, max_new_tokens)
    parsed = extract_json(text)
    detections = parsed.get("detections", [])
    if not isinstance(detections, list):
        detections = []
    normalized = [d for d in (normalize_detection(det, classes, width, height) for det in detections if isinstance(det, dict)) if d]
    if min_object_area_ratio > 0:
        for det in normalized:
            x1, y1, x2, y2 = det["bbox"]
            area_ratio = ((x2 - x1) * (y2 - y1)) / 1_000_000.0
            if area_ratio < min_object_area_ratio:
                det["notes"] = (det.get("notes", "") + f" flag: small-area-ratio={area_ratio:.6f}").strip()
    return {
        "width": width,
        "height": height,
        "bbox_format": "xyxy_1000",
        "instances": instances,
        "detections": normalized,
        "raw": text,
        "raw_count": count_raw,
    }


def pixel_box_to_1000(box: list[Any], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1 * 1000.0 / width, y1 * 1000.0 / height, x2 * 1000.0 / width, y2 * 1000.0 / height]


def review(
    model: Any,
    processor: Any,
    image_input: Union[Path, bytes],
    classes: list[str],
    detections: list[dict[str, Any]],
    width: int,
    height: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    if isinstance(image_input, Path):
        image = Image.open(image_input).convert("RGB")
    else:
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    listed = []
    for index, det in enumerate(detections, start=1):
        cls = det.get("class")
        if cls not in classes:
            continue
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            scaled_box = [round(value, 1) for value in pixel_box_to_1000(bbox, width, height)]
        except (TypeError, ValueError):
            continue
        listed.append({"id": index, "class": cls, "bbox": scaled_box, "confidence": det.get("confidence")})
    prompt = (
        "You are reviewing object-detection annotations. The image already has orange numbered boxes drawn on it. "
        "Target classes: "
        + ", ".join(classes)
        + ". Existing boxes on a 0 to 1000 image-relative xyxy scale are: "
        + json.dumps(listed, ensure_ascii=False, separators=(",", ":"))
        + ". Check whether each box tightly covers a visible target object. Remove false positives, merge duplicate boxes, "
        + "adjust shifted or loose boxes, and add any clearly missed target instances. Return only JSON in this exact schema: "
        + '{"detections":[{"class":"CLASS_NAME","bbox":[x1,y1,x2,y2],"confidence":0.0,"notes":"optional"}],'
        + '"review":{"status":"ok|changed","actions":["keep|adjust|drop|add"],"notes":"short reason"}}. '
        + "Coordinates must be final xyxy values on a 0 to 1000 scale relative to the full image. "
        + "Do not include boxes for labels, shadows, parts, or background objects."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    text = processor.decode(output[0][input_len:], skip_special_tokens=True)
    parsed = extract_json(text)
    reviewed = parsed.get("detections", [])
    if not isinstance(reviewed, list):
        reviewed = []
    normalized = [d for d in (normalize_detection(det, classes, width, height) for det in reviewed if isinstance(det, dict)) if d]
    review_info = parsed.get("review", {})
    if not isinstance(review_info, dict):
        review_info = {}
    return {
        "width": width,
        "height": height,
        "bbox_format": "xyxy_1000",
        "detections": normalized,
        "review": review_info,
        "raw": text,
    }


def chat(model: Any, processor: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt_messages = []
    system_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_parts.append(str(msg.get("content", "")))
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        prompt_messages.append({"role": role, "content": [{"type": "text", "text": str(msg.get("content", ""))}]})
    if system_parts:
        system_text = "\n\n".join(part for part in system_parts if part)
        if prompt_messages and prompt_messages[0]["role"] == "user":
            prompt_messages[0]["content"][0]["text"] = system_text + "\n\n" + prompt_messages[0]["content"][0]["text"]
        else:
            prompt_messages.insert(0, {"role": "user", "content": [{"type": "text", "text": system_text}]})
    if not prompt_messages:
        prompt_messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "Gemma4VisionService/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("GEMMA4_VERBOSE") == "1":
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        request_path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if request_path == "/health":
            json_response(
                self,
                200,
                {
                    "ok": LOAD_ERROR is None,
                    "loaded": MODEL is not None,
                    "load_error": LOAD_ERROR,
                    "backend": "gemma4-local",
                    "capabilities": ["open-vocabulary-detection", "count-first-detection", "annotation-review", "chat"],
                    "bbox_formats": ["xyxy_1000"],
                },
            )
            return
        json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        request_path = self.path.split("?", 1)[0].rstrip("/")
        if request_path not in {"/detect", "/review", "/chat"}:
            json_response(self, 404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            model, processor = load_model(self.server.model_dir)
            max_new_tokens = int(payload.get("max_new_tokens") or self.server.max_new_tokens)
            if request_path == "/chat":
                messages = payload.get("messages", [])
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                result = chat(model, processor, messages, max_new_tokens)
                json_response(self, 200, {"ok": True, "text": result})
                return
            classes = payload["classes"]
            if not isinstance(classes, list) or not all(isinstance(item, str) and item for item in classes):
                raise ValueError("classes must be a non-empty string list")

            if "image_base64" in payload:
                image_base64 = payload["image_base64"]
                if not isinstance(image_base64, str) or not image_base64:
                    raise ValueError("image_base64 must be a non-empty string")
                try:
                    image_input: Union[Path, bytes] = base64.b64decode(image_base64, validate=True)
                except binascii.Error as exc:
                    raise ValueError(f"invalid image_base64: {exc}") from exc
            elif "image_path" in payload:
                image_path = Path(str(payload["image_path"])).expanduser().resolve()
                if not image_path.exists():
                    raise ValueError(f"image not found: {image_path}")
                image_input = image_path
            else:
                raise ValueError("request must include image_base64 or image_path")

            if request_path == "/review":
                width = payload.get("width")
                height = payload.get("height")
                if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                    raise ValueError("review width and height must be positive integers")
                detections = payload.get("detections", [])
                if not isinstance(detections, list):
                    raise ValueError("review detections must be a list")
                overlay_base64 = payload.get("overlay_image_base64")
                if isinstance(overlay_base64, str) and overlay_base64:
                    try:
                        review_input: Union[Path, bytes] = base64.b64decode(overlay_base64, validate=True)
                    except binascii.Error as exc:
                        raise ValueError(f"invalid overlay_image_base64: {exc}") from exc
                else:
                    review_input = image_input
                result = review(model, processor, review_input, classes, detections, width, height, max_new_tokens)
                json_response(self, 200, {"ok": True, **result})
                return

            instance_policy = str(payload.get("instance_policy") or "all-visible")
            if instance_policy not in {"all-visible", "primary-subject"}:
                raise ValueError("instance_policy must be all-visible or primary-subject")
            min_object_area_ratio = payload.get("min_object_area_ratio", 0.0)
            if not isinstance(min_object_area_ratio, (int, float)) or min_object_area_ratio < 0:
                raise ValueError("min_object_area_ratio must be a non-negative number")
            result = detect(
                model,
                processor,
                image_input,
                classes,
                max_new_tokens,
                instance_policy,
                float(min_object_area_ratio),
            )
            confidence_threshold = payload.get("confidence_threshold")
            if confidence_threshold is not None:
                if not isinstance(confidence_threshold, (int, float)) or not (0 <= confidence_threshold <= 1):
                    raise ValueError("confidence_threshold must be in [0,1]")
                result["detections"] = [
                    det for det in result["detections"] if det.get("confidence", 0.0) >= float(confidence_threshold)
                ]
            json_response(self, 200, {"ok": True, **result})
        except Exception as exc:
            json_response(self, 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], model_dir: str, max_new_tokens: int):
        super().__init__(address, Handler)
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("GEMMA4_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GEMMA4_PORT", "11500")))
    parser.add_argument("--model-dir", default=os.environ.get("GEMMA4_MODEL_DIR", "/home/c301/models/gemma-4-31B-it"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--load-at-start", action="store_true")
    args = parser.parse_args()
    if args.load_at_start:
        load_model(args.model_dir)
    server = Server((args.host, args.port), args.model_dir, args.max_new_tokens)
    print(f"gemma4 vision service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
