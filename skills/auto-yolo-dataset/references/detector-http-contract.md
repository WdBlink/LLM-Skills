---
name: detector-http-contract
description: HTTP detector service contract for local model services and remote API adapters.
---

# Detector HTTP Contract

Use this contract between the agent host and the detector backend. The agent host should not inspect pixels itself; it sends images and target classes to a detector service, then converts the detector response into the annotation manifest.

## Health Check

```http
GET /health
```

Expected JSON:

```json
{
  "ok": true,
  "backend": "local-gemma4",
  "capabilities": ["open-vocabulary-detection", "annotation-review"],
  "bbox_formats": ["xyxy_1000"]
}
```

`ok: false` means the skill must stop and report the detector problem. It must not silently fall back to agent vision.

## Detection Request

```http
POST /detect
Content-Type: application/json
```

Request body:

```json
{
  "file": "sample.jpg",
  "classes": ["truck", "car"],
  "mode": "open-vocabulary",
  "mime_type": "image/jpeg",
  "bbox_format": "xyxy_pixel",
  "accepted_bbox_formats": ["xyxy_1000", "xyxy_normalized", "xyxy_pixel"],
  "instance_policy": "all-visible",
  "min_object_area_ratio": 0.001,
  "image_base64": "...",
  "confidence_threshold": 0.2
}
```

Same-machine services may also accept:

```json
{
  "file": "sample.jpg",
  "classes": ["truck"],
  "mode": "open-vocabulary",
  "mime_type": "image/jpeg",
  "image_path": "/absolute/path/to/sample.jpg"
}
```

Use `image_base64` for remote API adapters or when the service cannot read local paths. Use `image_path` only when the detector runs on the same machine with access to the same filesystem.

## Detection Response

```json
{
  "ok": true,
  "width": 1280,
  "height": 720,
  "bbox_format": "xyxy_pixel",
  "instances": [
    {
      "id": 1,
      "class": "truck",
      "description": "red box truck at the center",
      "visibility": "clear"
    }
  ],
  "detections": [
    {
      "class": "truck",
      "bbox": [120, 80, 540, 430],
      "confidence": 0.91,
      "source": "local-gemma4",
      "notes": "optional"
    }
  ]
}
```

Rules:

- `class` must be one of the requested class names.
- `bbox_format` declares the coordinate system used by every returned `bbox`.
- Supported response formats are:
  - `xyxy_pixel`: absolute pixel `[x_min, y_min, x_max, y_max]` in the original image.
  - `xyxy_1000`: `[x_min, y_min, x_max, y_max]` on a 0-1000 scale relative to the original image.
  - `xyxy_normalized`: `[x_min, y_min, x_max, y_max]` normalized to `[0,1]`.
- If `bbox_format` is omitted, clients must treat boxes as `xyxy_pixel` for backward compatibility.
- For `xyxy_pixel`, coordinates must satisfy `0 <= x_min < x_max <= width` and `0 <= y_min < y_max <= height`.
- `instance_policy` should be `all-visible` unless the user explicitly asks for only the primary subject.
- VLM backends should count/list target instances before boxing them, then return one full-object box per localizable instance.
- `min_object_area_ratio` is a quality hint and reporting threshold. It must not silently delete boxes unless the user explicitly requests filtering.
- `confidence` must be a number in `[0,1]`.
- If no target object is visible, return `{"ok": true, "width": ..., "height": ..., "detections": []}`.
- On failure, return `{"ok": false, "error": "..."}` with a clear diagnostic.

The detector may be a local model service, a remote API adapter, or a wrapper around a fixed-category detector. If it only supports fixed categories, expose that limitation through `capabilities` or `backend` documentation.

## Annotation Review Request

Detector services may also expose:

```http
POST /review
Content-Type: application/json
```

The review endpoint lets the model inspect the visualized annotation result and correct systematic shifts, loose boxes, duplicates, and false positives.

Request body:

```json
{
  "file": "sample.jpg",
  "classes": ["truck"],
  "mode": "annotation-review",
  "width": 1280,
  "height": 720,
  "mime_type": "image/jpeg",
  "image_base64": "...",
  "overlay_mime_type": "image/png",
  "overlay_image_base64": "...",
  "bbox_format": "xyxy_pixel",
  "accepted_bbox_formats": ["xyxy_1000", "xyxy_normalized", "xyxy_pixel"],
  "detections": [
    {
      "class": "truck",
      "bbox": [120, 80, 540, 430],
      "confidence": 0.91,
      "source": "local-gemma4"
    }
  ]
}
```

`overlay_image_base64` is a rendered preview image with boxes and labels drawn over the source image. Reviewers should inspect that preview, not just the raw JSON coordinates.

Response body:

```json
{
  "ok": true,
  "width": 1280,
  "height": 720,
  "bbox_format": "xyxy_1000",
  "detections": [
    {
      "class": "truck",
      "bbox": [94, 111, 422, 597],
      "confidence": 0.86,
      "source": "local-gemma4-review",
      "notes": "adjusted from shifted overlay"
    }
  ],
  "review": {
    "status": "changed",
    "actions": ["adjust", "drop", "add", "keep"],
    "notes": "Corrected shifted box and removed duplicate."
  }
}
```

Rules:

- The reviewer response follows the same `bbox_format` rules as `/detect`.
- The returned `detections` are the final corrected detections for that image.
- Dropped boxes are omitted from the final `detections` list.
- The review endpoint may return unchanged boxes when the visualization looks correct.
- Reviewer notes should explain meaningful `add`, `drop`, `adjust`, or `keep` actions.
- If review is unavailable, clients must report that the self-check was skipped.
