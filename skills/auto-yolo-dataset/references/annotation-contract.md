---
name: annotation-contract
description: Detection manifest schema shared by detector backends and the deterministic dataset builder.
---

# Annotation Contract

Use this JSON contract as the only boundary between detector output and dataset generation.

Detector adapters may receive boxes from different model coordinate systems, but this manifest must always store original-image absolute pixel `xyxy` boxes.

## Manifest Shape

```json
{
  "dataset": {
    "name": "sample-detection-dataset",
    "version": "0.1.0",
    "description": "Short purpose statement",
    "license": "UNSPECIFIED"
  },
  "classes": ["red-block", "blue-disc"],
  "images": [
    {
      "file": "images/sample-001.png",
      "width": 320,
      "height": 240,
      "split": "train",
      "detections": [
        {
          "class": "red-block",
          "bbox": [40, 50, 140, 150],
          "confidence": 0.93,
          "source": "http-detector",
          "notes": "optional"
        }
      ]
    }
  ]
}
```

## Field Rules

- `classes`: unique non-empty names. Order is the YOLO class id order.
- `images[].file`: relative image path under `--image-root` or the manifest directory.
- `width` and `height`: positive integers. If omitted, the builder reads PNG/JPEG dimensions.
- `split`: one of `train`, `val`, or `test`; defaults to `train`.
- `bbox`: absolute pixel `[x_min, y_min, x_max, y_max]`, using half-open extents where `x_max` and `y_max` are the exclusive lower-right bounds.
- Box bounds: `0 <= x_min < x_max <= width` and `0 <= y_min < y_max <= height`.
- `confidence`: optional number in `[0,1]`; not written into YOLO labels.
- Detector-specific coordinate systems such as 0-1000 boxes or normalized boxes must be converted before writing this manifest.

## Detector Annotation Procedure

1. Call a configured local or remote detector backend; do not depend on the host agent's own multimodal vision.
2. Request only the target classes relevant to the dataset.
3. Require tight boxes around visible object extents, not shadows or labels.
4. Prefer fewer reliable boxes over many uncertain boxes.
5. Save uncertainty in `confidence` or `notes`; never hide uncertainty inside class names.
6. Run the builder and correct any validation failures mechanically.
