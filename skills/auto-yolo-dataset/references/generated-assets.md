---
name: generated-assets
description: Complete output inventory written by build_yolo_dataset.py.
---

# Generated Assets

The builder writes a portable detection dataset under the chosen output directory.

## Directory Layout

```text
dataset/
├── data.yaml
├── classes.txt
├── dataset-card.md
├── manifest.normalized.json
├── validation.json
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── visualizations/              # only when --visualize is used
│   ├── index.html
│   └── {train,val,test}/*.svg
└── annotations/
    ├── coco.json
    ├── label-studio-tasks.json
    └── voc/{train,val,test}/
```

When `scripts/review_with_http_detector.py` is used before the builder, it writes a corrected manifest at the requested `--output-manifest` path and optionally a `review-report.json` with per-image before/after counts and reviewer notes.

## Standards

- YOLO: one `.txt` label file per image, with `class_id x_center y_center width height` normalized to `[0,1]`.
- Ultralytics-style `data.yaml`: `path`, split image directories, `names`, and `nc`.
- COCO: `images`, `annotations`, and `categories` arrays with absolute `xywh` boxes and areas.
- Pascal VOC: one XML file per image with absolute integer `xmin/ymin/xmax/ymax` boxes.
- Label Studio: task JSON with `predictions`, local-file image URLs, original image dimensions, rotation fields, and rectangle labels in percentages.
- Visualization overlays: optional SVG files that reference copied dataset images and draw bounding boxes plus labels on top for manual audit.

## Determinism

Given the same manifest, image root, and output directory, the builder produces stable file names, class ids, split placement, annotation ids, and visualization overlay paths.
