# Auto YOLO Dataset

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

Generate object-detection annotations and portable dataset assets from images through an explicit local or remote detector backend. It keeps detection as a backend responsibility, then validates and converts the returned manifest into YOLO labels, COCO JSON, Pascal VOC XML, Label Studio import data, dataset metadata, and optional visual review overlays.

Part of WdBlink LLM Skills.

## Install

```bash
mkdir -p ~/.codex/skills/auto-yolo-dataset
rsync -a skills/auto-yolo-dataset/ ~/.codex/skills/auto-yolo-dataset/
```

Requires: Python 3.8+ and a detector service that implements `references/detector-http-contract.md` when automatic detection is needed.

## Usage

Generate a detection manifest through the configured HTTP detector:

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.codex/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/detect_with_http_detector.py" \
  --image-dir images \
  --classes "truck,car" \
  --manifest detections.json \
  --instance-policy all-visible \
  --quality-report detections.quality-report.json
```

The optional quality report flags suspicious boxes, duplicate boxes, edge-touching boxes, and count mismatches. It does not remove detections automatically.

Optionally run detector-backed annotation review:

```bash
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/review_with_http_detector.py" \
  --manifest detections.json \
  --output-manifest detections.reviewed.json \
  --image-root images \
  --report review-report.json
```

Build the final dataset:

```bash
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/build_yolo_dataset.py" \
  --manifest detections.reviewed.json \
  --output dataset \
  --image-root images \
  --visualize
```

## Workflow

| Step | Description |
|------|-------------|
| Configure | Point `AUTO_YOLO_DETECTOR_URL` at a detector backend or use the default `http://127.0.0.1:11500/detect` |
| Detect | Call the backend with an instance policy and write a normalized detection manifest |
| Check | Inspect the quality report for suspicious boxes before final dataset generation |
| Review | Use `/review` when available to inspect rendered overlays and correct boxes |
| Build | Convert the manifest into YOLO, COCO, VOC, Label Studio, and dataset-card assets |
| Validate | Inspect `validation.json`, quality reports, review reports, and optional SVG/HTML overlays |

## References

| File | Purpose |
|------|---------|
| `references/annotation-contract.md` | Detection manifest schema and integration boundary |
| `references/detector-http-contract.md` | HTTP detector request and response contract |
| `references/generated-assets.md` | Files written by the dataset builder |
| `references/migration.md` | Portability rules across Codex, Claude Code, OpenCode, and other hosts |
| `references/skill-forge-checklist.md` | Distribution and audit checklist |

## License

Proprietary
