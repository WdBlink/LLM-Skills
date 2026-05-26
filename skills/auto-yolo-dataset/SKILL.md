---
name: auto-yolo-dataset
description: Generate object-detection annotations and dataset assets from images through an explicit local or remote detector backend. Use when Codex needs to call a configured detector service, draft bounding boxes, write standard YOLO labels, convert detections into COCO/Pascal VOC/Label Studio assets, validate detection manifests, or prepare portable local target-detection datasets for Claude Code, Codex, OpenCode, or other agent hosts.
---

# Auto YOLO Dataset

Use this skill to turn images into local object-detection dataset assets. The host agent only orchestrates files and tools. Visual detection must be performed by an explicit detector backend, usually a local model service or remote API service. Do not rely on the current agent's multimodal vision as the default path; many agent hosts only support chat and tool orchestration.

The detector writes the same manifest contract regardless of backend. The bundled dataset builder only validates and converts that manifest.

Detector backends may use different coordinate systems. The HTTP detector response must declare `bbox_format` when it does not return original-image pixels. The manifest and generated dataset always store original-image absolute pixel boxes.

For VLM backends, prefer count-first detection: list visible target instances first, then return one full-object box per instance. Do not use a fixed maximum box count as the primary quality control mechanism.

## Workflow

1. Identify the images, target classes, and output directory. If target classes are not supplied, ask for them unless the configured detector explicitly supports class discovery.
2. Confirm a detector backend is configured. Read `references/detector-http-contract.md` for the HTTP request/response shape. The default detector URL is `AUTO_YOLO_DETECTOR_URL`, or `http://127.0.0.1:11500/detect` when the environment variable is absent.
3. Generate the detection manifest through the detector service:

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.codex/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/detect_with_http_detector.py" \
  --image-dir images \
  --classes "truck,car" \
  --manifest detections.json \
  --instance-policy all-visible \
  --quality-report detections.quality-report.json
```

Use `--image-transfer path` only when the detector runs on the same machine and can read the same filesystem paths. The default `base64` mode works for local HTTP services and remote API adapters.

4. Inspect the optional quality report. It flags suspicious boxes such as tiny boxes, edge-touching boxes, duplicate boxes, and count mismatches. It does not drop boxes automatically.

5. If the backend supports annotation review, run a self-check pass before building the final dataset. This pass renders box overlays, asks the reviewer model to inspect the visualized annotations, and writes a corrected manifest:

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.codex/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/review_with_http_detector.py" \
  --manifest detections.json \
  --output-manifest detections.reviewed.json \
  --image-root images \
  --report review-report.json
```

Use the reviewed manifest for the final builder run. If the reviewer endpoint is unavailable, report that self-check was skipped rather than silently claiming the labels were model-reviewed.

6. Run the bundled builder from the installed skill directory:

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.codex/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/build_yolo_dataset.py" \
  --manifest detections.reviewed.json \
  --output dataset \
  --image-root images
```

Add `--visualize` when the user wants human-review overlays:

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.codex/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/build_yolo_dataset.py" \
  --manifest detections.reviewed.json \
  --output dataset \
  --image-root images \
  --visualize
```

If running under Claude Code only, use `$HOME/.claude/skills/auto-yolo-dataset` for `AUTO_YOLO_DATASET_SKILL`.

7. Inspect the generated `validation.json`, quality report, and review report. If `--visualize` was used, review `visualizations/index.html` or the per-image SVG overlays. Fix any manifest errors and rerun until the command exits 0.
8. Return the dataset path, class list, split counts, quality flags, review counts, and any limitations in the detector annotations.

## Detector Backend Mode

The preferred backend is a detector service that implements `references/detector-http-contract.md`. It may wrap a local model such as Gemma/Gamma, Qwen-VL, GroundingDINO, YOLO-World, or a remote model API. The service must return `xyxy` boxes for the requested class names and declare the response coordinate format when needed. If it supports `/review`, use it to inspect rendered overlay images and return corrected boxes before building the final dataset.

Supported instance policies:

- `all-visible`: label every visible, complete, localizable instance of the requested classes.
- `primary-subject`: label only the main subject instances and ignore background/incidental objects.

Supported detector response coordinate formats:

- `xyxy_pixel`: original-image absolute pixel boxes.
- `xyxy_1000`: 0-1000 image-relative boxes, common for some VLM prompt-based detectors.
- `xyxy_normalized`: normalized `[0,1]` image-relative boxes.

`detect_with_http_detector.py` converts these formats into original-image pixel boxes before writing the manifest. If a backend has another coordinate convention, add a backend adapter that converts it before returning the HTTP response or extend the shared contract deliberately.

Health check:

```bash
curl "${AUTO_YOLO_DETECTOR_URL%/detect}/health"
```

Example manifest generation:

```bash
AUTO_YOLO_DETECTOR_URL="http://127.0.0.1:11500/detect"
AUTO_YOLO_DATASET_SKILL="$HOME/.codex/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/detect_with_http_detector.py" \
  --image-dir images \
  --classes "卡车" \
  --manifest detections.json \
  --timeout 300 \
  --instance-policy all-visible \
  --min-object-area-ratio 0.001 \
  --quality-report detections.quality-report.json
```

## Gemma 4 Compatibility Mode

When using the bundled Gemma 4 service, the controller still only invokes tools. The detector service performs visual localization. Start or verify the service, then call either the generic HTTP script or the compatibility wrapper:

The Gemma 4 service returns `bbox_format: "xyxy_1000"` because this model is more reliable when prompted for 0-1000 image-relative coordinates. The generic detector script converts those boxes to original-image pixel coordinates before dataset generation. It also uses count-first prompting so Gemma 4 lists target instances before producing boxes.

The Gemma 4 service also supports `/review`. Use `review_with_http_detector.py` after detection when annotation quality matters:

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.claude/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/review_with_http_detector.py" \
  --manifest detections.json \
  --output-manifest detections.reviewed.json \
  --image-root images \
  --report review-report.json \
  --timeout 300 \
  --max-new-tokens 512
```

```bash
AUTO_YOLO_DATASET_SKILL="$HOME/.claude/skills/auto-yolo-dataset"
python3 "$AUTO_YOLO_DATASET_SKILL/scripts/detect_with_gemma4.py" \
  --image-dir images \
  --classes "卡车" \
  --manifest detections.json \
  --timeout 300 \
  --max-new-tokens 512 \
  --instance-policy all-visible \
  --quality-report detections.quality-report.json

python3 "$AUTO_YOLO_DATASET_SKILL/scripts/build_yolo_dataset.py" \
  --manifest detections.reviewed.json \
  --output dataset \
  --image-root images \
  --visualize
```

The local detector service must be running on `127.0.0.1:11500`:

```bash
systemctl --user status gemma4-vision.service
curl http://127.0.0.1:11500/health
```

## Model Contract

Read `references/annotation-contract.md` before writing or accepting a manifest. The same manifest is the integration boundary for Codex, Claude Code, OpenCode, local detector services, remote detector APIs, Gemma/Gamma adapters, or any other backend.

Detector migration rule: keep the user prompt and dataset command the same. Only replace the detector backend that emits the JSON fields.

## Generated Assets

Read `references/generated-assets.md` when the user asks what files are created or when checking dataset completeness. The builder writes YOLO, COCO, Pascal VOC, Label Studio import JSON, a dataset card, validation metadata, and optional visualization overlays.

## Portability Rules

Read `references/migration.md` before changing this skill for a specific host. The skill must not rely on Codex-only APIs, Claude-only slash commands, hardcoded personal paths, network access, or a specific Gamma 4 API shape.

## Skill Forge Review

When auditing or preparing this skill for distribution, apply `references/skill-forge-checklist.md`: discoverability, reliability, efficiency, trustworthiness, boundedness, value, structure, security, and cross-platform registration.
