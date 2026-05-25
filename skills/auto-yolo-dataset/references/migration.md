---
name: migration
description: Portability guidance for Codex, Claude Code, OpenCode, detector services, and other agent hosts.
---

# Migration

Keep the workflow stable across hosts:

1. Host agent calls an explicit detector backend.
2. Detector results are written as annotation contract JSON.
3. Host agent runs `scripts/build_yolo_dataset.py`.
4. Host agent reports generated assets and annotation limitations.

## Detector Adapter Boundary

A detector integration only needs to implement this function:

```text
images + class hints + output manifest path -> annotation-contract JSON
```

Do not let backend-specific request fields leak into the builder. Store backend details in optional `source` or `notes` fields if needed. Prefer the shared HTTP contract in `detector-http-contract.md` for local services and remote API adapters.

Backends may use different box coordinate systems. Normalize backend output to the shared HTTP `bbox_format` values, and let the manifest writer convert to original-image pixel coordinates before running the dataset builder.

For quality control, prefer backends that implement `/review`. The review step should inspect rendered overlays and return corrected boxes through the same HTTP coordinate-format contract. Keep review as a separate manifest-to-manifest step so hosts without review support can still run detection and clearly report that review was skipped.

For VLM backends, prefer count-first prompting over fixed box-count caps. Ask the model to list visible target instances, then ask for one full-object box per listed instance. Use script-level quality reports for suspicious cases rather than silently truncating detections.

## Cross-Host Constraints

- Use relative paths in manifests whenever possible.
- Do not require a slash command such as `/opc` or a Codex-only tool.
- Do not call network APIs from the dataset builder.
- Do not write outside the requested output directory except for reading source images.
- Keep scripts executable with the system Python standard library.
- Do not rely on the host agent having multimodal image-reading capability.
