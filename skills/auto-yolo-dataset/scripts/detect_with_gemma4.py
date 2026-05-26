#!/usr/bin/env python3
"""Compatibility wrapper for the local Gemma 4 detector service."""

from __future__ import annotations

from detect_with_http_detector import main


if __name__ == "__main__":
    main(
        default_detector_url="http://127.0.0.1:11500/detect",
        default_dataset_source="gemma4-local-detector",
        default_image_transfer="path",
    )
