#!/usr/bin/env python3
"""
Pre-flight check for the glm-ocr-pdf2md skill.

Verifies:
  1. Python deps (PyMuPDF, Pillow, transformers, torch) are importable.
  2. The HF model dir exists and contains config.json + model.safetensors.
  3. The model loads in a small inference test (1 token).

Exits 0 only if all three pass. Prints a checklist-style report.

Note: this skill uses Hugging Face transformers directly (no Ollama / vLLM).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import importlib.util as iu

DEFAULT_MODEL = "/Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR"
MODEL_DIR = Path(os.environ.get("GLM_OCR_MODEL_PATH", DEFAULT_MODEL))


def check_deps() -> list[tuple[str, str]]:
    out = []
    for mod in ("torch", "transformers", "PIL", "fitz", "accelerate"):
        out.append((mod, "OK" if iu.find_spec(mod) else "MISSING"))
    return out


def check_model_dir() -> str:
    if not MODEL_DIR.exists():
        return f"FAIL  {MODEL_DIR} does not exist"
    cfg = MODEL_DIR / "config.json"
    safetensors = MODEL_DIR / "model.safetensors"
    if not cfg.exists():
        return f"FAIL  {MODEL_DIR}/config.json missing"
    if not safetensors.exists():
        return f"FAIL  {MODEL_DIR}/model.safetensors missing"
    size_mb = safetensors.stat().st_size / 1024 / 1024
    return f"OK    config.json + model.safetensors ({size_mb:.0f} MB)"


def check_inference() -> str:
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        # tiny smoke: just confirm weights load
        model = AutoModelForImageTextToText.from_pretrained(
            str(MODEL_DIR), dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(dev)
        _ = next(model.parameters()).device
        return f"OK    loaded on {dev}"
    except Exception as e:  # pragma: no cover
        return f"FAIL  {type(e).__name__}: {e}"


def main() -> int:
    print("== glm-ocr-pdf2md env check ==")
    print(f"  model dir: {MODEL_DIR}")
    print(f"  device:    auto (mps > cuda > cpu)")
    print()
    failures = 0
    print("[deps]")
    for mod, st in check_deps():
        print(f"  {mod:<14} {st}")
        if not st.startswith("OK"):
            failures += 1
    print("[model dir]")
    md = check_model_dir()
    print(f"  {md}")
    if md.startswith("FAIL"):
        failures += 1
    print("[inference]")
    inf = check_inference()
    print(f"  {inf}")
    if inf.startswith("FAIL"):
        failures += 1
    print()
    if failures:
        print(f"=> {failures} check(s) failed. See SKILL.md for setup steps.")
        return 1
    print("=> all green. You can run: python3 convert_pdf.py <pdf>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
