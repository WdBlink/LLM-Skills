#!/usr/bin/env bash
# Manage the local GLM-OCR model + the PDF → Markdown skill.
# This script no longer wraps `ollama serve` — the skill runs the model
# directly through Hugging Face transformers on MPS / CUDA / CPU.
# We keep the script for parity with the original SKILL.md contract.
#
# Usage:
#   ./run_server.sh check     # verify model files + python deps
#   ./run_server.sh clean     # remove work dir for a given PDF
#   ./run_server.sh benchmark # time a 5-page OCR on /tmp/test_page1.png
set -euo pipefail

MODEL_DIR="${GLM_OCR_MODEL_DIR:-/Users/wdblink/Models/huggingface/ZhipuAI/GLM-OCR}"
VENV="${GLM_OCR_VENV:-/Users/wdblink/Code/glm-ocr-workspace/.venv}"

check() {
  echo "model dir:  $MODEL_DIR"
  echo "  weights:  $(ls -lh "$MODEL_DIR"/model.safetensors 2>/dev/null | awk '{print $5}')"
  echo "  config:   $(ls "$MODEL_DIR"/config.json 2>/dev/null && echo OK || echo MISSING)"
  echo "venv:       $VENV"
  if [[ -x "$VENV/bin/python" ]]; then
    echo "  python:   $("$VENV/bin/python" --version)"
  else
    echo "  python:   not found"
  fi
  echo ""
  "$VENV/bin/python" - <<'PY' 2>&1 | head -10
import importlib.util as iu, sys
missing = [m for m in ("torch","transformers","PIL","fitz","accelerate")
           if iu.find_spec(m) is None]
if missing:
    print(f"  missing: {missing}")
    sys.exit(1)
print("  all deps OK")
PY
}

clean() {
  local target="${1:?usage: clean <workdir>}"
  rm -rf "$target"
  echo "removed $target"
}

benchmark() {
  local page="${1:-/tmp/test_page1.png}"
  if [[ ! -f "$page" ]]; then
    echo "no page at $page — run the convert_pdf.py once first to populate it"
    return 1
  fi
  "$VENV/bin/python" - <<PY
import time, os
os.environ.setdefault("GLM_OCR_MODEL_PATH", "$MODEL_DIR")
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
proc = AutoProcessor.from_pretrained(os.environ["GLM_OCR_MODEL_PATH"])
model = AutoModelForImageTextToText.from_pretrained(
    os.environ["GLM_OCR_MODEL_PATH"], dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("mps" if torch.backends.mps.is_available() else "cpu")
model.eval()
img = Image.open("$page").convert("RGB")
img.thumbnail((768,768))
msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Text Recognition:"}]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inp.pop("token_type_ids", None)
inp = {k: v.to(model.device) for k,v in inp.items()}
t0 = time.time()
with torch.no_grad():
    g = model.generate(**inp, max_new_tokens=512, do_sample=False, num_beams=1)
print(f"{time.time()-t0:.1f}s, {g.shape[1]-inp['input_ids'].shape[1]} tokens")
PY
}

case "${1:-check}" in
  check)     check ;;
  clean)     clean "${2:-}" ;;
  benchmark) benchmark "${2:-/tmp/test_page1.png}" ;;
  *) echo "usage: $0 {check|clean|benchmark}"; exit 2 ;;
esac
