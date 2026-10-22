# MS Qwen VL

![Version](https://img.shields.io/badge/version-0.1.1-CC785C)

Analyze images and videos through Qwen-VL compatible APIs, including image description, OCR, visual QA, object detection, chart parsing, and YOLO dataset export.

Part of WdBlink LLM Skills.

## Install

```bash
cd skills/ms-qwen-vl
pip install -r requirements.txt
cp .env.example .env
```

Requires: Python 3.8+, an API key, and the dependencies in `requirements.txt`.

Edit `.env`:

```bash
QWEN_VL_PROVIDER=modelscope
MODELSCOPE_API_KEY=your_api_key_here
```

For SiliconFlow, use `QWEN_VL_PROVIDER=siliconflow` and `SILICONFLOW_API_KEY`.
ModelScope requests use `https://api-inference.modelscope.cn/v1/`; SiliconFlow
requests use `https://api.siliconflow.cn/v1/`. Each provider reads only its own key
and model variables. Without an explicit provider, a single configured key selects
its provider; two keys require `--provider` or `QWEN_VL_PROVIDER`. An explicit
`--api-key` also requires a provider selection.

Set `MODELSCOPE_MODEL` / `MODELSCOPE_MODEL_PRECISE` or
`SILICONFLOW_MODEL` / `SILICONFLOW_MODEL_PRECISE` for the selected provider.
The fallback model is `Qwen/Qwen3-VL-235B-A22B-Instruct` for both modes; use a model
currently available to your account. `--model` overrides either mode.

## Usage

Describe an image:

```bash
python scripts/ms_qwen_vl.py image.jpg --task describe
```

Run OCR:

```bash
python scripts/ms_qwen_vl.py image.jpg --task ocr
```

Ask a visual question:

```bash
python scripts/ms_qwen_vl.py image.jpg --task ask --question "图片里有什么？"
```

Detect objects and export a YOLO dataset:

```bash
python scripts/ms_qwen_vl.py ./images/ --task detect --categories car building --output_dir ./yolo_dataset
```

Extract video frames and detect objects:

```bash
python scripts/ms_qwen_vl.py video.mp4 --task detect --output_dir ./yolo_dataset --frame_interval 5
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--provider` | Environment or single configured key | `modelscope` or `siliconflow` |
| `--api-key` | Selected provider's environment variable | Override the key for the explicitly selected provider |
| `--model` | Selected provider's model variable | Override the model, including `--precise` |
| `--task` | `describe` | `describe`, `ocr`, `ask`, `detect`, or `chart` |
| `--question` | | Question for `ask` task |
| `--prompt` | | Custom prompt override |
| `--precise` | `false` | Use the high-precision model |
| `--output_dir` | | YOLO dataset output directory |
| `--frame_interval` | `5` | Video frame sampling interval in seconds |
| `--categories` | | Restrict detection to specific classes |

## Outputs

| Output | Description |
|--------|-------------|
| `images/` | Images saved for YOLO training |
| `labels/` | YOLO-format label files |
| `images_annotated/` | Annotated preview images |
| `raw/` | Raw model responses |
| `classes.txt` | Class list for the generated dataset |

## Offline verification

From the repository root:

```bash
python3 -B -m unittest discover -s skills/ms-qwen-vl/tests -v
```

These tests mock the API client and image dependencies. They verify provider,
key and model routing, ambiguous configuration rejection, and CLI forwarding.
They do not send images, call paid vision services, or verify model quality.

## Provider references

- [ModelScope API-Inference examples](https://www.modelscope.cn/posts/1010?pid=1009)
- [SiliconFlow quickstart](https://api-docs.siliconflow.cn/docs/userguide/quickstart)

## License

MIT
