# MS Qwen VL

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

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
MODELSCOPE_API_KEY=your_api_key_here
```

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

## License

MIT
