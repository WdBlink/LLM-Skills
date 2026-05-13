---
name: ms-qwen-vl
description: "调用魔搭社区（ModelScope）Qwen3-VL 多模态 API 对图片/视频进行视觉解析，支持图片描述、OCR、视觉问答、目标检测、图表解析，并可将检测结果按 YOLO 数据集格式保存。支持本地路径/URL、视频抽帧、文件夹批量处理、指定检测类别。使用 OpenAI SDK 兼容方式调用。用户提到\"硅基流动\"、\"ModelScope\"、\"Qwen-VL\"、\"多模态视觉\"、\"解析图片\"、“解析视频”、“生成yolo数据集”、“目标检测”、“批量处理”等关键词时应触发。"
---

# MS-Qwen-VL Skill

基于 ModelScope Qwen3-VL 系列模型的多模态视觉识别技能，使用 OpenAI SDK 兼容方式调用。

## 功能特点

- **OpenAI SDK 兼容**：使用标准 OpenAI SDK 调用 API
- **多种任务支持**：图像描述、OCR、视觉问答、目标检测、图表解析
- **双模型模式**：默认快速模型（30B）+ 精细高精度模型（235B）
- **灵活输入**：支持本地图片、URL、视频、文件夹批量
- **视频抽帧检测**：按时间间隔抽帧并检测
- **YOLO 数据集输出**：保存 images、labels、images_annotated、raw、classes.txt
- **下采样一致性**：模型输入与保存图片保持一致尺寸（长边 1024）
- **指定检测类别**：detect 任务可传入类别列表

## 安装与配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
```

编辑 `.env` 文件，填入从 https://modelscope.cn/my/myaccesstoken 获取的 API Key：

```
MODELSCOPE_API_KEY=your_api_key_here
```

## Claude Code 使用方式

### 重要：处理本地图片

当用户提供本地图片路径时（如桌面截图），**必须使用 Python 脚本处理**：

```bash
python scripts/ms_qwen_vl.py "<图片路径>" --task <任务类型>
```

脚本会自动将本地文件转换为 ModelScope API 需要的 base64 格式，并按长边 1024 进行下采样以保持一致性。

### 处理 URL 图片

当用户提供网络 URL 时，同样使用上述命令，脚本会自动识别：

```bash
python scripts/ms_qwen_vl.py "<URL>" --task <任务类型>
```

### Claude Code 对话示例

**场景 1：分析桌面截图**
```
用户: 请帮我描述这张图片 C:\Users\...\Desktop\screenshot.png
助手: [执行] python scripts/ms_qwen_vl.py "C:\Users\...\Desktop\screenshot.png"
```

**场景 2：OCR 识别本地图片**
```
用户: 识别这张图中的文字: D:\Documents\invoice.jpg
助手: [执行] python scripts/ms_qwen_vl.py "D:\Documents\invoice.jpg" --task ocr
```

**场景 3：分析网络图片**
```
用户: 分析这张图片 https://example.com/photo.jpg
助手: [执行] python scripts/ms_qwen_vl.py "https://example.com/photo.jpg" --task describe
```

**场景 4：视觉问答**
```
用户: 这张图里有几个人？C:\Users\...\Desktop\photo.png
助手: [执行] python scripts/ms_qwen_vl.py "C:\Users\...\Desktop\photo.png" --task ask --question "图片里有几个人？"
```

**场景 5：检测指定类别并生成 YOLO 数据集**
```
用户: 检测这批图片中的车和建筑，并保存为yolo数据集
助手: [执行] python scripts/ms_qwen_vl.py "D:\Images\dataset" --task detect --categories car building --output_dir ./yolo_dataset
```

**场景 6：视频抽帧检测**
```
用户: 抽帧检测视频中的车辆并生成yolo数据集
助手: [执行] python scripts/ms_qwen_vl.py "D:\Videos\demo.mp4" --task detect --output_dir ./yolo_dataset --frame_interval 5
```

### 任务类型对照

| 用户需求 | --task 参数 |
|---------|-------------|
| 描述图片内容 | describe |
| 识别文字/OCR | ocr |
| 回答关于图片的问题 | ask（需要 --question） |
| 检测物体 | detect |
| 解析图表 | chart |

## 快速使用

```bash
# 图像描述（默认）
python scripts/ms_qwen_vl.py image.jpg

# OCR 文字识别
python scripts/ms_qwen_vl.py image.jpg --task ocr

# 视觉问答
python scripts/ms_qwen_vl.py image.jpg --task ask --question "图片里有什么？"

# 使用精细模式（235B 模型）
python scripts/ms_qwen_vl.py image.jpg --task describe --precise

# 目标检测并生成 YOLO 数据集（文件夹批量）
python scripts/ms_qwen_vl.py ./images_folder/ --task detect --output_dir ./yolo_dataset

# 仅检测指定类别
python scripts/ms_qwen_vl.py image.jpg --task detect --categories car building --output_dir ./yolo_dataset

# 视频抽帧检测（每 5 秒一帧）
python scripts/ms_qwen_vl.py video.mp4 --task detect --output_dir ./yolo_dataset --frame_interval 5
```

Python 代码调用：

```python
from scripts.ms_qwen_vl import analyze_image

result = analyze_image("image.jpg", task="ocr")
print(result)
```

## 任务类型

| 任务 | 参数 | 说明 |
|------|------|------|
| 图像描述 | `describe` | 详细描述图片内容（默认） |
| OCR 识别 | `ocr` | 识别图片中的文字 |
| 视觉问答 | `ask` | 回答关于图片的问题 |
| 目标检测 | `detect` | 检测图片中的物体并生成 YOLO 数据 |
| 图表解析 | `chart` | 解析图表数据 |

## 环境变量

| 变量名 | 说明 |
|--------|------|
| `MODELSCOPE_API_KEY` | API 密钥（必需） |
| `MODELSCOPE_MODEL` | 默认模型（可选） |
| `MODELSCOPE_MODEL_PRECISE` | 精细模式模型（可选） |

## 主要参数

| 参数 | 说明 |
|------|------|
| `--task` | 任务类型 |
| `--question` | 视觉问答问题（ask 任务必需） |
| `--prompt` | 自定义 prompt |
| `--output_dir` | YOLO 数据集输出目录（detect 任务必需） |
| `--frame_interval` | 视频抽帧间隔（秒） |
| `--categories` | 指定检测类别（detect 任务可选） |

## Resources

### scripts/

**ms_qwen_vl.py** - 核心解析脚本，提供 `analyze_image()` 统一接口

### references/

**api-guide.md** - OpenAI SDK 兼容调用方式详细说明
**models.md** - Qwen3-VL 系列模型及推荐使用场景
