---
name: ms-qwen-vl
description: "通过 ModelScope 或 SiliconFlow 的 Qwen-VL API 解析图片和视频，支持描述、OCR、视觉问答、目标检测、图表解析及 YOLO 数据集导出。用于用户明确要求这些服务或批量视觉解析的任务。"
---

# MS-Qwen-VL Skill

通过明确选择的 ModelScope 或 SiliconFlow 服务调用 Qwen-VL 模型，使用 OpenAI SDK 兼容接口。

## 功能特点

- **OpenAI SDK 兼容**：使用标准 OpenAI SDK 调用 API
- **多种任务支持**：图像描述、OCR、视觉问答、目标检测、图表解析
- **可配置模型**：普通与精细模式分别读取所选服务的模型变量；未配置时均使用 `Qwen/Qwen3-VL-235B-A22B-Instruct`
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
QWEN_VL_PROVIDER=modelscope
MODELSCOPE_API_KEY=your_api_key_here
```

使用 SiliconFlow 时改为 `QWEN_VL_PROVIDER=siliconflow` 并配置 `SILICONFLOW_API_KEY`。ModelScope 固定请求 `https://api-inference.modelscope.cn/v1/`，SiliconFlow 固定请求 `https://api.siliconflow.cn/v1/`；不跨服务借用密钥或模型配置。

`--provider` 优先于 `QWEN_VL_PROVIDER`。两者都未设置时，仅有一套密钥可自动选择服务；同时存在两套密钥会报错。显式传入 `--api-key` 时也必须明确服务。缺少所选服务的密钥时停止调用，不尝试另一个服务。模型可用性以所选服务当前模型列表和账户权限为准，不能保证内置型号始终在线。

以下命令以 skill 目录为工作目录；从其他目录调用时使用脚本的绝对路径。脚本向所选服务发送图片内容；仅在当前任务已授权使用该外部视觉服务时执行。

## Agent 使用方式

### 重要：处理本地图片

当用户提供本地图片路径时（如桌面截图），**必须使用 Python 脚本处理**：

```bash
python scripts/ms_qwen_vl.py "<图片路径>" --task <任务类型>
```

脚本会自动将本地文件转换为 API 所需的 base64 格式，并按长边 1024 进行下采样以保持一致性。

### 处理 URL 图片

当用户提供网络 URL 时，同样使用上述命令，脚本会自动识别：

```bash
python scripts/ms_qwen_vl.py "<URL>" --task <任务类型>
```

### 对话示例

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

# 使用所选服务配置的精细模式模型
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
| `QWEN_VL_PROVIDER` | `modelscope` 或 `siliconflow`；可由 `--provider` 覆盖 |
| `MODELSCOPE_API_KEY` / `SILICONFLOW_API_KEY` | 所选服务对应的密钥（必需） |
| `MODELSCOPE_MODEL` / `SILICONFLOW_MODEL` | 所选服务的普通模式模型（可选） |
| `MODELSCOPE_MODEL_PRECISE` / `SILICONFLOW_MODEL_PRECISE` | 所选服务的精细模式模型（可选） |

## 主要参数

| 参数 | 说明 |
|------|------|
| `--provider` | 明确选择服务；同时配置两套密钥时必需 |
| `--api-key` | 覆盖所选服务的密钥，须明确服务 |
| `--model` | 覆盖模型，优先于普通和精细模式环境变量 |
| `--task` | 任务类型 |
| `--question` | 视觉问答问题（ask 任务必需） |
| `--prompt` | 自定义 prompt |
| `--output_dir` | YOLO 数据集输出目录（detect 任务必需） |
| `--frame_interval` | 视频抽帧间隔（秒） |
| `--categories` | 指定检测类别（detect 任务可选） |

## Resources

### scripts/

**ms_qwen_vl.py** - 核心解析脚本，提供 `analyze_image()` 统一接口

### 验证与来源

`tests/test_provider_routing.py` 在不发送网络请求的条件下验证密钥、地址、模型与 CLI 参数的绑定。从仓库根目录运行 `python3 -B -m unittest discover -s skills/ms-qwen-vl/tests -v`。离线通过不代表服务连通性、模型质量或 YOLO 标注准确率已验证。

服务地址依据 [ModelScope API-Inference 示例](https://www.modelscope.cn/posts/1010?pid=1009) 与 [SiliconFlow 官方快速上手](https://api-docs.siliconflow.cn/docs/userguide/quickstart)。
