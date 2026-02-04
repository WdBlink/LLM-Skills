# LLM Skills

本项目包含一系列 LLM 相关的技能（Skills），每个技能位于独立的文件夹中。

## 技能列表

### 1. MS-Qwen-VL Skill (yolo-dataset-generator)

基于 ModelScope Qwen3-VL 系列模型的多模态视觉识别技能，支持图片描述、OCR、视觉问答、目标检测、图表解析，并可将检测结果按 YOLO 数据集格式保存。

**功能特点:**
*   **OpenAI SDK 兼容**: 使用标准 OpenAI SDK 调用 API。
*   **多种任务支持**: 图像描述、OCR、视觉问答、目标检测、图表解析。
*   **YOLO 数据集输出**: 支持将检测结果保存为 YOLO 格式数据集。
*   **灵活输入**: 支持本地图片、URL、视频、文件夹批量处理。
*   **视频抽帧检测**: 支持对视频进行抽帧检测。

**使用方法:**

需要在 `yolo-dataset-generator` 目录下配置 `.env` 文件（参考 `.env.example`）。

```bash
# 安装依赖
pip install -r yolo-dataset-generator/requirements.txt

# 运行脚本
python yolo-dataset-generator/scripts/ms_qwen_vl.py "<图片路径或URL>" --task <任务类型> [其他参数]
```

**常用任务类型 (`--task`):**
*   `describe`: 描述图片内容
*   `ocr`: 识别文字
*   `ask`: 视觉问答 (需配合 `--question` 参数)
*   `detect`: 目标检测 (可配合 `--categories` 指定类别，`--output_dir` 指定输出目录)

**示例:**

*   **描述图片**:
    ```bash
    python yolo-dataset-generator/scripts/ms_qwen_vl.py "path/to/image.jpg" --task describe
    ```

*   **生成 YOLO 数据集**:
    ```bash
    python yolo-dataset-generator/scripts/ms_qwen_vl.py "path/to/images_folder" --task detect --categories car person --output_dir ./yolo_dataset
    ```

更多详细信息请参考 [yolo-dataset-generator/SKILL.md](yolo-dataset-generator/SKILL.md)。
