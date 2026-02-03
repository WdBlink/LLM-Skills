#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MS-Qwen-VL CLI
使用 OpenAI SDK 兼容方式调用 ModelScope Qwen-VL 多模态 API，对给定路径中的图片或者视频进行分析，并生成相应数据集。

支持功能：
- 图像描述 (describe)
- OCR 文字识别 (ocr)
- 视觉问答 (ask)
- 目标检测 (detect)，并可生成 YOLO 格式数据集
- 图表解析 (chart)

代码作者: wdblink
"""

import argparse
import base64
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import dotenv
import numpy as np
from openai import OpenAI
from PIL import Image

# 加载环境变量
dotenv.load_dotenv(Path(__file__).parent / ".env")

# 默认配置
DEFAULT_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DEFAULT_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
DEFAULT_MODEL_PRECISE = os.getenv("SILICONFLOW_MODEL_PRECISE", "Qwen/Qwen3-VL-235B-A22B-Instruct")
API_BASE_URL = "https://api.siliconflow.cn/v1/"

# 任务类型定义
TaskType = Literal["describe", "ocr", "ask", "detect", "chart"]


def load_and_resize_image(image_path: str, max_size: int = 1024) -> Tuple[Image.Image, str, str]:
    with Image.open(image_path) as img:
        fmt = img.format or "png"
        mime_type = f"image/{fmt.lower()}"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.mode == "L":
            img = img.convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        return img.copy(), fmt, mime_type


def encode_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """
    将本地图片编码为 base64 格式，并进行适当缩放以适应模型输入。

    Args:
        image_path (str): 图片路径。
        max_size (int): 图片长边的最大像素值，防止图片过大导致模型处理失败。

    Returns:
        str: data URI 格式的字符串 (data:image/{type};base64,{data})。
    """
    img, fmt, mime_type = load_and_resize_image(image_path, max_size)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{img_b64}"


def get_image_content(image_path: str) -> str:
    """
    获取图片内容，URL 直接使用，本地文件编码为 base64。

    Args:
        image_path (str): 图片路径或 URL。

    Returns:
        str: URL 或 base64 data URI。
    """
    if image_path.startswith(("http://", "https://")):
        return image_path
    return encode_image_to_base64(image_path)


def get_task_prompt(
    task: TaskType,
    question: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> str:
    """
    根据任务类型生成对应的 prompt。

    Args:
        task (TaskType): 任务类型。
        question (Optional[str]): 用户自定义问题 (仅用于 ask 任务)。
        custom_prompt (Optional[str]): 用户完全自定义的 prompt。

    Returns:
        str: 任务 prompt。
    """
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompts = {
            "describe": "请详细描述这张图片的内容，包括主要元素、场景、颜色等。",
            "ocr": "请识别图片中的所有文字内容，保持原文格式和排版。",
            "detect": """请检测图片中的所有物体。请以 JSON 格式输出结果，不要包含 markdown 代码块标记。
格式示例：
[
  {"name": "person", "bbox": [100, 200, 300, 400]},
  {"name": "car", "bbox": [500, 600, 700, 800]}
]
注意：bbox 坐标格式为 [xmin, ymin, xmax, ymax]，使用绝对像素坐标。""",
            "chart": "请分析这张图表，提取数据并说明图表的含义。",
        }
        prompt = prompts.get(task, prompts["describe"])

    if task == "ask":
        if not question:
            raise ValueError("--question 参数在 ask 任务中是必需的")
        return question

    if task == "detect" and categories:
        categories_text = "，".join(categories)
        prompt = f"{prompt}\n请仅检测以下类别：{categories_text}。"

    return prompt


def parse_and_save_yolo_detection(
    detection_result: str,
    image_path: str,
    output_dir: Path,
    classes_map: Dict[str, int],
) -> None:
    """
    解析模型返回的目标检测结果，并以 YOLO 格式保存。

    Args:
        detection_result (str): 模型返回的包含边界框的文本。
        image_path (str): 原始图片路径。
        output_dir (Path): YOLO 数据集保存目录。
        classes_map (Dict[str, int]): 类别名称到索引的映射。
    """
    try:
        resized_img, _, _ = load_and_resize_image(image_path)
    except Exception:
        print(f"警告: 无法读取图片 {image_path}，跳过处理。")
        return
    base_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    img = base_img.copy()
    img_height, img_width, _ = base_img.shape

    yolo_labels = []
    
    # 尝试解析 JSON 格式
    try:
        clean_result = (detection_result or "").strip()
        if clean_result:
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean_result, re.IGNORECASE)
            if fence_match:
                clean_result = fence_match.group(1).strip()
            json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", clean_result)
            if json_match:
                clean_result = json_match.group(1).strip()
            clean_result = re.sub(r",\s*([}\]])", r"\1", clean_result)
        data = json.loads(clean_result)
        
        # 兼容列表或字典格式（有些模型可能返回 {"objects": [...]}）
        if isinstance(data, dict):
            data = data.get("objects", []) or data.get("detections", [])
            
        if isinstance(data, list):
            for item in data:
                name = item.get("name") or item.get("label")
                bbox = item.get("bbox") or item.get("box")
                
                if name and bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(float, bbox)
                    
                    # 归一化处理
                    class_id = classes_map.get(name)
                    if class_id is None:
                        class_id = len(classes_map)
                        classes_map[name] = class_id
                        # 更新 classes.txt
                        with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                            for class_name in classes_map:
                                f.write(f"{class_name}\n")

                    # 转换 YOLO 格式
                    dw = 1.0 / img_width
                    dh = 1.0 / img_height
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1

                    x_center_norm = x_center * dw
                    y_center_norm = y_center * dh
                    w_norm = w * dw
                    h_norm = h * dh
                    
                    # 限制在 [0, 1] 范围内
                    x_center_norm = max(0.0, min(1.0, x_center_norm))
                    y_center_norm = max(0.0, min(1.0, y_center_norm))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

                    # 绘图
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except json.JSONDecodeError:
        print("DEBUG: JSON 解析失败，尝试对象逐条提取...")
        obj_texts = re.findall(r"\{[\s\S]*?\}", detection_result)
        for obj_text in obj_texts:
            obj_clean = re.sub(r",\s*}", "}", obj_text)
            try:
                item = json.loads(obj_clean)
            except Exception:
                continue
            name = item.get("name") or item.get("label")
            bbox = item.get("bbox") or item.get("box")
            if not (name and bbox and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = map(float, bbox)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            x1 = max(0.0, min(img_width - 1.0, x1))
            y1 = max(0.0, min(img_height - 1.0, y1))
            x2 = max(0.0, min(img_width - 1.0, x2))
            y2 = max(0.0, min(img_height - 1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = classes_map.get(name)
            if class_id is None:
                class_id = len(classes_map)
                classes_map[name] = class_id
                with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                    for class_name in classes_map:
                        f.write(f"{class_name}\n")
            dw = 1.0 / img_width
            dh = 1.0 / img_height
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            yolo_labels.append(f"{class_id} {x_center*dw:.6f} {y_center*dh:.6f} {w*dw:.6f} {h*dh:.6f}")
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if not yolo_labels:
            print("DEBUG: 对象提取无结果，尝试正则匹配 <box(...)>...")
            detections = re.findall(r"(.+?)\\s*<box\\(([-\\d.]+),\\s*([-\\d.]+),\\s*([-\\d.]+),\\s*([-\\d.]+)\\)>", detection_result)
            for name, x1, y1, x2, y2 in detections:
                name = name.strip()
                if not name:
                    continue
                if name not in classes_map:
                    classes_map[name] = len(classes_map)
                    with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                        for class_name in classes_map:
                            f.write(f"{class_name}\n")
                class_id = classes_map[name]
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                max_val = max(x1, y1, x2, y2)
                if max_val <= 1.5:
                    x1 *= img_width
                    x2 *= img_width
                    y1 *= img_height
                    y2 *= img_height
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                x1 = max(0.0, min(img_width - 1.0, x1))
                y1 = max(0.0, min(img_height - 1.0, y1))
                x2 = max(0.0, min(img_width - 1.0, x2))
                y2 = max(0.0, min(img_height - 1.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                yolo_labels.append(f"{class_id} {x_center*dw:.6f} {y_center*dh:.6f} {w*dw:.6f} {h*dh:.6f}")
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 无论是否有检测结果，都保存文件（前面已添加此逻辑，这里保持一致）

    label_filename = output_dir / "labels" / f"{Path(image_path).stem}.txt"
    label_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(label_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_labels))

    original_image_path = output_dir / "images" / f"{Path(image_path).stem}.jpg"
    original_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(original_image_path), base_img)

    if yolo_labels:
        annotated_image_path = output_dir / "images_annotated" / f"{Path(image_path).stem}.jpg"
        annotated_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_image_path), img)


def analyze_image(
    image_path: str,
    task: TaskType = "describe",
    question: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    categories: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    precise: bool = False,
) -> str:
    """
    调用 ModelScope API 分析图片。

    Args:
        image_path (str): 图片路径或 URL。
        task (TaskType): 任务类型。
        question (Optional[str]): 自定义问题 (ask 任务需要)。
        custom_prompt (Optional[str]): 用户完全自定义的 prompt。
        api_key (Optional[str]): ModelScope API Key。
        model (Optional[str]): 指定模型。
        precise (bool): 是否使用精细模式模型。

    Returns:
        str: 分析结果文本。
    """
    # 配置参数
    api_key = api_key or DEFAULT_API_KEY
    if not api_key:
        raise ValueError(
            "请设置 MODELSCOPE_API_KEY 环境变量或传入 --api-key 参数\n"
            "获取 API Key: https://modelscope.cn/my/myaccesstoken"
        )

    if model:
        model_name = model
    elif precise:
        model_name = DEFAULT_MODEL_PRECISE
    else:
        model_name = DEFAULT_MODEL

    # 获取图片内容
    image_content = get_image_content(image_path)

    # 获取任务 prompt
    prompt = get_task_prompt(task, question, custom_prompt, categories)

    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
    )

    # 调用 API
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_content}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.1,  # 降低温度，让输出更确定
            max_tokens=2048,
        )
        content = response.choices[0].message.content or ""
        # 调试输出：打印前100个字符，确认模型回复了什么
        print(f"DEBUG: 模型原始回复(前100字符): {content[:100].replace(chr(10), ' ')}...")
        return content
    except Exception as e:
        print(f"API 调用错误: {e}")
        return ""


def process_video(
    video_path: str,
    output_dir: Path,
    frame_interval: int,
    task: TaskType,
    args: argparse.Namespace,
    classes_map: Dict[str, int],
) -> None:
    """
    处理视频文件，抽帧并进行分析。

    Args:
        video_path (str): 视频文件路径。
        output_dir (Path): 输出目录。
        frame_interval (int): 抽帧间隔（秒）。
        task (TaskType): 要执行的任务。
        args (argparse.Namespace): 命令行参数。
        classes_map (Dict[str, int]): 类别映射。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * frame_interval)
    frame_count = 0
    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始处理视频: {video_path}, FPS: {fps}, 每 {frame_interval} 秒抽一帧...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = temp_dir / f"frame_{uuid.uuid4()}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            print(f"  - 正在分析第 {frame_count // fps} 秒的帧...")
            try:
                result = analyze_image(
                    image_path=str(frame_filename),
                    task=task,
                    question=args.question,
                    custom_prompt=args.prompt,
                    categories=args.categories,
                    api_key=args.api_key,
                    model=args.model,
                    precise=args.precise,
                )
                if task == "detect":
                    raw_dir = output_dir / "raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_file = raw_dir / f"{Path(frame_filename).stem}.txt"
                    raw_file.write_text(result, encoding="utf-8")
                    parse_and_save_yolo_detection(result, str(frame_filename), output_dir, classes_map)
                else:
                    result_file = output_dir / f"frame_{frame_count // fps}.txt"
                    result_file.write_text(result, encoding="utf-8")

            except Exception as e:
                print(f"    处理帧时出错: {e}")

        frame_count += 1

    cap.release()
    # 清理临时帧文件
    for f in temp_dir.glob("*.jpg"):
        f.unlink()
    temp_dir.rmdir()
    print("视频处理完成。")


def process_image(
    image_path: str,
    output_dir: Path,
    task: TaskType,
    args: argparse.Namespace,
    classes_map: Dict[str, int],
) -> None:
    """
    处理单个图片文件。

    Args:
        image_path (str): 图片文件路径。
        output_dir (Path): 输出目录。
        task (TaskType): 要执行的任务。
        args (argparse.Namespace): 命令行参数。
        classes_map (Dict[str, int]): 类别映射。
    """
    print(f"开始处理图片: {image_path}...")
    try:
        result = analyze_image(
            image_path=image_path,
            task=task,
            question=args.question,
            custom_prompt=args.prompt,
            categories=args.categories,
            api_key=args.api_key,
            model=args.model,
            precise=args.precise,
        )
        if task == "detect":
            raw_dir = output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"{Path(image_path).stem}.txt"
            raw_file.write_text(result, encoding="utf-8")
            parse_and_save_yolo_detection(result, image_path, output_dir, classes_map)
            print(f"检测完成，结果已保存到: {output_dir}")
        elif args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result, encoding="utf-8")
            print(f"结果已保存到: {args.output}")
        else:
            print(result)
    except Exception as e:
        print(f"处理图片时出错: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="MS-Qwen-VL CLI - 使用 OpenAI SDK 调用 ModelScope Qwen-VL 多模态 API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
任务类型:
  describe   图像描述 (默认)
  ocr        OCR 文字识别
  ask        视觉问答 (需要 --question 参数)
  detect     目标检测 (可生成 YOLO 数据集)
  chart      图表解析

示例:
  # 图像描述
  python scripts/ms_qwen_vl.py image.jpg

  # 批量处理文件夹中的图片并生成 YOLO 数据集
  python scripts/ms_qwen_vl.py ./images_folder/ --task detect --output_dir ./yolo_dataset

  # 视频抽帧检测并生成YOLO数据集 (每5秒一帧)
  python scripts/ms_qwen_vl.py video.mp4 --task detect --output_dir ./yolo_dataset --frame_interval 5

  # 视觉问答
  python scripts/ms_qwen_vl.py image.jpg --task ask --question "图片里有什么？"

  # 使用自定义 prompt 进行检测
  python scripts/ms_qwen_vl.py image.jpg --task detect --prompt "请检测图中的猫和狗" --output_dir ./cat_dog_dataset

  # 输出到文件
  python scripts/ms_qwen_vl.py image.jpg --task ocr --output result.txt
        """,
    )

    parser.add_argument(
        "input_path",
        help="要处理的图片、视频、文件夹路径或图片 URL",
    )

    parser.add_argument(
        "--task",
        "-t",
        default="describe",
        choices=["describe", "ocr", "ask", "detect", "chart"],
        help="任务类型",
    )

    parser.add_argument(
        "--question",
        help="ask 任务中的问题",
    )

    parser.add_argument(
        "--prompt",
        help="自定义 prompt，将覆盖默认模板",
    )

    parser.add_argument(
        "--output",
        help="将结果保存到指定文件（非 detect 任务）",
    )

    parser.add_argument(
        "--output_dir",
        help="YOLO 数据集输出目录（detect 任务需要）",
    )

    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="视频抽帧间隔（秒）",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        help="指定检测类别（detect 任务）",
    )

    parser.add_argument(
        "--api-key",
        help="ModelScope API Key（也可通过环境变量 MODELSCOPE_API_KEY 配置）",
    )

    parser.add_argument(
        "--model",
        help="指定使用的模型（默认使用环境变量或内置模型）",
    )

    parser.add_argument(
        "--precise",
        action="store_true",
        help="使用精细模式模型",
    )

    args = parser.parse_args()

    # 统一环境变量名称（兼容多种命名）
    global DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_MODEL_PRECISE
    DEFAULT_API_KEY = args.api_key or os.getenv("MODELSCOPE_API_KEY") or DEFAULT_API_KEY
    DEFAULT_MODEL = args.model or os.getenv("MODELSCOPE_MODEL") or DEFAULT_MODEL
    DEFAULT_MODEL_PRECISE = os.getenv("MODELSCOPE_MODEL_PRECISE") or DEFAULT_MODEL_PRECISE

    input_path = args.input_path
    output_dir = args.output_dir

    if args.task == "detect" and not output_dir:
        raise ValueError("detect 任务必须提供 --output_dir 参数，用于保存 YOLO 数据集")

    classes_map: Dict[str, int] = {}

    p = Path(input_path)
    if p.is_dir():
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(p.glob(ext))
        total = len(files)
        print(f"检测到 {total} 个文件，开始批量处理...")
        for idx, f in enumerate(sorted(files)):
            print(f"--- [{idx+1}/{total}] ---")
            process_image(str(f), Path(output_dir or "."), args.task, args, classes_map)
        print("文件夹处理完成。")
    elif p.is_file():
        if re.search(r"\.(mp4|avi|mov|mkv)$", input_path, re.IGNORECASE):
            process_video(input_path, Path(output_dir or "."), args.frame_interval, args.task, args, classes_map)
        else:
            process_image(input_path, Path(output_dir or "."), args.task, args, classes_map)
    else:
        # 输入可能是 URL
        if input_path.startswith(("http://", "https://")):
            # 对 URL 图片按文件处理
            process_image(input_path, Path(output_dir or "."), args.task, args, classes_map)
        else:
            raise FileNotFoundError(f"未找到输入路径: {input_path}")


if __name__ == "__main__":
    main()
