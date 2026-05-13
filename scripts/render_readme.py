#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict

from skill_index import ROOT, load_manifest


CATEGORY_ZH = {
    "Knowledge": "知识整理",
    "Multimodal": "多模态",
    "Developer Tools": "开发工具",
    "Presentations": "演示文稿",
}


def render_badge(license_name: str) -> str:
    if license_name.lower() == "proprietary":
        return "![Private](https://img.shields.io/badge/Private-blueviolet)"
    return "![Open](https://img.shields.io/badge/Open-green)"


def render() -> str:
    manifest = load_manifest()
    skills = list(manifest["skills"])
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for skill in skills:
        by_category[str(skill["category"])].append(skill)

    lines = [
        '<h1 align="center">LLM Skills</h1>',
        "",
        '<p align="center">',
        "  <strong>WdBlink 个人开发的 LLM / Agent Skills 聚合仓库。</strong><br>",
        "  <sub>面向 Codex、Claude Code、OpenClaw 等本地 Agent 工作流的可安装技能镜像。</sub>",
        "</p>",
        "",
        '<p align="center">',
        '  <a href="#技能列表">技能列表</a> ·',
        '  <a href="#仓库结构">仓库结构</a> ·',
        '  <a href="#安装">安装</a> ·',
        '  <a href="#维护方式">维护方式</a>',
        "</p>",
        "",
        "---",
        "",
        "## 这是什么",
        "",
        "本仓库模仿 `lovstudio/general-skills` 的组织方式，将个人开发或维护的技能统一放入 `skills/<name>/`，并用 `skills.yaml` 作为机器可读清单。",
        "",
        "本仓库包含：",
        "",
        "- `skills.yaml`：技能索引与展示元数据。",
        "- `skills/`：每个技能的安装镜像目录，目录名尽量与 `SKILL.md` frontmatter 的 `name` 一致。",
        "- `scripts/validate_skills.py`：校验清单与 `SKILL.md` 是否一致。",
        "- `scripts/render_readme.py`：从 `skills.yaml` 生成本 README。",
        "",
        "## 技能列表",
        "",
        f"> **{len(skills)} 个技能**",
        "",
        "| | 英文名 | 中文名 | 描述 |",
        "|---|---|---|---|",
    ]

    for category in sorted(by_category):
        lines.append(f"| **{CATEGORY_ZH.get(category, category)}** | | | |")
        for skill in sorted(by_category[category], key=lambda item: str(item["name"])):
            name = str(skill["name"])
            path = str(skill["path"])
            zh = str(skill["name_zh"])
            tagline = str(skill["tagline_zh"])
            badge = render_badge(str(skill["license"]))
            lines.append(f"| {badge} | [`{name}`]({path}) | [{zh}]({path}) | {tagline} |")

    lines.extend(
        [
            "",
            "<sub>上表由 `scripts/render_readme.py` 从 `skills.yaml` 生成。请编辑 `skills.yaml`，不要手动改表格。</sub>",
            "",
            "## 仓库结构",
            "",
            "```text",
            "LLM-Skills/",
            "├── README.md",
            "├── skills.yaml",
            "├── scripts/",
            "│   ├── render_readme.py",
            "│   └── validate_skills.py",
            "└── skills/",
            "    ├── claude-code/",
            "    ├── ms-qwen-vl/",
            "    ├── pv-forecast-pptx/",
            "    └── transcript-source-compiler/",
            "```",
            "",
            "## 安装",
            "",
            "### 安装单个技能到 Codex",
            "",
            "```bash",
            "mkdir -p ~/.codex/skills/<skill-name>",
            "rsync -a skills/<skill-name>/ ~/.codex/skills/<skill-name>/",
            "```",
            "",
            "例如：",
            "",
            "```bash",
            "mkdir -p ~/.codex/skills/transcript-source-compiler",
            "rsync -a skills/transcript-source-compiler/ ~/.codex/skills/transcript-source-compiler/",
            "```",
            "",
            "### 安装全部技能到 Codex",
            "",
            "```bash",
            "mkdir -p ~/.codex/skills",
            "rsync -a skills/ ~/.codex/skills/",
            "```",
            "",
            "## 维护方式",
            "",
            "新增或调整技能后运行：",
            "",
            "```bash",
            "python3 scripts/validate_skills.py",
            "python3 scripts/render_readme.py",
            "```",
            "",
            "如果 README 有变更，请提交生成后的 `README.md`。",
            "",
            "## 许可证",
            "",
            "各技能的许可见 `skills.yaml` 和各自目录内的说明。标记为 `Private` / `Proprietary` 的技能仅作为个人工作流镜像管理。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    (ROOT / "README.md").write_text(render(), encoding="utf-8")
    print("README.md rendered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
