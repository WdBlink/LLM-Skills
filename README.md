<h1 align="center">LLM Skills</h1>

<p align="center">
  <strong>WdBlink 个人开发的 LLM / Agent Skills 聚合仓库。</strong><br>
  <sub>面向 Codex、Claude Code、OpenClaw 等本地 Agent 工作流的可安装技能镜像。</sub>
</p>

<p align="center">
  <a href="#技能列表">技能列表</a> ·
  <a href="#仓库结构">仓库结构</a> ·
  <a href="#安装">安装</a> ·
  <a href="#维护方式">维护方式</a>
</p>

---

## 这是什么

本仓库模仿 `lovstudio/general-skills` 的组织方式，将个人开发或维护的技能统一放入 `skills/<name>/`，并用 `skills.yaml` 作为机器可读清单。

本仓库包含：

- `skills.yaml`：技能索引与展示元数据。
- `skills/`：每个技能的安装镜像目录，目录名尽量与 `SKILL.md` frontmatter 的 `name` 一致。
- `scripts/validate_skills.py`：校验清单与 `SKILL.md` 是否一致。
- `scripts/render_readme.py`：从 `skills.yaml` 生成本 README。

## 技能列表

> **4 个技能**

| | 英文名 | 中文名 | 描述 |
|---|---|---|---|
| **开发工具** | | | |
| ![Open](https://img.shields.io/badge/Open-green) | [`claude-code`](skills/claude-code) | [Claude Code 集成](skills/claude-code) | 查询 Claude Code 文档与工作流，辅助 OpenClaw 编程任务。 |
| **知识整理** | | | |
| ![Open](https://img.shields.io/badge/Open-green) | [`transcript-source-compiler`](skills/transcript-source-compiler) | [转写来源编译器](skills/transcript-source-compiler) | 将大段演讲转写编译成可追溯的来源底稿和可读文章。 |
| **多模态** | | | |
| ![Open](https://img.shields.io/badge/Open-green) | [`ms-qwen-vl`](skills/ms-qwen-vl) | [魔搭 Qwen-VL 视觉解析](skills/ms-qwen-vl) | 用 Qwen-VL 解析图片/视频，并可输出 YOLO 数据集。 |
| **演示文稿** | | | |
| ![Private](https://img.shields.io/badge/Private-blueviolet) | [`pv-forecast-pptx`](skills/pv-forecast-pptx) | [光伏预测汇报 PPTX](skills/pv-forecast-pptx) | 为光伏预测课题生成结构化科研汇报 PPT。 |

<sub>上表由 `scripts/render_readme.py` 从 `skills.yaml` 生成。请编辑 `skills.yaml`，不要手动改表格。</sub>

## 仓库结构

```text
LLM-Skills/
├── README.md
├── skills.yaml
├── scripts/
│   ├── render_readme.py
│   └── validate_skills.py
└── skills/
    ├── claude-code/
    ├── ms-qwen-vl/
    ├── pv-forecast-pptx/
    └── transcript-source-compiler/
```

## 安装

### 安装单个技能到 Codex

```bash
mkdir -p ~/.codex/skills/<skill-name>
rsync -a skills/<skill-name>/ ~/.codex/skills/<skill-name>/
```

例如：

```bash
mkdir -p ~/.codex/skills/transcript-source-compiler
rsync -a skills/transcript-source-compiler/ ~/.codex/skills/transcript-source-compiler/
```

### 安装全部技能到 Codex

```bash
mkdir -p ~/.codex/skills
rsync -a skills/ ~/.codex/skills/
```

## 维护方式

新增或调整技能后运行：

```bash
python3 scripts/validate_skills.py
python3 scripts/render_readme.py
```

如果 README 有变更，请提交生成后的 `README.md`。

## 许可证

各技能的许可见 `skills.yaml` 和各自目录内的说明。标记为 `Private` / `Proprietary` 的技能仅作为个人工作流镜像管理。
