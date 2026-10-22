<h1 align="center">LLM Skills</h1>

<p align="center">
  <strong>WdBlink 个人开发或维护的 LLM / Agent Skills 聚合仓库。</strong><br>
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

本仓库用于统一管理个人开发或维护的 LLM / Agent 技能。所有技能按 `skills/<name>/` 组织，并用 `skills.yaml` 作为机器可读清单。

本仓库包含：

- `skills.yaml`：技能索引与展示元数据。
- `skills/`：每个技能的安装镜像目录，目录名尽量与 `SKILL.md` frontmatter 的 `name` 一致。
- `skills/<name>/README.md`：面向人的技能用法说明，包含 version tag、安装和使用示例。
- `scripts/validate_skills.py`：校验清单、`SKILL.md` 与技能 README 是否一致。
- `scripts/render_readme.py`：从 `skills.yaml` 生成本 README。
- `scripts/inventory_skills.py`：只读盘点安装来源、符号链接和已审查的入口哈希。
- [来源与收录说明](docs/skill-collection-2026-10-23.md)：说明本地维护、第三方来源和待核实的区别。

## 技能列表

> **24 个技能**

| | 英文名 | 中文名 | 描述 |
|---|---|---|---|
| **知识整理** | | | |
| ![Open](https://img.shields.io/badge/Open-green) | [`academic-writing-storytelling`](skills/academic-writing-storytelling) | [学术写作讲故事框架](skills/academic-writing-storytelling) | 学术写作讲故事的四派综合框架——把"讲故事"拆成6个可操作心智模型与12步工作流。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`annie-duke-perspective`](skills/annie-duke-perspective) | [Annie Duke 对赌思维](skills/annie-duke-perspective) | 用 Annie Duke 的对赌思维做决策顾问——把决策质量与结果质量分开，强制概率式叙述与责任性预测。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`clean-code-refactoring-framework`](skills/clean-code-refactoring-framework) | [行为保持式代码重构](skills/clean-code-refactoring-framework) | 用书籍提炼的方法诊断坏味道，执行小步重构并验证行为保持。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`daily-todo-planner`](skills/daily-todo-planner) | [每日待办计划器](skills/daily-todo-planner) | 将杂乱每日描述转成可复盘学习、可提前预研的时间块计划。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`experience-distill`](skills/experience-distill) | [任务经验提炼](skills/experience-distill) | 从任务经历提炼有证据、有边界、可供下次任务复用的经验。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`good-benchmarks`](skills/good-benchmarks) | [Benchmark 设计与审计](skills/good-benchmarks) | 按场景选择 benchmark 形态与成功依据，交付可复用数据和评分器，验证可解性与比较公平性。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`human-understanding`](skills/human-understanding) | [结果理解与证据解释](skills/human-understanding) | 以明确主语、固定术语和分步示例解释复杂概念与结果，按需制作图解和交互页面。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`karpathy-autoresearch-adapter`](skills/karpathy-autoresearch-adapter) | [Karpathy 自动研究适配器](skills/karpathy-autoresearch-adapter) | 将现有仓库改造成以真实评价器为核心的自动研究协议项目。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`mind-map`](skills/mind-map) | [项目思维导图](skills/mind-map) | 将项目判断记录为 Obsidian 双链节点、紧凑 schema 索引、图谱中性的时间线/溯源和可交互 HTML 图谱。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`paper-deconstruction`](skills/paper-deconstruction) | [论文精读](skills/paper-deconstruction) | 用结构-故事-拆解-评价-视觉化五阶法把一篇论文精读成带配图的可视化笔记。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`project-handoff`](skills/project-handoff) | [项目交接](skills/project-handoff) | 核对项目资产与历史，经独立审查生成可单独阅读的项目交接文档。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`research-context`](skills/research-context) | [研究任务上下文](skills/research-context) | 以程序冻结证据、模型独立审查和受限修订，持续维护可恢复的研究上下文。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`template-writing`](skills/template-writing) | [模板写作](skills/template-writing) | 根据合同、技术协议和项目资料撰写并填充固定格式 DOCX 模板。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`thinking-in-bets`](skills/thinking-in-bets) | [对赌决策工作流](skills/thinking-in-bets) | 从《对赌》提炼35个工具与七条工作流，按情境自动辅助选择、信念校准、复盘和风险规划。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`transcript-source-compiler`](skills/transcript-source-compiler) | [转写来源编译器](skills/transcript-source-compiler) | 将大段演讲转写编译成可追溯的来源底稿和可读文章。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`weekly-report`](skills/weekly-report) | [中文周报生成器](skills/weekly-report) | 从工作日志生成可迁移的中文工作周报 DOCX。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`workflow-skill-creator`](skills/workflow-skill-creator) | [工作流 Skill 创作](skills/workflow-skill-creator) | 固化重复操作、参数化任务差异，按需由当前宿主判断，并验证交付质量与实际成本。 |
| **多模态** | | | |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`auto-yolo-dataset`](skills/auto-yolo-dataset) | [自动 YOLO 数据集](skills/auto-yolo-dataset) | 通过本地或远程检测器生成 YOLO、COCO、VOC 和 Label Studio 数据集资产。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`glm-ocr-pdf2md`](skills/glm-ocr-pdf2md) | [本地GLM-OCR扫描PDF转Markdown](skills/glm-ocr-pdf2md) | 本地ZhipuAI/GLM-OCR将扫描版PDF转为带目录和页码标记的Markdown书籍。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`ms-qwen-vl`](skills/ms-qwen-vl) | [魔搭 Qwen-VL 视觉解析](skills/ms-qwen-vl) | 用 Qwen-VL 解析图片/视频，并可输出 YOLO 数据集。 |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`photographer-style`](skills/photographer-style) | [摄影师风格拆解](skills/photographer-style) | 拆解摄影风格，交付介绍、参考图与双模式提示词，并按请求归档到 GitHub 私有仓库。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`resource2skill-distiller`](skills/resource2skill-distiller) | [资源方法技能提炼器](skills/resource2skill-distiller) | 调用本机 Codex 提炼网页、文本与视频证据，交付保留来源和验证记录的候选技能。 |
| ![Open](https://img.shields.io/badge/Open-green) | [`social-cover-maker`](skills/social-cover-maker) | [社交平台封面生成器](skills/social-cover-maker) | 基于 Open Design 风格系统为公众号、小红书、知乎等平台生成可校验的 SVG/PNG 封面。 |
| **演示文稿** | | | |
| ![Reserved](https://img.shields.io/badge/Rights_reserved-blueviolet) | [`pv-forecast-pptx`](skills/air-pptx) | [光伏预测汇报 PPTX](skills/air-pptx) | 基于模板资源生成光伏与能源预测研究汇报 PPTX。 |

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
    ├── academic-writing-storytelling/
    ├── air-pptx/
    ├── annie-duke-perspective/
    ├── auto-yolo-dataset/
    ├── clean-code-refactoring-framework/
    ├── daily-todo-planner/
    ├── experience-distill/
    ├── glm-ocr-pdf2md/
    ├── good-benchmarks/
    ├── human-understanding/
    ├── karpathy-autoresearch-adapter/
    ├── mind-map/
    ├── ms-qwen-vl/
    ├── paper-deconstruction/
    ├── photographer-style/
    ├── project-handoff/
    ├── research-context/
    ├── resource2skill-distiller/
    ├── social-cover-maker/
    ├── template-writing/
    ├── thinking-in-bets/
    ├── transcript-source-compiler/
    ├── weekly-report/
    └── workflow-skill-creator/
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

各技能的许可见 `skills.yaml` 和各自目录内的说明。标记为 `Reserved` / `Proprietary` 的技能保留权利；仓库公开可见不表示授予开源许可。第三方方法与素材的归属见各技能来源说明。
