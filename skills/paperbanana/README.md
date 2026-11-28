# PaperBanana

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)

通过 Agent 对话制作论文主图，包括方法示意图、系统架构图、pipeline 和原图局部修订。
用户提供材料、讨论构图并反馈修改；Agent 完成参考选择、规划、风格整理、实际生图和看图审查。
本 Skill 只有后端脚本，不提供浏览器页面、桌面客户端或独立聊天界面。

## 使用

```text
/paperbanana 根据 methods.md 画论文主图，突出训练与推理的区别，先讨论构图。
```

```text
用 paperbanana 修改这张主图，只把右下角箭头指向 Validator，保留画风、布局和其他内容。
```

```text
给这个方法画一张双栏论文主图，参考 references/ 的图片，减少文字和公式，输出 PNG 和 caption。
```

最终交付原始图片、所需 PNG/TIFF 导出、caption，以及来源、提示词、版本、工具回执和审图记录。
只有查看实际图片后才能判断主图是否可用。测量曲线与精确数值图交给真实数据 renderer。

## 安装

从仓库根目录将 `skills/paperbanana/` 复制到宿主的技能目录，例如
`~/.claude/skills/paperbanana/` 或 `~/.codex/skills/paperbanana/`。目标已存在时先检查，
不要直接覆盖另一份安装。也可将宿主目录链接到本仓库，实现单一维护源。

需要 Python 3.10+，图像处理依赖 Pillow：

```bash
python3 -m pip install -r skills/paperbanana/requirements.txt
```

生图需要宿主已有的图像工具或可用的本地 Magpie image MCP。Skill 不携带模型，
不要求新建 Gemini key，也不会安装上游 PaperBanana。当前会话无法发现刚安装技能时，
新开会话或让 Agent 直接读取本目录的 `SKILL.md`。

## 后端

| 入口 | 用途 |
| --- | --- |
| `scripts/paperbanana.py` | 候选图片归档、审图记录、选版、无损导出与文件一致性检查 |
| `scripts/magpie_image.py` | 可选的 Magpie stdio MCP 传输，同进程生成与取回异步结果 |
| `references/workflow.md` | 参考、构图、风格、生图和视觉修订流程 |
| `references/artifact-contract.md` | 文件契约和 CLI 用法 |

脚本由 Agent 调用，用户不必手工编写 JSON。运行时目录放在用户项目中，不写入 Skill 目录。
Magpie 可通过 `--request` 中的 `executable` 或 `PAPERBANANA_MAGPIE_BIN` 指定；
先发现真实 schema，再调用工具。不要把某次会话中的模型名和绝对路径当成默认配置。

## 验证

```bash
python3 -m unittest discover -s skills/paperbanana/tests -p 'test_*.py'
python3 scripts/validate_skills.py
```

测试中的本地 fixture 和模拟 MCP 只验证文件与传输管线。真实生图、局部编辑和视觉质量
需要单独执行并记录证据。PNG/TIFF 导出不放大像素，不把栅格转换包装成矢量化。

## 与 autoresearch-paper 的关系

可独立运行，不依赖研究 Controller。调用方也可把方法图规格、来源和 caption 交给本 Skill，
接收选中图片与记录。研究流程继续负责预算、WorkOrder、要求绑定、论文插入及独立审查。

## 来源与许可

MIT。复制自 Autoresearch 的图像产物实现保留其 MIT 许可声明。
流程借鉴 PaperBanana 的五个角色，未打包上游引擎或参考图数据集。
详细来源见 [provenance.md](references/provenance.md)。
