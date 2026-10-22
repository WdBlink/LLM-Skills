# Resource2Skill Distiller

![Version](https://img.shields.io/badge/version-0.3.0-CC785C)

使用本机 Codex CLI，将网页正文、文本或视频抽帧提炼为候选 Skill；保留证据、实施补充和真实测试记录。Resource2Skill 提供可选领域提示词。

## 环境

需要 Python 3.11 和已安装、登录的 Codex CLI。通过 `codex login status` 检查；认证由 Codex 管理，脚本不读取或复制凭据。默认采用本机基础配置的模型与推理强度；本机 MCP、插件配置和 shell 工具不用于提炼子调用。

使用上游领域时，需要已有的 `~/.local/share/resource2skill`（或 `RESOURCE2SKILL_HOME`）及它的 Python 虚拟环境。默认工作方法提示词和 Codex 调用只用 Python 标准库。视频获取另外使用 yt-dlp，抽帧使用 ffmpeg/ffprobe。

首次安装本包装包到个人技能目录，已有同名包时先比较：

```bash
test ! -e ~/.codex/skills/resource2skill-distiller && \
  cp -R skills/resource2skill-distiller ~/.codex/skills/resource2skill-distiller
```

## 使用

```text
$resource2skill-distiller
把这两篇网页的方法提炼为候选技能，保留来源，使用新任务验证。
链接：……
```

程序化调用已保存的网页正文：

```bash
python3.11 ~/.codex/skills/resource2skill-distiller/scripts/analyze_resource.py \
  --text-file /absolute/path/page.txt \
  --source-url 'https://example.org/article' \
  --goal '提炼可迁移的研究方法' \
  --output-dir /absolute/path/new-run
```

完整流程见 [SKILL.md](SKILL.md)，视频见 [输入说明](references/bilibili.md)。每次输出目录必须不存在。单次调用默认 900 秒超时，失败保留记录；无自动模型回退或无限重试。

## 检查

```bash
python3.11 skills/resource2skill-distiller/scripts/test_adapter.py
```

这是离线逻辑检查；真实 Codex 调用和生成技能内容另行验收。0.3.0 替换了 0.2.0 的 MiniMax 调用，视频改为显式抽帧加已有文本；旧 API 密钥、视频上传限额和服务端文件清理流程不再适用于当前入口。历史验证记录保留原样。

依据：[Codex 非交互调用](https://learn.chatgpt.com/docs/non-interactive-mode)。本包装技能采用 MIT；上游 Resource2Skill 来源与许可保留在独立仓库中。
