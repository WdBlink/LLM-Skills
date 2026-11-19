# AutoResearch Skill 创建器

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

从 [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) 完整克隆源码和 Git 历史，按目标改造为领域专用 Codex Skill。

## 安装到 Codex

需要 Python 3、Git 和可访问 GitHub 的网络。在本仓库根目录执行。

```bash
mkdir -p ~/.codex/skills/autoresearch-skill-creator
rsync -a skills/autoresearch-skill-creator/ ~/.codex/skills/autoresearch-skill-creator/
```

## 使用

```text
用 $autoresearch-skill-creator 按 AutoResearch 范式创建一个设备验收 Skill。
需要生成可打印的验收报告，并从 Codex 入口用一份样例验收记录验证交付流程。
```

脚本在新的开发目录保留上游历史，并用 `refs/autoresearch/base` 记录基线。默认基线和使用步骤见 [SKILL.md](SKILL.md)。工作流设计与验收见 [设计指南](references/workflow-design.md)，许可见 [LICENSE](LICENSE)。
