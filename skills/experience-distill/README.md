# Experience Distill

![Version](https://img.shields.io/badge/version-0.1.0-CC785C)
![License](https://img.shields.io/badge/license-MIT-green)

从已完成、失败或暂停的任务中，提炼有证据、有适用条件和反例边界的经验，供下一项任务复用。

## 安装

需要 Node.js，执行以下命令安装到 Codex：

```bash
npx -y skills add WdBlink/LLM-Skills --skill experience-distill --agent codex --global --yes
```

或在本仓库根目录手动安装：

```bash
mkdir -p ~/.codex/skills/experience-distill
rsync -a skills/experience-distill/ ~/.codex/skills/experience-distill/
```

## 使用

```text
用 $experience-distill 复盘本次任务，结合运行记录提炼成功与失败的经验，
写清证据、适用边界和待验证项，并为下一次同类任务生成短备忘录。
```

提供相关会话、任务材料或日志路径即可。输出围绕“触发条件 → 操作 → 证据 → 不适用边界”组织，可按要求保存为 Markdown。

## 内容与边界

- `SKILL.md`：取证、提炼、反证与复用流程。
- `references/examples.md`：条件化规则、反例与备忘录样式。
- `agents/openai.yaml`：Codex 展示及调用配置。

纯文本技能，无额外运行依赖。只处理可访问且与任务有关的证据；不会自动读取所有历史会话、重跑原任务或修改全局规则。模板中的历史案例不构成新任务的验证证据。
