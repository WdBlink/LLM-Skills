# Skill 来源盘点与收录（2026-10-23）

本次从 GitHub `main` 的 `d7b17e6` 出发，新增 6 个自建或本地维护技能，仓库由 18 个技能增至 24 个。这里的“本地维护”不等于已证明全部内容从零原创；方法、上游代码与素材的来源分别保留。逐项入口快照见 [skill-provenance.json](skill-provenance.json)。

## 如何区分来源

按证据检查安装记录中的来源仓库、符号链接真实指向、包内作者/许可证/来源声明、用户源仓库及 Git 状态。没有安装记录不代表自建；个人仓库也可能是 fork。安装 lock 只能按名称匹配，不能证明同名本地副本未被替换；脚本将这种证据标为 `name_only_not_content_verified`。`SKILL.md` 哈希只用于识别同一入口版本，不能证明作者或附件权利。

本次主要来源审查覆盖 `~/.codex/skills` 的 166 个顶层技能；另扫描 `.agents/skills` 和 `.claude/skills` 的安装副本。未遍历所有项目私有技能或所有插件缓存，也未读取无关聊天历史。因此这是可复核的候选定位，不是完整的作者身份认证。详细个人路径和全量安装清单保留在本机，不上传。

其中 55 项属于 poteto 的 [P-Stack（cursor/plugins/pstack）](https://github.com/cursor/plugins/tree/main/pstack) 安装适配；其余有第三方署名、用户仓库维护、本地定制、上游未定位和未知等情况。用户已确认部分女娲生成器署名来自模板，不能只凭这个尾注断言是下载品。本轮据此收录所询问的 clean-code-refactoring-framework，其他同署名包仍逐项核验。

从仓库根目录重新盘点：

```bash
python3 scripts/inventory_skills.py \
  --package "poteto-pstack=$HOME/.codex/pstack/skills" \
  --catalog docs/skill-provenance.json \
  --out .local-audit/installed-skills.json
```

`--package` 指向实际安装位置；未安装该包时省略。`--root` 可重复指定，默认扫描三个全局目录。别名按真实路径去重，物理副本分别保留；旧目录不等于仍在当前宿主启用。盘点脚本不会安装、卸载、同步或删除技能。输出含个人路径，应留在已忽略的 `.local-audit/`。

## 本次收录

| Skill | 收录理由 | 来源与边界 |
| --- | --- | --- |
| human-understanding | 将复杂解释拆成明确概念、证据和可检验的阅读结果，含 46 个离线测试 | 本地工作流与安装副本一致；作者身份仍以来源记录为准 |
| project-handoff | 从证据生成可独立阅读的项目交接，包含规划、评审、受限修订和恢复 | 保留 KSI 方法参考，区分本地实现与论文效果 |
| research-context | 长期科研上下文的冻结输入、受限修订和故障恢复已有程序实现 | README 声明原创技能/模板 MIT；外部资料另保留出处 |
| resource2skill-distiller | 已有资源提炼与隔离 Codex 调用入口，可从任意工作目录使用 | 保留 Resource2Skill 方法来源及外部资源边界 |
| photographer-style | 摄影风格研究、真实参考与双模式提示词、按请求归档形成完整工作流 | 分发版移除个人仓库常量及历史人物授权，保留方法来源和权利限制 |
| clean-code-refactoring-framework | 行为保持式重构、坏味道诊断与验证案例完整 | 用户确认通过工具生成；区分本地派生、女娲工具署名和原书作者，不包含整书 |

`experience-distill`、`good-benchmarks`、`thinking-in-bets` 和 `workflow-skill-creator` 原本已在远端 main，不计为新发现。原工作目录位于较早的开发分支，未跟踪文件并不代表未上传。归并使用独立 worktree，原目录及安装副本保持原状。

## 本次优化

- `workflow-skill-creator` 0.3.0：同步已安装的程序推进、局部恢复与实现责任边界；按 P-Stack 的实际 Codex adapter 读取角色配置，移除作者机器路径。增加 [Promptfoo 可选接线](../skills/workflow-skill-creator/references/promptfoo.md)，保持程序测试、语义判断与模型效果评估分开。
- `ms-qwen-vl` 0.1.1：provider、地址和密钥成组选择，阻止 ModelScope/SiliconFlow 误配；补充离线回归并替换不存在的文档引用。
- `weekly-report` 0.1.1：按 provider 成组选择凭证、地址和模型；自定义配置不完整时不请求服务。
- `mind-map` 0.2.14：移除其他用户的硬编码 vault 默认值；无明确配置或配置损坏时在写入前报错。
- `photographer-style` 0.6.1：从当前用户配置读取归档目标与操作授权，不沿用某次历史任务的授权。
- README 的 `Private` 徽章改为 `Reserved`。公开 GitHub 仓库中的 `Proprietary` 表示保留权利，不表示文件私密；没有给这些技能追加 MIT 许可。

P-Stack 负责工程执行方式，领域技能负责领域方法，Promptfoo 负责按配置执行评测；三者不是所有任务必须同时加载的依赖。

## 暂缓项

| 候选 | 暂缓原因 |
| --- | --- |
| project-cognition-compiler | 文档与程序有价值，但未找到足够作者来源记录；保留候选与本地原件 |
| iphone-film-recipe | 有完整 QR 与配方流程；系列进度、分享链接和附件应先分离，附件来源需逐一核验 |
| deli-autoresearch、research-to-paper | 明确有本地改编，但上游许可或内置 vendor 身份/授权未核实，不整包搬入 |
| token-firewall-team、investment-memo-craft | 功能有价值，作者与源项目关系仍缺足够证据 |
| skill-atlas-crawler、skill-atlas-extractor | 有独立项目源目录，但当前入口依赖该项目布局，不作为独立可安装包复制 |
| nas-web-publish | 含私人网络配置，未原样上传 |
| 旧 Mavis autoresearch-paper | 本地旧分支与当前独立 runtime 已分化，不能以同名为由覆盖或重加旧系统 |

P-Stack、ctx、huashu、lovstudio 以及有明确第三方来源的科学技能保留上游归属。已经有独立仓库的项目入口优先在源项目维护，不复制整个运行时形成第二套真相。

## 验证及限制

```bash
python3 scripts/validate_skills.py
python3 scripts/render_readme.py
python3 scripts/check_collected_skills.py
```

离线检查覆盖新增包程序合同、来源哈希/别名盘点、密钥路由、vault 选择、错误输入、旧判断拒绝和恢复。Promptfoo 的保存输出回放及评分自检见其说明；当前本机没有找到 Promptfoo CLI，不能声称执行过完整 CLI 评测或证明模型质量提升。真实研究、图像服务、付费模型、浏览器视觉质量和目标设备兼容性未在本轮执行。

本次新增内容不包含整本书、私人项目文档或模型运行产物。审查发现基线历史中已有书籍全文与项目附件；本轮未删除或改写这些历史，因此不能将仓库整体描述为已完成素材权利清理。
