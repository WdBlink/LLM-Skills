# Clean Code Refactoring Framework

Version: 0.1.0 · 用户确认自行生成的书籍方法派生技能，未声明开放许可证。

用于坏味道诊断、重构计划和行为保持式代码修改。方法来自 Robert C. Martin 的《代码整洁之道》和 Martin Fowler 的《重构》第2版，源定位与转述见 [source-map.md](references/source-map.md)。包中不包含原书全文；E 编号是原提炼阶段的本地证据编号，原始 chunks 未分发。

## 安装与使用

在仓库根目录复制 `skills/clean-code-refactoring-framework/` 到宿主的技能目录，然后请求：

```text
用 $clean-code-refactoring-framework 审查这个模块，先确定可观察行为和测试，再给出最小重构。
```

`SKILL.md` 是入口，`references/` 保存领域方法，`templates/` 提供任务模板。`evals/evals.json` 提供预期行为案例；它们尚未构成真实模型运行结果。执行时复用项目已有测试、构建与适用的 P-Stack 流程，不依赖作者机器上的生成器。

本库维护者于 2026-10-23 确认使用女娲等工具生成了这类技能，旧“创建者：花叔”文字沿用生成模板。本镜像明确区分生成工具署名、本地派生技能和原书方法归属；不将其他同署名技能一概认定为自建。
