# 来源与迁移边界

核验日期：2026-09-16。

来源：[OpenMMLab《AI科研加速营第3课｜建立可持续更新的研究上下文》](https://www.bilibili.com/video/BV11zec6NE3k/)，18:30。B 站公开元数据核对了标题、作者和时长；公开播放器接口返回无字幕轨。

实际阅读：公开可下载的 360P 视频，每 30 秒抽取一帧，共 37 帧，逐一查看课件和可辨认的内嵌字幕。下列时间为抽样附近位置，非逐字转写的精确起止时间。未完成音轨转写；未采样口述细节以及演示中的小字不作为已确认依据。没有宣称观看并理解每一秒。

## 能从画面确认的方法

| 视频附近位置 | 观察到的课程内容（概述） | Skill 中的落实 |
| --- | --- | --- |
| 02:15–03:45 | 资料、笔记、对话和工具结果分散，会让研究依据与进展难以还原 | 单个课题入口链接原始资料及关键认识 |
| 04:15–04:45 | 围绕明确课题组织信息及其关系，并持续维护 | 所有来源、方法和下一任务关联研究问题 |
| 05:15–08:45 | 课题上下文包含边界、证据、方法工具、当前认识和未解决问题 | 入口五要素及其字段合同 |
| 09:15–10:15 | 阶段记录应连接判断、依据、适用边界和后续行动 | 四段成果记录，不能只有结论 |
| 10:45 | 演示依次创建课题、加入材料、选择工具、开展对话、保存成果 | 建立 → 执行已授权研究 → 更新；工具服务具体问题 |
| 11:15–16:45 | 在课题空间整理材料、接入工具、提问并保留结果 | 转为宿主文件读写，未复制特定产品操作接口 |
| 17:15–18:15 | 用自己的方向组织课题、论文与可继续的研究状态 | 按实际项目使用，而非收藏一份通用提示词 |

## 原创迁移，不归因给讲者

以下是面向长期 Agent 任务的工程补充：建立/更新/恢复/检查四入口；稳定来源 ID 与具体定位；观察/解释/决定分开；更正关系、冲突并存、失败尝试及重试条件；更新前重读、重复输入不重写、写入中断恢复；按下一动作加载文件；默认使用本地 Markdown、不自动上传；与已有 ctx、mind-map、handoff 的边界。

这些机制没有课程中的效果量或对照实验支持。它们是可执行设计，不构成“提高科研成功率”或“解决所有上下文遗忘”的实证结论。

## 与已有技能的分工

- `ctx`：用户明确选择时承接现有存储与文档治理；本技能提供科研认识与证据关系的字段，不复制一套 ctx。
- `mind-map`：适合把研究判断做成图谱；本技能不生成图谱，也不重复其节点存储。
- `handoff` / `project-handoff`：已有一次性交接需求时可引用本入口；本技能持续维护研究认识与阶段成果，不重建交接平台。

技能指令与模板为原创整理；课程和平台素材权利属于原权利人。技能包不分发视频、完整转写或平台截图。

## ResearchStudio 的程序/模型分工（0.2.0）

固定来源：Microsoft ResearchStudio commit `0597891df1a153b8e4cbdc8c1c685f43a0a6abcf`。2026-09-16 实际读取该提交的 `next_step.py`、`merge_revisions.py` 和 `revise.txt`，与 workflow-skill-creator 的实现、派发、prompt 三份参考交叉核对。

| 核对的上游实现 | 迁移到本技能 | 未照搬的部分 |
| --- | --- | --- |
| [next_step.py](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/next_step.py)：只读导航与宿主动作包 | next 输出完整动作，宿主真正派发 child | 上游是文本协议，本技能是原创 JSON 协议 |
| [merge_revisions.py](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/scripts/merge_revisions.py)：受保护字段与程序合并 | 只允许八个叙述字段，patch 限 finding 点名范围 | 没有通用点路径、科研 kill-switch 例外或多类 patch 操作 |
| [revise.txt](https://github.com/microsoft/ResearchStudio/blob/0597891df1a153b8e4cbdc8c1c685f43a0a6abcf/ResearchStudio-Idea/skills/idea_spark/references/system-prompts/revise.txt)：critic/reviser 独立、仅输出修订 | 新上下文修订、程序 merge、另一新 critic 复审 | 不复制科研子模式、模型名称或上游重试预算 |

本技能的来源 hash、base 冲突拒绝、review_token、恢复性 apply 是针对研究上下文更新的原创工程实现，不把它们声称为 ResearchStudio 全局已有保证。实现源码未复制上游代码；也没有复现上游完整科研流水线或效果评测。
