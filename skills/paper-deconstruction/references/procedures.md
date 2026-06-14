# procedures.md — 六块拆解的标准问题清单

讲师在精读 ViLT 时, 把方法部分按 6 块拆。每块 5 个固定问题, 答完就懂。

---

## 1. 输入 (Input)

- 原始数据是什么? (图像? 文本? 音频? 视频?)
- 怎么变成模型输入的? (token 化? 离散化? embedding 矩阵?)
- 哪些预处理是"被作者砍掉" 的? (ViLT 砍了: 目标检测 / ResNet backbone)
- 输入序列长度多少? (ViLT: L+N+2, 其中 2 是两个 CLS token)
- 是否有 "模态标识" token? (ViLT: 文本=0, 图像=1, model type embedding)

→ 输出 1 行: `<数据> → <预处理> → <序列长度 L+...>`

---

## 2. 模型 (Architecture)

- 抄下模型图 (Fig 3)
- 主干是什么? (ViLT: 单个 Transformer encoder)
- 有几个子模块? 各自的作用?
- 参数集中在哪? (ViLT: 全在 transformer 里, 没有 vision tower)
- 单 stream 还是 dual stream? (ViLT: single, 序列 concat 后送一个 transformer)

→ 输出 1 句: `<主干> (single/dual stream), <参数量>, <关键 trick>`

---

## 3. 损失 (Loss)

- 主 loss 是什么? (ViLT: ITM + MLM)
- 辅助 loss? (ViLT: WPA, word-patch alignment via optimal transport)
- 每个 loss 在拉什么信号?
  - ITM: 整体图像-文本匹配
  - WPA: token 级对齐
  - MLM: 文本重建
- loss 加权怎么设? (论文里 ablation 一般会展示)
- 有没有 "反直觉" 的 loss? (ViLT 的 WPA 用 optimal transport, 不是简单点乘)

→ 输出 1 张表: `<loss 名> | <输入> | <信号>`

---

## 4. 训练技巧 (Tricks)

这部分讲师花了最多时间, 因为"tricks 是 follow 论文的导火索":

- **文本侧**: whole word masking (WWM) — 不 mask 子词, 整词 mask, 强制模型用图像信息
- **图像侧**: RandAugment — 但禁用 color jitter + cutout (避免破坏 image-text 配对)
- **学习率 / batch size / 训练步数**: ViLT = 64 V100 训 3 天, 100K steps
- **数据增强**: 凡是 multi-modal 都要考虑 "增强后是否还匹配" 的问题
- **预训练 → 微调**: 微调哪些层? 全参数还是 adapter?

→ 输出 1 张表: `<trick 名> | <来源领域> | <对什么任务有效>`

讲师原话: "你光看完主表没用, 一定要看消融。消融告诉你哪个 trick 真有用。"

---

## 5. 数据 (Data)

- 预训练集: 几个? 多大? (ViLT: COCO + VG + GCC + SBU ≈ 4M)
- 预训练数据的特点? (每个数据集 caption 长度、image-text pair 数)
- 下游任务数据集: 哪几个? (ViLT: VQA / NLVR2 / Flickr30K / MSCOCO retrieval)
- 数据集是否有可获得性问题? (ViLT 提到的 CC/SBU URL 已失效)
- 数据有没有"隐性数据泄漏"风险? (VQA 常用 COCO, 预训练也用 COCO)

→ 输出 1 张表: `<阶段> | <数据集> | <大小> | <特点>`

---

## 6. 实验 (Experiments)

- **主表怎么读**:
  - 按列读: 自己 vs 各类 baseline (region / grid / patch)
  - 不要按行读, 行是任务, 拿不到方法对比
- **运行时间** vs **精度**: ViLT 的卖点是时间, 不是精度。主表要单独看"速度列"
- **zero-shot vs fine-tune**: retrieval 分两表, 这是公平比较的标准做法
- **消融实验**:
  - 训练步数 ablation (25K/50K/100K) — 决定能不能跑
  - 每个 trick 的贡献 (WWM / RandAugment / 两者组合)
  - 大多论文会先在 25K 上做消融 (省成本), 最后报 100K 主结果
- **可视化**: 一般是 attention map / retrieval 例子, 用来给读者"看得到"的成果

→ 输出: 主表截图 (或复述列结构) + 消融表 + 一句"哪个 trick 最关键"

---

## 阅读顺序建议 (讲师路径)

1. **结构** (2 min): 翻目录, 画章节地图
2. **摘要 + 标题** (3 min): 抓卖点
3. **引言** (5 min): 抓痛点 + 前人 + 贡献
4. **Fig 1 / Fig 2** (5 min): 抓"对比图" (ViLT Fig 1 = 速度, Fig 2 = 分类)
5. **方法** (15 min): 6 块拆
6. **实验** (10 min): 主表 + 消融
7. **结论 + future work** (3 min): 找坑

总计 45 min 一篇, 是组会汇报的标准时长。
