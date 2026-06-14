# examples.md — ViLT 走通四阶法的样例

以 ViLT (ICML 2021, "Vision-and-Language Transformer Without Convolution or Region Supervision") 为例, 演示四阶法的完整输出。

---

## 阶段 1 输出: 结构 (2 min)

```
摘要 (~0.5 页) → 引言 (1.5 页) → 相关工作 (2 页) → 方法 (1 段) → 实验 (3 页) → 结论 (0.5 页)
                                                              ↑
                                              主表 (VQA/NLVR2/Retrieval) + 消融
```

**非典型**: 相关工作 2 页 (综述味), 方法 1 段 (极简), 实验 3 页 (重工程)。
**判断**: 典型的"重实验 + 极简方法"配比 = 工程型 SOTA。

---

## 阶段 2 输出: 故事 (5 min)

**痛点 (3 个)**:
1. 视觉特征抽取太贵 (885ms vs 文本 15ms, 千倍差距)
2. 预训练好的目标检测器表达力受限 (类别数不够, 几百类 vs 文本无限词)
3. 之前的多模态工作不敢用强数据增强 (怕破坏 image-text 配对)

**前人**:
- VSE 系: 视觉太贵, 融合太浅
- CLIP: 视觉文本等算力, 但融合太浅 (只点乘), 不适合分类任务
- VisualBERT / UNITER / Oscar: 视觉太贵 (目标检测), 融合够深 → 性能好

**本文卖点**: 砍掉 region / convolution supervision, 用 patch projection 替代。**1000× 加速, 性能只掉几个点**。

**故事走向**: 先抛"为什么之前的工作不对" (3 痛点) → 再抛 ViLT 的极简设计 → 卖点单独列在 Fig 1 → 三个贡献最后总结。

---

## 阶段 3 输出: 拆解 (20 min)

### 3.1 输入
- 文本: tokenize → word embedding (BERT tokenizer, 长度 L)
- 图像: 打成 patch → linear projection (ViT 风格, 长度 N)
- 模态标识: model type embedding (文本=0, 图像=1)
- 序列总长: 1 (text CLS) + L + 1 (image CLS) + N
- **位置编码**: 文本 1..L, 图像 1..N, position embedding 加在 token + model type 上

### 3.2 模型
- 主体: 单个 Transformer encoder (ViT-B/32 规模)
- 参数量: 集中在 transformer, 没有 vision tower / detection head
- 单 stream: 文本 + 图像 concat 送入同一 transformer

### 3.3 损失
| 损失 | 作用 | 信号粒度 |
|---|---|---|
| ITM (Image-Text Matching) | 整体匹配 | 序列级 |
| WPA (Word-Patch Alignment) | token 级对齐, 用 optimal transport | 细粒度 |
| MLM (Masked Language Modeling) | 文本重建 (WWM 变体) | token 级 |

**为什么加 WPA**: 单纯 ITM 信号太粗, 加 WPA 强迫 token 级对齐。

### 3.4 训练技巧
| Trick | 来源 | 对什么有效 |
|---|---|---|
| Whole Word Masking (WWM) | NLP (BERT 变体) | 强制模型看图 (因为整词 mask 很难靠语言猜) |
| RandAugment (改版) | 视觉 | 图像侧强增强; 禁用 color jitter + cutout, 防破坏配对 |
| 64 V100 训 3 天 | 算力堆 | 必要条件, 但讲师认为这是"过奢侈", 后续 ALBEF 8 卡就行 |

### 3.5 数据
| 阶段 | 数据集 | 大小 | 特点 |
|---|---|---|---|
| 预训练 | COCO + VG + GCC + SBU | 4M 图 | 各数据集 caption 长度差异大 |
| 下游分类 | VQA / NLVR2 | — | 标准 benchmark |
| 下游检索 | Flickr30K / MSCOCO | — | zero-shot 和 fine-tune 两套 |

**坑**: CC 和 SBU 的 URL 已失效, 复现时只能下到部分。

### 3.6 实验
- **主表 (VQA/NLVR2)**: ViLT 跑 15ms (vs region feature 900ms), 性能 76.1/83.5 vs 75.8/85.9 (差距很小, 但快 60×)
- **主表 (Retrieval)**: zero-shot 落后 SOTA, fine-tune 也落后, 但速度碾压
- **消融**:
  - 训练步数 25K → 100K 性能提升有限 (讲师认为可以省成本)
  - RandAugment > WWM (图像增强更关键)
  - 组合最佳

---

## 阶段 4 输出: 评价 (5 min)

| 维度 | 分 | 理由 |
|---|---|---|
| 故事 | 2/2 | "天下苦目标检测久矣" 直击痛点, 卖点明确 |
| 方法 | 1/2 | 极简是优势, 但没有新 insight (ViT + BERT + 多模态) |
| 实验 | 2/2 | 全面, 诚实报告了低于 SOTA, 有 25K→100K 的诚实曲线 |
| 代码 | 1/2 | PyTorch Lightning 写的, 上手有难度 (讲师吐槽) |
| 留白 | 2/2 | 留了 3 个未来方向, 1 年内全被填上 |
| **总分** | **8/10** | 经典里程碑工作, 性能让位给速度, 影响力大 |

**结论**: 值得 follow。具体 follow 哪个 future work (scalability / 图像重建 / 增强), 看手速和算力。

---

## 笔记最终版 (用 `templates/output-template.md` 的格式)

```markdown
# 《ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision》精读笔记

**一句话卖点**: 砍掉 region supervision, 1000× 加速, 性能只掉几点。
**结构特点**: 相关工作 2 页 (综述味), 方法 1 段, 实验 3 页。
**性能坐标**: VQA/NLVR2 距 SOTA 几个点, 速度 60× 领先; retrieval 落后 SOTA 较多。

## 1. 故事
- 痛点 1: 视觉特征抽取太贵 (885ms vs 15ms)
- 痛点 2: 预训练检测器表达力受限
- 痛点 3: 多模态不能用强数据增强
- 前人: VSE / CLIP (融合太浅), UNITER / Oscar (视觉太贵)
- 卖点: patch projection 替代目标检测

## 2. 方法
- 输入: 文本 + 图像 concat, model type + position embedding
- 模型: 单个 Transformer encoder (single stream)
- 损失: ITM + WPA (optimal transport) + MLM (WWM)
- 训练技巧: WWM + RandAugment (改版)

## 3. 实验
- 数据: 4M (COCO + VG + GCC + SBU)
- 主表: VQA / NLVR2 / Retrieval (zero-shot + fine-tune)
- 消融: 25K 够用, RandAugment > WWM

## 4. 评价
| 维度 | 分 |
|---|---|
| 故事 | 2 |
| 方法 | 1 |
| 实验 | 2 |
| 代码 | 1 |
| 留白 | 2 |
| **总** | **8/10** |

## 5. 接下来
值得 follow。3 个未来方向: scalability / 图像重建 / 数据增强。
讲师推荐: 如果算力不够, 跟 ALBEF (8 卡) 而非 ViLT (64 卡)。
```

---

## 用这套笔记能回答什么?

- 30 秒讲清 ViLT: "砍 region feature, 1000× 加速, 性能小掉。"
- 5 分钟讲清: 上面 5 段
- 决定 follow: 选 future work 1 (scalability) 还是 3 (增强) — 取决于你的算力
- 写 related work: 引用 ViLT 的 Fig 2 (taxonomy)
