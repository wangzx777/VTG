# Paper

一句话：
解决 DETR 在时序语句定位中因相似目标片段和全局—局部目标不一致而产生的查询冲突，通过查询分组与排序（QGR）和查询到帧的全局—局部桥接（GLB），实现更稳定的跨层匹配与更准确的 Temporal Boundary 预测。

## Problem

- 增加 DETR 查询数或解码器层数在 TSG 中反而可能降低性能，原因不是简单的查询冗余。
- 语义相似但边界不同的目标片段造成查询间随机匹配；单个查询还面临全局语义匹配与局部边界定位之间的冲突。

## Method

Video / Query
↓
CLIP + SlowFast 帧特征 / CLIP 文本词元
↓
跨模态 DETR Encoder
↓
带 QGR 的 Decoder Self-Attention：按跨度距离软分组，并以分类置信度 × IoU 预测排序查询
↓
GLB Query-to-Frame Alignment + Span / IoU Heads
↓
Temporal Boundary

核心创新：
1. 用预测跨度距离形成软查询组，并联合全局分类置信度与局部 IoU 预测重塑查询间自注意力，降低跨片段查询冲突和跨层匹配抖动。
2. 用查询到帧的桥接损失同时拉近片段内所有帧、压低片段外帧相似度，把全局语义与局部边界连接起来。
3. 仅对标准时序 DETR 解码器做轻量修改，就使增加查询数和层数不再导致异常性能下降，并加快收敛。

## Training

- Training paradigm: 监督式端到端微调；单张 NVIDIA A40 上训练 200 epochs，AdamW，初始学习率 1e-4。
- Loss / Reward: $\mathcal{L}_{MD}+\lambda_{bridge}\mathcal{L}_{bridge}+\lambda_{iou}\mathcal{L}_{iou}$；其中 $\mathcal{L}_{MD}$ 含 L1、gIoU、分类与显著性损失。
- 特殊训练策略: 每层使用二分匹配进行一对一标签分配；QGR 根据预测跨度、分类置信度和 IoU 预测调制自注意力；GLB 施加帧级正负对齐约束。

## Experiment

Dataset: QVHighlights；Charades-STA；TACoS
Metric: QVHighlights 使用 R1、mAP；Charades-STA 与 TACoS 使用 R1、mIoU。
主要结果 / 结论: QVHighlights 验证集达到 R1@0.5 69.48、Avg. mAP 49.50；TACoS 与 Charades-STA 的 mIoU 分别为 39.44 和 52.56。消融中 QGR 与 GLB 均有效，联合后把基线 mAP 从 44.97 提升到 49.50。

## Tags

task: task:mr
role: N/A
training: train:finetune
time: reason:boundary; reason:multisegment; out:regression
visual: modal:vt; setting:finetuned

## Key Figure / Table

- Fig.6：Sim-DETR 核心架构，展示标准 Temporal DETR、QGR 和 GLB 两项解码器修改。
- Table.1：QVHighlights 上的主要结果；验证集 Avg. mAP 为 49.50，测试集 Avg. mAP 为 46.93。

## Code

Repo: Not Checked
关键文件: Not Checked

## 我还没懂

1. $\mathcal{L}_{bridge}$ 的分式形式在不同片段长度、片段外帧很多或早期相似度不可靠时，如何避免梯度尺度失衡或退化解？
2. QGR 依赖尚在学习中的预测跨度、分类置信度和 IoU 预测；训练早期这些信号的噪声如何影响分组、排序与跨层稳定性？
3. 正文引用了附录 C–G 来支持收敛、排序、超参数和距离选择，但当前 PDF 未包含这些附录，相关敏感性与实现细节仍待确认。
