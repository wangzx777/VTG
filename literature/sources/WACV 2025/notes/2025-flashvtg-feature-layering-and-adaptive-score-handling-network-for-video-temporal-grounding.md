# Paper

一句话：
解决 DETR 稀疏查询难以覆盖短时刻、孤立置信度难以正确排序相邻预测的问题，通过时序特征分层（TFL）、尺度内/尺度间自适应评分细化（ASR）与片段感知评分损失，实现兼顾 MR 与 HD 的多尺度 Temporal Boundary 预测。

## Problem

- DETR 解码器查询稀疏且有限，对短于 10 秒的时刻容易漏检；直接增加查询数又会提高计算复杂度。
- 仅依据单个预测时刻生成置信度，无法利用相邻时刻和跨尺度上下文，导致首个检索结果排序不准。

## Method

Video / Query
↓
冻结的 CLIP / SlowFast（或 InternVideo2）视频编码器与 CLIP / GloVe（或 LLaMA）文本编码器
↓
Dummy Tokens + Adaptive Cross-Attention + Transformer Encoder 完成视频—文本融合
↓
TFL：步幅 2 的 Conv1D 构建多时间尺度特征金字塔
↓
统一 Moment Head 回归多尺度边界；ASR 融合尺度内与尺度间置信度；HD Head 输出显著性分数
↓
Temporal Boundary

核心创新：
1. 用密集多尺度特征金字塔取代稀疏 DETR 解码器查询，使不同持续时长、尤其短时刻获得对应分辨率的预测位置。
2. ASR 同时汇聚同尺度相邻位置与跨尺度特征，并以可学习权重融合两类置信度，改善预测排序与首个时刻召回。
3. Clip-Aware Score Loss 将 HD 显著性标签迁移为 MR 的片段级细粒度监督，专门增强短时刻置信度学习。

## Training

- Training paradigm: 监督式端到端微调；冻结预训练特征编码器，AdamW，QVHighlights 上单张 RTX 4090 训练 150 epochs。
- Loss / Reward: MR 使用 Focal、L1、Clip-Aware Score Loss；HD 使用 SampledNCE 与 Saliency Loss，五项加权求和。
- 特殊训练策略: 对预测置信度和真实显著性标签分别做 min-max 归一化，再以 MSE 对齐其视频片段级相对分布；MR 与 HD 联合监督。

## Experiment

Dataset: QVHighlights；TACoS；Charades-STA；TVSum；YouTube-HL
Metric: MR 使用 R1@X、mAP、mIoU；HD 使用 mAP、Hit@1。
主要结果 / 结论: QVHighlights 上采用 InternVideo2 时，验证集 MR Avg. mAP 为 52.84、HD mAP 为 44.15；相同 SlowFast+CLIP 骨干下短时刻 mAP 达 15.73，相比此前最佳 12.62 提升至约 125%。消融中 TFL 将 MR mAP 从 46.84 提升到 52.47，加入 ASR 后达到 52.84。

## Tags

task: task:mr; task:hd
role: N/A
training: train:finetune
time: reason:boundary; out:regression; out:score
visual: visual:coarse-to-fine; setting:multitask; modal:vt

## Key Figure / Table

- Fig.2：FlashVTG 总体架构，展示特征提取与融合、TFL 特征金字塔、ASR 双评分路径以及 MR/HD 输出头。
- Table.7：QVHighlights 短于 10 秒的时刻检索结果，FlashVTG 的 short-mAP 为 15.73。

## Code

Repo: https://github.com/Zhuo-Cao/FlashVTG
关键文件: Not Checked

## 我还没懂

1. 不同尺度的每个位置如何严格对应到原视频时间坐标，式（5）中的转置与尺度参数 $C_k$ 在实现中具体怎样完成边界解码？
2. ASR 直接拼接不同长度的金字塔特征后，尺度间评分头如何区分尺度身份，并避免较长的细粒度层在数量上主导 $c_{\mathrm{inter}}$？
3. 对 $c_{\mathrm{final}}$ 与 $s_{\mathrm{gt}}$ 分别做 min-max 归一化时，常数分数或极小动态范围如何处理，其稳定性与损失权重敏感性仍待补充材料确认。
