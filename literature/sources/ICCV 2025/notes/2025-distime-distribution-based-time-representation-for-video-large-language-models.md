# Paper

一句话：
解决 Video-LLM 离散时间词元精度不足、事件边界含糊和时间感知训练数据有限的问题，通过单一分布式时间词元、轻量时间编码—解码器及 InternVid-TG 伪标注数据，实现连续时间回归与自回归时间上下文细化。

## Problem

- 文本数值会混淆时间与普通数字，多时间词元受长尾训练与量化误差影响，专用时序头则参数较重且需重复输入视觉特征。
- 现有自动时间标注依赖镜头边界或固定间隔，难以覆盖镜头内部事件和短时事件的细粒度边界。

## Method

Video / Query
↓
均匀采样帧 → Visual Encoder + Projector；归一化帧时间 → Distribution-based Time Encoder
↓
时间词元与视觉词元交错输入 Video-LLM；输出单一 `<TIME_STAMP>`
↓
Time Decoder 预测起止时间分布，经锚点加权得到连续 $(st,et)$
↓
将解码时间戳高斯化并重新编码，更新后续自回归的时间上下文
↓
Temporal Boundary / Dense Events / Grounded Answer

核心创新：
1. 用单一专用词元承载连续时间分布，避免文本数字混淆和多时间词元的离散量化问题。
2. 轻量解码器预测边界概率分布并以锚点期望回归时间，编码器再将时间戳高斯化回注，实现迭代时间细化。
3. 结合 GPT-4o 事件描述与 UniMD、Mr.BLIP、TFVTG 定位结果，经相似度评分集成构建 17.9 万视频、125 万事件的 InternVid-TG。

## Training

- Training paradigm: 单阶段 LoRA 微调；冻结视觉主干与中间层，完整训练 LLM token embedding、LLM head、时间编码器和解码器。
- Loss / Reward: 下一词元损失 $\mathcal L_{ntp}$ + 1d-IoU 回归损失 $\mathcal L_{reg}$ + Distribution Focal Loss $\mathcal L_{dist}$，三项权重均为 1。
- 特殊训练策略: 基于 InternVL2.5-1B/8B 与 LLaVA-OneVision-7B；时间编码器和解码器均为三层 ReLU MLP，额外参数占比约 0.34%–0.84%。

## Experiment

Dataset: Charades-STA；ANet-Caption；QVHighlights；YouCook2；NExT-GQA；MVBench；Video-MME；LongVideoBench。
Metric: MR 的 R@1@IoU 与 mIoU；DVC 的 SODA_c、CIDEr、F1、METEOR；GQA 的 Acc、IoP、IoU；通用视频理解平均分。
主要结果 / 结论: 零样本 Charades-STA 上 DisTime-InternVL-1B 的 R@1@0.3/0.5/0.7 与 mIoU 为 78.1/56.3/29.7/51.6，8B 达 81.0/60.3/30.8/53.1；分布表示将 YouCook2 F1 从 2.2 提至 16.3，再编码后升至 20.5；8B 在 NExT-GQA 全部指标领先，但 Video-MME 有所下降。

## Tags

task: task:multi; task:mr; task:dc; task:gqa
role: role:executor
training: train:finetune; opt:sft; data:pseudo-label; data:instruction
time: time:explicit-token; reason:boundary; reason:iterative-refine; out:temporal-token; out:decoder-head; out:regression
visual: setting:zeroshot; setting:multitask; modal:vt

## Key Figure / Table

- Fig.2：DisTime 总体结构，展示输入时间词元注入、单一 `<TIME_STAMP>` 解码和时间词元重新编码。
- Fig.3：时间解码器的边界分布预测与时间编码器的高斯投影。
- Table.5：Charades-STA 与 ANet-Caption 时刻检索主结果。

## Code

Repo: https://github.com/josephzpng/DisTime
关键文件: Not Checked

## 我还没懂

1. 文中 $\delta=1$ 与归一化 $[0,1]$ 时间轴的尺度关系、Gaussian Projector 的离散化细节及 $reg_{max}$ 默认值未在主文明确说明。
2. 对解码分布取期望可能落在多峰之间的低概率区域，模型在真正多模态边界下为何仍能保持校准尚缺少分析。
3. InternVid-TG 按最高相似度选择三个定位器之一，但模型置信度、跨模型一致性与人工标注误差之间的关系仍需补充材料验证。
