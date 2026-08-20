# Paper

一句话：
解决 VTG 评测与训练数据噪声大、MLLM 后训练方案缺乏可靠共识的问题，通过 TimeLens-Bench/100K 重标注数据、交错文本时间戳和无思维过程 GRPO-RLVR，实现高性能且高效率的时序定位基线。

## Problem

- Charades-STA、ActivityNet Captions、QVHighlights 存在查询重复/不存在/不清楚、边界不准或不穷尽等严重错误，旧基准会扭曲开源与闭源模型排序。
- 时间戳编码、SFT/RLVR、显式思维过程、训练时长与数据采样等设计缺乏统一数据基础上的公平比较和最佳实践。

## Method

Legacy VTG Data
↓
人工 diagnose-then-refine + 交叉复核 → TimeLens-Bench；自动视频重标注 → TimeLens-100K
↓
每帧原始时间戳经文本 tokenizer 编码，并作为前缀与对应视觉词元交错排列
↓
Qwen2.5-VL / Qwen3-VL + thinking-free GRPO-RLVR
↓
奖励平台期早停 + 基于离线 IoU 难度估计的高难样本高斯采样
↓
Temporal Boundary

核心创新：
1. 提出唯一性、存在性、清晰度、无泄漏、边界精确与穷尽等 VTG 标注标准，并重标三大基准，揭示旧排行榜不可靠。
2. 在统一高质量数据上发现“原始时间戳的交错文本前缀”优于位置嵌入、视觉叠字和非交错文本方案。
3. 证明纯 thinking-free RLVR 无需先行 SFT，配合奖励平台早停和高难样本采样，可兼顾效果与效率。

## Training

- Training paradigm: 以 Qwen2.5-VL-7B 为主要实验基线，使用 GRPO 进行无显式思维过程的 RLVR；最终另构建基于 Qwen3-VL-8B 的 TimeLens-8B。
- Loss / Reward: 可验证的 VTG 时间 IoU 奖励；主文未给出完整奖励公式，记为未确认。
- 特殊训练策略: 待训练模型先离线推理，以 IoU 估计样本难度并进行高斯采样；选择相对难度超过约 0.75 的样本，当平均奖励与组内奖励标准差进入平台期时早停。

## Experiment

Dataset: TimeLens-Bench（Charades-TimeLens、ActivityNet-TimeLens、QVHighlights-TimeLens）；TimeLens-100K。
Metric: R1@0.3/0.5/0.7；mIoU；训练时间。
主要结果 / 结论: TimeLens-8B 在三套基准上的 mIoU 为 55.2/53.2/65.5，达到开源 SOTA 并超过 GPT-5 与 Gemini-2.5-Flash；TimeLens-7B 相比 Qwen2.5-VL-7B 的 mIoU 分别从 39.3/31.4/31.6 提升至 48.8/46.2/56.0。Thinking-free RLVR 用约 4 小时 10 分钟训练，在平均表现与效率上优于 SFT、thinking-based RLVR 及 SFT+RLVR。

## Tags

task: task:mr
role: role:executor
training: train:posttrain; opt:rl; opt:grpo; data:pseudo-label; data:instruction
time: time:explicit-token; reason:boundary; out:timestamp-text
visual: setting:multidomain; modal:vt

## Key Figure / Table

- Fig.2：展示重标注前后的模型重排序，以及从数据整理到 RLVR 训练方案的累计增益。
- Fig.5 / Table.2：比较交错文本前缀、视觉叠加、位置嵌入与时间戳格式，原始时间戳交错文本前缀最佳。
- Table.1：TimeLens-Bench 主结果，TimeLens-8B 在开源模型中达到 SOTA。

## Code

Repo: https://timelens-arc-lab.github.io/
关键文件: Not Checked

## 我还没懂

1. 主文把时间 IoU 作为 RLVR 奖励，但奖励塑形、格式奖励、GRPO 超参数与输出解析细节均留在未附于 PDF 的补充材料中。
2. TimeLens-100K 的自动重标模型、提示、边界生成与质量过滤流程未在主文展开，训练收益究竟来自重写查询还是更准时间戳尚难分离。
3. 论文把 VTG 定性为感知驱动任务并据此去除显式思维过程，但对长视频、多事件消歧和组合时序查询是否仍成立尚未单独评估。
