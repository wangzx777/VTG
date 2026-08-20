# Paper

一句话：
解决 Video LLM 直接生成时间戳监督稀疏且交叉熵不感知时间距离的问题，通过多轮偏移量驱动的粗到细时间细化和训练期辅助 L1 边界头，实现可插拔的逐步自纠错 Temporal Boundary 预测。

## Problem

- 每个片段仅有起点、终点两个时间词元提供监督，面对数百视觉与文本词元时信号过于稀疏，长视频中直接预测精确时间戳困难。
- 交叉熵把错误的“21 秒”和“100 秒”视为同等错误，不能表达预测与真实时间之间的连续距离。

## Method

Video / Query
↓
Visual Encoder + Adapter / Tokenizer → Video LLM
↓
先生成粗略 $(s_0,e_0)$，再输出到目标边界的 $(o_0^s,o_0^e)$
↓
用 `<refine>` 重复预测新边界与偏移量，执行 4 轮粗到细自纠错
↓
训练期 `<refine>` 隐状态经辅助 Linear Head 接受 L1 边界监督；推理取最后一步序列结果
↓
Temporal Boundary

核心创新：
1. 将一次性时间戳生成改写为多轮“边界 + 偏移”序列，用递减高斯噪声构造从粗到细的指令监督。
2. 在不改变 Video LLM 主干的前提下，为 `<refine>` 词元增加训练期 L1 回归头，使损失显式感知时间距离。
3. 方法可直接接入 VTimeLLM、VTG-LLM 等不同时间词元编码框架，推理时丢弃辅助头。

## Training

- Training paradigm: 指令微调 / LoRA；VTimeLLM 版本从第一阶段检查点初始化，再分别训练 4000 与 1000 steps。
- Loss / Reward: 答案词元交叉熵 + $\lambda=10$ 的辅助 L1 片段边界损失。
- 特殊训练策略: 每段使用 $K=4$ 次细化；从固定标准差 $\{5,3,1,0\}$ 秒的高斯分布独立采样起止偏移，越到后期噪声越小。

## Experiment

Dataset: ActivityNet Captions；Charades-STA
Metric: R@1 IoU=0.3/0.5/0.7；mIoU；另报告密集字幕 SODA_c、CIDEr、METEOR。
主要结果 / 结论: 基于 VTimeLLM 时，ActivityNet Captions mIoU 从 30.4 提升到 34.0，Charades-STA 从 31.2 提升到 36.2；基于 VTG-LLM 时，Charades-STA mIoU 从 34.4 提升到 35.6。最终细化步骤显著优于初始步骤。

## Tags

task: task:mr
role: role:executor
training: train:finetune; data:instruction
time: time:reasoning; reason:iterative-refine; out:timestamp-text; out:iterative-refine
visual: setting:zeroshot; modal:vt

## Key Figure / Table

- Fig.2：TIMEREFINE 总体框架，展示时间细化序列、下一词元交叉熵与 `<refine>` 辅助 L1 边界头。
- Table.1：ActivityNet Captions 与 Charades-STA 的主要时序定位结果。

## Code

Repo: https://github.com/SJTUwxz/TimeRefine_code
关键文件: Not Checked

## 我还没懂

1. 起点与终点偏移独立采样时，除越界外是否还显式避免 $s_k>e_k$，以及异常训练序列如何处理？
2. 辅助头连接每个 `<refine>` 隐状态时，各步是否都回归同一真实片段，损失在多步和多片段答案间如何精确聚合？
3. 自回归推理中若早期边界或偏移词元格式错误，后续细化能否恢复，控制词元约束解码的鲁棒性尚未报告。
