# Sim-DETR Reproduction

## 状态

QVHighlights 正式训练已完成。2026-08-19 的运行完成 200 epoch，best checkpoint 位于 epoch 160。训练结束后自动执行的独立评测曾因 PyTorch 2.6 将 `torch.load(weights_only)` 默认改为 `True` 而退出；使用项目自身生成、可信 checkpoint 并设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 后，完整 1550 样本验证成功。

## 当前结果

| 指标 | 复现 | 论文 | 差值 |
|---|---:|---:|---:|
| R1@0.5 | 68.32 | 69.48 | -1.16 |
| R1@0.7 | 53.23 | N/A | N/A |
| Avg. mAP | 48.94 | 49.50 | -0.56 |

详细记录见 `records/20260819-official-seed2017.md`。

## 路径

- 官方源码：`repos/Sim-DETR/`
- 启动脚本：`scripts/train_qvhighlights.sh`
- 独立评测：`scripts/eval_qvhighlights.sh`
- 运行产物：`runs/qvhighlights/<run-id>/`
