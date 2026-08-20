# FlashVTG Reproduction

## 状态

QVHighlights 正式训练与验证已完成。2026-08-19 的运行完成 150 epoch，串行流水线中 FlashVTG 阶段 exit code 为 0。

## 当前结果

| 指标 | 复现（NMS 0.7） | 论文 | 差值 |
|---|---:|---:|---:|
| MR Avg. mAP | 53.49 | 52.84 | +0.65 |
| MR mIoU | 67.19 | N/A | N/A |
| MR short mAP | 19.29 | 15.73 | +3.56 |
| HL VeryGood mAP | 43.55 | 44.15 | -0.60 |

详细记录见 `records/20260819-official-seed2024.md`。

## 路径

- 官方源码：`repos/FlashVTG/`
- 启动脚本：`scripts/train_qvhighlights.sh`
- 运行产物：`runs/qvhighlights/<run-id>/`
