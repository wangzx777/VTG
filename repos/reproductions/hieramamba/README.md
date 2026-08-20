# HieraMamba Reproduction

## 状态

TACoS 正式训练与测试已完成。运行包含 5 个 warmup epoch 和 10 个正式训练 epoch，共 15 epoch、9285 iteration。最终 checkpoint 可加载，完整 25 视频测试已完成。

## 当前结果

| 指标 | 复现 | 论文 | 差值 |
|---|---:|---:|---:|
| R1@0.3 | 58.79 | 59.59 | -0.80 |
| R1@0.5 | 48.64 | 48.99 | -0.35 |
| R5@0.3 | 84.10 | 83.75 | +0.35 |
| R5@0.5 | 73.93 | 74.28 | -0.35 |
| 四项平均 | 66.36 | 66.65 | -0.29 |

详细记录见 `records/20260820-official-seed1234567891.md`。

## 路径

- 官方源码：`repos/HieraMamba/`
- Python 3.12/NumPy 2 兼容入口：`scripts/train_compat.py`
- TACoS 启动脚本：`scripts/train_tacos.sh`
- 运行产物：`runs/tacos/<run-id>/`
