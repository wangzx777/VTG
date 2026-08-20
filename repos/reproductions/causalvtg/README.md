# CausalVTG Reproduction

## 状态

目前仅完成 QVHighlights 真实特征 smoke test，尚未执行正式 50 epoch 训练或生成正式 256 簇结果。smoke test 已验证 4 条训练样本的前向与反向、checkpoint 写入，以及 4 条验证样本的 MR/HL 指标流程。

smoke 指标只用于验证代码路径，不代表模型性能。详细记录见 `records/20260819-smoke.md`。

## 路径

- 官方源码：`repos/CausalVTG/`
- Smoke 配置：`configs/qv_internvideo_smoke.py`
- 运行产物：`runs/qvhighlights/<run-id>/`
