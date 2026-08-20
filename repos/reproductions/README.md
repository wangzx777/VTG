# VTG Reproductions

这里统一保存官方/参考项目的复现配置、启动脚本、运行记录和训练产物。`project/` 保留给后续自研代码，不再承载复现运行结果。

## 项目总览

| 项目 | 数据集 | 当前级别 | 最近运行 | 状态 | 核心结果 |
|---|---|---|---|---|---|
| Sim-DETR | QVHighlights | 正式训练与验证 | `20260819-official-seed2017` | 完成 | R1@0.5 68.32；R1@0.7 53.23；Avg. mAP 48.94 |
| FlashVTG | QVHighlights | 正式训练与验证 | `20260819-official-seed2024` | 完成 | NMS Avg. mAP 53.49；mIoU 67.19 |
| CausalVTG | QVHighlights | Smoke test | `20260819-smoke` | 完成 | 前向、反向、checkpoint 和 4 样本验证通过；无正式结果 |
| HieraMamba | TACoS | 正式训练与测试 | `20260820-official-seed1234567891` | 完成 | R1@0.3/0.5 58.79/48.64；R5@0.3/0.5 84.10/73.93 |

## 目录约定

每个项目使用同样的结构：

```text
<project>/
├── README.md
├── configs/
├── scripts/
├── records/
└── runs/
    └── <dataset>/
        └── <YYYYMMDD>-<purpose>-seed<seed>/
```

`configs/`、`scripts/` 和 `records/` 是可复用、可提交的小文件。`runs/` 保存日志、checkpoint、预测、指标和 TensorBoard 数据，其内容由 Git 忽略，仅保存在实际运行主机。

运行目录一经完成即视为不可变。重复实验必须创建新的 run ID，不在旧目录中覆盖训练产物。官方仓库位于 `repos/`，保持只读，所有 `--results_root`、工作目录或 job name 都必须指向本目录下对应项目的 `runs/`。

## 运行命名

正式运行使用 `YYYYMMDD-official-seed<seed>`，冒烟测试使用 `YYYYMMDD-smoke`，正式训练前的完整契约检查使用 `YYYYMMDD-preflight-smoke`。同一天重复运行时，在日期后加入时间或递增后缀。

## QVHighlights 串行训练

运行 `scripts/train_qvhighlights_sequential.sh` 会依次调用 Sim-DETR 和 FlashVTG 各自的启动脚本；两个项目的输出分别进入自己的项目目录，不再写入共同的 overnight 结果目录。
