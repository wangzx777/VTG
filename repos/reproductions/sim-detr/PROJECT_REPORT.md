# Sim-DETR 项目与数据报告

> 对象：ICCV 2025《Sim-DETR: Unlock DETR for Temporal Sentence Grounding》官方实现  
> 官方代码 revision：`1965e994bd4ef486c2a9b137ef4b0b57837330c3`  
> 当前复现范围：QVHighlights，seed 2017，200 epoch  
> 本报告依据官方代码、论文、4090 上的真实数据目录及本次运行产物编写。

## 1. 一句话理解这个项目

Sim-DETR 接收“视频的预提取特征”和“一句自然语言的预提取特征”，输出：

1. Moment Retrieval（MR）：与句子对应的若干时间区间；
2. Highlight Detection（HL）：视频中每个 2 秒 clip 的显著性分数。

它不在训练时读取原始视频，也不负责提取 CLIP/SlowFast 特征。当前训练真正优化的是一个以 DETR 为骨架的时序定位网络。

```text
标注 JSONL
  ├─ qid ───────> CLIP 文本特征 <qid>.npy
  └─ vid ───────> SlowFast <vid>.npz + CLIP 视频 <vid>.npy
                         │
                         v
               StartEndDataset 拼接与补齐
                         │
                         v
             文本/视频投影 + 跨模态融合
                         │
                         v
              Transformer Encoder/Decoder
                ├─ QGR：约束 query 自注意力
                └─ GLB：query-to-frame 对齐参与匹配
                         │
                         v
       spans / class / IoU / saliency / frame masks
                         │
            ┌────────────┴────────────┐
            v                         v
      训练损失与 checkpoint      预测 JSONL 与指标 JSON
```

## 2. 当前工作区中的位置

| 内容 | Mac | 4090 |
|---|---|---|
| 官方代码 | `repos/Sim-DETR/` | `repos/Sim-DETR/` |
| 标注 JSONL | 官方仓库的 `data/` 内 | 官方仓库的 `data/` 内 |
| 大型特征 | 未同步 | `datasets/qvhighlights_features/` |
| 启动脚本与说明 | `repos/reproductions/sim-detr/` | 同路径 |
| checkpoint、预测、日志 | 未同步 | `repos/reproductions/sim-detr/runs/` |

官方仓库以 Git submodule 固定 revision。大型特征、环境、checkpoint 和日志不进入 Git。

## 3. 官方仓库逐文件说明

### 3.1 根目录

| 文件 | 作用 |
|---|---|
| `.gitignore` | 官方仓库自己的忽略规则；内容很少，并未完整忽略 Python 缓存。 |
| `README.md` | 官方快速开始，只详细说明 QVHighlights 训练、val/test 推理和 CodaLab 提交。 |
| `LICENSE` | 项目代码许可证。 |
| `assets/framework.png` | README 中展示的模型框架图。 |

### 3.2 `data/`：标注，不是视频特征

| 文件 | 作用 | 当前数据量 |
|---|---|---:|
| `data/README.md` | 解释 QVHighlights JSONL 字段和 test 无公开标签的事实。 | - |
| `data/LICENSE` | 数据标注的许可说明。 | - |
| `data/highlight_train_release.jsonl` | QVHighlights 训练标注，含 MR 窗口和 HL 显著性标签。 | 7,218 条 |
| `data/highlight_val_release.jsonl` | QVHighlights 验证标注，字段与 train 相同；训练选 best 和本地评测都使用它。 | 1,550 条 |
| `data/highlight_test_release.jsonl` | QVHighlights 测试输入；不含 GT，只能生成提交文件。 | 1,542 条 |
| `data/charades/charades_sta_train_tvr_format.jsonl` | Charades-STA 训练标注，只有时序窗口，没有 HL 标签。 | 12,404 条 |
| `data/charades/charades_sta_val_tvr_format.jsonl` | Charades-STA 验证标注。 | 3,720 条 |

`data/README.md` 还提到弱监督预训练用的 `subs_train.jsonl`，但当前 revision 并没有提交这个文件。

### 3.3 `sim_detr/`：模型、训练和推理主体

| 文件 | 作用 | 当前 QVHighlights 运行是否使用 |
|---|---|---|
| `__init__.py` | 将目录标记为 Python package；文件为空。 | 是，作为包结构 |
| `config.py` | 定义全部命令行参数；生成结果目录；保存 `opt.json`；复制 `model.py`、`transformer.py`；打包 `code.zip`。 | 是 |
| `start_end_dataset.py` | 读取 JSONL；按 `qid/vid` 加载文本和视频特征；归一化、拼接、生成 TEF、span 标签和显著性标签；提供 collate。 | 是 |
| `start_end_dataset_audio.py` | 上一文件的音频扩展版本，可以额外拼接音频特征。 | 否，本次未传 `a_feat_dir` |
| `model.py` | `SimDETR` 主模型、输出头、`SetCriterion` 和损失汇总，是理解模型输出的第一核心文件。 | 是 |
| `transformer.py` | Encoder/Decoder 主体、query 迭代更新、QGR competition matrix 和自注意力约束，是理解论文创新的第二核心文件。 | 是 |
| `attention.py` | 从 PyTorch MHA 改写的多头注意力；增加 `sa_decay`，用 QGR 关系矩阵缩放 query-query attention logits。 | 是 |
| `matcher.py` | Hungarian 一对一匹配；综合分类、span L1、gIoU 和 query-to-frame mask IoU 成本。 | 是 |
| `position_encoding.py` | 正弦或可学习的位置编码构造。 | 是 |
| `interaction/test_CQA.py` | 名字像测试文件，实际是正在使用的跨模态融合模块：`CQAttention`、`CQConcatenate`、`VSLFuser`。 | 是 |
| `loss_fun/CTCLoss.py` | clip-text alignment；使正相关视频 clip 与整句文本特征更相似。源码类名是 `CTC_Loss`。 | 是，权重 0.5 |
| `loss_fun/VTCLoss.py` | batch 内 video-text 对比学习，双向交叉熵。 | 是，权重 0.3 |
| `train.py` | 训练入口；建 DataLoader、生成 frame mask、前向/反向、逐 epoch val、保存 best/latest/周期 checkpoint，训练后调用 best 推理。 | 是 |
| `inference.py` | 加载 checkpoint；生成 MR 区间和 HL 分数；保存 submission；val 有 GT 时计算 metrics，test 只保存预测。 | 是 |
| `postprocessing_sim_detr.py` | 把预测时间裁剪到 `[0,150]`，并对齐到 2 秒 clip 边界。 | 是 |
| `span_utils.py` | `[start,end]` 与 `[center,width]` 互转；temporal IoU、generalized IoU。 | 是 |
| `misc.py` | 小型分类 accuracy 工具，主要用于记录 class error。 | 是 |
| `text_encoder.py` | 一个额外的 Transformer 文本编码器实现。当前主模型直接读取 CLIP 特征，没有调用它。 | 否 |

### 3.4 `sim_detr/scripts/`

| 文件 | 作用 |
|---|---|
| `scripts/train.sh` | 官方 QVHighlights 训练参数模板；必须手工改 `feat_root`。 |
| `scripts/inference.sh` | 根据 `val` 或 `test` 拼接标注路径并调用推理。 |
| `scripts/tvsum/train_tvsum.sh` | TVSum 纯视频 HL 分支的训练脚本。 |
| `scripts/tvsum/train_tvsum_audio.sh` | TVSum 视频+音频分支训练脚本。 |

我们没有直接使用官方 `train.sh`，而是使用工作区的 `repos/reproductions/sim-detr/scripts/train_qvhighlights.sh`，以便固定数据路径、seed 和输出目录。

### 3.5 `standalone_eval/`：脱离训练代码的评测器

| 文件 | 作用 |
|---|---|
| `README.md` | 定义预测 JSONL 格式，以及 val/test 打包上传 CodaLab 的方法。 |
| `eval.py` | 计算 MR mAP/R1 和 HL mAP/Hit@1；输出完整及 `brief` 指标。 |
| `utils.py` | AP、precision-recall、批量 temporal IoU 等底层函数。 |
| `eval_sample.sh` | 使用示例预测和 val GT 运行独立评测。 |
| `sample_val_preds.jsonl` | 官方提供的预测格式样例。 |
| `sample_val_preds_metrics.json` | 对样例预测重新计算得到的指标。 |
| `sample_val_preds_metrics_raw.json` | 官方随仓库提供的预期指标，用来核对评测环境。 |

### 3.6 `utils/`：通用工具

| 文件 | 作用 |
|---|---|
| `basic_utils.py` | JSON/JSONL/Pickle 读写、目录创建、L2 归一化、`code.zip` 打包、AverageMeter。 |
| `model_utils.py` | 统计模型参数量。 |
| `temporal_nms.py` | 一维时间窗口的 NMS。当前正式运行 `nms_thd=-1`，没有启用 NMS。 |
| `tensor_utils.py` | 对不同长度的一维/二维序列进行 padding 并生成 mask。 |
| `windows_utils.py` | clip id 与时间窗口之间的转换。当前主训练链很少直接使用。 |

### 3.7 为什么仓库里有大量 `__pycache__/*.pyc`

这些是 Python 3.7/3.9 编译缓存，不是另一套源码，也不应该作为研究入口。官方仓库错误地把不少 `.pyc` 提交到了 Git；其中甚至有只有 `.pyc`、没有对应 `.py` 的旧后处理名称。阅读时全部忽略，以 `.py` 文件为准。

## 4. QVHighlights 数据到底长什么样

### 4.1 一个 train/val JSONL 对象

```json
{
  "qid": 2579,
  "query": "A girl and her mother cooked while talking with each other on facetime.",
  "duration": 150,
  "vid": "NUsG9BgSes0_210.0_360.0",
  "relevant_clip_ids": [41, 42, 43],
  "saliency_scores": [[1, 1, 2], [1, 1, 3], [2, 1, 4]],
  "relevant_windows": [[82, 150]]
}
```

| 字段 | 含义 |
|---|---|
| `qid` | query 的唯一整数 ID，同时用于定位 `<qid>.npy` 文本特征。 |
| `query` | 自然语言描述；模型不现场 tokenize，而是读取预提取特征。 |
| `duration` | 视频片段秒数；QVHighlights 最大按 150 秒处理。 |
| `vid` | 视频 ID，格式通常为 `YouTubeID_原视频起点_终点`；用于定位视频特征文件。 |
| `relevant_windows` | 一个 query 对应的一个或多个 `[start,end]` 秒级 GT 区间。 |
| `relevant_clip_ids` | 落在相关区间中的 2 秒 clip 序号。 |
| `saliency_scores` | 每个相关 clip 的 3 位标注者打分；4 为 Very Good。 |

test 对象只有 `qid/query/duration/vid`，缺少最后三个 GT 字段。因此本地代码不能计算 test 指标。

### 4.2 split 统计

| split | query 数 | 不同视频数 | 有无 GT | 用途 |
|---|---:|---:|---|---|
| train | 7,218 | 7,100 | 有 | 参数优化 |
| val | 1,550 | 1,519 | 有 | 每轮验证、选 best、本地报告指标 |
| test | 1,542 | 1,529 | 无 | 生成 CodaLab submission |
| 合计 | 10,310 | - | - | 文本特征文件也正好有 10,310 个 |

### 4.3 4090 上实际使用的特征

| 目录 | 文件命名 | 单样本原始形状 | Dataset 中的处理 | 解压大小 |
|---|---|---|---|---:|
| `clip_b32_txt_k4/` | `<qid>.npy` | `(Lq, 4, 512)` float16 | 截到最多 32 token，L2 normalize，展平为 `(Lq, 2048)` | 587 MB |
| `clip_b32_vid_k4/` | `<vid>.npy` | `(Lv, 4, 768)` float16 | L2 normalize，展平为 `(Lv, 3072)` | 4.4 GB |
| `slowfast_features/` | `<vid>.npz`，key=`features` | `(Lv, 2304)` float16 | 读取 `features` 并 L2 normalize | 4.1 GB |

CLIP 视频和 SlowFast 在最后一维拼接为 `3072+2304=5376`。`ctx_mode=video_tef` 还会增加 `[clip_start, clip_end]` 两维 Temporal Endpoint Feature，因此送入投影层的实际视频维度是 5,378。

`max_v_l=75`、`clip_length=2`，所以最多处理 `75×2=150` 秒。不同特征长度不一致时，代码截到两者的最短长度。

三个 split 的文本、CLIP 视频、SlowFast 特征均已核对为零缺失。压缩包仍保留在同目录：510 MB、4.1 GB、3.7 GB；它们与解压目录是重复存储，不参与训练。

Mac 只有代码和标注，没有上述大型特征；真正训练必须在 4090 或另行准备相同目录。

## 5. 模型与训练链路

### 5.1 输入和融合

1. `StartEndDataset` 根据 `qid` 读文本特征，根据 `vid` 读两类视频特征。
2. 文本 2,048 维、视频+TEF 5,378 维分别通过 `LinearLayer` 投影到 256 维。
3. `VSLFuser` 先做 context-query attention，再把整句池化结果拼回每个视频位置。
4. 融合后的视频 token 与文本 token 拼成一条序列，交给 Transformer Encoder。

### 5.2 Decoder 输出

当前配置有 10 个 query、2 层 encoder、4 层 decoder。模型输出：

| 输出 key | 形状/含义 |
|---|---|
| `pred_logits` | `(B,10,2)`；每个 query 的前景/背景 logits。实际代码把类别 0 当前景。 |
| `pred_spans` | `(B,10,2)`；归一化 `[center,width]`，推理时转成秒级 `[start,end]`。 |
| `iou_scores` | `(B,10,1)`；每个 query 的预测定位质量。 |
| `pred_masks` | `(B,10,Lv)`；query 与每个视频 clip 的相似度。 |
| `saliency_scores` | `(B,Lv)`；HL 每个 clip 的分数。 |
| `aux_outputs` | 前几个 decoder 层的辅助输出，用于逐层监督。 |

推理时最终 MR 排名分数是：

```text
foreground softmax probability × sigmoid(predicted IoU)
```

每个 query 给出一个候选窗口，因此默认每个样本最多输出 10 个窗口。

### 5.3 两个核心设计如何落到代码

#### QGR：Query Grouping and Ranking

位置：`transformer.py` 的 `TransformerDecoder.forward` 和 `TransformerDecoderLayer.forward`。

从第二个 decoder layer 开始，代码：

1. 用预测 span 边界距离估计 query 两两空间相关性；
2. 用 `class_score × iou_score` 决定 query 两两排名；
3. 两者组合成 `competition_matrix`；
4. 经一个小 MLP 和 sigmoid 变成 `sa_decay`；
5. 在 `attention.py` 中直接乘到 query-query attention logits 上。

目的是让对应相似片段的 query 不再随机争抢目标，并让高质量 query 聚合同组信息。

#### GLB：Global-Local Bridging

位置：`model.py` 的 query-to-frame `pred_masks`、`matcher.py` 的 mask IoU cost，以及 `train.py` 生成的 frame mask GT。

每个匹配 query 与每个视频 clip 做余弦相似度，得到一条 frame mask；GT mask 根据 `relevant_windows` 将片段内部 clip 标 1、外部标 0。Hungarian matcher 使用 mask IoU cost 参与 query 与 GT 窗口的一对一匹配。

需要注意一个直接来自当前代码的事实：`build_model()` 最后强制把 `loss_mask_iou` 的优化权重设为 0，但 Hungarian matching 中 `set_cost_mask=6` 仍然生效。因此，当前 revision 中 frame mask 明确影响“匹配关系”，却没有作为一个非零的直接反向传播 loss 加入总损失。这一点阅读或修改代码时必须留意。

### 5.4 当前训练的损失

| 损失 | 当前权重 | 作用 |
|---|---:|---|
| span L1 | 10 | 预测中心和宽度接近 GT。 |
| generalized temporal IoU | 1 | 时间区间重合质量。 |
| focal classification | 4 | query 前景/背景分类。 |
| saliency | 1 | HL 排序，包括正负 clip margin 与 rank contrastive 部分。 |
| predicted IoU regression | 2 | 训练 IoU head，为 QGR 排名及最终置信度提供局部质量。 |
| frame mask IoU | 0 | 被计算但总损失权重被强制设为 0；matcher cost 仍为 6。 |
| VTC | 0.3 | batch 内 video-text 对比学习。 |
| CTC | 0.5 | clip 与整句文本的相关/不相关二分类对齐。 |

Hungarian matcher 的 cost 权重为 span 10、gIoU 1、class 4、mask IoU 6。

### 5.5 当前正式训练配置

| 项目 | 数值 |
|---|---|
| seed | 2017 |
| epoch | 200 |
| batch size | 32 |
| optimizer | AdamW |
| learning rate | `1e-4` |
| LR drop | 第 100 epoch 起降为 1/10 |
| queries | 10 |
| encoder / decoder | 2 / 4 |
| hidden dimension | 256 |
| max query/video length | 32 token / 75 clips |
| NMS | 关闭，`nms_thd=-1` |

每个 epoch 都在 val 上计算指标。`MR-full-mAP` 创新高时保存 `model_best.ckpt`，每轮覆盖 `model_latest.ckpt`，每 50 epoch 额外保存一个周期 checkpoint。

## 6. 那么多 JSON/JSONL 到底是什么

### 6.1 JSON 与 JSONL 的区别

- `.json`：整个文件是一个 JSON 对象，适合配置或汇总指标；
- `.jsonl`：每一行是一个独立 JSON 对象，适合 1,550 条预测流式读写。

不能把 `.jsonl` 当成一个大 JSON 数组直接 `json.load()`；代码使用 `load_jsonl()` 逐行解析。

### 6.2 输入标注 JSONL

| 文件 | 谁读取 | 作用 |
|---|---|---|
| `highlight_train_release.jsonl` | `StartEndDataset` | 训练输入和 GT。 |
| `highlight_val_release.jsonl` | 训练期 eval 与独立 inference | 选 best、计算本地 MR/HL 指标。 |
| `highlight_test_release.jsonl` | test inference | 只有输入元数据，生成 submission，不计算指标。 |

### 6.3 正式运行目录中的预测 JSONL

运行目录：

```text
repos/reproductions/sim-detr/runs/qvhighlights/20260819-official-seed2017/
```

| 文件 | 内容 | 当前关系 |
|---|---|---|
| `best_hl_val_preds.jsonl` | 训练期间 val mAP 最优时保存的 1,550 条预测。 | best checkpoint 预测 |
| `latest_hl_val_preds.jsonl` | 第 200 epoch 的最后一次 val 预测。 | latest checkpoint 预测 |
| `hl_val_submission.jsonl` | 训练结束后重新加载 best checkpoint 生成的标准提交命名文件。 | 与 `best_hl_val_preds.jsonl` 字节完全一致 |

每行预测包含：

```json
{
  "qid": 2579,
  "query": "...",
  "vid": "...",
  "pred_relevant_windows": [[118.0, 150.0, 0.846], [92.0, 148.0, 0.0256]],
  "pred_saliency_scores": [-0.667, -0.657, -0.682]
}
```

- `pred_relevant_windows`：`[开始秒, 结束秒, MR 排名分数]`，默认最多 10 个；
- `pred_saliency_scores`：按时间排列，每个 2 秒 clip 一个 HL 分数；只看相对大小，不要求在 `[0,1]`。

### 6.4 指标 JSON

| 文件 | 对应预测 | 当前关系 |
|---|---|---|
| `best_hl_val_preds_metrics.json` | `best_hl_val_preds.jsonl` | best 指标 |
| `latest_hl_val_preds_metrics.json` | `latest_hl_val_preds.jsonl` | epoch 200 指标 |
| `hl_val_submission_metrics.json` | `hl_val_submission.jsonl` | 与 best metrics 字节完全一致 |

指标 JSON 顶层结构：

| key | 含义 |
|---|---|
| `brief` | 最常看的扁平汇总指标。 |
| `full` | 全时长样本的 MR mAP 与 R1，包含多个 IoU 阈值。 |
| `short` | GT 窗口长度 `(0,10]` 秒。 |
| `middle` | GT 窗口长度 `(10,30]` 秒。 |
| `long` | GT 窗口长度 `(30,150]` 秒。 |
| `HL-min-Fair` | 将标注分数至少为 2 的 clip 当正例。 |
| `HL-min-Good` | 至少为 3。 |
| `HL-min-VeryGood` | 至少为 4。 |

MR 的 `Avg. mAP` 是 IoU 0.50、0.55、…、0.95 十个阈值的平均；R1@0.5 表示每条 query 排名第一的窗口是否以至少 0.5 IoU 命中任一 GT。HL 报告 mAP 和 Hit@1。

### 6.5 `opt.json`

这是本次运行的完整参数快照。测试/独立推理加载 checkpoint 时，也会从 checkpoint 所在目录读取它，以恢复特征路径、维度和模型结构。

其中的 `results_root/results_dir` 仍是迁移前的历史绝对路径，因为运行产物后来整体移动到了 `repos/reproductions/`。这是历史记录，不代表数据仍在旧位置。输入特征绝对路径目前仍然有效。

### 6.6 为什么没有 test JSON

正式目录目前没有 `hl_test_submission.jsonl`，说明 test inference 尚未执行。完整 test 流程应生成：

```text
hl_val_submission.jsonl
hl_test_submission.jsonl
```

然后将两者放在一个无外层目录的 ZIP 中上传 CodaLab。test 没有公开 GT，因此不会产生 `hl_test_submission_metrics.json`；真正指标由服务器返回。

## 7. 运行目录中其他文件

| 文件/目录 | 作用 | 是否核心保留 |
|---|---|---|
| `model_best.ckpt` | val MR Avg. mAP 最好的模型、优化器、scheduler、epoch 和 opt；约 142 MB。 | 必留 |
| `model_latest.ckpt` | epoch 200 最后模型，可续训；约 142 MB。 | 建议保留 |
| `model_e0049/e0099/e0149/e0199.ckpt` | 第 50/100/150/200 epoch 周期快照。 | 完成分析后可归档或删除 |
| `train.log.txt` | 每个 epoch 的训练 loss 摘要。 | 建议保留 |
| `eval.log.txt` | 每个 epoch 的 val 指标。 | 建议保留 |
| `console.log` | stdout/stderr 完整输出，包括进度条和报错。 | 建议保留 |
| `tensorboard_log/` | TensorBoard event，画 loss/metric 曲线。 | 需要看曲线时保留 |
| `model.py` | `config.py` 自动复制的训练时源码快照。 | 与固定 revision 重复 |
| `transformer.py` | 同上。 | 与固定 revision 重复 |
| `code.zip` | 整个 `sim_detr/` 训练时代码快照；已包含前两个文件。 | 与固定 revision 重复 |
| `legacy-launcher.log/.pid` | 迁移前外层串行启动器的日志和 PID。 | 只具历史意义 |
| `legacy-pipeline-status.tsv` | 迁移前 Sim-DETR/FlashVTG 串行任务状态。 | 只具历史意义 |

视觉上文件很多，但空间主要由 6 个约 142 MB 的 checkpoint 占用；源码快照只有约 0.3 MB。

## 8. 当前结果应怎样解释

best checkpoint 对应第 160 个 epoch。val 结果：

| 指标 | 本次 val | 论文 val | 论文 test |
|---|---:|---:|---:|
| R1@0.5 | 68.32 | 69.48 | 67.64 |
| R1@0.7 | 53.23 | 54.06 | 50.91 |
| mAP@0.5 | 69.17 | 69.70 | 67.81 |
| mAP@0.75 | 49.77 | 51.11 | 47.59 |
| Avg. mAP | 48.94 | 49.50 | 46.93 |

结论是：本次 val 略低于论文 val；虽然数值高于论文 test，但二者 split 不同，不能据此声称 test 超过论文。当前还没有本次模型的 test server 指标。

`latest_hl_val_preds_metrics.json` 的 MR Avg. mAP 为 48.81，略低于 best 的 48.94；这解释了为何应该使用 `model_best.ckpt` 而不是 `model_latest.ckpt` 做最终提交。

## 9. 阅读代码时需要特别留意的问题

1. **README 不完整**：没有锁定依赖版本，也没有完整说明论文中的 Charades-STA/TACoS 复现。
2. **官方仓库提交了大量 `.pyc`**：忽略它们，只读 `.py`。
3. **`test_CQA.py` 不是测试**：它是主模型实际调用的视觉-文本融合器。
4. **类别注释与实际实现相反**：`model.py` 某处注释写“0 background, 1 foreground”，但 criterion、matcher 和 inference 实际都把类别 0 当 foreground。以执行代码为准。
5. **frame mask loss 权重为 0**：mask IoU 仍进入 Hungarian matching，但不作为非零直接损失回传。
6. **硬编码 CUDA**：多处直接 `.cuda()` 或 `torch.cuda.set_device()`，不能仅靠 `--device -1` 完整切到 CPU。
7. **test 只生成预测**：`inference.py` 明确在 `eval_split_name == 'test'` 时跳过本地 metrics。
8. **训练输出自动膨胀**：`config.py` 每次运行都复制两个源码文件并创建 `code.zip`。
9. **PyTorch 2.6+ checkpoint 兼容**：checkpoint 内保存了 `argparse.Namespace`；当前环境需对可信自生成 checkpoint 设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
10. **`opt.json` 有历史输出路径**：结果迁移后没有改写参数快照，避免把历史配置误认为当前目录结构。

## 10. 建议的代码阅读顺序

如果目标是尽快掌握项目，不要从 900 行的 `transformer.py` 第一行顺读。建议：

1. `repos/reproductions/sim-detr/scripts/train_qvhighlights.sh`：先看本次到底传了什么参数；
2. `sim_detr/train.py:start_training()`：理解对象怎样组装；
3. `sim_detr/start_end_dataset.py`：理解 JSONL 如何变成 tensor；
4. `sim_detr/model.py:SimDETR.forward()`：理解输入输出；
5. `sim_detr/transformer.py:TransformerDecoder.forward()`：看 QGR competition matrix；
6. `sim_detr/transformer.py:TransformerDecoderLayer.forward()` 与 `attention.py`：看 `sa_decay` 如何进入自注意力；
7. `sim_detr/matcher.py` 和 `model.py:SetCriterion`：看匹配与所有 loss；
8. `sim_detr/inference.py:compute_mr_results()`：看预测 JSONL 如何生成；
9. `standalone_eval/eval.py`：看论文指标如何计算。

读完这九处，就能掌握当前 QVHighlights 复现的完整主路径。TVSum、音频分支、`text_encoder.py` 和旧 `.pyc` 可以最后再看，当前正式运行没有使用。
