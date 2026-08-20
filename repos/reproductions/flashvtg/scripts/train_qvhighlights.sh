#!/usr/bin/env bash

set -u

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$script_dir/../../../.." && pwd)
run_id=${RUN_ID:-$(date '+%Y%m%d-%H%M%S')-official-seed2024}
run_root="$workspace/repos/reproductions/flashvtg/runs/qvhighlights/$run_id"
status_file="$run_root/status.tsv"
feature_root=${FLASHVTG_QV_FEATURE_ROOT:-/media/jia/MyProject/flashvtg/QVHighlights/features}

mkdir -p "$run_root"
printf 'event\ttimestamp\texit_code\nstarted\t%s\t-\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$status_file"

(
  source "$workspace/repos/FlashVTG/.venv/bin/activate"
  cd "$workspace/repos/FlashVTG"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD" \
    python FlashVTG/train.py \
      data/MR_16.py \
      --seed 2024 \
      --dset_name qv_internvideo2 \
      --ctx_mode video_tef \
      --train_path data/highlight_train_release_IV2.jsonl \
      --eval_path data/highlight_val_release.jsonl \
      --eval_split_name val \
      --v_feat_dirs "$feature_root/internvideo2_video/qvhighlight_6b" \
      --v_feat_dim 768 \
      --t_feat_dir "$feature_root/llama_text/qvhighlight_llama_text_feature" \
      --t_feat_dim 4096 \
      --enc_layers 3 \
      --results_root "$run_root" \
      --bsz 64 \
      --exp_id official-qvhighlights-internvideo2 \
      --t2v_layers 6 \
      --dummy_layers 2 \
      --max_v_l 75 \
      --max_q_l 40 \
      --n_epoch 150 \
      --lr_drop 400 \
      --eval_epoch 5 \
      --wd 0.0001 \
      --eval_bsz 1 \
      --lw_reg 1 \
      --lw_cls 5 \
      --lw_sal 0.1 \
      --lw_saliency 0.8 \
      --nms_thd 0.7 \
      --use_neg \
      --num_dummies 40 \
      --kernel_size 5 \
      --num_conv_layers 1 \
      --num_mlp_layers 5 \
      --label_loss_coef 0
) > "$run_root/console.log" 2>&1
exit_code=$?

printf 'finished\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$exit_code" >> "$status_file"
exit "$exit_code"
