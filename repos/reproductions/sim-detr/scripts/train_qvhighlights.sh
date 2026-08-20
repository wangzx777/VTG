#!/usr/bin/env bash

set -u

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$script_dir/../../../.." && pwd)
run_id=${RUN_ID:-$(date '+%Y%m%d-%H%M%S')-official-seed2017}
run_root="$workspace/repos/reproductions/sim-detr/runs/qvhighlights/$run_id"
status_file="$run_root/status.tsv"

mkdir -p "$run_root"
printf 'event\ttimestamp\texit_code\nstarted\t%s\t-\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$status_file"

(
  source "$workspace/.venv/bin/activate"
  cd "$workspace/repos/Sim-DETR"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD" \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    python sim_detr/train.py \
      --seed 2017 \
      --label_loss_coef 4 \
      --VTC_loss_coef 0.3 \
      --CTC_loss_coef 0.5 \
      --dset_name hl \
      --ctx_mode video_tef \
      --train_path data/highlight_train_release.jsonl \
      --eval_path data/highlight_val_release.jsonl \
      --eval_split_name val \
      --v_feat_dirs \
        "$workspace/datasets/qvhighlights_features/slowfast_features" \
        "$workspace/datasets/qvhighlights_features/clip_b32_vid_k4" \
      --v_feat_dim 5376 \
      --t_feat_dir "$workspace/datasets/qvhighlights_features/clip_b32_txt_k4" \
      --t_feat_dim 2048 \
      --bsz 32 \
      --results_root "$run_root" \
      --exp_id official-qvhighlights \
      --lr 0.0001 \
      --n_epoch 200 \
      --lw_saliency 1.0 \
      --lr_drop 100 \
      --dec_layers 4 \
      --enc_layers 2
) > "$run_root/console.log" 2>&1
exit_code=$?

printf 'finished\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$exit_code" >> "$status_file"
exit "$exit_code"
