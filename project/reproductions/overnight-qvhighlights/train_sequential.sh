#!/usr/bin/env bash

set -u

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$script_dir/../../.." && pwd)
run_id=${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}
run_root="$script_dir/runs/$run_id"
status_file="$run_root/status.tsv"

mkdir -p "$run_root/sim-detr" "$run_root/flashvtg"
printf 'stage\tevent\ttimestamp\texit_code\n' > "$status_file"

record_status() {
  local stage=$1
  local event=$2
  local exit_code=${3:--}
  printf '%s\t%s\t%s\t%s\n' \
    "$stage" "$event" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$exit_code" \
    >> "$status_file"
}

run_sim_detr() {
  local output_root="$run_root/sim-detr"
  record_status sim-detr started

  (
    source "$workspace/.venv/bin/activate"
    cd "$workspace/repos/Sim-DETR"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
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
        --results_root "$output_root" \
        --exp_id official-qvhighlights \
        --lr 0.0001 \
        --n_epoch 200 \
        --lw_saliency 1.0 \
        --lr_drop 100 \
        --dec_layers 4 \
        --enc_layers 2
  ) > "$run_root/sim-detr-console.log" 2>&1
  local exit_code=$?
  record_status sim-detr finished "$exit_code"
  return "$exit_code"
}

run_flashvtg() {
  local output_root="$run_root/flashvtg"
  record_status flashvtg started

  (
    source "$workspace/repos/FlashVTG/.venv/bin/activate"
    cd "$workspace/repos/FlashVTG"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
      python FlashVTG/train.py \
        data/MR_16.py \
        --dset_name qv_internvideo2 \
        --ctx_mode video_tef \
        --train_path data/highlight_train_release_IV2.jsonl \
        --eval_path data/highlight_val_release.jsonl \
        --eval_split_name val \
        --v_feat_dirs \
          /media/jia/MyProject/flashvtg/QVHighlights/features/internvideo2_video/qvhighlight_6b \
        --v_feat_dim 768 \
        --t_feat_dir \
          /media/jia/MyProject/flashvtg/QVHighlights/features/llama_text/qvhighlight_llama_text_feature \
        --t_feat_dim 4096 \
        --enc_layers 3 \
        --results_root "$output_root" \
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
  ) > "$run_root/flashvtg-console.log" 2>&1
  local exit_code=$?
  record_status flashvtg finished "$exit_code"
  return "$exit_code"
}

record_status pipeline started
run_sim_detr
sim_exit=$?
run_flashvtg
flash_exit=$?

if [[ $sim_exit -eq 0 && $flash_exit -eq 0 ]]; then
  pipeline_exit=0
else
  pipeline_exit=1
fi

record_status pipeline finished "$pipeline_exit"
exit "$pipeline_exit"
