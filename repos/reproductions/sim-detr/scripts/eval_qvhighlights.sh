#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/model_best.ckpt" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$script_dir/../../../.." && pwd)
checkpoint=$1

source "$workspace/.venv/bin/activate"
cd "$workspace/repos/Sim-DETR"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH="$PWD" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python -m sim_detr.inference \
    --resume "$checkpoint" \
    --eval_split_name val \
    --eval_path data/highlight_val_release.jsonl
