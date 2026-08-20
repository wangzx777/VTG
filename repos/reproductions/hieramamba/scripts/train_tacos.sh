#!/usr/bin/env bash

set -u

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$script_dir/../../../.." && pwd)
run_id=${RUN_ID:-$(date '+%Y%m%d-%H%M%S')-official-seed1234567891}
run_root="$workspace/repo/reproductions/hieramamba/runs/tacos/$run_id"
status_file="$run_root/status.tsv"

mkdir -p "$run_root"
printf 'event\ttimestamp\texit_code\nstarted\t%s\t-\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$status_file"

(
  source "$workspace/.venv/bin/activate"
  cd "$workspace/repos/HieraMamba"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONUNBUFFERED=1 \
    python "$script_dir/train_compat.py" \
      --opt tacos_hieramamba.yaml \
      --name "$run_root"
) > "$run_root/console.log" 2>&1
exit_code=$?

printf 'finished\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$exit_code" >> "$status_file"
exit "$exit_code"
