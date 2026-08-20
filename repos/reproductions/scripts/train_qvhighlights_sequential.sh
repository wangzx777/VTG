#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
reproduction_root=$(cd "$script_dir/.." && pwd)
run_date=${RUN_DATE:-$(date '+%Y%m%d')}

RUN_ID="${SIM_DETR_RUN_ID:-${run_date}-official-seed2017}" \
  "$reproduction_root/sim-detr/scripts/train_qvhighlights.sh"

RUN_ID="${FLASHVTG_RUN_ID:-${run_date}-official-seed2024}" \
  "$reproduction_root/flashvtg/scripts/train_qvhighlights.sh"
