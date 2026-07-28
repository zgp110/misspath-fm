#!/usr/bin/env bash
# Component ablation on a single (dataset, mechanism, rate) cell.
#   usage: bash scripts/run_ablation.sh [GPU] [DATASET] [MECHANISM] [RATE]
#
# Variants:
#   full        interpolation prior + missingness-aware path + full context
#   no_prior    standard Gaussian source          (prior ablation)
#   no_path     standard linear path              (path ablation)
#   mask_only   velocity sees mask but not gaps   (context ablation)
#   gap_only    velocity sees gaps but not mask   (context ablation)
#   no_context  velocity sees x_t only            (context ablation)
set -euo pipefail

GPU="${1:-0}"
DATASET="${2:-ettm1}"
MECH="${3:-mcar}"
RATE="${4:-0.3}"
DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./logs/ablation}"
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")/.."

run () {  # name, extra args...
  local name="$1"; shift
  local tag="ablation_${DATASET}_${MECH}${RATE/./}_${name}"
  echo "[$(date +%H:%M:%S)] >>> ${tag}  (gpu=${GPU})"
  python -u train.py \
    --dataset "${DATASET}" --mechanism "${MECH}" --missing_rate "${RATE}" \
    --gpu "${GPU}" --data_dir "${DATA_DIR}" --tag "${tag}" "$@" \
    2>&1 | tee "${LOG_DIR}/${tag}.log"
}

run full
run no_prior    --prior gaussian
run no_path     --path standard
run mask_only   --context mask_only
run gap_only    --context gap_only
run no_context  --context no_mask
