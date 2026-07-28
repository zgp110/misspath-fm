#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./logs/physionet}"
mkdir -p "$LOG_DIR"

DATASET="physionet"
RATES=(0.1 0.2 0.3 0.5)

cd "$(dirname "$0")/.."

for rate in "${RATES[@]}"; do
  tag="${DATASET}_art${rate/./}"
  echo "[$(date +%H:%M:%S)] >>> ${tag}  (gpu=${GPU})"
  python -u train.py \
    --dataset "${DATASET}" \
    --missing_rate "${rate}" \
    --gpu "${GPU}" \
    --data_dir "${DATA_DIR}" \
    --tag "${tag}" \
    2>&1 | tee "${LOG_DIR}/${tag}.log"
done
