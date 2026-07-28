#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./logs/ettm1}"
mkdir -p "$LOG_DIR"

DATASET="ettm1"
MECHANISMS=(mcar block mnar irregular)
RATES=(0.3 0.5)

cd "$(dirname "$0")/.."

for mech in "${MECHANISMS[@]}"; do
  for rate in "${RATES[@]}"; do
    tag="${DATASET}_${mech}${rate/./}"
    echo "[$(date +%H:%M:%S)] >>> ${tag}  (gpu=${GPU})"
    python -u train.py \
      --dataset "${DATASET}" \
      --mechanism "${mech}" \
      --missing_rate "${rate}" \
      --gpu "${GPU}" \
      --data_dir "${DATA_DIR}" \
      --tag "${tag}" \
      2>&1 | tee "${LOG_DIR}/${tag}.log"
  done
done
