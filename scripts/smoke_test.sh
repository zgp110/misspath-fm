#!/usr/bin/env bash
# Fast end-to-end sanity check (a few minutes on one GPU, also runs on CPU).
# Trains a tiny model on the synthetic dataset, then reloads the checkpoint
# and evaluates it. No external data required.
set -euo pipefail

GPU="${1:-0}"
cd "$(dirname "$0")/.."

python -u train.py \
  --dataset synthetic \
  --mechanism mcar \
  --missing_rate 0.3 \
  --epochs 10 \
  --eval_every 5 \
  --hidden_dim 64 \
  --n_layers 2 \
  --n_samples 3 \
  --gpu "${GPU}" \
  --tag smoke \
  --ckpt_dir ./checkpoints/smoke \
  --result_dir ./results/smoke

python -u evaluate.py \
  --ckpt ./checkpoints/smoke/smoke.pt \
  --n_samples 3 \
  --gpu "${GPU}" \
  --result_json ./results/smoke/smoke_eval.json

echo "[smoke] OK"
