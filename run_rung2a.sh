#!/usr/bin/env bash
# run_rung2a.sh — full rung-2a pipeline: data -> both arms -> plot.
# Runs nice (Miu is a daily-use workstation). Activate the venv first.
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NICE="nice -n 10"

echo "== 1/4 data prep (cached; skips if present) =="
$NICE python data_prep.py

echo "== 2/4 train FULL attention (baseline) =="
$NICE python train_rung2a.py --attn full "$@"

echo "== 3/4 train NSA =="
$NICE python train_rung2a.py --attn nsa "$@"

echo "== 4/4 plot val-loss overlay =="
$NICE python plot_val.py

echo "done. see plots/val_loss.png"
