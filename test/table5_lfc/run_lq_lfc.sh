#!/bin/bash
# Table 5 LFC reproduction: ResNet26 LQ 4-bit/8-bit, genuine LFC-QAT (r0r8)
# checkpoints, over all 15 CIFAR10-C corruptions.
# Usage: run_lq_lfc.sh [gpu_id]
set -u
GPU="${1:-0}"
RESULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RESULT_ROOT/../.." && pwd)"
PY=python
SCRIPT="$REPO_ROOT/coda/test_tta_per_corruption.py"
CORRS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
YAMLS="table5_resnet26_lq4bit_lfc table5_resnet26_lq8bit_lfc"

mkdir -p "$RESULT_ROOT/run_logs"
cd "$REPO_ROOT" || exit 1
for y in $YAMLS; do
  YAML="$RESULT_ROOT/yaml/${y}.yaml"
  for c in $CORRS; do
    LOG="$RESULT_ROOT/run_logs/${y}__${c}.log"
    echo "[$(date +%H:%M:%S)] GPU${GPU} ${y} ${c} start"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT" --yaml "$YAML" --tta_method coda --corr_type "$c" > "$LOG" 2>&1
    status=$?
    echo "[$(date +%H:%M:%S)] GPU${GPU} ${y} ${c} done (exit ${status})"
  done
done
echo "LANE_LQ ALL DONE at $(date)"
