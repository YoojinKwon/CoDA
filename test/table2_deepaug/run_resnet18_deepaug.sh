#!/bin/bash
# Table 2+A (DeepAug) reproduction: ResNet18 FP/LSQ-2/4/8bit, DeepAug-trained
# checkpoints, over all 15 ImageNet-C corruptions.
# Usage: run_resnet18_deepaug.sh [gpu_id]
set -u

GPU="${1:-0}"
RESULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RESULT_ROOT/../.." && pwd)"
PY=python
SCRIPT="$REPO_ROOT/coda/test_tta_per_corruption.py"
YAML_DIR="$RESULT_ROOT/yaml"

CORRS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
YAMLS="table2_resnet18_fp_deepaug table2_resnet18_lsq2bit_deepaug table2_resnet18_lsq4bit_deepaug table2_resnet18_lsq8bit_deepaug"

mkdir -p "$RESULT_ROOT/run_logs"
cd "$REPO_ROOT" || exit 1
for y in $YAMLS; do
  YAML="$YAML_DIR/${y}.yaml"
  for c in $CORRS; do
    LOG="$RESULT_ROOT/run_logs/${y}__${c}.log"
    echo "[$(date +%H:%M:%S)] GPU${GPU} ${y} ${c} start"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT" --yaml "$YAML" --tta_method coda --corr_type "$c" > "$LOG" 2>&1
    status=$?
    echo "[$(date +%H:%M:%S)] GPU${GPU} ${y} ${c} done (exit ${status})"
  done
done
echo "LANE_A (resnet18) ALL DONE at $(date)"
