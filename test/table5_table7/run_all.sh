#!/bin/bash
# Table 5 (ResNet26 fp/lsq2bit/lq2bit) + Table 7 (MobileNetV3-small/large,
# EfficientNet-b0, lsq4bit/8bit, k=4/k=64) reproduction, sequentially on one GPU.
# Usage: run_all.sh [gpu_id]
set -u
GPU="${1:-0}"
RESULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RESULT_ROOT/../.." && pwd)"
PY=python
SCRIPT="$REPO_ROOT/coda/test_tta_per_corruption.py"

CORRS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

YAMLS="table5_resnet26_fp table5_resnet26_lsq2bit table5_resnet26_lq2bit \
table7_mbv3small_lsq4bit_k4 table7_mbv3small_lsq8bit_k4 table7_mbv3small_lsq4bit_k64 table7_mbv3small_lsq8bit_k64 \
table7_mbv3large_lsq4bit_k4 table7_mbv3large_lsq8bit_k4 table7_mbv3large_lsq4bit_k64 table7_mbv3large_lsq8bit_k64 \
table7_effb0_lsq4bit_k4 table7_effb0_lsq8bit_k4 table7_effb0_lsq4bit_k64 table7_effb0_lsq8bit_k64"

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
echo "ALL DONE at $(date)"
