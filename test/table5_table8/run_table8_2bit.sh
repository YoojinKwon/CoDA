#!/bin/bash
# Table 8, 2-bit combos (MobileNetV3-small/large, EfficientNet-b0, k=4/k=64).
# Usage: run_table8_2bit.sh [gpu_id]
set -u
GPU="${1:-0}"
RESULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RESULT_ROOT/../.." && pwd)"
PY=python
SCRIPT="$REPO_ROOT/coda/test_tta_per_corruption.py"
CORRS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
YAMLS="table8_mbv3small_lsq2bit_k4 table8_mbv3small_lsq2bit_k64 table8_mbv3large_lsq2bit_k4 table8_mbv3large_lsq2bit_k64 table8_effb0_lsq2bit_k4 table8_effb0_lsq2bit_k64"

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
