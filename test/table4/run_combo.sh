#!/bin/bash
# Table 4 reproduction: run one (model, q_scheme, bits) combo over all 15
# ImageNet-C corruption types, sequentially, on one GPU.
# Usage: run_combo.sh <yaml_path> <gpu_id>
set -u

YAML="$1"
GPU="$2"

RESULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RESULT_ROOT/../.." && pwd)"
PY=python
SCRIPT="$REPO_ROOT/coda/test_tta_per_corruption.py"
NAME=$(basename "$YAML" .yaml)

CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

mkdir -p "$RESULT_ROOT/run_logs"
cd "$REPO_ROOT" || exit 1

for c in $CORRUPTIONS; do
  LOG="$RESULT_ROOT/run_logs/${NAME}__${c}.log"
  echo "[$(date +%H:%M:%S)] GPU${GPU} ${NAME} ${c} start"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT" \
    --yaml "$YAML" \
    --tta_method coda \
    --corr_type "$c" \
    > "$LOG" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] GPU${GPU} ${NAME} ${c} done (exit ${status})"
done
