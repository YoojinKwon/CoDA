#!/bin/bash
# Table 4 (ResNet50, batch_size=4) dispatcher: 7 combos over 4 GPUs, one
# combo at a time per GPU.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_COMBO="$ROOT/run_combo.sh"
Y="$ROOT/yaml"

mkdir -p "$ROOT/run_logs"

run_lane() {
  gpu_id=$1; shift
  for yaml in "$@"; do bash "$RUN_COMBO" "$yaml" "$gpu_id"; done
}

run_lane 0 "$Y/table4_resnet50_fp.yaml" "$Y/table4_resnet50_lsq2bit.yaml" > "$ROOT/run_logs/lane_gpu0.master.log" 2>&1 &
run_lane 1 "$Y/table4_resnet50_lq2bit.yaml" "$Y/table4_resnet50_lsq4bit.yaml" > "$ROOT/run_logs/lane_gpu1.master.log" 2>&1 &
run_lane 2 "$Y/table4_resnet50_lq4bit.yaml" "$Y/table4_resnet50_lsq8bit.yaml" > "$ROOT/run_logs/lane_gpu2.master.log" 2>&1 &
run_lane 3 "$Y/table4_resnet50_lq8bit.yaml" > "$ROOT/run_logs/lane_gpu3.master.log" 2>&1 &
wait
fails=$(grep -l Traceback "$ROOT"/run_logs/table4_*.log 2>/dev/null | wc -l)
echo "Table 4 ALL DONE at $(date); jobs with Traceback: $fails"
