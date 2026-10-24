#!/bin/bash
# Table 2 dispatcher: fan the 14 combos out across GPUs (default 4), one
# combo running at a time per GPU (its 15 corruptions run sequentially
# inside run_combo.sh to avoid concurrent writers on the same adapt_result.csv).
# Usage: launch_phase1.sh [num_gpus]
set -u

RESULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="$RESULT_ROOT/yaml"
RUN_COMBO="$RESULT_ROOT/run_combo.sh"
NUM_GPUS="${1:-4}"

mkdir -p "$RESULT_ROOT/run_logs"
mapfile -t YAMLS < <(ls "$YAML_DIR"/table2_*.yaml | sort)
echo "Found ${#YAMLS[@]} Table 2 combos"

declare -a LANES
for ((g=0; g<NUM_GPUS; g++)); do LANES[$g]=""; done

for i in "${!YAMLS[@]}"; do
  g=$((i % NUM_GPUS))
  LANES[$g]="${LANES[$g]} ${YAMLS[$i]}"
done

run_lane() {
  gpu_id=$1
  shift
  for yaml in "$@"; do
    bash "$RUN_COMBO" "$yaml" "$gpu_id"
  done
}

pids=()
for ((g=0; g<NUM_GPUS; g++)); do
  if [ -n "${LANES[$g]}" ]; then
    run_lane "$g" ${LANES[$g]} > "$RESULT_ROOT/run_logs/lane_gpu${g}.master.log" 2>&1 &
    pids+=($!)
    echo "Launched lane GPU${g} (pid $!) with combos:${LANES[$g]}"
  fi
done

echo "Waiting for ${#pids[@]} lanes to finish..."
wait "${pids[@]}"
echo "Table 2 ALL DONE at $(date)"
