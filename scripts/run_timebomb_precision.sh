#!/usr/bin/env bash
# Launch a 28M LM-head precision comparison.

set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
wandb_mode="${WANDB_MODE:-offline}"
run_prefix="${RUN_PREFIX:-timebomb-precision}"
seeds_string="${SEEDS:-0 1 2 3 4}"
precisions_string="${PRECISIONS:-bf16_true bf16_mixed fp32_lm_head}"
lrs_string="${LRS:-0.1}"

read -r -a seeds <<< "${seeds_string}"
read -r -a precisions <<< "${precisions_string}"
read -r -a lrs <<< "${lrs_string}"

for precision in "${precisions[@]}"; do
  for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${run_prefix}-${precision}-lr${lr}-seed${seed}"
      "${python_bin}" main.py \
        +model=lm28m \
        +dataset=fw_gpt2_100B \
        +experiment=timebomb_28m \
        +precision="${precision}" \
        +method=baseline \
        opt.peak_lr="${lr}" \
        seed="${seed}" \
        wandb_mode="${wandb_mode}" \
        run_name="${run_name}" \
        "$@"
    done
  done
done
