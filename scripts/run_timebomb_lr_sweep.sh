#!/usr/bin/env bash
# Launch a 404M long-horizon learning-rate sweep.

set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
wandb_mode="${WANDB_MODE:-offline}"
run_prefix="${RUN_PREFIX:-timebomb-lr}"
precision="${PRECISION:-bf16_mixed}"
seeds_string="${SEEDS:-0}"
methods_string="${METHODS:-baseline}"
lrs_string="${LRS:-3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1}"

read -r -a seeds <<< "${seeds_string}"
read -r -a methods <<< "${methods_string}"
read -r -a lrs <<< "${lrs_string}"

for method in "${methods[@]}"; do
  for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${run_prefix}-${precision}-${method}-lr${lr}-seed${seed}"
      "${python_bin}" main.py \
        +model=lm404m \
        +dataset=fw_gpt2_100B \
        +experiment=timebomb_404m \
        +precision="${precision}" \
        +method="${method}" \
        opt.peak_lr="${lr}" \
        seed="${seed}" \
        wandb_mode="${wandb_mode}" \
        run_name="${run_name}" \
        "$@"
    done
  done
done
