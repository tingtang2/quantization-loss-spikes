#!/usr/bin/env bash
# Launch a 162M fast-spike learning-rate sweep.

set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
wandb_mode="${WANDB_MODE:-offline}"
run_prefix="${RUN_PREFIX:-fast-spike}"
seeds_string="${SEEDS:-0}"
methods_string="${METHODS:-baseline mu_centering z_loss row_centering sgdm}"
lrs_string="${LRS:-0.1 0.12 0.145 0.175 0.21 0.25 0.3}"

read -r -a seeds <<< "${seeds_string}"
read -r -a methods <<< "${methods_string}"
read -r -a lrs <<< "${lrs_string}"

for method in "${methods[@]}"; do
  for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${run_prefix}-${method}-lr${lr}-seed${seed}"
      "${python_bin}" main.py \
        +model=lm162m \
        +dataset=fw_gpt2 \
        +experiment=fast_spike_162m \
        +precision=bf16_mixed \
        +method="${method}" \
        opt.peak_lr="${lr}" \
        seed="${seed}" \
        wandb_mode="${wandb_mode}" \
        run_name="${run_name}" \
        "$@"
    done
  done
done
