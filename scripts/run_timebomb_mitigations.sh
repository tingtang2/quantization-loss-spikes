#!/usr/bin/env bash
# Launch a 28M optimizer-method comparison.

set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
wandb_mode="${WANDB_MODE:-offline}"
run_prefix="${RUN_PREFIX:-timebomb-mitigation}"
precision="${PRECISION:-bf16_mixed}"
seeds_string="${SEEDS:-0 1 2 3 4}"
methods_string="${METHODS:-baseline mu_centering z_loss sgdm}"

read -r -a seeds <<< "${seeds_string}"
read -r -a methods <<< "${methods_string}"

for method in "${methods[@]}"; do
  for seed in "${seeds[@]}"; do
    run_name="${run_prefix}-${precision}-${method}-seed${seed}"
    "${python_bin}" main.py \
      +model=lm28m \
      +dataset=fw_gpt2_100B \
      +experiment=timebomb_28m \
      +precision="${precision}" \
      +method="${method}" \
      seed="${seed}" \
      wandb_mode="${wandb_mode}" \
      run_name="${run_name}" \
      "$@"
  done
done
