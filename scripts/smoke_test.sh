#!/usr/bin/env bash
# Compile and execute two optimizer steps of the paper baseline.

set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"

"${python_bin}" main.py \
  +model=lm28m \
  +dataset=fw_gpt2 \
  +experiment=timebomb_28m \
  +precision=bf16_mixed \
  +method=baseline \
  num_tokens_train=1_048_576 \
  num_tokens_valid=131_072 \
  diagnostics.end_step=2 \
  eval_every_steps=-1 \
  log_every_tokens=131_072 \
  use_single_vm_among_multiple=true \
  checkpoint.turn_on=false \
  wandb_mode=disabled \
  run_name=smoke-test \
  "$@"
