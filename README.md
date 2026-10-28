# The Role of Quantization in Loss Spikes

This repository accompanies *The Role of Quantization in Loss Spikes*, which
studies how finite-precision computation in the LM head contributes to delayed
loss spikes. The paper also compares AdamW with SGD with momentum for optimizing
the LM head.

The repository includes the training implementation, configurable model and
optimizer settings, precision controls for the LM head, and tools for measuring
logit growth, quantization error, and parameter updates.

See [docs/experiments.md](docs/experiments.md) for a guide to composing and
overriding the experiment configurations.

## Installation

The experiments were run with Python 3.11. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then create a
uv-managed virtual environment and install the requirements for your
accelerator. If Python 3.11 is not already available, uv will download it
automatically.

```bash
uv venv --python 3.11
source .venv/bin/activate

# NVIDIA CUDA 12
uv pip install -r requirements_cuda_gpu.txt

# Google TPU
# uv pip install -r requirements.txt

# AMD ROCm 7
# uv pip install -r requirements_rocm_gpu.txt
```

The experiments were implemented with JAX 0.8.1 and Flax 0.12.0. Large runs
also require a configured Weights & Biases account when `WANDB_MODE=online` and
Google Cloud credentials when writing checkpoints to GCS.

## Dataset

All runs use FineWeb tokenized with the GPT-2 tokenizer. The downloader writes
a headerless `uint16` token stream under `~/datasets`.

For the 28M and 404M time-bomb experiments, download FineWeb-100B. It requires
roughly 206 GB for the token file, plus temporary/cache space:

```bash
python3 download_fineweb.py --full_fineweb100b --stream_write
```

For the shorter fast-spike experiments, the 10B-token version is sufficient:

```bash
python3 download_fineweb.py --stream_write
```

Existing NanoGPT-format FineWeb shards can instead be converted with
`prepare_nanogpt_shards.py`.

## Running experiments

Run a two-step accelerator smoke test first:

```bash
scripts/smoke_test.sh
```

The 28M precision launcher defaults to five seeds and a 16B-token horizon:

```bash
WANDB_MODE=online scripts/run_timebomb_precision.sh
```

The optimizer-comparison launcher defaults to BF16 mixed precision:

```bash
WANDB_MODE=online scripts/run_timebomb_mitigations.sh

# Run the same comparison with BF16 stored LM-head weights.
PRECISION=bf16_true WANDB_MODE=online \
  scripts/run_timebomb_mitigations.sh
```

The 404M, 96B-token baseline learning-rate sweep is substantially more
expensive:

```bash
WANDB_MODE=online scripts/run_timebomb_lr_sweep.sh

# Sweep the available optimizer methods as well as the baseline.
METHODS="baseline mu_centering z_loss sgdm" WANDB_MODE=online \
  scripts/run_timebomb_lr_sweep.sh
```

Scripts run jobs sequentially and accept Hydra overrides as positional
arguments. For example, to enable checkpoints:

```bash
SEEDS="0" scripts/run_timebomb_precision.sh \
  checkpoint.turn_on=true \
  checkpoint.workdir=/path/to/checkpoints
```

## One-off runs

The configuration is composed from five explicit groups:

```text
model + dataset + experiment + precision + method
```

For example, a BF16-mixed 28M run with SGD-M on the LM head is:

```bash
python3 main.py \
  +model=lm28m \
  +dataset=fw_gpt2_100B \
  +experiment=timebomb_28m \
  +precision=bf16_mixed \
  +method=sgdm \
  seed=0 \
  wandb_mode=online \
  run_name=timebomb-bf16-mixed-sgdm-seed0
```

Use `--cfg job --resolve` with the same arguments to inspect the fully resolved
configuration without starting training.

## Repository layout

- `configs/`: composable model, dataset, experiment, precision, and method settings.
- `scripts/`: sequential examples for launching common sweeps.
- `main.py`, `train.py`: experiment entrypoint and setup orchestration.
- `training/`: losses, compiled steps, optimizer construction, checkpointing,
  metric logging, evaluation, distributed batch placement, and the stateful loop.
- `model.py`: Transformer architecture and model initialization.
- `diagnostics/`: LM-head, quantization, and parameter-update measurements used
  by the paper experiments.
- `utils.py`: shared configuration and model-state utilities.
- `evaluate_checkpoint.py`: find high-loss validation tokens and test context swaps.
- `diagnostics/reverse_bad_updates.py`: skip batches and restore earlier optimizer states.

## License and attribution

This work is derived from Martin Marek's
[Picodo](https://github.com/martin-marek/picodo). The original MIT copyright
and license notice are retained in [LICENSE.txt](LICENSE.txt).
