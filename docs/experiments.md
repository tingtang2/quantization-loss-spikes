# Experiment configuration guide

This guide describes the experiment configurations and launch scripts included
in the repository. Model size, training setup, numerical precision, and
optimizer options are configured separately and can be overridden from the
command line. The shell scripts run several common sweeps sequentially.

Unless overridden, the release uses FineWeb with the GPT-2 tokenizer, QK-Norm,
AdamW with `beta1=0.9`, `beta2=0.95`, and `eps=1e-8`, linear warmup followed by
cosine decay, and global gradient clipping at 1.0.

## Configuration groups

Each run is assembled from five Hydra groups:

```text
model + dataset + experiment + precision + method
```

### Model sizes

| Config | Width | Layers | Sequence length | Approximate parameters |
| --- | ---: | ---: | ---: | ---: |
| `lm28m` | 256 | 3 | 512 | 28M |
| `lm162m` | 768 | 12 | 1,024 | 162M |
| `lm404m` | 1,024 | 24 | 1,024 | 404M |
| `lm1p4b` | 2,048 | 24 | 2,048 | 1.4B |

The counts include untied input and output embeddings.

### Precision modes

| Config | LM-head compute | Stored LM-head weights |
| --- | --- | --- |
| `bf16_true` | BF16 | BF16 |
| `bf16_mixed` | BF16 | FP32 |
| `fp32_lm_head` | FP32 | FP32 |

In `fp32_lm_head`, only the final normalization and LM head switch to FP32; the
transformer trunk remains BF16.

### Optimizer options

| Config | LM-head configuration |
| --- | --- |
| `baseline` | AdamW with the trunk learning rate |
| `mu_centering` | AdamW followed by vocabulary-axis mean removal |
| `z_loss` | AdamW and z-loss with coefficient `1e-4` |
| `sgdm` | SGD-M with independently configurable head learning rate and momentum |
| `adamw_head_b1_099` | AdamW with LM-head `beta1=0.99` |
| `row_centering` | AdamW followed by feature-axis row centering |

The method files contain useful defaults, but optimizer hyperparameters can be
overridden independently. For example:

```bash
opt.peak_lr=1e-3 \
opt.lm_head_optimizer.peak_lr=3e-4
```

## Long-horizon experiments

The `timebomb_28m` and `timebomb_404m` configurations define the two
long-horizon setups. They set the token horizon, batch size, normalization
parameters, and weight decay. Precision and the LM-head optimizer are selected
separately.

A single 28M run can be launched with:

```bash
python3 main.py \
  +model=lm28m \
  +dataset=fw_gpt2_100B \
  +experiment=timebomb_28m \
  +precision=bf16_mixed \
  +method=baseline \
  seed=0 \
  wandb_mode=online
```

The supplied sweep examples are:

```bash
# Precision modes across seeds.
scripts/run_timebomb_precision.sh

# Optimizer methods across seeds.
scripts/run_timebomb_mitigations.sh

# Long-horizon learning-rate sweep.
scripts/run_timebomb_lr_sweep.sh
```

All launcher axes can be changed with environment variables such as `SEEDS`,
`LRS`, `METHODS`, and `PRECISIONS`. Additional Hydra overrides are forwarded as
positional arguments. Shorter horizons can be selected directly, for example:

```bash
scripts/run_timebomb_precision.sh \
  num_tokens_train=4_000_000_000 \
  eval_every_steps=2000
```

## Fast-spike experiments

The `fast_spike_162m`, `fast_spike_404m`, and `fast_spike_1p4b` configurations
use parameterless RMSNorms and the fast-spike batch and weight-decay settings.
The 162M learning-rate launcher is:

```bash
scripts/run_fast_spike_sweep.sh
```

For a one-off run at another scale, compose the corresponding model and
experiment groups directly. Set `model.rmsnorm_use_scale=true` to compare
against learnable RMSNorm gains.

Muon on the transformer trunk is selected with
`+optimizer=muon_adamw_embeddings`. Use
`+optimizer=muon_input_adamw_lm_head_sgd` to combine a Muon trunk with SGD-M on
the LM head.

## Analysis options

Quantization-error logging is enabled with
`log_lm_head_quantization_error_metrics=true`. The native and FP32-shadow logits
are computed from the same hidden states, and the resulting metrics are logged
under the `lm_head_quantization/` namespace.

Other available options include:

- `opt.loss_skip.enabled=true` for high-loss-token masking;
- `diagnostics/reverse_bad_updates.py` for skipping batches and restoring an
  earlier optimizer state;
- `evaluate_checkpoint.py` for finding high-loss validation tokens and
  measuring the effect of replacing context tokens;
- `log_parameter_update_metrics=true` and the other `log_*_metrics` flags for
  additional metrics.

The checkpoint analysis scripts require `checkpoint.turn_on=true` and either
`checkpoint.workdir` or `checkpoint.gcp_bucket`.

## Running on a cluster

The shell scripts run one job at a time. When using a cluster scheduler, submit
each `(method, precision, learning rate, seed)` combination as a separate job.
Device layout, checkpoint paths, and logging and evaluation frequency can all
be overridden on the command line.

Add `--cfg job --resolve` to any `main.py` command to print its configuration
without starting training.
