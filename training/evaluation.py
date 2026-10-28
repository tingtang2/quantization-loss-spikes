"""Evaluation with method-independent cross-entropy reporting."""

import jax.numpy as jnp

from diagnostics.lm_head import get_logits_by_lm_head, get_mean_and_norm_output_logit
from training.distributed import _maybe_shard_multihost_batch
from training.losses import loss_fn


def eval_step(
    c,
    model_state,
    model_graphdef,
    dataset,
    collect_qkv_stats: bool = True,
    mesh=None,
    multihost_batch_sharding: bool = False,
    force_data_sharding: bool = False,
):
    loss_sum = jnp.zeros([], dtype=jnp.float32)
    raw_losses = []
    total_logits = []
    logit_mean_sum = jnp.zeros([], dtype=jnp.float32)
    logit_std_sum = jnp.zeros([], dtype=jnp.float32)
    logit_entropy_sum = jnp.zeros([], dtype=jnp.float32)

    
    for i in range(len(dataset)):
        batch = dataset[i]
        batch = _maybe_shard_multihost_batch(
            batch,
            mesh,
            multihost_batch_sharding,
            force_data_sharding,
        )
        # Report the same cross-entropy metric for every training method. In
        # particular, z-loss is an optimization penalty, not part of the
        # language-modeling loss used for method comparisons in the paper.
        batch_loss, (raw_loss, _) = loss_fn(
            model_state, model_graphdef, batch, collect_qkv_stats
        )
        loss_sum += batch_loss
        raw_losses.append(raw_loss)
        output_logit_mean, _, output_logit_std, output_logit_entropy = get_mean_and_norm_output_logit(
            model_state, model_graphdef, batch
        )
        logit_mean_sum += output_logit_mean
        logit_std_sum += output_logit_std
        logit_entropy_sum += output_logit_entropy
        if c.diagnostics.save_raw_losses:
            total_logits.append(get_logits_by_lm_head(model_state, model_graphdef, batch).astype(jnp.float32))

    mean_loss = loss_sum / len(dataset)
    mean_output_logit = logit_mean_sum / len(dataset)
    mean_output_logit_std = logit_std_sum / len(dataset)
    mean_output_logit_entropy = logit_entropy_sum / len(dataset)
    
    return (
        mean_loss,
        raw_losses,
        total_logits,
        mean_output_logit,
        mean_output_logit_std,
        mean_output_logit_entropy,
    )

