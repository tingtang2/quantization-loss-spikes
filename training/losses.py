"""Cross-entropy objectives and optional loss variants."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

import utils


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn(model_state, model_graphdef, x, collect_qkv_stats: bool = True): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True) # [B, T, V]
    else:
        logits = model(x, return_qkv=False) # [B, T, V]
        qkv_dict = None

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}
    
    return losses.mean(), (losses, qkv_stats)

@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_z_loss(model_state, model_graphdef, x, collect_qkv_stats: bool = True): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True) # [B, T, V]
    else:
        logits = model(x, return_qkv=False) # [B, T, V]
        qkv_dict = None
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)  

    z_loss = (z**2).mean()
    lam = 1e-4

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}
    
    return losses.mean() + lam * z_loss, (losses, qkv_stats)


def _build_loss_skip_weights(
    losses,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
):
    use_log_loss = jnp.asarray(use_log_loss, dtype=jnp.bool_)
    stats_losses = jnp.where(
        use_log_loss,
        jnp.log1p(jnp.maximum(losses, 0.0)),
        losses,
    )
    scale = jnp.maximum(1.4826 * mad, eps)
    z = (stats_losses - center) / scale
    weights = jnp.ones_like(losses, dtype=jnp.float32)
    soft_band = jnp.logical_and(z > z_soft, z <= z_hard)
    weights = jnp.where(soft_band, jnp.asarray(soft_weight, dtype=jnp.float32), weights)
    hard_mask = jnp.logical_or(z > z_hard, stats_losses > abs_hard)
    weights = jnp.where(hard_mask, 0.0, weights)
    gate = jnp.asarray(apply_gate, dtype=weights.dtype)
    weights = gate * weights + (1.0 - gate) * jnp.ones_like(weights)
    return weights


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_with_skip(
    model_state,
    model_graphdef,
    x,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    weights = _build_loss_skip_weights(
        losses, center, mad, z_soft, z_hard, abs_hard, soft_weight, eps, apply_gate, use_log_loss
    )
    denom = jnp.maximum(weights.sum(), 1.0)
    masked_loss = (losses * weights).sum() / denom
    unmasked_loss = losses.mean()

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    skip_stats = {
        'loss_skip_keep_frac': jnp.mean(weights > 0),
        'loss_skip_weight_mean': jnp.mean(weights),
        'loss_skip_masked_loss': masked_loss,
        'loss_skip_unmasked_loss': unmasked_loss,
    }

    return masked_loss, (losses, qkv_stats, skip_stats)


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_z_loss_with_skip(
    model_state,
    model_graphdef,
    x,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    weights = _build_loss_skip_weights(
        losses, center, mad, z_soft, z_hard, abs_hard, soft_weight, eps, apply_gate, use_log_loss
    )
    denom = jnp.maximum(weights.sum(), 1.0)
    masked_ce = (losses * weights).sum() / denom
    unmasked_ce = losses.mean()

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)
    z_loss = (z**2).mean()
    lam = 1e-4

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    skip_stats = {
        'loss_skip_keep_frac': jnp.mean(weights > 0),
        'loss_skip_weight_mean': jnp.mean(weights),
        'loss_skip_masked_loss': masked_ce,
        'loss_skip_unmasked_loss': unmasked_ce,
    }

    return masked_ce + lam * z_loss, (losses, qkv_stats, skip_stats)


def _apply_tanh_soft_cap(losses, soft_cap):
    soft_cap = jnp.maximum(jnp.asarray(soft_cap, dtype=losses.dtype), jnp.asarray(1e-6, dtype=losses.dtype))
    return soft_cap * jnp.tanh(losses / soft_cap)


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_with_soft_cap(
    model_state,
    model_graphdef,
    x,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    capped_losses = _apply_tanh_soft_cap(losses, soft_cap)
    apply_cap_f = jnp.asarray(apply_cap, dtype=capped_losses.dtype)
    capped_losses = apply_cap_f * capped_losses + (1.0 - apply_cap_f) * losses
    capped_loss = capped_losses.mean()
    uncapped_loss = losses.mean()

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    cap_stats = {
        'loss_cap_applied': jnp.asarray(apply_cap, dtype=jnp.float32),
        'loss_cap_capped_loss': capped_loss,
        'loss_cap_uncapped_loss': uncapped_loss,
    }

    return capped_loss, (losses, qkv_stats, cap_stats)


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_z_loss_with_soft_cap(
    model_state,
    model_graphdef,
    x,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    capped_losses = _apply_tanh_soft_cap(losses, soft_cap)
    apply_cap_f = jnp.asarray(apply_cap, dtype=capped_losses.dtype)
    capped_losses = apply_cap_f * capped_losses + (1.0 - apply_cap_f) * losses
    capped_ce = capped_losses.mean()
    uncapped_ce = losses.mean()

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)
    z_loss = (z**2).mean()
    lam = 1e-4

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    cap_stats = {
        'loss_cap_applied': jnp.asarray(apply_cap, dtype=jnp.float32),
        'loss_cap_capped_loss': capped_ce,
        'loss_cap_uncapped_loss': uncapped_ce,
    }

    return capped_ce + lam * z_loss, (losses, qkv_stats, cap_stats)

