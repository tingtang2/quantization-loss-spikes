"""Cancellation-resistant softmax and cross-entropy from arXiv:2610.31642."""

import jax
import jax.numpy as jnp


def fixed_log_softmax(logits):
    """Computes log-softmax while preserving an exact zero max-logit shift."""
    max_indices = jnp.argmax(logits, axis=-1)
    max_logits = jnp.take_along_axis(logits, max_indices[..., None], axis=-1)
    is_max = jax.nn.one_hot(max_indices, logits.shape[-1], dtype=bool)
    shifted = jnp.where(is_max, 0.0, logits - max_logits)
    exp_others = jnp.where(is_max, 0.0, jnp.exp(shifted))
    sum_exp_others = jnp.sum(exp_others, axis=-1, keepdims=True)
    return shifted - jnp.log1p(sum_exp_others)


def fixed_softmax(logits):
    return jnp.exp(fixed_log_softmax(logits))


def fixed_cross_entropy(logits, targets):
    """Computes target-margin cross-entropy without subtractive cancellation."""
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)
    is_target = jax.nn.one_hot(targets, logits.shape[-1], dtype=bool)
    shifted = jnp.where(is_target, 0.0, logits - target_logits)
    max_shift = jnp.max(shifted, axis=-1, keepdims=True)
    sum_exp_others = jnp.sum(
        jnp.where(is_target, 0.0, jnp.exp(shifted - max_shift)),
        axis=-1,
    )
    max_shift = max_shift[..., 0]
    return max_shift + jnp.log1p(sum_exp_others + jnp.expm1(-max_shift))


def select_log_softmax(logits, use_fixed_softmax: bool = False):
    if use_fixed_softmax:
        return fixed_log_softmax(logits)
    return jax.nn.log_softmax(logits, axis=-1)


def select_cross_entropy(logits, targets, use_fixed_softmax: bool = False):
    if use_fixed_softmax:
        return fixed_cross_entropy(logits, targets)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1,
    ).squeeze(-1)
