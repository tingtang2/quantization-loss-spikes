"""LM-head statistics used for training and appendix diagnostics."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

import utils


@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logits_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    return logits.reshape(-1, logits.shape[-1]).astype(jnp.float32).mean(axis=0) # [B * T, V] -> [V]

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_gaps_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x).astype(jnp.float32)[:, :-1, :] # [B, T, V]
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    gaps = logits - max_logits

    mean_gaps = jnp.mean(gaps, axis=(0, 1)) # [V]
    y = jnp.roll(x, -1, axis=1)[:, :-1]
    target_logits = jnp.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
    target_gaps = target_logits - max_logits.squeeze(-1) # [B, T]

    return mean_gaps, target_gaps

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_mean_and_norm_output_logit(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    logits = logits[:, :-1].astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    return logits.mean(), utils.get_l2_norm(logits), logits.std(), entropy


@partial(jax.jit, static_argnames=('model_graphdef',))
def get_lm_head_diagnostic_metrics(model_state, model_graphdef, x): # [B, T]
    """Measures opt-in LM-head diagnostics without modifying the train step."""
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x)[:, :-1].astype(jnp.float32) # [B, T-1, V]
    targets = x[:, 1:]

    def mean_cross_entropy(logits_arg):
        log_probs = jax.nn.log_softmax(logits_arg, axis=-1)
        losses = -jnp.take_along_axis(
            log_probs, targets[..., None], axis=-1
        ).squeeze(-1)
        return losses.mean()

    # Autodiff returns (p - y) / num_tokens because the loss is a mean. Undo
    # only that reduction so this diagnostic measures the LM-head residual.
    grad_logits = jax.grad(mean_cross_entropy)(logits)
    num_tokens = logits.shape[0] * logits.shape[1]
    residual_sum = jnp.sum(grad_logits, axis=-1) * jnp.asarray(
        num_tokens, dtype=grad_logits.dtype
    )

    per_token_logit_std = jnp.std(logits, axis=-1)
    max_logit_over_std = jnp.max(logits, axis=-1) / jnp.maximum(
        per_token_logit_std,
        jnp.asarray(1e-30, dtype=logits.dtype),
    )
    output_embedding = jnp.asarray(
        model.token_embed_out.embedding.value, dtype=jnp.float32
    )

    return {
        'lm_head/mean_output_embedding_l2_norm': jnp.linalg.norm(
            jnp.mean(output_embedding, axis=0)
        ),
        'lm_head/max_logit_over_logit_std_mean': jnp.mean(max_logit_over_std),
        'lm_head/probability_target_residual_sum_mean': jnp.mean(residual_sum),
    }

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_head_scale_metrics(model_state, model_graphdef, x):
    """Measures the relative activation and parameter scales around the LM head."""
    model = nnx.merge(model_graphdef, model_state)
    raw_final_norm_input = model(x, return_final_norm_input=True)
    lm_head_input = model.out_ln(raw_final_norm_input)
    if model.final_hidden_mean_centering:
        lm_head_input = lm_head_input - (
            model.final_hidden_mean_centering_coeff * jnp.mean(lm_head_input, axis=-1, keepdims=True)
        )
    if model.lm_head_oblique_learn_target_rms:
        target_rms = jnp.exp(
            jnp.asarray(model.lm_head_oblique_target_rms_log.value, dtype=lm_head_input.dtype)
        )
        lm_head_input = lm_head_input * target_rms

    input_norm = utils.get_l2_norm(lm_head_input.astype(jnp.float32))
    raw_input_norm = utils.get_l2_norm(raw_final_norm_input.astype(jnp.float32))
    output_embedding_norm = utils.get_l2_norm(
        model.token_embed_out.embedding.value.astype(jnp.float32)
    )
    final_norm_scale_norm = utils.get_l2_norm(model.out_ln.scale.value.astype(jnp.float32))
    eps = jnp.asarray(1e-30, dtype=jnp.float32)
    return {
        'head_scale/input_ratio': input_norm / jnp.maximum(raw_input_norm, eps),
        'head_scale/weight_product': output_embedding_norm * final_norm_scale_norm,
        'head_scale/weight_ratio': output_embedding_norm / jnp.maximum(final_norm_scale_norm, eps),
    }

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_grad_sum_stats(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32) # [B, T, V]

    def loss_from_logits(l):
        log_probs = jax.nn.log_softmax(l, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        losses = losses[:, :-1]
        return losses.mean()

    grad_logits = jax.grad(loss_from_logits)(logits) # dL/dz
    grad_sum = grad_logits.sum(axis=-1)[:, :-1] # [B, T-1]

    return {
        'logit_grad_sum_mean_abs': jnp.mean(jnp.abs(grad_sum)),
        'logit_grad_sum_max_abs': jnp.max(jnp.abs(grad_sum)),
        'logit_grad_sum_mean': jnp.mean(grad_sum),
    }


@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_grad_scaling_stats(model_state, model_graphdef, x): # [B, T]
    """Compares autodiff CE logits grads vs closed-form grads to detect scaling issues."""
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32) # [B, T, V]

    mask = jnp.ones_like(y, dtype=logits.dtype)
    mask = mask.at[:, -1].set(0)
    denom = jnp.maximum(mask.sum(), 1.0)

    def loss_from_logits(l):
        log_probs = jax.nn.log_softmax(l, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        return (losses * mask).sum() / denom

    grad_auto = jax.grad(loss_from_logits)(logits)

    probs = jax.nn.softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y, logits.shape[-1], dtype=logits.dtype)
    grad_closed_form = (probs - one_hot) * mask[..., None] / denom

    grad_diff = grad_auto - grad_closed_form
    auto_norm = utils.get_l2_norm(grad_auto)
    diff_norm = utils.get_l2_norm(grad_diff)

    grad_sum = grad_auto.sum(axis=-1)[:, :-1]
    closed_form_sum = grad_closed_form.sum(axis=-1)[:, :-1]

    return {
        'logit_grad_rel_l2_err': diff_norm / jnp.maximum(auto_norm, 1e-30),
        'logit_grad_max_abs_err': jnp.max(jnp.abs(grad_diff[:, :-1])),
        'logit_grad_mean_abs_err': jnp.mean(jnp.abs(grad_diff[:, :-1])),
        'logit_grad_closed_form_sum_mean_abs': jnp.mean(jnp.abs(closed_form_sum)),
        'logit_grad_closed_form_sum_max_abs': jnp.max(jnp.abs(closed_form_sum)),
        'logit_grad_auto_sum_mean_abs': jnp.mean(jnp.abs(grad_sum)),
        'logit_grad_auto_sum_max_abs': jnp.max(jnp.abs(grad_sum)),
    }

