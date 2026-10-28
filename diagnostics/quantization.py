"""Finite-precision LM-head measurements reported in the paper."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

import model as model_lib
import utils


@partial(
    jax.jit,
    static_argnames=(
        'model_graphdef',
        'collect_final_norm_channel_metrics',
        'collect_high_confidence_error_metrics',
        'collect_lm_head_quantization_error_metrics',
    ),
)
def get_output_logit_and_optional_metrics(
    model_state,
    model_graphdef,
    x,
    collect_final_norm_channel_metrics: bool,
    collect_high_confidence_error_metrics: bool,
    collect_lm_head_quantization_error_metrics: bool,
):
    model = nnx.merge(model_graphdef, model_state)
    logits, final_norm_input = model(x, return_final_norm_input_with_logits=True)
    logits = logits[:, :-1].astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    output_stats = (logits.mean(), utils.get_l2_norm(logits), logits.std(), entropy)

    metrics = {}
    if collect_final_norm_channel_metrics or collect_high_confidence_error_metrics:
        gamma_abs = jnp.abs(jnp.asarray(model.out_ln.scale.value, dtype=jnp.float32))
        top_k = min(10, gamma_abs.shape[0])
        top_gamma_indices = jax.lax.top_k(gamma_abs, top_k)[1]

    if collect_final_norm_channel_metrics:
        h_hat = model_lib._rmsnorm_operator_input(model.out_ln, final_norm_input)
        channel_energy = jnp.sum(jnp.square(h_hat), axis=tuple(range(h_hat.ndim - 1)))
        total_energy = jnp.maximum(jnp.sum(channel_energy), jnp.asarray(1e-30, dtype=jnp.float32))
        top_activation_energy = jnp.sum(jax.lax.top_k(channel_energy, top_k)[0])
        top_gamma_energy = jnp.sum(jnp.take(channel_energy, top_gamma_indices))
        metrics.update({
            'final_norm/gamma_abs_max': jnp.max(gamma_abs),
            'final_norm/gamma_abs_median': jnp.median(gamma_abs),
            'final_norm/top_activation_energy_frac': top_activation_energy / total_energy,
            'final_norm/top_gamma_energy_frac': top_gamma_energy / total_energy,
        })

    lm_head_input = None
    if collect_high_confidence_error_metrics or collect_lm_head_quantization_error_metrics:
        # Reconstruct the exact activation consumed by the ordinary LM-head
        # forward, then promote only the shadow copy to FP32.
        lm_head_input = model.out_ln(final_norm_input)
        if model.final_hidden_mean_centering:
            lm_head_input = lm_head_input - (
                model.final_hidden_mean_centering_coeff
                * jnp.mean(lm_head_input, axis=-1, keepdims=True)
            )
        if model.lm_head_oblique_learn_target_rms:
            target_rms = jnp.exp(
                jnp.asarray(model.lm_head_oblique_target_rms_log.value, dtype=lm_head_input.dtype)
            )
            lm_head_input = lm_head_input * target_rms

    if collect_lm_head_quantization_error_metrics:
        shadow_input = jnp.asarray(lm_head_input[:, :-1], dtype=jnp.float32)
        shadow_embedding = jnp.asarray(
            model.token_embed_out.embedding.value, dtype=jnp.float32
        )
        shadow_logits = jnp.dot(
            shadow_input,
            shadow_embedding.T,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        logit_difference = logits - shadow_logits
        reference_l2 = jnp.sqrt(jnp.sum(jnp.square(shadow_logits), axis=-1))
        difference_l2 = jnp.sqrt(jnp.sum(jnp.square(logit_difference), axis=-1))
        relative_l2 = difference_l2 / jnp.maximum(
            reference_l2, jnp.asarray(1e-30, dtype=jnp.float32)
        )

        reference_log_probs = jax.nn.log_softmax(shadow_logits, axis=-1)
        reference_probs = jnp.exp(reference_log_probs)
        kl_per_token = jnp.sum(
            reference_probs * (reference_log_probs - log_probs), axis=-1
        )

        targets = x[:, 1:]
        native_losses = -jnp.take_along_axis(
            log_probs, targets[..., None], axis=-1
        ).squeeze(-1)
        reference_losses = -jnp.take_along_axis(
            reference_log_probs, targets[..., None], axis=-1
        ).squeeze(-1)

        metrics.update({
            'lm_head_quantization/fp32_shadow_logit_std': jnp.std(shadow_logits),
            'lm_head_quantization/mean_relative_logit_l2_error': jnp.mean(relative_l2),
            'lm_head_quantization/logit_difference_rms': jnp.sqrt(
                jnp.mean(jnp.square(logit_difference))
            ),
            'lm_head_quantization/kl_fp32_reference_to_native_mean': jnp.mean(kl_per_token),
            'lm_head_quantization/cross_entropy_difference_native_minus_fp32': jnp.mean(
                native_losses - reference_losses
            ),
        })

    if collect_high_confidence_error_metrics:
        predictions = jnp.argmax(logits, axis=-1)
        confidence = jnp.take_along_axis(probs, predictions[..., None], axis=-1)[..., 0]
        targets = x[:, 1:]
        high_confidence_errors = (predictions != targets) & (confidence > 0.9)
        error_mask = high_confidence_errors.astype(jnp.float32)
        error_count = jnp.sum(error_mask)

        lm_head_input = lm_head_input[:, :-1].astype(jnp.float32)

        embedding = jnp.asarray(model.token_embed_out.embedding.value, dtype=jnp.float32)
        predicted_weights = jnp.take(embedding, predictions, axis=0)
        target_weights = jnp.take(embedding, targets, axis=0)
        channel_contributions = lm_head_input * (predicted_weights - target_weights)
        top_gamma_contributions = jnp.take(channel_contributions, top_gamma_indices, axis=-1)

        positive_contributions = jnp.maximum(channel_contributions, 0.0)
        positive_support = jnp.sum(positive_contributions, axis=-1)
        top_gamma_positive_support = jnp.sum(jnp.maximum(top_gamma_contributions, 0.0), axis=-1)
        eps = jnp.asarray(1e-30, dtype=jnp.float32)
        positive_share = top_gamma_positive_support / jnp.maximum(positive_support, eps)

        predicted_logits = jnp.take_along_axis(logits, predictions[..., None], axis=-1)[..., 0]
        target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        positive_logit_gap = jnp.maximum(predicted_logits - target_logits, eps)
        net_share = jnp.sum(top_gamma_contributions, axis=-1) / positive_logit_gap
        conditional_denominator = jnp.maximum(error_count, 1.0)
        metrics.update({
            'high_conf_error/rate': jnp.mean(error_mask),
            'high_conf_error/top_gamma_positive_share': (
                jnp.sum(jnp.where(high_confidence_errors, positive_share, 0.0))
                / conditional_denominator
            ),
            'high_conf_error/top_gamma_net_share': (
                jnp.sum(jnp.where(high_confidence_errors, net_share, 0.0))
                / conditional_denominator
            ),
        })
    return output_stats, metrics

