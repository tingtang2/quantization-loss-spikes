"""Compiled optimizer steps and small state-manipulation helpers."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

import model as model_lib
from diagnostics.parameter_updates import compute_parameter_update_metrics
from training.losses import (
    _apply_tanh_soft_cap,
    _build_loss_skip_weights,
    loss_fn,
    loss_fn_with_skip,
    loss_fn_with_soft_cap,
    loss_fn_z_loss,
    loss_fn_z_loss_with_skip,
    loss_fn_z_loss_with_soft_cap,
)


def _prepare_parameter_update_metrics(model_state, model_graphdef, batch, collect_metrics: bool):
    if not collect_metrics:
        return None

    # Clone the variable containers so the in-place NNX optimizer update does not
    # overwrite the pre-update values referenced by the diagnostics graph.
    old_model = nnx.clone(nnx.merge(model_graphdef, model_state))
    _, captures = old_model(
        batch,
        return_update_inputs=True,
        skip_output_logits=True,
    )
    return old_model, captures, {}


def _finish_parameter_update_metrics(metric_context, new_model):
    if metric_context is None:
        return {}
    old_model, captures, metrics = metric_context
    metrics.update(compute_parameter_update_metrics(old_model, new_model, captures))
    return metrics


@jax.jit
def _copy_model_state_for_metrics(model_state):
    """Makes non-aliased parameter buffers that survive donation of the optimizer state."""
    return jax.tree.map(lambda x: jnp.array(x, copy=True), model_state)


@partial(jax.jit, static_argnames=('model_graphdef',))
def get_parameter_update_metrics(old_model_state, new_model_state, model_graphdef, batch):
    """Runs update diagnostics separately from the loss/backward executable."""
    old_model = nnx.merge(model_graphdef, old_model_state)
    new_model = nnx.merge(model_graphdef, new_model_state)
    _, captures = old_model(
        batch,
        return_update_inputs=True,
        skip_output_logits=True,
    )
    return compute_parameter_update_metrics(old_model, new_model, captures)


# Large enough to keep the LM-head GEMMs efficient, while capping the temporary
# float32 logits buffer at roughly 400 MiB for a 50k-token vocabulary.
_FINAL_NORM_GRAD_TOKEN_CHUNK_SIZE = 2048


@partial(
    jax.jit,
    static_argnames=('model_graphdef', 'use_z_loss', 'objective_mode'),
)
def get_final_norm_activation_grad_metrics(
    model_state,
    model_graphdef,
    batch,
    use_z_loss: bool = False,
    objective_mode: str = 'standard',
    skip_center=0.0,
    skip_mad=1.0,
    skip_z_soft=5.0,
    skip_z_hard=8.0,
    skip_abs_hard=float('inf'),
    skip_soft_weight=1.0,
    skip_eps=1e-6,
    skip_apply_gate=False,
    skip_use_log_loss=False,
    loss_cap=30.0,
    loss_cap_apply=False,
):
    """Measures loss gradients on both sides of the final RMSNorm.

    The transformer backbone is only run forward. The vocabulary projection and
    loss are evaluated in token chunks so this diagnostic never materializes a
    [batch, sequence, vocabulary] logits tensor.
    """
    if objective_mode not in ('standard', 'skip', 'soft_cap'):
        raise ValueError(f"Unsupported final-norm gradient objective: {objective_mode}")

    model = nnx.merge(model_graphdef, model_state)
    norm_input = model(batch, return_final_norm_input=True)
    norm_output, norm_pullback = jax.vjp(model.out_ln, norm_input)

    def tail_loss_and_stats(final_norm_output):
        h = final_norm_output
        if model.final_hidden_mean_centering:
            h = h - model.final_hidden_mean_centering_coeff * jnp.mean(h, axis=-1, keepdims=True)
        if model.lm_head_oblique_learn_target_rms:
            target_rms = jnp.exp(jnp.asarray(model.lm_head_oblique_target_rms_log.value, dtype=h.dtype))
            h = h * target_rms

        # The last token in each sequence has no next-token target.
        flat_h = h[:, :-1, :].reshape(-1, h.shape[-1])
        flat_targets = batch[:, 1:].reshape(-1)
        num_tokens = flat_h.shape[0]
        chunk_size = _FINAL_NORM_GRAD_TOKEN_CHUNK_SIZE
        padding = (-num_tokens) % chunk_size
        flat_h = jnp.pad(flat_h, ((0, padding), (0, 0)))
        flat_targets = jnp.pad(flat_targets, ((0, padding),))
        valid = jnp.arange(flat_h.shape[0]) < num_tokens
        hidden_chunks = flat_h.reshape(-1, chunk_size, flat_h.shape[-1])
        target_chunks = flat_targets.reshape(-1, chunk_size)
        valid_chunks = valid.reshape(-1, chunk_size)

        def chunk_statistics(chunk):
            hidden, targets, valid_mask = chunk
            logits = model.token_embed_out.attend(hidden).astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            losses = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
            mask = valid_mask.astype(jnp.float32)

            if objective_mode == 'skip':
                weights = _build_loss_skip_weights(
                    losses,
                    skip_center,
                    skip_mad,
                    skip_z_soft,
                    skip_z_hard,
                    skip_abs_hard,
                    skip_soft_weight,
                    skip_eps,
                    skip_apply_gate,
                    skip_use_log_loss,
                ) * mask
                ce_numerator = jnp.sum(losses * weights)
                ce_denominator = jnp.sum(weights)
            elif objective_mode == 'soft_cap':
                capped_losses = _apply_tanh_soft_cap(losses, loss_cap)
                apply_cap = jnp.asarray(loss_cap_apply, dtype=losses.dtype)
                objective_losses = apply_cap * capped_losses + (1.0 - apply_cap) * losses
                ce_numerator = jnp.sum(objective_losses * mask)
                ce_denominator = jnp.sum(mask)
            else:
                ce_numerator = jnp.sum(losses * mask)
                ce_denominator = jnp.sum(mask)

            log_z = jax.nn.logsumexp(logits, axis=-1)
            probabilities = jnp.exp(log_probs)
            entropy = -jnp.sum(probabilities * log_probs, axis=-1)
            logit_mask = mask[:, None]
            return (
                ce_numerator,
                ce_denominator,
                jnp.sum(jnp.square(log_z) * mask),
                jnp.sum(logits * logit_mask),
                jnp.sum(jnp.square(logits) * logit_mask),
                jnp.sum(entropy * mask),
                jnp.sum(mask),
            )

        chunk_totals = jax.lax.map(
            # Rematerialize one chunk during the reverse pass instead of saving
            # every chunk's logits/softmax residuals for the whole sequence.
            jax.checkpoint(chunk_statistics),
            (hidden_chunks, target_chunks, valid_chunks),
        )
        totals = tuple(jnp.sum(value, axis=0) for value in chunk_totals)
        ce_numerator, ce_denominator, z_squared_sum, logit_sum, logit_squared_sum, entropy_sum, token_count = totals
        loss = ce_numerator / jnp.maximum(ce_denominator, 1.0)
        if use_z_loss:
            loss = loss + 1e-4 * z_squared_sum / jnp.maximum(token_count, 1.0)

        logit_count = token_count * model.token_embed_out.num_embeddings
        logit_mean = logit_sum / jnp.maximum(logit_count, 1.0)
        logit_mean_square = logit_squared_sum / jnp.maximum(logit_count, 1.0)
        logit_std = jnp.sqrt(jnp.maximum(logit_mean_square - jnp.square(logit_mean), 0.0))
        stats = (
            logit_mean,
            jnp.sqrt(jnp.maximum(logit_squared_sum, 0.0)),
            logit_std,
            entropy_sum / jnp.maximum(token_count, 1.0),
        )
        return loss, stats

    (_, output_stats), grad_output = jax.value_and_grad(tail_loss_and_stats, has_aux=True)(norm_output)
    grad_input = norm_pullback(grad_output)[0]

    grad_input_f32 = jnp.asarray(grad_input, dtype=jnp.float32)
    grad_output_f32 = jnp.asarray(grad_output, dtype=jnp.float32)
    grad_input_fro = jnp.sqrt(jnp.sum(jnp.square(grad_input_f32)))
    grad_output_fro = jnp.sqrt(jnp.sum(jnp.square(grad_output_f32)))
    metrics = {
        'activation_grad/out_ln/input_fro': grad_input_fro,
        'activation_grad/out_ln/output_fro': grad_output_fro,
        'activation_grad/out_ln/input_rms': jnp.sqrt(jnp.mean(jnp.square(grad_input_f32))),
        'activation_grad/out_ln/output_rms': jnp.sqrt(jnp.mean(jnp.square(grad_output_f32))),
        'activation_grad/out_ln/input_to_output_fro_ratio': (
            grad_input_fro / jnp.maximum(grad_output_fro, 1e-30)
        ),
    }
    return output_stats, metrics


@partial(
    jax.jit,
    static_argnames=(
        'opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'return_grads', 'collect_parameter_update_metrics'
    ),
    donate_argnames=('opt_state'),
)
def train_step(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    collect_qkv_stats: bool = True,
    return_grads: bool = False,
    collect_parameter_update_metrics: bool = False,
):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    
    return opt_state, loss, raw_loss, grads if return_grads else None, update_metrics

@partial(
    jax.jit,
    static_argnames=(
        'opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'return_grads', 'collect_parameter_update_metrics'
    ),
    donate_argnames=('opt_state'),
)
def train_step_z_loss(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    collect_qkv_stats: bool = True,
    return_grads: bool = False,
    collect_parameter_update_metrics: bool = False,
):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    
    return opt_state, loss, raw_loss, grads if return_grads else None, update_metrics

@partial(
    jax.jit,
    static_argnames=(
        'opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'return_grads', 'collect_parameter_update_metrics'
    ),
    donate_argnames=('opt_state'),
)
def train_step_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    collect_qkv_stats: bool = True,
    return_grads: bool = False,
    collect_parameter_update_metrics: bool = False,
):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    
    return opt_state, loss, raw_loss, grads if return_grads else None, update_metrics

@partial(
    jax.jit,
    static_argnames=(
        'opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'return_grads', 'collect_parameter_update_metrics'
    ),
    donate_argnames=('opt_state'),
)
def train_step_z_loss_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    collect_qkv_stats: bool = True,
    return_grads: bool = False,
    collect_parameter_update_metrics: bool = False,
):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    
    return opt_state, loss, raw_loss, grads if return_grads else None, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_with_skip(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
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
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_z_loss_with_skip(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
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
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_with_skip_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
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
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_z_loss_with_skip_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
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
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_with_soft_cap(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_z_loss_with_soft_cap(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_with_soft_cap_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats', 'collect_parameter_update_metrics'),
    donate_argnames=('opt_state'),
)
def train_step_z_loss_with_soft_cap_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
    collect_parameter_update_metrics: bool = False,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    metric_context = _prepare_parameter_update_metrics(
        opt_state.model, model_graphdef, batch, collect_parameter_update_metrics
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    update_metrics = _finish_parameter_update_metrics(metric_context, optimizer.model)
    return opt_state, loss, aux, grads, update_metrics


@partial(
    jax.jit,
    static_argnames=('opt_graphdef',),
    donate_argnames=('opt_state',),
)
def amplify_lm_head_mean(opt_state, opt_graphdef, coefficient):
    """Applies the scheduled post-update LM-head mean amplification."""
    optimizer = nnx.merge(opt_graphdef, opt_state)
    model_lib.amplify_output_embedding_mean(optimizer.model, coefficient)
    return nnx.state(optimizer)


@partial(jax.jit, static_argnames=('model_graphdef',))
def snapshot_lm_head_parameters(model_state, model_graphdef):
    """Copies the trainable parameters that determine the LM-head output."""
    model = nnx.merge(model_graphdef, model_state)
    parameters = {
        'embedding': jnp.array(model.token_embed_out.embedding.value, copy=True),
    }
    if model.lm_head_oblique_learn_target_rms:
        parameters['oblique_target_rms_log'] = jnp.array(
            model.lm_head_oblique_target_rms_log.value, copy=True
        )
    return parameters


@partial(
    jax.jit,
    static_argnames=('opt_graphdef',),
    donate_argnames=('opt_state',),
)
def restore_lm_head_parameters(opt_state, opt_graphdef, frozen_parameters):
    """Restores a frozen LM head after the optimizer and post-update hooks."""
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.model.token_embed_out.embedding.value = frozen_parameters['embedding']
    if 'oblique_target_rms_log' in frozen_parameters:
        optimizer.model.lm_head_oblique_target_rms_log.value = frozen_parameters[
            'oblique_target_rms_log'
        ]
    return nnx.state(optimizer)


@partial(jax.jit, static_argnames=('model_graphdef',))
def snapshot_out_ln_scale(model_state, model_graphdef):
    """Copies the final output RMSNorm scale parameter."""
    model = nnx.merge(model_graphdef, model_state)
    return jnp.array(model.out_ln.scale.value, copy=True)


@partial(
    jax.jit,
    static_argnames=('opt_graphdef',),
    donate_argnames=('opt_state',),
)
def restore_out_ln_scale(opt_state, opt_graphdef, frozen_scale):
    """Restores a frozen final output RMSNorm scale after an optimizer step."""
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.model.out_ln.scale.value = frozen_scale
    return nnx.state(optimizer)

