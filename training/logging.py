"""Host-side metric preparation and Weights & Biases logging."""

import jax
import numpy as np
import wandb
from jax.experimental import multihost_utils

import utils


def _to_host_numpy(x, dtype=np.float32, flatten: bool = False):
    """Converts possibly non-addressable JAX arrays to host NumPy arrays."""
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        host_value = multihost_utils.process_allgather(x, tiled=True)
    else:
        host_value = jax.device_get(x)
    arr = np.asarray(host_value, dtype=dtype)
    return arr.reshape(-1) if flatten else arr


def _prepare_output_embedding_metric_groups(c):
    if not bool(getattr(c, "log_output_embedding_group_metrics", True)):
        return ()

    groups_cfg = getattr(c, "output_embedding_groups", None)
    if not groups_cfg:
        return ()

    groups = []
    for group_cfg in groups_cfg:
        name = getattr(group_cfg, "name", None)
        token_ids_cfg = getattr(group_cfg, "token_ids", None)
        if name is None or token_ids_cfg is None:
            continue
        token_ids = tuple(int(token_id) for token_id in token_ids_cfg)
        if not token_ids:
            continue
        groups.append({
            "name": str(name),
            "token_ids": np.asarray(token_ids, dtype=np.int32),
        })

    return tuple(groups)


class _AsyncMetricWriter:
    """Queues one step of metrics so host logging stays off the step critical path."""

    def __init__(self):
        self.prev_step = None
        self.prev_metrics = None

    def _kick_host_transfer(self, x):
        if isinstance(x, jax.Array) and x.is_fully_addressable and hasattr(x, "copy_to_host_async"):
            x.copy_to_host_async()
        return x

    def _materialize_scalar(self, x):
        if isinstance(x, jax.Array):
            arr = _to_host_numpy(x, flatten=False)
            return arr.item() if arr.ndim == 0 else arr
        if isinstance(x, np.generic):
            return x.item()
        return x

    def enqueue(self, step, metrics, pbar):
        if jax.process_index() != 0:
            return
        jax.tree_util.tree_map(self._kick_host_transfer, metrics)
        log_step, log_metrics = self.prev_step, self.prev_metrics
        self.prev_step, self.prev_metrics = step, metrics
        if log_metrics is not None:
            materialized = {k: self._materialize_scalar(v) for k, v in log_metrics.items()}
            _log_metrics_if_primary(materialized, log_step, pbar)

    def flush(self, pbar):
        if jax.process_index() != 0 or self.prev_metrics is None:
            return
        materialized = {k: self._materialize_scalar(v) for k, v in self.prev_metrics.items()}
        _log_metrics_if_primary(materialized, self.prev_step, pbar)
        self.prev_step = None
        self.prev_metrics = None
def _build_lightweight_train_metrics(
    step,
    tokens_per_opt_step,
    train_loss,
    train_med_loss,
    train_lower_90th_mean_loss,
    lr_schedule,
    b2_schedule,
    lm_head_lr_schedule=None,
    loss_skip_stats=None,
):
    metrics = {}
    metrics['train_loss'] = train_loss
    metrics['train_med_loss'] = train_med_loss
    metrics['train_lower_90th_mean_loss'] = train_lower_90th_mean_loss
    metrics['train_tokens_seen'] = (step + 1) * tokens_per_opt_step
    metrics['lr'] = lr_schedule(step)
    if lm_head_lr_schedule is not None:
        metrics['lm_head_lr'] = lm_head_lr_schedule(step)
    if b2_schedule is not None:
        metrics['b2'] = b2_schedule(step)
    if loss_skip_stats is not None:
        metrics.update(loss_skip_stats)
    return metrics


def _build_heavy_train_metrics(
    output_logit_mean,
    output_logit_norm,
    output_logit_std,
    output_logit_entropy,
    opt_state,
    grads,
    qkv_stats,
    logit_grad_stats,
    logit_grad_scaling_stats,
    loss_skip_stats=None,
    include_model_diagnostics: bool = True,
    output_embedding_metric_groups=(),
    adam_eps: float = 1e-8,
    parameter_update_metrics=None,
    head_scale_metrics=None,
    final_norm_channel_metrics=None,
    high_confidence_error_metrics=None,
    lm_head_diagnostic_metrics=None,
):
    metrics = {}
    if output_logit_mean is not None:
        metrics['train_output_logit_mean'] = output_logit_mean
        metrics['train_output_logit_norm'] = output_logit_norm
        metrics['train_output_logit_std'] = output_logit_std
        metrics['train_output_logit_entropy'] = output_logit_entropy
    if include_model_diagnostics:
        metrics.update(utils.get_layer_weight_norms_split(opt_state.model))
        metrics.update(utils.get_layer_moment_norms(opt_state))
        metrics.update(utils.get_output_embedding_group_metrics(opt_state, output_embedding_metric_groups, eps=adam_eps))
        metrics.update(utils.get_lm_head_oblique_target_metrics(opt_state.model))
        if grads is not None:
            metrics.update(utils.get_layer_grad_norms_split(grads))
            metrics.update(utils.get_layer_second_moment_rms_metrics(grads, opt_state))
    if logit_grad_stats is not None:
        metrics.update(logit_grad_stats)
    if logit_grad_scaling_stats is not None:
        metrics.update(logit_grad_scaling_stats)
    metrics.update(qkv_stats)
    if loss_skip_stats is not None:
        metrics.update(loss_skip_stats)
    if parameter_update_metrics:
        metrics.update(parameter_update_metrics)
    if head_scale_metrics:
        metrics.update(head_scale_metrics)
    if final_norm_channel_metrics:
        metrics.update(final_norm_channel_metrics)
    if high_confidence_error_metrics:
        metrics.update(high_confidence_error_metrics)
    if lm_head_diagnostic_metrics:
        metrics.update(lm_head_diagnostic_metrics)
    return metrics


def _build_train_metrics(
    step,
    tokens_per_opt_step,
    train_loss,
    train_med_loss,
    train_lower_90th_mean_loss,
    output_logit_mean,
    output_logit_norm,
    output_logit_std,
    output_logit_entropy,
    opt_state,
    grads,
    lr_schedule,
    b2_schedule,
    lm_head_lr_schedule,
    qkv_stats,
    logit_grad_stats,
    logit_grad_scaling_stats,
    loss_skip_stats=None,
    include_model_diagnostics: bool = True,
    output_embedding_metric_groups=(),
    adam_eps: float = 1e-8,
    parameter_update_metrics=None,
    head_scale_metrics=None,
    final_norm_channel_metrics=None,
    high_confidence_error_metrics=None,
    lm_head_diagnostic_metrics=None,
):
    metrics = _build_lightweight_train_metrics(
        step,
        tokens_per_opt_step,
        train_loss,
        train_med_loss,
        train_lower_90th_mean_loss,
        lr_schedule,
        b2_schedule,
        lm_head_lr_schedule,
        loss_skip_stats,
    )
    metrics.update(_build_heavy_train_metrics(
        output_logit_mean,
        output_logit_norm,
        output_logit_std,
        output_logit_entropy,
        opt_state,
        grads,
        qkv_stats,
        logit_grad_stats,
        logit_grad_scaling_stats,
        loss_skip_stats=None,
        include_model_diagnostics=include_model_diagnostics,
        output_embedding_metric_groups=output_embedding_metric_groups,
        adam_eps=adam_eps,
        parameter_update_metrics=parameter_update_metrics,
        head_scale_metrics=head_scale_metrics,
        final_norm_channel_metrics=final_norm_channel_metrics,
        high_confidence_error_metrics=high_confidence_error_metrics,
        lm_head_diagnostic_metrics=lm_head_diagnostic_metrics,
    ))
    return metrics


def _log_metrics_if_primary(metrics, step, pbar):
    if jax.process_index() == 0:
        wandb.log(metrics, step)
        if 'train_loss' in metrics:
            pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')

