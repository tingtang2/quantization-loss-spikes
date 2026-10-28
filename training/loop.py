"""The stateful training loop for a fully initialized experiment."""

from collections import deque
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm.auto import tqdm

import model as model_lib
import utils
from diagnostics.lm_head import (
    get_head_scale_metrics,
    get_lm_head_diagnostic_metrics,
    get_logit_gaps_by_lm_head,
    get_logit_grad_scaling_stats,
    get_logit_grad_sum_stats,
    get_mean_and_norm_output_logit,
)
from diagnostics.quantization import get_output_logit_and_optional_metrics
from training.checkpoints import (
    _resolve_diagnostics_dir,
    _save_checkpoint,
    _save_eval_diagnostics,
    _should_checkpoint,
)
from training.distributed import _maybe_shard_multihost_batch
from training.evaluation import eval_step
from training.logging import (
    _AsyncMetricWriter,
    _build_heavy_train_metrics,
    _build_lightweight_train_metrics,
    _build_train_metrics,
    _log_metrics_if_primary,
    _prepare_output_embedding_metric_groups,
    _to_host_numpy,
)
from training.losses import loss_fn, loss_fn_z_loss
from training.steps import (
    _copy_model_state_for_metrics,
    amplify_lm_head_mean,
    get_final_norm_activation_grad_metrics,
    get_parameter_update_metrics,
    restore_lm_head_parameters,
    restore_out_ln_scale,
    snapshot_lm_head_parameters,
    snapshot_out_ln_scale,
    train_step,
    train_step_centered,
    train_step_with_skip,
    train_step_with_skip_centered,
    train_step_with_soft_cap,
    train_step_with_soft_cap_centered,
    train_step_z_loss,
    train_step_z_loss_centered,
    train_step_z_loss_with_skip,
    train_step_z_loss_with_skip_centered,
    train_step_z_loss_with_soft_cap,
    train_step_z_loss_with_soft_cap_centered,
)


@dataclass
class TrainingContext:
    c: object
    n_params: dict
    ds_train: object
    ds_valid: object
    mesh: object
    model_graphdef: object
    opt_graphdef: object
    opt_state: object
    num_opt_steps: int
    start_step: int
    tokens_per_opt_step: int
    lr_schedule: object
    lm_head_tx_lr_schedule: object
    lm_head_metric_lr_schedule: object | None
    b2_schedule: object | None
    ckpt_dir: str | None
    ckpt_mngr: object | None
    multihost_batch_sharding: bool
    conventional_data_parallel: bool
    profiling_enabled: bool
    profiling_start_step: int
    profiling_num_steps: int
    profiling_log_dir: str
    profiling_create_perfetto_trace: bool


def run_training(context: TrainingContext):
    c = context.c
    n_params = context.n_params
    ds_train = context.ds_train
    ds_valid = context.ds_valid
    mesh = context.mesh
    model_graphdef = context.model_graphdef
    opt_graphdef = context.opt_graphdef
    opt_state = context.opt_state
    num_opt_steps = context.num_opt_steps
    start_step = context.start_step
    tokens_per_opt_step = context.tokens_per_opt_step
    lr_schedule = context.lr_schedule
    lm_head_tx_lr_schedule = context.lm_head_tx_lr_schedule
    lm_head_metric_lr_schedule = context.lm_head_metric_lr_schedule
    b2_schedule = context.b2_schedule
    ckpt_dir = context.ckpt_dir
    ckpt_mngr = context.ckpt_mngr
    multihost_batch_sharding = context.multihost_batch_sharding
    conventional_data_parallel = context.conventional_data_parallel
    profiling_enabled = context.profiling_enabled
    profiling_start_step = context.profiling_start_step
    profiling_num_steps = context.profiling_num_steps
    profiling_log_dir = context.profiling_log_dir
    profiling_create_perfetto_trace = context.profiling_create_perfetto_trace

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
    log_metrics_per_step = bool(getattr(c, "log_metrics_per_step", False))
    log_metrics_per_step_full = bool(getattr(c, "log_metrics_per_step_full", False))
    log_effective_lr_every_steps = int(getattr(c, "log_effective_lr_every_steps", -1))
    if log_effective_lr_every_steps == 0 or log_effective_lr_every_steps < -1:
        raise ValueError("log_effective_lr_every_steps must be -1 (disabled) or positive.")
    use_async_metric_writer = log_metrics_per_step and jax.process_count() == 1
    metric_writer = _AsyncMetricWriter() if use_async_metric_writer else None
    output_embedding_metric_groups = _prepare_output_embedding_metric_groups(c)
    if log_metrics_per_step and (not use_async_metric_writer) and jax.process_index() == 0:
        print("disabling async per-step metric writer in multi-host mode; using direct logging instead")
    if output_embedding_metric_groups and jax.process_index() == 0:
        group_summary = ", ".join(
            f"{group['name']}({len(group['token_ids'])})" for group in output_embedding_metric_groups
        )
        print(f"output embedding group metrics enabled: {group_summary}")
    log_logit_grad_stats = bool(getattr(c, "log_logit_grad_stats", False))
    log_logit_grad_scaling_stats = bool(getattr(c, "log_logit_grad_scaling_stats", False))
    log_lm_head_diagnostic_metrics = bool(
        getattr(c, "log_lm_head_diagnostic_metrics", False)
    )
    log_lm_head_quantization_error_metrics = bool(
        getattr(c, "log_lm_head_quantization_error_metrics", False)
    )
    log_parameter_update_metrics = bool(getattr(c, "log_parameter_update_metrics", False))
    log_head_scale_metrics = bool(getattr(c, "log_head_scale_metrics", False))
    log_final_norm_channel_metrics = bool(getattr(c, "log_final_norm_channel_metrics", False))
    log_high_confidence_error_metrics = bool(getattr(c, "log_high_confidence_error_metrics", False))
    log_final_norm_activation_grad_metrics = bool(
        getattr(c, "log_final_norm_activation_grad_metrics", False)
    )
    out_ln_use_scale = model_lib.resolve_out_ln_use_scale(c.model)
    if log_head_scale_metrics and not out_ln_use_scale:
        raise ValueError("log_head_scale_metrics requires the final out_ln scale to be enabled.")
    if log_head_scale_metrics and jax.process_index() == 0:
        print("head scale metrics enabled on heavy logging steps")
    if log_final_norm_channel_metrics and not out_ln_use_scale:
        raise ValueError("log_final_norm_channel_metrics requires the final out_ln scale to be enabled.")
    if log_final_norm_channel_metrics and jax.process_index() == 0:
        print("final norm channel metrics enabled on heavy logging steps")
    if log_high_confidence_error_metrics and not out_ln_use_scale:
        raise ValueError("log_high_confidence_error_metrics requires the final out_ln scale to be enabled.")
    if log_high_confidence_error_metrics and jax.process_index() == 0:
        print("high-confidence error metrics enabled on heavy logging steps")
    if log_parameter_update_metrics and jax.process_index() == 0:
        print("parameter update metrics enabled on heavy logging steps")
    if log_final_norm_activation_grad_metrics and jax.process_index() == 0:
        print("final norm activation gradient metrics enabled on heavy logging steps")
    if log_lm_head_diagnostic_metrics and jax.process_index() == 0:
        print("LM-head diagnostic metrics enabled on heavy logging steps")
    if log_lm_head_quantization_error_metrics and jax.process_index() == 0:
        print(
            "LM-head BF16/native versus FP32 shadow quantization-error metrics "
            "enabled on heavy logging steps"
        )
    collect_qkv_stats = bool(getattr(c.diagnostics, "collect_qkv_stats", True))
    loss_skip_cfg = getattr(c.opt, "loss_skip", None)
    loss_skip_enabled = bool(getattr(loss_skip_cfg, "enabled", False))
    loss_skip_warmup_steps = int(getattr(loss_skip_cfg, "warmup_steps", 1000))
    loss_skip_window_size = int(getattr(loss_skip_cfg, "window_size", 1000))
    loss_skip_min_history = int(getattr(loss_skip_cfg, "min_history", 256))
    loss_skip_z_soft = float(getattr(loss_skip_cfg, "z_soft", 5.0))
    loss_skip_z_hard = float(getattr(loss_skip_cfg, "z_hard", 8.0))
    loss_skip_soft_weight = float(getattr(loss_skip_cfg, "soft_weight", 1.0))
    loss_skip_eps = float(getattr(loss_skip_cfg, "eps", 1e-6))
    loss_skip_use_log_loss = bool(getattr(loss_skip_cfg, "use_log_loss", False))
    loss_skip_abs_hard_cfg = getattr(loss_skip_cfg, "abs_hard", None)
    loss_skip_abs_hard_fixed = None if loss_skip_abs_hard_cfg is None else float(loss_skip_abs_hard_cfg)
    loss_skip_history = deque(maxlen=loss_skip_window_size)
    loss_skip_cutoff_logged = False
    loss_cap_cfg = getattr(c.opt, "loss_cap", None)
    loss_cap_enabled = bool(getattr(loss_cap_cfg, "enabled", False))
    loss_cap_soft_cap = float(getattr(loss_cap_cfg, "soft_cap", 30.0))
    loss_cap_warmup_steps = int(getattr(loss_cap_cfg, "warmup_steps", 0))
    loss_rewrite_cfg = getattr(c.opt, "loss_rewrite", None)
    loss_rewrite_enabled = bool(getattr(loss_rewrite_cfg, "enabled", False))
    loss_rewrite_replacement_token_id = int(getattr(loss_rewrite_cfg, "replacement_token_id", 11))
    loss_rewrite_warmup_steps = int(getattr(loss_rewrite_cfg, "warmup_steps", 1000))
    loss_rewrite_window_size = int(getattr(loss_rewrite_cfg, "window_size", 1000))
    loss_rewrite_min_history = int(getattr(loss_rewrite_cfg, "min_history", 256))
    loss_rewrite_z_hard = float(getattr(loss_rewrite_cfg, "z_hard", 8.0))
    loss_rewrite_eps = float(getattr(loss_rewrite_cfg, "eps", 1e-6))
    loss_rewrite_use_log_loss = bool(getattr(loss_rewrite_cfg, "use_log_loss", False))
    loss_rewrite_history = deque(maxlen=loss_rewrite_window_size)
    loss_rewrite_cutoff_logged = False
    if loss_skip_enabled and loss_cap_enabled:
        raise ValueError("opt.loss_skip.enabled and opt.loss_cap.enabled cannot both be True.")
    if (loss_skip_enabled and loss_rewrite_enabled) or (loss_cap_enabled and loss_rewrite_enabled):
        raise ValueError(
            "opt.loss_rewrite.enabled cannot be combined with opt.loss_skip.enabled or opt.loss_cap.enabled."
        )
    if loss_skip_enabled and jax.process_index() == 0:
        print(
            "loss-skip enabled: "
            f"warmup_steps={loss_skip_warmup_steps}, window_size={loss_skip_window_size}, "
            f"min_history={loss_skip_min_history}, z_soft={loss_skip_z_soft}, z_hard={loss_skip_z_hard}, "
            f"soft_weight={loss_skip_soft_weight}, use_log_loss={loss_skip_use_log_loss}, abs_hard="
            f"{loss_skip_abs_hard_fixed if loss_skip_abs_hard_fixed is not None else 'inf'}"
        )
    if loss_cap_enabled and jax.process_index() == 0:
        print(f"loss-cap enabled: soft_cap={loss_cap_soft_cap}, warmup_steps={loss_cap_warmup_steps}")
    if loss_rewrite_enabled and jax.process_index() == 0:
        print(
            "loss-rewrite enabled: "
            f"replacement_token_id={loss_rewrite_replacement_token_id}, "
            f"warmup_steps={loss_rewrite_warmup_steps}, window_size={loss_rewrite_window_size}, "
            f"min_history={loss_rewrite_min_history}, z_hard={loss_rewrite_z_hard}, "
            f"use_log_loss={loss_rewrite_use_log_loss}"
        )

    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step

    if profiling_enabled and profiling_start_step + profiling_num_steps > num_opt_steps:
        raise ValueError(
            'The profiling window must fit within diagnostics.end_step/num_opt_steps; '
            f'window ends at step {profiling_start_step + profiling_num_steps}, '
            f'but training stops at {num_opt_steps}.'
        )
    
    mucentering_configured = bool(getattr(c.opt, "mucentering", False))
    mean_amplification_cfg = getattr(c.opt, "lm_head_mean_amplification", None)
    mean_amplification_enabled = bool(
        getattr(mean_amplification_cfg, "enabled", False)
    )
    mean_amplification_start_step = int(
        getattr(mean_amplification_cfg, "start_step", 0)
    )
    mean_amplification_coefficient = float(
        getattr(mean_amplification_cfg, "coefficient", 1.0)
    )
    lm_head_freeze_cfg = getattr(c.opt, "lm_head_freeze", None)
    lm_head_freeze_enabled = bool(getattr(lm_head_freeze_cfg, "enabled", False))
    lm_head_freeze_start_step = int(getattr(lm_head_freeze_cfg, "start_step", 0))
    out_ln_scale_freeze_cfg = getattr(c.opt, "out_ln_scale_freeze", None)
    out_ln_scale_freeze_enabled = bool(
        getattr(out_ln_scale_freeze_cfg, "enabled", False)
    )
    out_ln_scale_freeze_start_step = int(
        getattr(out_ln_scale_freeze_cfg, "start_step", 0)
    )
    if mean_amplification_enabled:
        if mean_amplification_start_step < 0:
            raise ValueError(
                "opt.lm_head_mean_amplification.start_step must be non-negative."
            )
        if (
            not math.isfinite(mean_amplification_coefficient)
            or mean_amplification_coefficient <= 0.0
        ):
            raise ValueError(
                "opt.lm_head_mean_amplification.coefficient must be finite and positive."
            )
        if jax.process_index() == 0:
            centering_transition = (
                "; mu centering will stop at that step" if mucentering_configured else ""
            )
            print(
                "LM-head mean amplification enabled: "
                f"start_step={mean_amplification_start_step}, "
                f"coefficient={mean_amplification_coefficient}, "
                f"per-step mean multiplier={1.0 + mean_amplification_coefficient}"
                f"{centering_transition}"
            )
    if lm_head_freeze_enabled:
        if lm_head_freeze_start_step < 0:
            raise ValueError("opt.lm_head_freeze.start_step must be non-negative.")
        if jax.process_index() == 0:
            print(
                "LM head freeze enabled: "
                f"updates are suppressed starting at step {lm_head_freeze_start_step}"
            )
    if out_ln_scale_freeze_enabled:
        if not out_ln_use_scale:
            raise ValueError(
                "opt.out_ln_scale_freeze.enabled requires the final out_ln scale to be enabled."
            )
        if out_ln_scale_freeze_start_step < 0:
            raise ValueError("opt.out_ln_scale_freeze.start_step must be non-negative.")
        if jax.process_index() == 0:
            print(
                "Final out_ln scale freeze enabled: "
                f"updates are suppressed starting at step {out_ln_scale_freeze_start_step}"
            )

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)
    profiler_active = False
    for step in pbar:
        mean_amplification_active = (
            mean_amplification_enabled and step >= mean_amplification_start_step
        )
        lm_head_freeze_active = (
            lm_head_freeze_enabled and step >= lm_head_freeze_start_step
        )
        out_ln_scale_freeze_active = (
            out_ln_scale_freeze_enabled and step >= out_ln_scale_freeze_start_step
        )
        mucentering = mucentering_configured and not mean_amplification_active
        batch = ds_train[step]
        batch = _maybe_shard_multihost_batch(
            batch,
            mesh,
            multihost_batch_sharding,
            conventional_data_parallel,
        )
        will_log_heavy_metrics = log_metrics_per_step_full or (
            (train_loss_num + 1) * tokens_per_opt_step >= c.log_every_tokens
        )
        should_collect_parameter_update_metrics = log_parameter_update_metrics and will_log_heavy_metrics
        should_collect_head_scale_metrics = log_head_scale_metrics and will_log_heavy_metrics
        should_collect_final_norm_activation_grads = (
            log_final_norm_activation_grad_metrics and will_log_heavy_metrics
        )
        # Keep the training executable on its ordinary, low-memory compilation path.
        collect_step_parameter_update_metrics = False
        pre_update_model_state = None
        parameter_update_batch = batch
        parameter_update_metrics = {}
        head_scale_metrics = {}
        need_step_grads = False
        pre_output_logit_mean = None
        pre_output_logit_norm = None
        pre_output_logit_std = None
        pre_output_logit_entropy = None
        final_norm_channel_metrics = {}
        high_confidence_error_metrics = {}
        lm_head_diagnostic_metrics = {}
        if will_log_heavy_metrics and (
            log_final_norm_channel_metrics
            or log_high_confidence_error_metrics
            or log_lm_head_quantization_error_metrics
        ):
            output_stats, optional_metrics = get_output_logit_and_optional_metrics(
                opt_state.model,
                model_graphdef,
                batch,
                log_final_norm_channel_metrics,
                log_high_confidence_error_metrics,
                log_lm_head_quantization_error_metrics,
            )
            if log_final_norm_channel_metrics:
                final_norm_channel_metrics = {
                    name: value for name, value in optional_metrics.items() if name.startswith('final_norm/')
                }
            if log_high_confidence_error_metrics:
                high_confidence_error_metrics = {
                    name: value for name, value in optional_metrics.items() if name.startswith('high_conf_error/')
                }
            if log_lm_head_quantization_error_metrics:
                lm_head_diagnostic_metrics.update({
                    name: value
                    for name, value in optional_metrics.items()
                    if name.startswith('lm_head_quantization/')
                })
            (
                pre_output_logit_mean,
                pre_output_logit_norm,
                pre_output_logit_std,
                pre_output_logit_entropy,
            ) = output_stats
        elif will_log_heavy_metrics and not should_collect_final_norm_activation_grads:
            (
                pre_output_logit_mean,
                pre_output_logit_norm,
                pre_output_logit_std,
                pre_output_logit_entropy,
            ) = get_mean_and_norm_output_logit(opt_state.model, model_graphdef, batch)
        logit_grad_stats = None
        logit_grad_scaling_stats = None
        if log_logit_grad_stats:
            logit_grad_stats = get_logit_grad_sum_stats(opt_state.model, model_graphdef, batch)
        if log_logit_grad_scaling_stats:
            logit_grad_scaling_stats = get_logit_grad_scaling_stats(opt_state.model, model_graphdef, batch)
        if log_lm_head_diagnostic_metrics and will_log_heavy_metrics:
            lm_head_diagnostic_metrics.update(
                get_lm_head_diagnostic_metrics(opt_state.model, model_graphdef, batch)
            )
        loss_skip_stats = None
        gate_apply = False
        gate_center = 0.0
        gate_mad = 1.0
        gate_abs_hard = float("inf")
        cap_apply = False
        rewrite_gate_apply = False
        rewrite_gate_center = 0.0
        rewrite_gate_mad = 1.0
        rewrite_hard_mask_np = None
        rewrite_probe_raw_np = None
        rewrite_probe_loss = None
        if loss_skip_enabled and step >= loss_skip_warmup_steps and len(loss_skip_history) >= loss_skip_min_history:
            hist = np.asarray(loss_skip_history, dtype=np.float32)
            gate_center = float(np.median(hist))
            gate_mad = float(np.median(np.abs(hist - gate_center)))
            if loss_skip_abs_hard_fixed is not None:
                gate_abs_hard = float(loss_skip_abs_hard_fixed)
            else:
                gate_abs_hard = float("inf")
            gate_apply = True
            if (not loss_skip_cutoff_logged) and jax.process_index() == 0:
                scale = max(1.4826 * gate_mad, loss_skip_eps)
                z_hard_cutoff = gate_center + loss_skip_z_hard * scale
                domain = "log1p(loss)" if loss_skip_use_log_loss else "loss"
                print(
                    "loss-skip z_hard cutoff active: "
                    f"domain={domain}, z_hard={loss_skip_z_hard}, "
                    f"cutoff={z_hard_cutoff:.6f}, abs_hard={gate_abs_hard}"
                )
                loss_skip_cutoff_logged = True
        if (
            loss_rewrite_enabled
            and step >= loss_rewrite_warmup_steps
            and len(loss_rewrite_history) >= loss_rewrite_min_history
        ):
            hist = np.asarray(loss_rewrite_history, dtype=np.float32)
            rewrite_gate_center = float(np.median(hist))
            rewrite_gate_mad = float(np.median(np.abs(hist - rewrite_gate_center)))
            rewrite_gate_apply = True
            if (not loss_rewrite_cutoff_logged) and jax.process_index() == 0:
                scale = max(1.4826 * rewrite_gate_mad, loss_rewrite_eps)
                z_hard_cutoff = rewrite_gate_center + loss_rewrite_z_hard * scale
                domain = "log1p(loss)" if loss_rewrite_use_log_loss else "loss"
                print(
                    "loss-rewrite z_hard cutoff active: "
                    f"domain={domain}, z_hard={loss_rewrite_z_hard}, "
                    f"cutoff={z_hard_cutoff:.6f}"
                )
                loss_rewrite_cutoff_logged = True
        # Copy only the parameters, immediately before the donated training step.
        if should_collect_parameter_update_metrics or should_collect_final_norm_activation_grads:
            pre_update_model_state = _copy_model_state_for_metrics(opt_state.model)
        frozen_lm_head_parameters = None
        if lm_head_freeze_active:
            frozen_lm_head_parameters = snapshot_lm_head_parameters(
                opt_state.model,
                model_graphdef,
            )
        frozen_out_ln_scale = None
        if out_ln_scale_freeze_active:
            frozen_out_ln_scale = snapshot_out_ln_scale(
                opt_state.model,
                model_graphdef,
            )

        # Profile only warmed-up optimizer steps, excluding setup/compilation and
        # expensive optional diagnostics. The trace remains useful for both the
        # GPU execution timeline and Python-side input/logging overhead.
        if profiling_enabled and step == profiling_start_step:
            jax.profiler.start_trace(
                profiling_log_dir,
                create_perfetto_trace=profiling_create_perfetto_trace,
            )
            profiler_active = True
            print(f'Started JAX profiler trace at training step {step}.')

        # training step
        if loss_skip_enabled:
            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads, parameter_update_metrics = train_step_z_loss_with_skip_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads, parameter_update_metrics = train_step_z_loss_with_skip(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads, parameter_update_metrics = train_step_with_skip_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads, parameter_update_metrics = train_step_with_skip(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
            skip_stats_host = {k: float(v) for k, v in jax.device_get(skip_stats_device).items()}
            skip_stats_host['loss_skip_gate_applied'] = float(gate_apply)
            skip_stats_host['loss_skip_center'] = float(gate_center)
            skip_stats_host['loss_skip_mad'] = float(gate_mad)
            skip_stats_host['loss_skip_abs_hard'] = float(gate_abs_hard)
            skip_stats_host['loss_skip_use_log_loss'] = float(loss_skip_use_log_loss)
            skip_stats_host['loss_skip_z_hard_cutoff'] = float(
                gate_center + loss_skip_z_hard * max(1.4826 * gate_mad, loss_skip_eps)
            )
            loss_skip_stats = skip_stats_host
        elif loss_cap_enabled:
            cap_apply = step >= loss_cap_warmup_steps
            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads, parameter_update_metrics = train_step_z_loss_with_soft_cap_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads, parameter_update_metrics = train_step_z_loss_with_soft_cap(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads, parameter_update_metrics = train_step_with_soft_cap_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads, parameter_update_metrics = train_step_with_soft_cap(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                        collect_step_parameter_update_metrics,
                    )
            loss_skip_stats = {k: float(v) for k, v in jax.device_get(cap_stats_device).items()}
        elif loss_rewrite_enabled:
            if c.opt.use_z_loss:
                rewrite_probe_loss, (rewrite_probe_raw_loss, _) = loss_fn_z_loss(
                    opt_state.model, model_graphdef, batch, False
                )
            else:
                rewrite_probe_loss, (rewrite_probe_raw_loss, _) = loss_fn(
                    opt_state.model, model_graphdef, batch, False
                )
            rewrite_probe_raw_np = _to_host_numpy(rewrite_probe_raw_loss, dtype=np.float32)
            rewrite_stats_np = (
                np.log1p(np.maximum(rewrite_probe_raw_np, 0.0))
                if loss_rewrite_use_log_loss
                else rewrite_probe_raw_np
            )
            if rewrite_gate_apply:
                scale = max(1.4826 * rewrite_gate_mad, loss_rewrite_eps)
                rewrite_z = (rewrite_stats_np - rewrite_gate_center) / scale
                rewrite_hard_mask_np = rewrite_z > loss_rewrite_z_hard
            else:
                rewrite_hard_mask_np = np.zeros_like(rewrite_stats_np, dtype=bool)

            rewrite_hard_mask = jnp.asarray(rewrite_hard_mask_np, dtype=jnp.bool_)
            replacement_token = jnp.asarray(loss_rewrite_replacement_token_id, dtype=batch.dtype)
            rewritten_inputs = jnp.where(rewrite_hard_mask, replacement_token, batch[:, :-1])
            rewritten_batch = batch.at[:, :-1].set(rewritten_inputs)
            parameter_update_batch = rewritten_batch

            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step_z_loss_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        rewritten_batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step_z_loss(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        rewritten_batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        rewritten_batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        rewritten_batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
            num_replaced = int(np.count_nonzero(rewrite_hard_mask_np))
            total_rewrite_tokens = int(rewrite_hard_mask_np.size)
            loss_skip_stats = {
                'loss_rewrite_gate_applied': float(rewrite_gate_apply),
                'loss_rewrite_center': float(rewrite_gate_center),
                'loss_rewrite_mad': float(rewrite_gate_mad),
                'loss_rewrite_z_hard': float(loss_rewrite_z_hard),
                'loss_rewrite_z_hard_cutoff': float(
                    rewrite_gate_center + loss_rewrite_z_hard * max(1.4826 * rewrite_gate_mad, loss_rewrite_eps)
                ),
                'loss_rewrite_num_replaced': float(num_replaced),
                'loss_rewrite_frac_replaced': float(num_replaced / max(total_rewrite_tokens, 1)),
                'loss_rewrite_probe_loss': float(jax.device_get(rewrite_probe_loss)),
                'loss_rewrite_use_log_loss': float(loss_rewrite_use_log_loss),
            }
        else:
            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step_z_loss_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step_z_loss(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads, parameter_update_metrics = train_step(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        collect_qkv_stats,
                        need_step_grads,
                        collect_step_parameter_update_metrics,
                    )
        if mean_amplification_active and not lm_head_freeze_active:
            opt_state = amplify_lm_head_mean(
                opt_state,
                opt_graphdef,
                mean_amplification_coefficient,
            )
        if lm_head_freeze_active:
            opt_state = restore_lm_head_parameters(
                opt_state,
                opt_graphdef,
                frozen_lm_head_parameters,
            )
        if out_ln_scale_freeze_active:
            opt_state = restore_out_ln_scale(
                opt_state,
                opt_graphdef,
                frozen_out_ln_scale,
            )
        if profiler_active and step + 1 == profiling_start_step + profiling_num_steps:
            # Dispatches are asynchronous on GPU. Wait for the final profiled
            # step before exporting, otherwise its kernels may be omitted.
            batch_loss.block_until_ready()
            jax.profiler.stop_trace()
            profiler_active = False
            print(f'Saved JAX profiler trace to {profiling_log_dir}.')
        if should_collect_parameter_update_metrics:
            parameter_update_metrics = get_parameter_update_metrics(
                pre_update_model_state,
                opt_state.model,
                model_graphdef,
                parameter_update_batch,
            )
        if should_collect_final_norm_activation_grads:
            objective_mode = 'skip' if loss_skip_enabled else 'soft_cap' if loss_cap_enabled else 'standard'
            output_stats, activation_grad_metrics = get_final_norm_activation_grad_metrics(
                pre_update_model_state,
                model_graphdef,
                parameter_update_batch,
                bool(c.opt.use_z_loss),
                objective_mode,
                gate_center,
                gate_mad,
                loss_skip_z_soft,
                loss_skip_z_hard,
                gate_abs_hard,
                loss_skip_soft_weight,
                loss_skip_eps,
                gate_apply,
                loss_skip_use_log_loss,
                loss_cap_soft_cap,
                cap_apply,
            )
            (
                pre_output_logit_mean,
                pre_output_logit_norm,
                pre_output_logit_std,
                pre_output_logit_entropy,
            ) = output_stats
            parameter_update_metrics.update(activation_grad_metrics)
        if should_collect_head_scale_metrics:
            head_scale_metrics = get_head_scale_metrics(opt_state.model, model_graphdef, batch)
        if loss_skip_enabled:
            raw_np = _to_host_numpy(train_raw_loss, dtype=np.float32, flatten=True)
            stats_np = np.log1p(np.maximum(raw_np, 0.0)) if loss_skip_use_log_loss else raw_np
            if gate_apply:
                scale = max(1.4826 * gate_mad, loss_skip_eps)
                z = (stats_np - gate_center) / scale
                hard = np.logical_or(z > loss_skip_z_hard, stats_np > gate_abs_hard)
                keep = np.logical_not(hard)
            else:
                keep = np.ones_like(stats_np, dtype=bool)
            if keep.any():
                loss_skip_history.extend(stats_np[keep].tolist())
        if loss_rewrite_enabled and rewrite_probe_raw_np is not None:
            rewrite_stats_np = (
                np.log1p(np.maximum(rewrite_probe_raw_np, 0.0))
                if loss_rewrite_use_log_loss
                else rewrite_probe_raw_np
            ).reshape(-1)
            rewrite_keep = np.logical_not(rewrite_hard_mask_np.reshape(-1))
            if rewrite_gate_apply:
                if rewrite_keep.any():
                    loss_rewrite_history.extend(rewrite_stats_np[rewrite_keep].tolist())
            else:
                loss_rewrite_history.extend(rewrite_stats_np.tolist())
        batch_loss_to_log = (
            jnp.asarray(loss_skip_stats['loss_skip_unmasked_loss'], dtype=jnp.float32)
            if (loss_skip_enabled and loss_skip_stats is not None and 'loss_skip_unmasked_loss' in loss_skip_stats)
            else jnp.asarray(loss_skip_stats['loss_cap_uncapped_loss'], dtype=jnp.float32)
            if (loss_cap_enabled and loss_skip_stats is not None and 'loss_cap_uncapped_loss' in loss_skip_stats)
            else jnp.asarray(loss_skip_stats['loss_rewrite_probe_loss'], dtype=jnp.float32)
            if (loss_rewrite_enabled and loss_skip_stats is not None and 'loss_rewrite_probe_loss' in loss_skip_stats)
            else jnp.mean(train_raw_loss)
        )
        # if jax.process_index() == 0:
        #     min_train_loss = float(jax.device_get(jnp.min(train_raw_loss)))
        #     assert min_train_loss >= 0.0, f"negative train loss: {min_train_loss}"
        
        if c.diagnostics.save_raw_losses:
            train_logit_gaps, train_target_gaps = get_logit_gaps_by_lm_head(opt_state.model, model_graphdef, batch)

        train_loss_sum += batch_loss_to_log
        train_med_loss_sum += jnp.median(train_raw_loss)
        train_lower_90th_mean_loss_sum += utils.compute_lower_90th_percentile_mean(train_raw_loss)
        train_loss_num += 1
        should_log_interval_metrics = train_loss_num * tokens_per_opt_step >= c.log_every_tokens
        effective_lr_metrics = {}
        if log_effective_lr_every_steps > 0 and step % log_effective_lr_every_steps == 0:
            effective_lr_metrics = utils.get_effective_learning_rate_metrics(
                opt_state,
                learning_rate=lr_schedule(step),
                lm_head_learning_rate=lm_head_tx_lr_schedule(step),
                beta2=b2_schedule(step) if b2_schedule is not None else c.opt.b2,
                eps=c.opt.eps,
            )

        # logging
        if log_metrics_per_step:
            metrics = _build_lightweight_train_metrics(
                step,
                tokens_per_opt_step,
                batch_loss_to_log,
                jnp.median(train_raw_loss),
                utils.compute_lower_90th_percentile_mean(train_raw_loss),
                lr_schedule,
                b2_schedule,
                lm_head_metric_lr_schedule,
                loss_skip_stats,
            )
            metrics.update(effective_lr_metrics)
            if log_metrics_per_step_full:
                metrics.update(_build_heavy_train_metrics(
                    pre_output_logit_mean,
                    pre_output_logit_norm,
                    pre_output_logit_std,
                    pre_output_logit_entropy,
                    opt_state,
                    grads,
                    qkv_stats,
                    logit_grad_stats,
                    logit_grad_scaling_stats,
                    include_model_diagnostics=True,
                    output_embedding_metric_groups=output_embedding_metric_groups,
                    adam_eps=c.opt.eps,
                    parameter_update_metrics=parameter_update_metrics,
                    head_scale_metrics=head_scale_metrics,
                    final_norm_channel_metrics=final_norm_channel_metrics,
                    high_confidence_error_metrics=high_confidence_error_metrics,
                    lm_head_diagnostic_metrics=lm_head_diagnostic_metrics,
                ))
            if metric_writer is not None:
                metric_writer.enqueue(step, metrics, pbar)
            else:
                _log_metrics_if_primary(metrics, step, pbar)
        elif effective_lr_metrics:
            _log_metrics_if_primary(effective_lr_metrics, step, pbar)
        if should_log_interval_metrics:
            if log_metrics_per_step:
                if not log_metrics_per_step_full:
                    metrics = _build_heavy_train_metrics(
                        pre_output_logit_mean,
                        pre_output_logit_norm,
                        pre_output_logit_std,
                        pre_output_logit_entropy,
                        opt_state,
                        grads,
                        qkv_stats,
                        logit_grad_stats,
                        logit_grad_scaling_stats,
                        include_model_diagnostics=True,
                        output_embedding_metric_groups=output_embedding_metric_groups,
                        adam_eps=c.opt.eps,
                        parameter_update_metrics=parameter_update_metrics,
                        head_scale_metrics=head_scale_metrics,
                        final_norm_channel_metrics=final_norm_channel_metrics,
                        high_confidence_error_metrics=high_confidence_error_metrics,
                        lm_head_diagnostic_metrics=lm_head_diagnostic_metrics,
                    )
                    _log_metrics_if_primary(metrics, step, pbar)
            else:
                metrics = _build_train_metrics(
                    step,
                    tokens_per_opt_step,
                    train_loss_sum / train_loss_num,
                    train_med_loss_sum / train_loss_num,
                    train_lower_90th_mean_loss_sum / train_loss_num,
                    pre_output_logit_mean,
                    pre_output_logit_norm,
                    pre_output_logit_std,
                    pre_output_logit_entropy,
                    opt_state,
                    grads,
                    lr_schedule,
                    b2_schedule,
                    lm_head_metric_lr_schedule,
                    qkv_stats,
                    logit_grad_stats,
                    logit_grad_scaling_stats,
                    loss_skip_stats,
                    include_model_diagnostics=True,
                    output_embedding_metric_groups=output_embedding_metric_groups,
                    adam_eps=c.opt.eps,
                    parameter_update_metrics=parameter_update_metrics,
                    head_scale_metrics=head_scale_metrics,
                    final_norm_channel_metrics=final_norm_channel_metrics,
                    high_confidence_error_metrics=high_confidence_error_metrics,
                    lm_head_diagnostic_metrics=lm_head_diagnostic_metrics,
                )
                _log_metrics_if_primary(metrics, step, pbar)
            train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
        
        # eval and checkpointing
        if c.eval_every_steps > 0 and step > 0 and step % c.eval_every_steps == 0:
            (
                eval_loss,
                eval_raw_loss,
                eval_logits,
                mean_eval_output_logit,
                mean_eval_output_logit_std,
                mean_eval_output_logit_entropy,
            ) = eval_step(
                c,
                opt_state.model,
                model_graphdef,
                ds_valid,
                collect_qkv_stats,
                mesh,
                multihost_batch_sharding,
                conventional_data_parallel,
            )
            flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
            metrics = {}
            metrics['eval_loss'] = eval_loss
            metrics['eval_output_logit_mean'] = mean_eval_output_logit
            metrics['eval_output_logit_std'] = mean_eval_output_logit_std
            metrics['eval_output_logit_entropy'] = mean_eval_output_logit_entropy
            metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
            metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            if jax.process_index() == 0:
                wandb.log(metrics, step)

            # diagnostics
            if c.diagnostics.save_raw_losses:
                diagnostics_dir = _resolve_diagnostics_dir(ckpt_dir)
                _save_eval_diagnostics(
                    diagnostics_dir,
                    step,
                    train_raw_loss,
                    eval_raw_loss,
                    train_logit_gaps,
                    train_target_gaps,
                    eval_logits,
                )

        if _should_checkpoint(c, step):
            _save_checkpoint(ckpt_mngr, step, opt_state)
            # Wait for async checkpoint to complete in multihost setting
            if jax.process_count() > 1:
                ckpt_mngr.wait_until_finished()

    if metric_writer is not None:
        metric_writer.flush(pbar)
    if profiler_active:
        # This path is only expected if training was interrupted during the
        # requested window; preserve the partial trace rather than losing it.
        jax.profiler.stop_trace()
    
    if num_opt_steps != len(ds_train):
        print('exiting early')
        wandb.finish()
        if ckpt_mngr is not None:
            ckpt_mngr.close()
        return

    # eval at end of training
    (
        eval_loss,
        eval_raw_loss,
        eval_logits,
        mean_eval_output_logit,
        mean_eval_output_logit_std,
        mean_eval_output_logit_entropy,
    ) = eval_step(
        c,
        opt_state.model,
        model_graphdef,
        ds_valid,
        collect_qkv_stats,
        mesh,
        multihost_batch_sharding,
        conventional_data_parallel,
    )
    metrics = {}
    flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
    metrics['eval_loss'] = eval_loss
    metrics['eval_output_logit_mean'] = mean_eval_output_logit
    metrics['eval_output_logit_std'] = mean_eval_output_logit_std
    metrics['eval_output_logit_entropy'] = mean_eval_output_logit_entropy
    metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
    metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
    if jax.process_index() == 0:
        wandb.log(metrics)
        wandb.finish()
        if c.diagnostics.save_raw_losses:
            diagnostics_dir = _resolve_diagnostics_dir(ckpt_dir)
            if diagnostics_dir:
                utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{num_opt_steps}.npy', data=eval_raw_loss)
            
    # final checkpoint
    if c.checkpoint.turn_on and not c.diagnostics.save_raw_losses:
        final_step = max(num_opt_steps - 1, 0)
        _save_checkpoint(ckpt_mngr, final_step, opt_state)

        ckpt_mngr.wait_until_finished()
        if jax.process_index() == 0:
            print(f'Saved final checkpoint at step {final_step} to {ckpt_mngr.directory}')
        ckpt_mngr.close()

