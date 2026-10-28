import gc
import math
import os
import sys
from collections import deque
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint.checkpoint_managers import preservation_policy
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import model as model_lib
import utils
from configs import resolver_setup  # noqa: F401 - registers OmegaConf resolvers
from training.checkpoints import (
    _StandardNameFormatHNS,
    _save_checkpoint,
    _should_checkpoint,
)
from training.evaluation import eval_step
from training.steps import (
    train_step,
    train_step_centered,
    train_step_z_loss,
    train_step_z_loss_centered,
)


@partial(jax.jit, static_argnames=("model_graphdef",))
def loss_fn_light(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x, return_qkv=False).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]
    return losses.mean(), losses


@partial(jax.jit, static_argnames=("model_graphdef",))
def get_loss_and_output_logit_mean(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x, return_qkv=False).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]
    mean_output_logit = logits[:, :-1].mean()
    return losses.mean(), mean_output_logit


def _compute_rolling_zscore_skip(history, value: float, threshold: float):
    if history.maxlen is None:
        raise ValueError("Rolling z-score history must have a finite maxlen.")

    if len(history) < history.maxlen:
        return False, {
            "skip_detector_ready": 0.0,
            "skip_detector_value": float(value),
            "skip_detector_mean": 0.0,
            "skip_detector_std": 0.0,
            "skip_detector_zscore": 0.0,
            "skip_detector_threshold": float(threshold),
        }

    window = np.asarray(history, dtype=np.float64)
    mean = float(np.mean(window))
    std = float(np.std(window))
    zscore = 0.0 if std <= 0.0 else abs(float(value) - mean) / std
    should_skip = std > 0.0 and zscore >= threshold
    return should_skip, {
        "skip_detector_ready": 1.0,
        "skip_detector_value": float(value),
        "skip_detector_mean": mean,
        "skip_detector_std": std,
        "skip_detector_zscore": zscore,
        "skip_detector_threshold": float(threshold),
    }


def _normalize_checkpoint_path(path: str) -> tuple[str, bool]:
    normalized = str(path).rstrip("/")
    if normalized.startswith("gs://"):
        return normalized, True
    if normalized.startswith("/") or normalized.startswith("./") or normalized.startswith("../"):
        return normalized, False
    return f"gs://{normalized}", True


def _parse_checkpoint_load_path(load_path: str) -> tuple[str, int | None, str, bool]:
    normalized_path, is_gcs = _normalize_checkpoint_path(load_path)
    path_without_scheme = normalized_path[5:] if normalized_path.startswith("gs://") else normalized_path
    parts = [part for part in path_without_scheme.split("/") if part]
    if not parts:
        raise ValueError(f"Invalid checkpoint.load_path: {load_path!r}")

    step_to_load = None
    if parts[-1].isdigit():
        step_to_load = int(parts[-1])
        parts = parts[:-1]

    if not parts:
        raise ValueError(
            "checkpoint.load_path must include at least the checkpoint directory, "
            f"got {load_path!r}."
        )

    ckpt_dir = "/".join(parts)
    if is_gcs:
        ckpt_dir = f"gs://{ckpt_dir}"
    run_name = parts[-1]
    return ckpt_dir, step_to_load, run_name, is_gcs


def _resolve_checkpoint_source(c: DictConfig) -> tuple[str, str, int | None, bool]:
    load_path = getattr(c.checkpoint, "load_path", None)
    if load_path:
        ckpt_dir, inferred_step, inferred_run_name, is_gcs = _parse_checkpoint_load_path(load_path)
        run_name = c.run_name if c.run_name else inferred_run_name
        step_to_load = (
            c.checkpoint.start_step
            if c.checkpoint.start_step is not None
            else inferred_step
        )
        return run_name, ckpt_dir, step_to_load, is_gcs

    run_name = c.run_name if c.run_name else "picodo_run"
    gcp_bucket = getattr(c.checkpoint, "gcp_bucket", None)
    if gcp_bucket:
        normalized_bucket, _ = _normalize_checkpoint_path(gcp_bucket)
        return run_name, f"{normalized_bucket.rstrip('/')}/{run_name}", c.checkpoint.start_step, True

    return run_name, os.path.join(c.checkpoint.workdir, run_name), c.checkpoint.start_step, False


def _checkpoint_restore_explicitly_requested(c: DictConfig) -> bool:
    return (
        getattr(c.checkpoint, "load_path", None) is not None
        or getattr(c.checkpoint, "start_step", None) is not None
    )


def _build_optimizer(c: DictConfig, model, num_opt_steps: int):
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = utils.build_learning_rate_schedule(
        c.opt, c.opt.peak_lr, warmup_steps, num_opt_steps
    )
    lm_head_peak_lr = utils.get_lm_head_peak_lr(c.opt)
    lm_head_tx_lr_schedule = lr_schedule
    lm_head_metric_lr_schedule = None
    if not math.isclose(lm_head_peak_lr, float(c.opt.peak_lr)):
        lm_head_tx_lr_schedule = utils.build_learning_rate_schedule(
            c.opt, lm_head_peak_lr, warmup_steps, num_opt_steps
        )
        lm_head_metric_lr_schedule = lm_head_tx_lr_schedule
        if jax.process_index() == 0:
            print(
                "split lm-head peak lr enabled: "
                f"default_peak_lr={c.opt.peak_lr}, lm_head_peak_lr={lm_head_peak_lr}, "
                f"warmup_steps={warmup_steps}"
            )

    b2_schedule = None
    b2_hparam = c.opt.b2
    b2_cosine_cfg = getattr(c.opt, "b2_cosine_anneal", None)
    if bool(getattr(b2_cosine_cfg, "enabled", False)):
        final_b2 = float(getattr(b2_cosine_cfg, "final_b2", c.opt.b2))
        b2_schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=c.opt.b2,
            peak_value=c.opt.b2,
            warmup_steps=0,
            decay_steps=num_opt_steps,
            end_value=final_b2,
        )
        b2_hparam = b2_schedule
        if jax.process_index() == 0:
            print(
                "b2 cosine annealing enabled: "
                f"start_b2={c.opt.b2}, final_b2={final_b2}, warmup_steps=0"
            )

    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    adamw_tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        b2_hparam,
        eps=c.opt.eps,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )

    pre_transforms, post_transforms, output_embedding_mask, lm_head_transform_logs = (
        utils.build_lm_head_update_transforms(model, c.opt)
    )
    if jax.process_index() == 0:
        for log_message in lm_head_transform_logs:
            print(log_message)

    lm_head_optimizer_cfg = getattr(c.opt, "lm_head_optimizer", None)
    lm_head_optimizer_type = utils.get_lm_head_optimizer_type(c.opt)
    resolved_lm_head_target_rms = None
    if lm_head_optimizer_type == "adamw":
        if lm_head_metric_lr_schedule is None:
            base_optimizer_tx = adamw_tx
        else:
            if output_embedding_mask is None:
                output_embedding_mask = utils.build_output_embedding_mask(model)
            non_output_embedding_mask = jax.tree_util.tree_map(
                lambda is_output_embedding: not is_output_embedding,
                output_embedding_mask,
            )
            if wd_mask is None:
                rest_wd_mask = non_output_embedding_mask
                lm_head_wd_mask = output_embedding_mask
            else:
                rest_wd_mask = jax.tree_util.tree_map(
                    lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                    wd_mask,
                    non_output_embedding_mask,
                )
                lm_head_wd_mask = jax.tree_util.tree_map(
                    lambda use_wd, is_output_embedding: bool(use_wd and is_output_embedding),
                    wd_mask,
                    output_embedding_mask,
                )

            lm_head_adamw_tx = optax.inject_hyperparams(optax.adamw)(
                lm_head_tx_lr_schedule,
                c.opt.b1,
                b2_hparam,
                eps=c.opt.eps,
                weight_decay=c.opt.weight_decay,
                mask=lm_head_wd_mask,
            )
            rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
                lr_schedule,
                c.opt.b1,
                b2_hparam,
                eps=c.opt.eps,
                weight_decay=c.opt.weight_decay,
                mask=rest_wd_mask,
            )
            base_optimizer_tx = optax.chain(
                optax.masked(rest_adamw_tx, non_output_embedding_mask),
                optax.masked(lm_head_adamw_tx, output_embedding_mask),
            )
            if jax.process_index() == 0:
                print(
                    "split lm-head optimizer enabled: "
                    "default=adamw, lm_head=adamw, "
                    f"lm_head_peak_lr={lm_head_peak_lr}; "
                    "all other hparams shared"
                )
    elif lm_head_optimizer_type == "sgd_momentum":
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        non_output_embedding_mask = jax.tree_util.tree_map(
            lambda is_output_embedding: not is_output_embedding,
            output_embedding_mask,
        )
        if wd_mask is None:
            rest_wd_mask = non_output_embedding_mask
        else:
            rest_wd_mask = jax.tree_util.tree_map(
                lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                wd_mask,
                non_output_embedding_mask,
            )

        lm_head_sgd_tx = optax.inject_hyperparams(optax.sgd)(
            learning_rate=lm_head_tx_lr_schedule,
            momentum=float(getattr(lm_head_optimizer_cfg, "momentum", 0.9)),
            nesterov=bool(getattr(lm_head_optimizer_cfg, "nesterov", False)),
        )
        rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=c.opt.eps,
            weight_decay=c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_sgd_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split lm-head optimizer enabled: "
                "default=adamw, lm_head=sgd_momentum, "
                f"lm_head_peak_lr={lm_head_peak_lr}, "
                f"momentum={float(getattr(lm_head_optimizer_cfg, 'momentum', 0.9))}, "
                f"nesterov={bool(getattr(lm_head_optimizer_cfg, 'nesterov', False))}"
            )
    elif lm_head_optimizer_type == "adamw_b1":
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        non_output_embedding_mask = jax.tree_util.tree_map(
            lambda is_output_embedding: not is_output_embedding,
            output_embedding_mask,
        )
        if wd_mask is None:
            rest_wd_mask = non_output_embedding_mask
        else:
            rest_wd_mask = jax.tree_util.tree_map(
                lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                wd_mask,
                non_output_embedding_mask,
            )

        lm_head_b1 = float(getattr(lm_head_optimizer_cfg, "b1", c.opt.b1))
        lm_head_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lm_head_tx_lr_schedule,
            lm_head_b1,
            b2_hparam,
            eps=c.opt.eps,
            weight_decay=c.opt.weight_decay,
        )
        rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=c.opt.eps,
            weight_decay=c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_adamw_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split lm-head optimizer enabled: "
                f"default=adamw(b1={c.opt.b1}), "
                f"lm_head=adamw(b1={lm_head_b1}, peak_lr={lm_head_peak_lr}); "
                "all other hparams (b2, eps, weight_decay) shared"
            )
    elif lm_head_optimizer_type in {"row_oblique", "column_oblique"}:
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        lm_head_target_rms_mask = utils.build_lm_head_oblique_target_rms_mask(model)
        non_lm_head_mask = jax.tree_util.tree_map(
            lambda is_output_embedding, is_target_rms: not (is_output_embedding or is_target_rms),
            output_embedding_mask,
            lm_head_target_rms_mask,
        )
        if wd_mask is None:
            rest_wd_mask = non_lm_head_mask
        else:
            rest_wd_mask = jax.tree_util.tree_map(
                lambda use_wd, is_non_lm_head: bool(use_wd and is_non_lm_head),
                wd_mask,
                non_lm_head_mask,
            )

        lm_head_momentum = float(getattr(lm_head_optimizer_cfg, "momentum", 0.9))
        if utils.lm_head_uses_learned_target_rms(c.opt):
            resolved_lm_head_target_rms = utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
        elif utils.lm_head_target_rms_from_random_init(c.opt):
            resolved_lm_head_target_rms = (
                float(model_lib.get_average_output_embedding_row_rms(model))
                if lm_head_optimizer_type == "row_oblique"
                else float(model_lib.get_average_output_embedding_col_rms(model))
            )
        else:
            resolved_lm_head_target_rms = utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
        lm_head_direction_target_rms = utils.get_lm_head_oblique_direction_target_rms(c.opt)
        lm_head_oblique_eps = float(getattr(lm_head_optimizer_cfg, "eps", c.opt.eps))
        lm_head_oblique_tx = optax.chain(
            utils.scale_by_ema_momentum(lm_head_momentum),
            (
                utils.row_oblique_steepest_descent
                if lm_head_optimizer_type == "row_oblique"
                else utils.column_oblique_steepest_descent
            )(
                learning_rate=lm_head_tx_lr_schedule,
                target_rms=resolved_lm_head_target_rms,
                direction_target_rms=lm_head_direction_target_rms,
                eps=lm_head_oblique_eps,
            )
        )
        rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=c.opt.eps,
            weight_decay=c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        lm_head_target_rms_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lm_head_tx_lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=c.opt.eps,
            weight_decay=0.0,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_lm_head_mask),
            optax.masked(lm_head_target_rms_adamw_tx, lm_head_target_rms_mask),
            optax.masked(lm_head_oblique_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split lm-head optimizer enabled: "
                f"default=adamw, lm_head={lm_head_optimizer_type}, "
                f"lm_head_peak_lr={lm_head_peak_lr}, momentum={lm_head_momentum}, "
                f"target_rms={resolved_lm_head_target_rms}, "
                f"direction_target_rms={lm_head_direction_target_rms}, "
                f"learn_target_rms={utils.lm_head_uses_learned_target_rms(c.opt)}, "
                f"target_rms_log_peak_lr={lm_head_peak_lr}, "
                f"eps={lm_head_oblique_eps}, weight_decay=off_for_lm_head"
            )
    else:
        raise ValueError(
            "Expected `opt.lm_head_optimizer.type` to be one of "
            "{'adamw', 'sgd_momentum', 'adamw_b1', 'row_oblique', 'column_oblique'}, "
            f"got {lm_head_optimizer_type!r}."
        )

    base_optimizer_tx, rmsnorm_scale_optimizer_log = utils.maybe_partition_rmsnorm_scale_sgd(
        model, c.opt, base_optimizer_tx, lr_schedule
    )
    if rmsnorm_scale_optimizer_log is not None and jax.process_index() == 0:
        print(rmsnorm_scale_optimizer_log)

    tx_chain = [*pre_transforms, base_optimizer_tx, *post_transforms]
    tx = tx_chain[0] if len(tx_chain) == 1 else optax.chain(*tx_chain)

    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(clip_by_global_norm), tx)

    return tx, lr_schedule, lm_head_metric_lr_schedule, b2_schedule


def _restore_start_step(step_to_load: int, training_metadata: dict) -> int:
    meta_step = training_metadata.get("step")
    meta_next_step = training_metadata.get("next_step")
    if meta_next_step is not None:
        return meta_next_step
    if meta_step is not None:
        if meta_step == step_to_load + 1:
            return meta_step
        if meta_step == step_to_load:
            return meta_step + 1
        print(
            "Warning: checkpoint metadata step does not match checkpoint id "
            f"(metadata step={meta_step}, checkpoint id={step_to_load}). "
            "Falling back to resume from checkpoint id + 1."
        )
        return step_to_load + 1
    print(
        "Warning: checkpoint metadata missing step/next_step; "
        "falling back to resume from checkpoint id + 1."
    )
    return step_to_load + 1


def _run_current_train_step(c, opt_state, opt_graphdef, model_graphdef, batch):
    collect_qkv_stats = False
    mucentering = bool(getattr(c.opt, "mucentering", False))
    if c.opt.use_z_loss:
        if mucentering:
            return train_step_z_loss_centered(
                opt_state,
                opt_graphdef,
                model_graphdef,
                batch,
                collect_qkv_stats,
            )
        return train_step_z_loss(
            opt_state,
            opt_graphdef,
            model_graphdef,
            batch,
            collect_qkv_stats,
        )
    if mucentering:
        return train_step_centered(
            opt_state,
            opt_graphdef,
            model_graphdef,
            batch,
            collect_qkv_stats,
        )
    return train_step(
        opt_state,
        opt_graphdef,
        model_graphdef,
        batch,
        collect_qkv_stats,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(c: DictConfig):
    OmegaConf.resolve(c)
    print(OmegaConf.to_yaml(c))

    jax.distributed.initialize()

    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ("data", "model"))
    jax.set_mesh(mesh)
    print("sharding mesh:", ", ".join(f"{k}={v}" for k, v in mesh.shape.items()))

    print("initializing base model...")
    utils.sync_lm_head_oblique_model_config(c)
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    model = model_lib.create_sharded_model(c.model, key_model)
    utils.validate_row_oblique_lm_head_options(c.opt)
    lm_head_optimizer_type = utils.get_lm_head_optimizer_type(c.opt)
    resolved_lm_head_target_rms = None
    if lm_head_optimizer_type in {"row_oblique", "column_oblique"}:
        learn_target_rms = utils.lm_head_uses_learned_target_rms(c.opt)
        target_rms_from_random_init = utils.lm_head_target_rms_from_random_init(c.opt)
        if learn_target_rms:
            resolved_lm_head_target_rms = (
                utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
            )
        elif target_rms_from_random_init:
            resolved_lm_head_target_rms = (
                float(model_lib.get_average_output_embedding_row_rms(model))
                if lm_head_optimizer_type == "row_oblique"
                else float(model_lib.get_average_output_embedding_col_rms(model))
            )
        else:
            resolved_lm_head_target_rms = (
                utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
            )
        initial_target_rms = (
            float(jnp.exp(jnp.asarray(model.lm_head_oblique_target_rms_log.value, dtype=jnp.float32)))
            if learn_target_rms and hasattr(model, "lm_head_oblique_target_rms_log")
            else resolved_lm_head_target_rms
        )
        initial_target_rms_source = (
            "random_init_row_rms"
            if utils.lm_head_initial_target_rms_from_random_init(c.opt) and lm_head_optimizer_type == "row_oblique"
            else "random_init_col_rms"
            if utils.lm_head_initial_target_rms_from_random_init(c.opt) and lm_head_optimizer_type == "column_oblique"
            else "fixed_target_rms_from_random_init"
            if target_rms_from_random_init
            else "config"
        )
        if lm_head_optimizer_type == "row_oblique":
            model_lib.row_normalize_output_embeddings(
                model,
                target_rms=resolved_lm_head_target_rms,
                eps=float(getattr(getattr(c.opt, "lm_head_optimizer", None), "eps", c.opt.eps)),
            )
            actual_row_l2 = float(
                jnp.sqrt(jnp.asarray(c.model.D, dtype=jnp.float32)) * jnp.asarray(initial_target_rms, dtype=jnp.float32)
            )
            init_log_message = (
                "row-oblique lm-head initialization enabled: "
                f"target_rms={resolved_lm_head_target_rms}, "
                f"learn_target_rms={learn_target_rms}, initial_target_rms={initial_target_rms}, "
                f"initial_target_rms_source={initial_target_rms_source}, "
                f"initial_actual_row_l2={actual_row_l2:.6g}"
            )
        else:
            model_lib.column_normalize_output_embeddings(
                model,
                target_rms=resolved_lm_head_target_rms,
                eps=float(getattr(getattr(c.opt, "lm_head_optimizer", None), "eps", c.opt.eps)),
            )
            actual_col_l2 = float(
                jnp.sqrt(jnp.asarray(c.model.V, dtype=jnp.float32)) * jnp.asarray(initial_target_rms, dtype=jnp.float32)
            )
            init_log_message = (
                "column-oblique lm-head initialization enabled: "
                f"target_rms={resolved_lm_head_target_rms}, "
                f"learn_target_rms={learn_target_rms}, initial_target_rms={initial_target_rms}, "
                f"initial_target_rms_source={initial_target_rms_source}, "
                f"initial_actual_col_l2={actual_col_l2:.6g}"
            )
        if jax.process_index() == 0:
            print(init_log_message)
    model_graphdef = nnx.graphdef(model)

    n_params = {
        "n_param_nonembed": 12 * c.model.L * c.model.D**2,
        "n_param_embed": c.model.D * c.model.V,
        "n_param_actual": utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f"{k}={v:_}")

    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (
            n_params["n_param_nonembed"] + n_params["n_param_embed"]
        )
    ds_train, ds_valid = data.load_ds(
        key_dataset,
        mesh,
        c.ds_path,
        c.model.T,
        c.opt.batch_size,
        c.num_tokens_valid,
        c.num_tokens_train,
    )
    if c.num_tokens_train is None:
        c.num_tokens_train = ds_train.size

    num_opt_steps = len(ds_train)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    tx, lr_schedule, lm_head_lr_schedule, b2_schedule = _build_optimizer(c, model, num_opt_steps)
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
    del optimizer
    del model

    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    run_name, load_ckpt_dir, requested_step_to_load, load_from_gcs = _resolve_checkpoint_source(c)
    explicit_restore_request = _checkpoint_restore_explicitly_requested(c)
    step_prefix = getattr(c.checkpoint, "step_prefix", None)
    if load_from_gcs:
        restore_options = ocp.CheckpointManagerOptions(
            create=False,
            step_name_format=_StandardNameFormatHNS(step_prefix=step_prefix),
        )
    else:
        restore_options = ocp.CheckpointManagerOptions(create=False)
    restore_mngr = ocp.CheckpointManager(load_ckpt_dir, options=restore_options)

    step_to_load = (
        requested_step_to_load
        if requested_step_to_load is not None
        else restore_mngr.latest_step()
    )
    if step_to_load is None:
        restore_mngr.close()
        if explicit_restore_request:
            raise ValueError(f"No checkpoint found in {load_ckpt_dir} to load base model from.")
        start_step = 0
        print(f"No checkpoint found in {load_ckpt_dir}. Starting from scratch.")
    else:
        print(f"Restoring base model from step {step_to_load} in {load_ckpt_dir}...")
        restored_data = restore_mngr.restore(
            step_to_load,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_opt_state),
                training_metadata=ocp.args.JsonRestore(),
            ),
        )
        opt_state = restored_data["state"]
        start_step = _restore_start_step(step_to_load, restored_data.get("training_metadata", {}))
        del restored_data
        restore_mngr.close()
        print(f"Successfully restored checkpoint. Resuming from step {start_step}.")
    del abstract_opt_state
    gc.collect()

    save_ckpt_mngr = None
    if c.checkpoint.turn_on:
        if not c.checkpoint.workdir:
            raise ValueError("checkpoint.workdir must be set when checkpoint.turn_on is True.")
        reverse_run_name = f"{run_name}_reverse_bad_checkpoints_only_{c.opt.peak_lr}"
        save_ckpt_dir = os.path.join(c.checkpoint.workdir, reverse_run_name)
        save_options = ocp.CheckpointManagerOptions(
            create=True,
            preservation_policy=preservation_policy.LatestN(c.checkpoint.max_to_keep),
        )
        save_ckpt_mngr = ocp.CheckpointManager(save_ckpt_dir, options=save_options)
        print(f"New checkpoints will be saved to: {save_ckpt_dir}")

    if jax.process_index() == 0:
        wandb.init(
            project=c.wandb_project,
            config=utils.flatten_dict(c),
            mode=c.wandb_mode,
            name=f"{run_name}_reverse_checkpoint_{c.opt.peak_lr}",
        )
        wandb.summary.update(n_params)

    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step

    eval_every_steps = int(getattr(c, "eval_every_steps", 100))
    skip_bad_batch = bool(getattr(c, "skip_bad_batch", False))
    revert_idx = int(getattr(c, "revert_idx", -1))
    skip_bad_batch_metric = str(getattr(c, "skip_bad_batch_metric", "relative_jump")).lower()
    skip_bad_batch_relative_threshold = float(getattr(c, "skip_bad_batch_relative_threshold", 0.5))
    skip_bad_batch_window = int(getattr(c, "skip_bad_batch_window", 1000))
    skip_bad_batch_z_threshold = float(getattr(c, "skip_bad_batch_z_threshold", 7.0))
    rollback_buffer_len = max(revert_idx + 1, 1) if revert_idx >= 0 else max(abs(revert_idx), 1)
    rolling_skip_history = deque(maxlen=skip_bad_batch_window)
    if skip_bad_batch_metric not in {"relative_jump", "rolling_zscore"}:
        raise ValueError(
            "skip_bad_batch_metric must be one of {'relative_jump', 'rolling_zscore'}, "
            f"got {skip_bad_batch_metric!r}."
        )

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0:
        pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)

    last_opt_states = deque(maxlen=rollback_buffer_len)
    for step in pbar:
        batch = ds_train[step]
        if skip_bad_batch:
            snapshot_state = jax.tree_util.tree_map(np.asarray, jax.device_get(opt_state))
            last_opt_states.append(snapshot_state)

        opt_state, objective_loss, (train_raw_loss, _), _grads, _update_metrics = _run_current_train_step(
            c,
            opt_state,
            opt_graphdef,
            model_graphdef,
            batch,
        )

        train_loss_before_update = train_raw_loss.mean()
        train_loss_after_update, output_logit_mean = get_loss_and_output_logit_mean(
            opt_state.model,
            model_graphdef,
            batch,
        )

        metrics = {
            "train_loss": train_loss_before_update,
            "train_objective_loss": objective_loss,
            "train_loss_after_update": train_loss_after_update,
            "train_med_loss": jnp.median(train_raw_loss),
            "train_lower_90th_mean_loss": utils.compute_lower_90th_percentile_mean(train_raw_loss),
            "train_tokens_seen": (step + 1) * tokens_per_opt_step,
            "train_output_logit_mean": output_logit_mean,
            "lr": lr_schedule(step),
        }
        if lm_head_lr_schedule is not None:
            metrics["lm_head_lr"] = lm_head_lr_schedule(step)
        if b2_schedule is not None:
            metrics["b2"] = b2_schedule(step)

        relative_loss_jump = (
            (train_loss_after_update - train_loss_before_update)
            / (train_loss_before_update + 1e-6)
        )
        rollback_ready = len(last_opt_states) >= rollback_buffer_len
        should_skip = False
        metrics["skip_bad_batch_metric"] = 0.0 if skip_bad_batch_metric == "relative_jump" else 1.0
        metrics["skip_bad_batch_relative_jump"] = relative_loss_jump

        if skip_bad_batch_metric == "relative_jump":
            metrics["skip_detector_ready"] = 1.0
            metrics["skip_detector_value"] = relative_loss_jump
            metrics["skip_detector_mean"] = 0.0
            metrics["skip_detector_std"] = 0.0
            metrics["skip_detector_zscore"] = 0.0
            metrics["skip_detector_threshold"] = skip_bad_batch_relative_threshold
            should_skip = bool(relative_loss_jump > skip_bad_batch_relative_threshold)
        else:
            should_skip, rolling_metrics = _compute_rolling_zscore_skip(
                rolling_skip_history,
                float(train_loss_after_update),
                skip_bad_batch_z_threshold,
            )
            metrics.update(rolling_metrics)

        metrics["skip_bad_batch_triggered"] = float(skip_bad_batch and should_skip)
        metrics["skip_bad_batch_rollback_ready"] = float(rollback_ready)

        if jax.process_index() == 0:
            wandb.log(metrics, step)
            pbar.set_postfix_str(f"loss={float(train_loss_before_update):.2f}")

        if should_skip and rollback_ready and skip_bad_batch:
            rollback_state = list(last_opt_states)[revert_idx]
            opt_state = jax.tree_util.tree_map(
                lambda arr, ref: (
                    jax.device_put(arr, ref.sharding)
                    if hasattr(ref, "sharding")
                    else jax.device_put(arr)
                ),
                rollback_state,
                opt_state,
            )
            if jax.process_index() == 0:
                if skip_bad_batch_metric == "relative_jump":
                    pct_jump = 100.0 * float(relative_loss_jump)
                    print(
                        f"Reversed optimizer update at step {step} due to train loss jump of "
                        f"{pct_jump:.1f}%"
                    )
                else:
                    print(
                        f"Reversed optimizer update at step {step} due to rolling loss z-score "
                        f"{float(metrics['skip_detector_zscore']):.2f} "
                        f"(threshold={float(metrics['skip_detector_threshold']):.2f})"
                    )
        elif skip_bad_batch_metric == "rolling_zscore":
            rolling_skip_history.append(float(train_loss_after_update))

        if step % eval_every_steps == 0:
            (
                eval_loss,
                eval_raw_loss,
                _eval_logits,
                mean_eval_output_logit,
                _mean_eval_output_logit_std,
                _mean_eval_output_logit_entropy,
            ) = eval_step(
                c,
                opt_state.model,
                model_graphdef,
                ds_valid,
                collect_qkv_stats=False,
            )
            flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
            eval_metrics = {
                "eval_loss": eval_loss,
                "eval_output_logit_mean": mean_eval_output_logit,
                "eval_med_loss": jnp.median(flattened_eval_raw_loss),
                "eval_lower_90th_mean_loss": utils.compute_lower_90th_percentile_mean(
                    flattened_eval_raw_loss
                ),
                "train_tokens_seen": (step + 1) * tokens_per_opt_step,
            }
            if jax.process_index() == 0:
                wandb.log(eval_metrics, step)

        if save_ckpt_mngr is not None and _should_checkpoint(c, step):
            _save_checkpoint(save_ckpt_mngr, step, opt_state)

    if jax.process_index() == 0:
        wandb.finish()
    if save_ckpt_mngr is not None:
        save_ckpt_mngr.close()


if __name__ == "__main__":
    main()
