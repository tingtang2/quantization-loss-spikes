"""Optimizer schedules, parameter partitions, and optimizer state creation."""

from dataclasses import dataclass
import math

import jax
import optax
from flax import nnx

import utils


@dataclass(frozen=True)
class OptimizerSetup:
    opt_graphdef: object
    opt_state: object
    lr_schedule: object
    lm_head_tx_lr_schedule: object
    lm_head_metric_lr_schedule: object | None
    b2_schedule: object | None
    tokens_per_opt_step: int


def build_optimizer(c, model, num_opt_steps: int, resolved_lm_head_target_rms=None):
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    lr_schedule = utils.build_learning_rate_schedule(c.opt, c.opt.peak_lr, warmup_steps, num_opt_steps)
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
    b2_cosine_enabled = bool(getattr(b2_cosine_cfg, "enabled", False))
    if b2_cosine_enabled:
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
    adamc_cfg = getattr(c.opt, "adamc", None)
    adamc_enabled = bool(getattr(adamc_cfg, "enabled", False))
    adamc_normalized_linear_mask = None
    if adamc_enabled:
        if float(c.opt.peak_lr) <= 0.0:
            raise ValueError("opt.adamc.enabled=true requires opt.peak_lr to be positive.")
        adamc_normalized_linear_mask = utils.build_adamc_normalized_linear_mask(model)
        if jax.process_index() == 0:
            print(
                "AdamC enabled: transformer linear weight decay is scaled by lr / peak_lr; "
                "LM head and other parameters retain regular AdamW decay"
            )

    def build_adam_tx(learning_rate, peak_learning_rate, b1, weight_decay, mask=None):
        if not adamc_enabled:
            return optax.inject_hyperparams(optax.adamw)(
                learning_rate,
                b1,
                b2_hparam,
                eps=c.opt.eps,
                weight_decay=weight_decay,
                mask=mask,
            )
        return optax.inject_hyperparams(utils.adamc)(
            learning_rate=learning_rate,
            peak_learning_rate=peak_learning_rate,
            b1=b1,
            b2=b2_hparam,
            eps=c.opt.eps,
            weight_decay=weight_decay,
            mask=mask,
            normalized_linear_mask=adamc_normalized_linear_mask,
        )

    base_adam_name = "adamc" if adamc_enabled else "adamw"
    adamw_tx = build_adam_tx(
        lr_schedule,
        c.opt.peak_lr,
        c.opt.b1,
        c.opt.weight_decay,
        mask=wd_mask,
    )

    def masked_weight_decay_mask(target_mask):
        if wd_mask is None:
            return target_mask
        return utils.and_masks(wd_mask, target_mask)

    def build_sgd_momentum_tx(optimizer_cfg, learning_rate):
        return optax.inject_hyperparams(optax.sgd)(
            learning_rate=learning_rate,
            momentum=float(getattr(optimizer_cfg, "momentum", 0.9)),
            nesterov=bool(getattr(optimizer_cfg, "nesterov", False)),
        )

    pre_transforms, post_transforms, output_embedding_mask, lm_head_transform_logs = (
        utils.build_lm_head_update_transforms(model, c.opt)
    )
    if jax.process_index() == 0:
        for log_message in lm_head_transform_logs:
            print(log_message)

    def normalize_embedding_optimizer_type(optimizer_cfg, field_name):
        optimizer_type = str(getattr(optimizer_cfg, "type", "adamw")).lower()
        if optimizer_type not in {"adamw", "sgd_momentum"}:
            raise ValueError(
                f"Expected `{field_name}.type` to be one of "
                "{'adamw', 'sgd_momentum'}, "
                f"got {optimizer_type!r}."
            )
        return optimizer_type

    def build_embedding_optimizer_tx(
        optimizer_cfg, target_mask, field_name, learning_rate, peak_learning_rate
    ):
        optimizer_type = normalize_embedding_optimizer_type(optimizer_cfg, field_name)
        if optimizer_type == "adamw":
            return build_adam_tx(
                learning_rate,
                peak_learning_rate,
                c.opt.b1,
                c.opt.weight_decay,
                mask=masked_weight_decay_mask(target_mask),
            ), optimizer_type
        return build_sgd_momentum_tx(optimizer_cfg, learning_rate), optimizer_type

    default_optimizer_cfg = getattr(c.opt, "default_optimizer", None)
    default_optimizer_type = str(getattr(default_optimizer_cfg, "type", "adamw")).lower()
    input_embedding_optimizer_cfg = getattr(c.opt, "input_embedding_optimizer", None)
    lm_head_optimizer_cfg = getattr(c.opt, "lm_head_optimizer", None)
    lm_head_optimizer_type = utils.get_lm_head_optimizer_type(c.opt)
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

            lm_head_adamw_tx = build_adam_tx(
                lm_head_tx_lr_schedule,
                lm_head_peak_lr,
                c.opt.b1,
                c.opt.weight_decay,
                mask=lm_head_wd_mask,
            )
            rest_adamw_tx = build_adam_tx(
                lr_schedule,
                c.opt.peak_lr,
                c.opt.b1,
                c.opt.weight_decay,
                mask=rest_wd_mask,
            )
            base_optimizer_tx = optax.chain(
                optax.masked(rest_adamw_tx, non_output_embedding_mask),
                optax.masked(lm_head_adamw_tx, output_embedding_mask),
            )
            if jax.process_index() == 0:
                print(
                    "split lm-head optimizer enabled: "
                    f"default={base_adam_name}, lm_head=adamw, "
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
        rest_adamw_tx = build_adam_tx(
            lr_schedule,
            c.opt.peak_lr,
            c.opt.b1,
            c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_sgd_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split lm-head optimizer enabled: "
                f"default={base_adam_name}, lm_head=sgd_momentum, "
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

        lm_head_adamw_tx = build_adam_tx(
            lm_head_tx_lr_schedule,
            lm_head_peak_lr,
            lm_head_b1,
            c.opt.weight_decay,
        )
        rest_adamw_tx = build_adam_tx(
            lr_schedule,
            c.opt.peak_lr,
            c.opt.b1,
            c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_adamw_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split lm-head optimizer enabled: "
                f"default={base_adam_name}(b1={c.opt.b1}), "
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
        lm_head_target_rms = (
            resolved_lm_head_target_rms
            if resolved_lm_head_target_rms is not None
            else utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
        )
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
                target_rms=lm_head_target_rms,
                direction_target_rms=lm_head_direction_target_rms,
                eps=lm_head_oblique_eps,
            )
        )
        rest_adamw_tx = build_adam_tx(
            lr_schedule,
            c.opt.peak_lr,
            c.opt.b1,
            c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        lm_head_target_rms_adamw_tx = build_adam_tx(
            lm_head_tx_lr_schedule,
            lm_head_peak_lr,
            c.opt.b1,
            0.0,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_lm_head_mask),
            optax.masked(lm_head_target_rms_adamw_tx, lm_head_target_rms_mask),
            optax.masked(lm_head_oblique_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split lm-head optimizer enabled: "
                f"default={base_adam_name}, lm_head={lm_head_optimizer_type}, "
                f"lm_head_peak_lr={lm_head_peak_lr}, momentum={lm_head_momentum}, "
                f"target_rms={lm_head_target_rms}, "
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

    if default_optimizer_type not in {"adamw", "muon"}:
        raise ValueError(
            "Expected `opt.default_optimizer.type` to be one of "
            "{'adamw', 'muon'}, "
            f"got {default_optimizer_type!r}."
        )

    if default_optimizer_type == "muon":
        if not hasattr(optax, "contrib") or not hasattr(optax.contrib, "muon"):
            raise ValueError(
                "`opt.default_optimizer.type=muon` requires an Optax version with "
                "`optax.contrib.muon`."
            )

        input_embedding_optimizer_type = normalize_embedding_optimizer_type(
            input_embedding_optimizer_cfg,
            "opt.input_embedding_optimizer",
        )
        muon_lm_head_optimizer_type = normalize_embedding_optimizer_type(
            lm_head_optimizer_cfg,
            "opt.lm_head_optimizer",
        )
        input_embedding_mask = utils.build_input_embedding_mask(model)
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        non_embedding_mask = utils.build_non_embedding_mask(model)

        muon_cfg = getattr(default_optimizer_cfg, "muon", None)
        muon_peak_lr = float(getattr(muon_cfg, "peak_lr", c.opt.peak_lr))
        muon_lr_schedule = lr_schedule
        if not math.isclose(muon_peak_lr, float(c.opt.peak_lr)):
            muon_lr_schedule = utils.build_learning_rate_schedule(
                c.opt, muon_peak_lr, warmup_steps, num_opt_steps
            )
        muon_tx = optax.contrib.muon(
            learning_rate=muon_lr_schedule,
            beta=float(getattr(muon_cfg, "beta", 0.95)),
            ns_steps=int(getattr(muon_cfg, "ns_steps", 5)),
            eps=float(getattr(muon_cfg, "eps", c.opt.eps)),
            weight_decay=float(getattr(muon_cfg, "weight_decay", c.opt.weight_decay)),
            weight_decay_mask=masked_weight_decay_mask(non_embedding_mask),
            mu_dtype=getattr(muon_cfg, "mu_dtype", None),
            nesterov=bool(getattr(muon_cfg, "nesterov", True)),
            adaptive=bool(getattr(muon_cfg, "adaptive", False)),
            adam_b1=float(getattr(muon_cfg, "adam_b1", c.opt.b1)),
            adam_b2=float(getattr(muon_cfg, "adam_b2", c.opt.b2)),
            adam_weight_decay=float(
                getattr(muon_cfg, "adam_weight_decay", c.opt.weight_decay)
            ),
            muon_weight_dimension_numbers=utils.build_muon_dimension_numbers(model),
            consistent_rms=getattr(muon_cfg, "consistent_rms", None),
        )
        input_embedding_tx, _ = build_embedding_optimizer_tx(
            input_embedding_optimizer_cfg,
            input_embedding_mask,
            "opt.input_embedding_optimizer",
            lr_schedule,
            c.opt.peak_lr,
        )
        lm_head_tx, _ = build_embedding_optimizer_tx(
            lm_head_optimizer_cfg,
            output_embedding_mask,
            "opt.lm_head_optimizer",
            lm_head_tx_lr_schedule,
            lm_head_peak_lr,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(muon_tx, non_embedding_mask),
            optax.masked(input_embedding_tx, input_embedding_mask),
            optax.masked(lm_head_tx, output_embedding_mask),
        )
        if jax.process_index() == 0:
            print(
                "split Muon optimizer enabled: "
                f"default=muon_non_embeddings, input_embedding={input_embedding_optimizer_type}, "
                f"lm_head={muon_lm_head_optimizer_type}, "
                f"muon_peak_lr={muon_peak_lr}, "
                f"muon_beta={float(getattr(muon_cfg, 'beta', 0.95))}, "
                f"muon_nesterov={bool(getattr(muon_cfg, 'nesterov', True))}"
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
        tx = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
    del optimizer
    return OptimizerSetup(
        opt_graphdef=opt_graphdef,
        opt_state=opt_state,
        lr_schedule=lr_schedule,
        lm_head_tx_lr_schedule=lm_head_tx_lr_schedule,
        lm_head_metric_lr_schedule=lm_head_metric_lr_schedule,
        b2_schedule=b2_schedule,
        tokens_per_opt_step=tokens_per_opt_step,
    )

