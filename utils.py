import jax
import jax.numpy as jnp
import optax
from flax import nnx
from collections.abc import Mapping
from functools import partial
import os
import numpy as np
from typing import NamedTuple
from omegaconf import DictConfig, open_dict


def flatten_dict(d, prefix=None, sep='.'):
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            nested_prefix = k if prefix is None else f'{prefix}{sep}{k}'
            out |= flatten_dict(v, nested_prefix, sep)
        return out
    else:
        return {prefix: d}


def get_num_model_params(model: nnx.Module):
    graphdef, params = nnx.split(model, nnx.Param)
    n_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), params, 0)
    return n_params


def _is_output_embedding_path(path) -> bool:
    key = jax.tree_util.keystr(path, simple=True, separator='/')
    return key == 'token_embed_out/embedding'


def _is_lm_head_oblique_target_rms_log_path(path) -> bool:
    key = jax.tree_util.keystr(path, simple=True, separator='/')
    return key == 'lm_head_oblique_target_rms_log'


def _is_input_embedding_path(path) -> bool:
    key = jax.tree_util.keystr(path, simple=True, separator='/')
    return key == 'token_embed_in/embedding'


def _is_embedding_path(path) -> bool:
    return _is_input_embedding_path(path) or _is_output_embedding_path(path)


def build_weight_decay_mask(model: nnx.Module, exclude_input_embedding: bool):
    _, params = nnx.split(model, nnx.Param)
    learned_target_rms_present = False

    def mask_leaf(path, _):
        nonlocal learned_target_rms_present
        key = jax.tree_util.keystr(path, simple=True, separator='/')
        if key == 'lm_head_oblique_target_rms_log':
            learned_target_rms_present = True
            return False
        if exclude_input_embedding and 'token_embed_in' in key:
            return False
        return True

    if not exclude_input_embedding:
        for path, leaf in jax.tree_util.tree_leaves_with_path(
            params,
            is_leaf=lambda x: isinstance(x, nnx.Param),
        ):
            key = jax.tree_util.keystr(path, simple=True, separator='/')
            if key == 'lm_head_oblique_target_rms_log':
                learned_target_rms_present = True
                break
        if not learned_target_rms_present:
            return None

    return jax.tree_util.tree_map_with_path(
        mask_leaf,
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def _build_path_mask(model: nnx.Module, predicate):
    _, params = nnx.split(model, nnx.Param)

    return jax.tree_util.tree_map_with_path(
        lambda path, _: bool(predicate(path)),
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def _is_adamc_normalized_linear_path(key: str) -> bool:
    """The transformer-linear mask used for the paper's LLM experiments."""
    parts = key.split('/')
    if len(parts) != 5 or parts[0] != 'blocks' or not parts[1].isdigit():
        return False
    if parts[2] == 'attn':
        return parts[3] in {'qkv_proj', 'gate_proj', 'out_proj'} and parts[4] == 'kernel'
    if parts[2] == 'mlp':
        return parts[3] in {'up_proj', 'down_proj'} and parts[4] == 'kernel'
    return False


def build_adamc_normalized_linear_mask(model: nnx.Module):
    """Select every transformer linear weight, excluding the LM output head."""
    _, params = nnx.split(model, nnx.Param)
    return jax.tree_util.tree_map_with_path(
        lambda path, _: _is_adamc_normalized_linear_path(
            jax.tree_util.keystr(path, simple=True, separator='/')
        ),
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def adamc(
    learning_rate,
    peak_learning_rate,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    weight_decay=1e-4,
    mask=None,
    normalized_linear_mask=None,
):
    """AdamW with LR-corrected decay on normalized transformer linears.

    For selected parameters, the decay coefficient inside AdamW is
    ``weight_decay * learning_rate / peak_learning_rate``. Since Optax then
    scales the complete update by ``learning_rate``, the resulting decay step
    is ``learning_rate**2 / peak_learning_rate * weight_decay * param``.
    """
    if normalized_linear_mask is None:
        raise ValueError("adamc requires normalized_linear_mask.")

    if mask is None:
        corrected_decay_mask = normalized_linear_mask
        regular_decay_mask = jax.tree_util.tree_map(
            lambda corrected: not corrected,
            normalized_linear_mask,
        )
    else:
        corrected_decay_mask = jax.tree_util.tree_map(
            lambda use_decay, corrected: bool(use_decay and corrected),
            mask,
            normalized_linear_mask,
        )
        regular_decay_mask = jax.tree_util.tree_map(
            lambda use_decay, corrected: bool(use_decay and not corrected),
            mask,
            normalized_linear_mask,
        )

    corrected_weight_decay = weight_decay * learning_rate / peak_learning_rate
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.add_decayed_weights(weight_decay, regular_decay_mask),
        optax.add_decayed_weights(corrected_weight_decay, corrected_decay_mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def invert_mask(mask):
    return jax.tree_util.tree_map(lambda include: not include, mask)


def and_masks(*masks):
    if not masks:
        raise ValueError("and_masks requires at least one mask.")
    return jax.tree_util.tree_map(lambda *values: all(values), *masks)


def or_masks(*masks):
    if not masks:
        raise ValueError("or_masks requires at least one mask.")
    return jax.tree_util.tree_map(lambda *values: any(values), *masks)


def build_input_embedding_mask(model: nnx.Module):
    return _build_path_mask(model, _is_input_embedding_path)


def build_output_embedding_mask(model: nnx.Module):
    _, params = nnx.split(model, nnx.Param)

    def mask_leaf(path, leaf):
        if not _is_output_embedding_path(path):
            return False
        value = getattr(leaf, 'value', leaf)
        if value.ndim != 2:
            raise ValueError(
                "Expected `token_embed_out.embedding` to be a rank-2 matrix for row-wise centering, "
                f"got shape={value.shape}."
            )
        return True

    return jax.tree_util.tree_map_with_path(
        mask_leaf,
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def build_lm_head_oblique_target_rms_mask(model: nnx.Module):
    _, params = nnx.split(model, nnx.Param)

    return jax.tree_util.tree_map_with_path(
        lambda path, _: _is_lm_head_oblique_target_rms_log_path(path),
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def build_non_embedding_mask(model: nnx.Module):
    return _build_path_mask(model, lambda path: not _is_embedding_path(path))


def build_muon_dimension_numbers(model: nnx.Module):
    """Selects every non-embedding parameter for Muon, including projection tensors."""
    _, params = nnx.split(model, nnx.Param)

    def spec_leaf(path, leaf):
        if _is_embedding_path(path):
            return None

        key = jax.tree_util.keystr(path, simple=True, separator='/')
        value = getattr(leaf, 'value', leaf)
        ndim = value.ndim

        if 'qkv_proj' in key:
            return optax.contrib.MuonDimensionNumbers(
                reduction_axis=(2,),
                output_axis=(0, 1, 3),
            )
        if 'out_proj' in key:
            return optax.contrib.MuonDimensionNumbers(
                reduction_axis=(0, 1),
                output_axis=(2,),
            )
        if 'gate_proj' in key:
            return optax.contrib.MuonDimensionNumbers(
                reduction_axis=(1,),
                output_axis=(0, 2),
            )
        if ndim == 2:
            return optax.contrib.MuonDimensionNumbers()
        return None

    return jax.tree_util.tree_map_with_path(
        spec_leaf,
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def _is_rmsnorm_scale_path(key: str) -> bool:
    if key == 'out_ln/scale':
        return True
    parts = key.split('/')
    if len(parts) == 4 and parts[0] == 'blocks' and parts[1].isdigit():
        return parts[2] in {'ln1', 'ln2'} and parts[3] == 'scale'
    if len(parts) == 5 and parts[0] == 'blocks' and parts[1].isdigit() and parts[2] == 'attn':
        return parts[3] in {'query_norm', 'key_norm'} and parts[4] == 'scale'
    return False


def build_rmsnorm_scale_mask(model: nnx.Module, all_rmsnorm_scales: bool):
    """Selects either out_ln.scale or every RMSNorm scale parameter."""
    _, params = nnx.split(model, nnx.Param)
    found = False

    def mask_leaf(path, _):
        nonlocal found
        key = jax.tree_util.keystr(path, simple=True, separator='/')
        selected = _is_rmsnorm_scale_path(key) if all_rmsnorm_scales else key == 'out_ln/scale'
        found |= selected
        return selected

    mask = jax.tree_util.tree_map_with_path(
        mask_leaf,
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )
    if not found:
        config_name = 'all_rmsnorm_scales_sgd' if all_rmsnorm_scales else 'out_ln_scale_sgd'
        raise ValueError(
            f"opt.{config_name}.enabled=true requires model.rmsnorm_use_scale=true "
            "so that RMSNorm scale parameters exist."
        )
    return mask


def maybe_partition_rmsnorm_scale_sgd(model, opt_cfg, base_optimizer_tx, learning_rate):
    """Optionally routes selected RMSNorm scales to momentum SGD."""
    out_ln_cfg = getattr(opt_cfg, 'out_ln_scale_sgd', None)
    all_scales_cfg = getattr(opt_cfg, 'all_rmsnorm_scales_sgd', None)
    out_ln_enabled = bool(getattr(out_ln_cfg, 'enabled', False))
    all_scales_enabled = bool(getattr(all_scales_cfg, 'enabled', False))
    if out_ln_enabled and all_scales_enabled:
        raise ValueError(
            "opt.out_ln_scale_sgd.enabled and opt.all_rmsnorm_scales_sgd.enabled "
            "cannot both be true."
        )
    if not (out_ln_enabled or all_scales_enabled):
        return base_optimizer_tx, None

    all_rmsnorm_scales = all_scales_enabled
    sgd_cfg = all_scales_cfg if all_rmsnorm_scales else out_ln_cfg
    config_name = 'all_rmsnorm_scales_sgd' if all_rmsnorm_scales else 'out_ln_scale_sgd'
    momentum = float(getattr(sgd_cfg, 'momentum', 0.9))
    nesterov = bool(getattr(sgd_cfg, 'nesterov', False))
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"Expected opt.{config_name}.momentum in [0, 1), got {momentum}.")

    rmsnorm_scale_mask = build_rmsnorm_scale_mask(model, all_rmsnorm_scales)
    parameter_labels = jax.tree_util.tree_map(
        lambda selected: 'rmsnorm_scale_sgd' if selected else 'base',
        rmsnorm_scale_mask,
    )
    rmsnorm_scale_tx = optax.inject_hyperparams(optax.sgd)(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )
    partitioned_tx = optax.partition(
        {
            'base': base_optimizer_tx,
            'rmsnorm_scale_sgd': rmsnorm_scale_tx,
        },
        parameter_labels,
    )
    selected_parameters = 'all RMSNorm scales' if all_rmsnorm_scales else 'out_ln.scale'
    log_message = (
        "RMSNorm scale optimizer enabled: "
        f"{selected_parameters}=sgd(momentum={momentum}, nesterov={nesterov}), "
        "learning_rate=global_schedule, weight_decay=off"
    )
    return partitioned_tx, log_message


def normalize_lm_head_centering_mode(raw_value, field_name: str) -> str:
    if isinstance(raw_value, bool):
        if raw_value:
            raise ValueError(
                f"Expected `{field_name}` to be one of "
                "{'off', 'pre', 'post'} or legacy `false`, got `true`."
            )
        return "off"
    mode = str(raw_value).lower()
    if mode not in {"off", "pre", "post"}:
        raise ValueError(
            f"Expected `{field_name}` to be one of "
            "{'off', 'pre', 'post'} or legacy `false`, "
            f"got {raw_value!r}."
        )
    return mode


def get_lm_head_optimizer_type(opt_cfg) -> str:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return str(getattr(lm_head_optimizer_cfg, "type", "adamw")).lower()


def get_lm_head_peak_lr(opt_cfg) -> float:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return float(getattr(lm_head_optimizer_cfg, "peak_lr", opt_cfg.peak_lr))


def build_learning_rate_schedule(opt_cfg, peak_value, warmup_steps, decay_steps):
    """Builds the configured warmup + cosine learning-rate schedule.

    getattr defaults retain compatibility with configs saved before the
    ``opt.lr_schedule`` block was introduced.
    """
    schedule_cfg = getattr(opt_cfg, "lr_schedule", None)
    exponent = float(getattr(schedule_cfg, "exponent", 1.0))
    end_value = float(getattr(schedule_cfg, "end_value", 0.0))
    if exponent <= 0.0:
        raise ValueError(f"Expected opt.lr_schedule.exponent to be positive, got {exponent}.")
    if end_value < 0.0:
        raise ValueError(f"Expected opt.lr_schedule.end_value to be non-negative, got {end_value}.")
    return optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value,
        exponent=exponent,
    )


def lm_head_uses_learned_target_rms(opt_cfg) -> bool:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return bool(getattr(lm_head_optimizer_cfg, "learn_target_rms", False))


def lm_head_initial_target_rms_from_random_init(opt_cfg) -> bool:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return bool(getattr(lm_head_optimizer_cfg, "initial_target_rms_from_random_init", False))


def lm_head_target_rms_from_random_init(opt_cfg) -> bool:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return bool(getattr(lm_head_optimizer_cfg, "target_rms_from_random_init", False))


def get_lm_head_oblique_initial_target_rms(opt_cfg) -> float:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return float(getattr(lm_head_optimizer_cfg, "initial_target_rms", get_lm_head_oblique_target_rms(opt_cfg)))


def get_lm_head_oblique_optimizer_target_rms(opt_cfg) -> float:
    return 1.0 if lm_head_uses_learned_target_rms(opt_cfg) else get_lm_head_oblique_target_rms(opt_cfg)


def sync_lm_head_oblique_model_config(c):
    if not hasattr(c, "model") or not hasattr(c, "opt"):
        return

    lm_head_optimizer_type = get_lm_head_optimizer_type(c.opt)
    use_oblique_optimizer = lm_head_optimizer_type in {"row_oblique", "column_oblique"}
    learn_target_rms = use_oblique_optimizer and lm_head_uses_learned_target_rms(c.opt)
    initial_target_rms_from_random_init = (
        use_oblique_optimizer and learn_target_rms and lm_head_initial_target_rms_from_random_init(c.opt)
    )
    initial_target_rms = get_lm_head_oblique_initial_target_rms(c.opt)
    if (not initial_target_rms_from_random_init) and initial_target_rms <= 0.0:
        raise ValueError(f"Expected initial_target_rms to be positive, got {initial_target_rms}.")

    if isinstance(c.model, DictConfig):
        with open_dict(c.model):
            c.model.lm_head_oblique_learn_target_rms = learn_target_rms
            c.model.lm_head_oblique_initial_target_rms = initial_target_rms
            c.model.lm_head_oblique_initial_target_rms_from_random_init = initial_target_rms_from_random_init
            c.model.lm_head_oblique_optimizer_type = lm_head_optimizer_type
    else:
        c.model.lm_head_oblique_learn_target_rms = learn_target_rms
        c.model.lm_head_oblique_initial_target_rms = initial_target_rms
        c.model.lm_head_oblique_initial_target_rms_from_random_init = initial_target_rms_from_random_init
        c.model.lm_head_oblique_optimizer_type = lm_head_optimizer_type


def validate_row_oblique_lm_head_options(opt_cfg):
    lm_head_optimizer_type = get_lm_head_optimizer_type(opt_cfg)
    lm_head_peak_lr = get_lm_head_peak_lr(opt_cfg)
    learn_target_rms = lm_head_uses_learned_target_rms(opt_cfg)
    initial_target_rms_from_random_init = lm_head_initial_target_rms_from_random_init(opt_cfg)
    target_rms_from_random_init = lm_head_target_rms_from_random_init(opt_cfg)
    if lm_head_peak_lr <= 0.0:
        raise ValueError(
            f"Expected opt.lm_head_optimizer.peak_lr to be positive, got {lm_head_peak_lr}."
        )
    if lm_head_optimizer_type not in {"row_oblique", "column_oblique"}:
        if learn_target_rms:
            raise ValueError(
                "opt.lm_head_optimizer.learn_target_rms requires "
                "opt.lm_head_optimizer.type to be 'row_oblique' or 'column_oblique'."
            )
        if initial_target_rms_from_random_init:
            raise ValueError(
                "opt.lm_head_optimizer.initial_target_rms_from_random_init requires "
                "opt.lm_head_optimizer.type to be 'row_oblique' or 'column_oblique'."
            )
        if target_rms_from_random_init:
            raise ValueError(
                "opt.lm_head_optimizer.target_rms_from_random_init requires "
                "opt.lm_head_optimizer.type to be 'row_oblique' or 'column_oblique'."
            )
        return
    manifold_name = "Row-Oblique" if lm_head_optimizer_type == "row_oblique" else "Column-Oblique"
    if initial_target_rms_from_random_init and not learn_target_rms:
        raise ValueError(
            "opt.lm_head_optimizer.initial_target_rms_from_random_init requires "
            "opt.lm_head_optimizer.learn_target_rms=true."
        )
    if target_rms_from_random_init and learn_target_rms:
        raise ValueError(
            "opt.lm_head_optimizer.target_rms_from_random_init is only for the fixed target_rms path. "
            "Use opt.lm_head_optimizer.initial_target_rms_from_random_init when "
            "opt.lm_head_optimizer.learn_target_rms=true."
        )
    initial_target_rms = get_lm_head_oblique_initial_target_rms(opt_cfg)
    if (not initial_target_rms_from_random_init) and initial_target_rms <= 0.0:
        raise ValueError(
            f"Expected opt.lm_head_optimizer.initial_target_rms to be positive, got {initial_target_rms}."
        )
    direction_target_rms = get_lm_head_oblique_direction_target_rms(opt_cfg)
    if direction_target_rms <= 0.0:
        raise ValueError(
            "Expected opt.lm_head_optimizer.direction_target_rms to be positive, "
            f"got {direction_target_rms}."
        )

    if bool(getattr(opt_cfg, "mucentering", False)):
        raise ValueError(
            f"opt.mucentering is incompatible with opt.lm_head_optimizer.type={lm_head_optimizer_type!r}. "
            f"The {manifold_name} optimizer already retracts `token_embed_out.embedding` onto a fixed-scale manifold."
        )

    incompatible_modes = (
        ("opt.lm_head_gradient_centering", getattr(opt_cfg, "lm_head_gradient_centering", "off")),
        (
            "opt.lm_head_weighted_columnwise_gradient_centering",
            getattr(opt_cfg, "lm_head_weighted_columnwise_gradient_centering", "off"),
        ),
    )
    for field_name, raw_value in incompatible_modes:
        if normalize_lm_head_centering_mode(raw_value, field_name) != "off":
            raise ValueError(
                f"{field_name} is incompatible with opt.lm_head_optimizer.type={lm_head_optimizer_type!r}. "
                f"Use the {manifold_name} tangent projection instead of extra LM-head centering transforms."
            )

    lm_head_adaptive_tuc_cfg = getattr(opt_cfg, "lm_head_adaptive_tuc", None)
    if bool(getattr(lm_head_adaptive_tuc_cfg, "enabled", False)):
        raise ValueError(
            "opt.lm_head_adaptive_tuc.enabled is incompatible with "
            f"opt.lm_head_optimizer.type={lm_head_optimizer_type!r}."
        )


def get_lm_head_oblique_target_rms(opt_cfg) -> float:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return float(getattr(lm_head_optimizer_cfg, "target_rms", 1.0))


def get_lm_head_oblique_direction_target_rms(opt_cfg) -> float:
    lm_head_optimizer_cfg = getattr(opt_cfg, "lm_head_optimizer", None)
    return float(getattr(lm_head_optimizer_cfg, "direction_target_rms", 1.0))


def get_row_oblique_target_rms(opt_cfg) -> float:
    return get_lm_head_oblique_target_rms(opt_cfg)


def _oblique_target_norm(width: int, target_rms: float):
    return jnp.asarray(target_rms, dtype=jnp.float32) * jnp.sqrt(
        jnp.asarray(width, dtype=jnp.float32)
    )


def row_wise_normalize_to_norm(matrix, target_norm: float = 1.0, eps: float = 1e-8):
    matrix_f32 = jnp.asarray(matrix, dtype=jnp.float32)
    if matrix_f32.ndim != 2:
        raise ValueError(
            "Expected a rank-2 matrix for row-wise normalization, "
            f"got shape={matrix_f32.shape}."
        )
    row_norms = jnp.linalg.norm(matrix_f32, axis=1, keepdims=True)
    safe_row_norms = jnp.maximum(row_norms, jnp.asarray(eps, dtype=jnp.float32))
    return matrix_f32 * (jnp.asarray(target_norm, dtype=jnp.float32) / safe_row_norms)


def row_wise_normalize(matrix, target_rms: float = 1.0, eps: float = 1e-8):
    matrix_f32 = jnp.asarray(matrix, dtype=jnp.float32)
    return row_wise_normalize_to_norm(
        matrix_f32,
        target_norm=_oblique_target_norm(matrix_f32.shape[1], target_rms),
        eps=eps,
    )


def column_wise_normalize(matrix, target_rms: float = 1.0, eps: float = 1e-8):
    matrix_f32 = jnp.asarray(matrix, dtype=jnp.float32)
    return column_wise_normalize_to_norm(
        matrix_f32,
        target_norm=_oblique_target_norm(matrix_f32.shape[0], target_rms),
        eps=eps,
    )


def column_wise_normalize_to_norm(matrix, target_norm: float = 1.0, eps: float = 1e-8):
    matrix_f32 = jnp.asarray(matrix, dtype=jnp.float32)
    if matrix_f32.ndim != 2:
        raise ValueError(
            "Expected a rank-2 matrix for column-wise normalization, "
            f"got shape={matrix_f32.shape}."
        )
    col_norms = jnp.linalg.norm(matrix_f32, axis=0, keepdims=True)
    safe_col_norms = jnp.maximum(col_norms, jnp.asarray(eps, dtype=jnp.float32))
    return matrix_f32 * (jnp.asarray(target_norm, dtype=jnp.float32) / safe_col_norms)


def project_to_row_oblique_tangent_space(update, param, target_rms: float = 1.0):
    update_f32 = jnp.asarray(update, dtype=jnp.float32)
    param_f32 = jnp.asarray(param, dtype=jnp.float32)
    if update_f32.ndim != 2 or param_f32.ndim != 2:
        raise ValueError(
            "Expected rank-2 matrices for Row-Oblique tangent projection, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )
    if update_f32.shape != param_f32.shape:
        raise ValueError(
            "Row-Oblique tangent projection requires matching update/parameter shapes, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )

    width = param_f32.shape[1]
    denom = jnp.asarray((target_rms ** 2) * width, dtype=jnp.float32)
    row_alignment = jnp.sum(update_f32 * param_f32, axis=1, keepdims=True)
    return update_f32 - param_f32 * (row_alignment / denom)


def project_to_column_oblique_tangent_space(update, param, target_rms: float = 1.0):
    update_f32 = jnp.asarray(update, dtype=jnp.float32)
    param_f32 = jnp.asarray(param, dtype=jnp.float32)
    if update_f32.ndim != 2 or param_f32.ndim != 2:
        raise ValueError(
            "Expected rank-2 matrices for Column-Oblique tangent projection, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )
    if update_f32.shape != param_f32.shape:
        raise ValueError(
            "Column-Oblique tangent projection requires matching update/parameter shapes, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )

    height = param_f32.shape[0]
    col_alignment = jnp.sum(update_f32 * param_f32, axis=0, keepdims=True)
    denom = jnp.asarray((target_rms ** 2) * height, dtype=jnp.float32)
    return update_f32 - param_f32 * (col_alignment / denom)


def apply_row_oblique_update(
    update,
    param,
    learning_rate,
    target_rms: float = 1.0,
    direction_target_rms: float = 1.0,
    eps: float = 1e-8,
):
    param_f32 = jnp.asarray(param, dtype=jnp.float32)
    update_f32 = jnp.asarray(update, dtype=jnp.float32)
    tangent_update = project_to_row_oblique_tangent_space(
        update_f32,
        param_f32,
        target_rms=target_rms,
    )
    steepest_direction = row_wise_normalize(
        tangent_update,
        target_rms=direction_target_rms,
        eps=eps,
    )
    lr_f32 = jnp.asarray(learning_rate, dtype=jnp.float32)
    next_param = row_wise_normalize(
        param_f32 - lr_f32 * steepest_direction,
        target_rms=target_rms,
        eps=eps,
    )
    return next_param.astype(jnp.asarray(param).dtype) - jnp.asarray(param)


def apply_column_oblique_update(
    update,
    param,
    learning_rate,
    target_rms: float = 1.0,
    direction_target_rms: float = 1.0,
    eps: float = 1e-8,
):
    param_f32 = jnp.asarray(param, dtype=jnp.float32)
    update_f32 = jnp.asarray(update, dtype=jnp.float32)
    tangent_update = project_to_column_oblique_tangent_space(
        update_f32,
        param_f32,
        target_rms=target_rms,
    )
    steepest_direction = column_wise_normalize(
        tangent_update,
        target_rms=direction_target_rms,
        eps=eps,
    )
    lr_f32 = jnp.asarray(learning_rate, dtype=jnp.float32)
    next_param = column_wise_normalize(
        param_f32 - lr_f32 * steepest_direction,
        target_rms=target_rms,
        eps=eps,
    )
    return next_param.astype(jnp.asarray(param).dtype) - jnp.asarray(param)


class EmaMomentumState(NamedTuple):
    mu: optax.Updates


class RowObliqueState(NamedTuple):
    count: jax.Array


def scale_by_ema_momentum(decay: float) -> optax.GradientTransformation:
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"Expected `decay` to be in [0, 1), got {decay}.")

    def init_leaf(param):
        if isinstance(param, optax.MaskedNode):
            return param
        value = getattr(param, "value", param)
        return jnp.zeros_like(value)

    def init_fn(params):
        mu = jax.tree_util.tree_map(
            init_leaf,
            params,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return EmaMomentumState(mu=mu)

    def update_leaf(update, mu):
        if isinstance(update, optax.MaskedNode) or isinstance(mu, optax.MaskedNode):
            return update
        new_mu = decay * mu + (1.0 - decay) * jnp.asarray(update)
        return new_mu.astype(jnp.asarray(mu).dtype)

    def update_fn(updates, state, params=None):
        del params
        new_mu = jax.tree_util.tree_map(
            update_leaf,
            updates,
            state.mu,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return new_mu, EmaMomentumState(mu=new_mu)

    return optax.GradientTransformation(init_fn, update_fn)


def row_oblique_steepest_descent(
    learning_rate,
    target_rms: float = 1.0,
    direction_target_rms: float = 1.0,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    if target_rms <= 0.0:
        raise ValueError(f"Expected `target_rms` to be positive, got {target_rms}.")
    if direction_target_rms <= 0.0:
        raise ValueError(
            f"Expected `direction_target_rms` to be positive, got {direction_target_rms}."
        )

    lr_schedule = learning_rate if callable(learning_rate) else lambda _: learning_rate

    def init_fn(_):
        return RowObliqueState(count=jnp.zeros([], dtype=jnp.int32))

    def transform_leaf(update, param, lr):
        if isinstance(update, optax.MaskedNode) or isinstance(param, optax.MaskedNode):
            return update
        return apply_row_oblique_update(
            update,
            param,
            learning_rate=lr,
            target_rms=target_rms,
            direction_target_rms=direction_target_rms,
            eps=eps,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("row_oblique_steepest_descent requires `params`.")
        lr = jnp.asarray(lr_schedule(state.count), dtype=jnp.float32)
        transformed_updates = jax.tree_util.tree_map(
            lambda update, param: transform_leaf(update, param, lr),
            updates,
            params,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return transformed_updates, RowObliqueState(count=state.count + jnp.asarray(1, dtype=jnp.int32))

    return optax.GradientTransformation(init_fn, update_fn)


def column_oblique_steepest_descent(
    learning_rate,
    target_rms: float = 1.0,
    direction_target_rms: float = 1.0,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    if target_rms <= 0.0:
        raise ValueError(f"Expected `target_rms` to be positive, got {target_rms}.")
    if direction_target_rms <= 0.0:
        raise ValueError(
            f"Expected `direction_target_rms` to be positive, got {direction_target_rms}."
        )

    lr_schedule = learning_rate if callable(learning_rate) else lambda _: learning_rate

    def init_fn(_):
        return RowObliqueState(count=jnp.zeros([], dtype=jnp.int32))

    def transform_leaf(update, param, lr):
        if isinstance(update, optax.MaskedNode) or isinstance(param, optax.MaskedNode):
            return update
        return apply_column_oblique_update(
            update,
            param,
            learning_rate=lr,
            target_rms=target_rms,
            direction_target_rms=direction_target_rms,
            eps=eps,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("column_oblique_steepest_descent requires `params`.")
        lr = jnp.asarray(lr_schedule(state.count), dtype=jnp.float32)
        transformed_updates = jax.tree_util.tree_map(
            lambda update, param: transform_leaf(update, param, lr),
            updates,
            params,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return transformed_updates, RowObliqueState(count=state.count + jnp.asarray(1, dtype=jnp.int32))

    return optax.GradientTransformation(init_fn, update_fn)


def _row_wise_mean_center(update):
    update_f32 = jnp.asarray(update, dtype=jnp.float32)
    if update_f32.ndim != 2:
        raise ValueError(
            "Expected a rank-2 update matrix for row-wise centering, "
            f"got shape={update_f32.shape}."
        )
    return update_f32 - jnp.mean(update_f32, axis=1, keepdims=True)


def apply_row_wise_adaptive_tuc(update, param, lambda_scale: float, eps: float = 1e-8):
    """Applies row-wise mean centering followed by an adaptive trust-region clip."""
    update_f32 = _row_wise_mean_center(update)
    param_f32 = jnp.asarray(param, dtype=jnp.float32)
    if param_f32.ndim != 2:
        raise ValueError(
            "Expected a rank-2 parameter matrix for adaptive TUC, "
            f"got shape={param_f32.shape}."
        )
    if update_f32.shape != param_f32.shape:
        raise ValueError(
            "Adaptive TUC requires update and parameter matrices with matching shapes, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )

    weight_norm = jnp.linalg.norm(param_f32, axis=1, keepdims=True)
    update_norm = jnp.linalg.norm(update_f32, axis=1, keepdims=True)
    safe_update_norm = jnp.maximum(update_norm, jnp.asarray(eps, dtype=jnp.float32))
    scale = jnp.minimum(1.0, jnp.asarray(lambda_scale, dtype=jnp.float32) * weight_norm / safe_update_norm)
    return update_f32 * scale


def apply_magnitude_weighted_column_wise_centering(update, param, eps: float = 1e-8):
    """Zero-centers per-column drift, weighted by squared row norms of the params."""
    update_f32 = jnp.asarray(update, dtype=jnp.float32)
    param_f32 = jnp.asarray(param, dtype=jnp.float32)
    if update_f32.ndim != 2 or param_f32.ndim != 2:
        raise ValueError(
            "Expected rank-2 matrices for weighted column-wise centering, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )
    if update_f32.shape != param_f32.shape:
        raise ValueError(
            "Weighted column-wise centering requires matching update/parameter shapes, "
            f"got update.shape={update_f32.shape}, param.shape={param_f32.shape}."
        )

    total_drift = jnp.sum(update_f32, axis=0, keepdims=True)
    sq_norms = jnp.sum(jnp.square(param_f32), axis=1, keepdims=True)
    sq_norm_sum = jnp.sum(sq_norms)
    uniform = jnp.full_like(sq_norms, 1.0 / sq_norms.shape[0])
    proportions = jnp.where(sq_norm_sum > eps, sq_norms / sq_norm_sum, uniform)
    return update_f32 - proportions * total_drift


def row_wise_mean_centering() -> optax.GradientTransformation:
    """Subtracts the per-row mean from masked 2D updates."""

    def init_fn(_):
        return optax.EmptyState()

    def center_update(update):
        if isinstance(update, optax.MaskedNode):
            return update
        return _row_wise_mean_center(update).astype(update.dtype)

    def update_fn(updates, state, params=None):
        del params
        centered_updates = jax.tree_util.tree_map(
            center_update,
            updates,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return centered_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def adaptive_tuc_row_wise(lambda_scale: float, eps: float = 1e-8) -> optax.GradientTransformation:
    """Row-wise mean-centers masked 2D updates, then clips them to a weight-relative trust region."""
    if lambda_scale < 0.0:
        raise ValueError(f"Expected `lambda_scale` to be non-negative, got {lambda_scale}.")

    def init_fn(_):
        return optax.EmptyState()

    def transform_update(update, param):
        if isinstance(update, optax.MaskedNode) or isinstance(param, optax.MaskedNode):
            return update
        transformed = apply_row_wise_adaptive_tuc(update, param, lambda_scale=lambda_scale, eps=eps)
        return transformed.astype(update.dtype)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("adaptive_tuc_row_wise requires `params` to compute row norms.")
        transformed_updates = jax.tree_util.tree_map(
            transform_update,
            updates,
            params,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return transformed_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def magnitude_weighted_column_wise_centering(eps: float = 1e-8) -> optax.GradientTransformation:
    """Zero-centers per-column drift, weighted by squared row norms of the params."""

    def init_fn(_):
        return optax.EmptyState()

    def center_update(update, param):
        if isinstance(update, optax.MaskedNode) or isinstance(param, optax.MaskedNode):
            return update
        centered = apply_magnitude_weighted_column_wise_centering(update, param, eps=eps)
        return centered.astype(update.dtype)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "magnitude_weighted_column_wise_centering requires `params` to compute row norms."
            )
        centered_updates = jax.tree_util.tree_map(
            center_update,
            updates,
            params,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return centered_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def apply_lm_head_output_update_post_transforms(update, param, opt_cfg):
    """Applies the configured post-optimizer LM-head transforms to a single update matrix."""
    transformed = jnp.asarray(update, dtype=jnp.float32)
    lm_head_gc_mode = normalize_lm_head_centering_mode(
        getattr(opt_cfg, "lm_head_gradient_centering", "off"),
        "opt.lm_head_gradient_centering",
    )
    if lm_head_gc_mode == "post":
        transformed = _row_wise_mean_center(transformed)

    lm_head_weighted_gc_mode = normalize_lm_head_centering_mode(
        getattr(opt_cfg, "lm_head_weighted_columnwise_gradient_centering", "off"),
        "opt.lm_head_weighted_columnwise_gradient_centering",
    )
    if lm_head_weighted_gc_mode == "post":
        transformed = apply_magnitude_weighted_column_wise_centering(transformed, param)

    lm_head_adaptive_tuc_cfg = getattr(opt_cfg, "lm_head_adaptive_tuc", None)
    if bool(getattr(lm_head_adaptive_tuc_cfg, "enabled", False)):
        transformed = apply_row_wise_adaptive_tuc(
            transformed,
            param,
            lambda_scale=float(getattr(lm_head_adaptive_tuc_cfg, "lambda_scale", 0.05)),
            eps=float(getattr(lm_head_adaptive_tuc_cfg, "eps", getattr(opt_cfg, "eps", 1e-8))),
        )

    return transformed.astype(jnp.asarray(update).dtype)


def build_lm_head_update_transforms(model: nnx.Module, opt_cfg):
    validate_row_oblique_lm_head_options(opt_cfg)
    pre_transforms = []
    post_transforms = []
    output_embedding_mask = None
    log_messages = []

    lm_head_gc_mode = normalize_lm_head_centering_mode(
        getattr(opt_cfg, "lm_head_gradient_centering", "off"),
        "opt.lm_head_gradient_centering",
    )
    if lm_head_gc_mode != "off":
        output_embedding_mask = build_output_embedding_mask(model)
        lm_head_gc_tx = optax.masked(
            row_wise_mean_centering(),
            output_embedding_mask,
        )
        if lm_head_gc_mode == "pre":
            pre_transforms.append(lm_head_gc_tx)
        else:
            post_transforms.append(lm_head_gc_tx)
        log_messages.append(
            "lm-head row-wise gradient centering enabled: "
            f"mode={lm_head_gc_mode}, target=token_embed_out.embedding"
        )

    lm_head_weighted_gc_mode = normalize_lm_head_centering_mode(
        getattr(opt_cfg, "lm_head_weighted_columnwise_gradient_centering", "off"),
        "opt.lm_head_weighted_columnwise_gradient_centering",
    )
    if lm_head_weighted_gc_mode != "off":
        if output_embedding_mask is None:
            output_embedding_mask = build_output_embedding_mask(model)
        lm_head_weighted_gc_tx = optax.masked(
            magnitude_weighted_column_wise_centering(),
            output_embedding_mask,
        )
        if lm_head_weighted_gc_mode == "pre":
            pre_transforms.append(lm_head_weighted_gc_tx)
        else:
            post_transforms.append(lm_head_weighted_gc_tx)
        log_messages.append(
            "lm-head weighted column-wise gradient centering enabled: "
            f"mode={lm_head_weighted_gc_mode}, target=token_embed_out.embedding"
        )

    lm_head_adaptive_tuc_cfg = getattr(opt_cfg, "lm_head_adaptive_tuc", None)
    lm_head_adaptive_tuc_enabled = bool(getattr(lm_head_adaptive_tuc_cfg, "enabled", False))
    if lm_head_adaptive_tuc_enabled:
        if output_embedding_mask is None:
            output_embedding_mask = build_output_embedding_mask(model)
        lambda_scale = float(getattr(lm_head_adaptive_tuc_cfg, "lambda_scale", 0.05))
        adaptive_tuc_eps = float(getattr(lm_head_adaptive_tuc_cfg, "eps", getattr(opt_cfg, "eps", 1e-8)))
        lm_head_adaptive_tuc_tx = optax.masked(
            adaptive_tuc_row_wise(lambda_scale=lambda_scale, eps=adaptive_tuc_eps),
            output_embedding_mask,
        )
        post_transforms.append(lm_head_adaptive_tuc_tx)
        log_messages.append(
            "lm-head adaptive TUC enabled: "
            f"lambda_scale={lambda_scale}, eps={adaptive_tuc_eps}, target=token_embed_out.embedding"
        )

    return tuple(pre_transforms), tuple(post_transforms), output_embedding_mask, tuple(log_messages)


def save_to_numpy(save_dir: str, name: str, data):
    path = os.path.join(save_dir, name)
    np.save(path, np.array(data))
    

def compute_lower_90th_percentile_mean(x):
    k = int(0.9 * x.size)
    top_90th = jnp.partition(x.flatten(), k)[:k]
    return top_90th.mean()


def get_l2_norm(data):
    sq_sum = jax.tree_util.tree_reduce(lambda acc, g: acc + jnp.sum(g * g), data, initializer=0.)
    return jnp.sqrt(sq_sum)

def get_layer_grad_norms(grads):
    norms = {}
    norms['global_grad_norm'] = get_l2_norm(grads)
    
    for key, value in grads.items():
        if key == 'blocks':
            for i, block_grads in value.items():
                norms[f'grad_norm/blocks.{i}'] = get_l2_norm(block_grads)
        else:
            norms[f'grad_norm/{key}'] = get_l2_norm(value)
            
    return norms


def get_layer_grad_norms_split(grads):
    norms = {}
    norms['grad_norm/global'] = float(get_l2_norm(grads))
    def visit(path, node):
        # Case 1: Param leaf
        if hasattr(node, "value"):      # nnx.Param
            norms[f"grad_norm/{path}"] = float(get_l2_norm(node.value))
            return
        # Case 2: State or dict-like object
        if hasattr(node, "items"):      # nnx.State, nested dicts
            for key, value in node.items():
                key = str(key)
                new_path = key if path == "" else f"{path}.{key}"
                visit(new_path, value)
            return
        # Case 3: raw array leaf
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            norms[f"grad_norm/{path}"] = float(get_l2_norm(node))
            return
    visit("", grads)
    return norms

def get_layer_weight_norms(params):
    leaves = _collect_weight_leaves_with_effective_lm_head(params)
    norms = {}
    global_sq_sum = sum((jnp.sum(value * value) for value in leaves.values()), jnp.asarray(0.0, dtype=jnp.float32))
    norms['weight_norm/global'] = jnp.sqrt(global_sq_sum)

    grouped_sq_sums = {}
    for path, value in leaves.items():
        path_parts = path.split(".")
        if path_parts[0] == "blocks" and len(path_parts) > 1:
            group_key = f"blocks.{path_parts[1]}"
        else:
            group_key = path_parts[0]
        grouped_sq_sums[group_key] = grouped_sq_sums.get(group_key, jnp.asarray(0.0, dtype=jnp.float32)) + jnp.sum(
            value * value
        )

    for key, sq_sum in grouped_sq_sums.items():
        norms[f'weight_norm/{key}'] = jnp.sqrt(sq_sum)

    return norms

def get_layer_weight_norms_split(params):
    leaves = _collect_weight_leaves_with_effective_lm_head(params)
    norms = {}
    global_sq_sum = sum((jnp.sum(value * value) for value in leaves.values()), jnp.asarray(0.0, dtype=jnp.float32))
    norms['weight_norm/global'] = float(jnp.sqrt(global_sq_sum))
    for path, value in leaves.items():
        norms[f"weight_norm/{path}"] = float(get_l2_norm(value))
    return norms


def _find_moment_state(opt_state):
    def visit(tree):
        if isinstance(tree, Mapping):
            if "mu" in tree and "nu" in tree:
                return tree
            for value in tree.values():
                found = visit(value)
                if found is not None:
                    return found
        elif hasattr(tree, "mu") and hasattr(tree, "nu"):
            return tree
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                found = visit(value)
                if found is not None:
                    return found
        return None

    return visit(opt_state)


def _find_moment_states(opt_state):
    """Returns every Adam moment state, including states inside masked chains."""
    states = []

    def visit(tree):
        if isinstance(tree, Mapping):
            if "mu" in tree and "nu" in tree:
                states.append(tree)
                return
            for value in tree.values():
                visit(value)
        elif hasattr(tree, "mu") and hasattr(tree, "nu"):
            states.append(tree)
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                visit(value)

    visit(opt_state)
    return states


def _get_state_component(state, key):
    if isinstance(state, Mapping):
        return state[key]
    return getattr(state, key)


def _get_nested_state_item(tree, path):
    node = tree
    for key in path:
        if isinstance(node, Mapping) or hasattr(node, "items"):
            node = node[key]
        else:
            node = getattr(node, key)
    return node


def _state_leaf_to_array(node, dtype=jnp.float32):
    value = node.value if hasattr(node, "value") else node
    return jnp.asarray(value, dtype=dtype)


def _get_lm_head_effective_scale(model_state, dtype=jnp.float32):
    try:
        log_target_rms = _state_leaf_to_array(
            _get_nested_state_item(model_state, ("lm_head_oblique_target_rms_log",)),
            dtype=dtype,
        )
    except (AttributeError, KeyError, TypeError):
        return None
    return jnp.exp(jnp.asarray(log_target_rms, dtype=dtype))


def _maybe_scale_output_embedding_array(path: str, value, lm_head_effective_scale):
    value = jnp.asarray(value, dtype=jnp.float32)
    if path == "token_embed_out.embedding" and lm_head_effective_scale is not None:
        return value * jnp.asarray(lm_head_effective_scale, dtype=value.dtype)
    return value


def _collect_weight_leaves_with_effective_lm_head(tree):
    leaves = {}
    lm_head_effective_scale = _get_lm_head_effective_scale(tree, dtype=jnp.float32)

    def visit(path, node):
        if hasattr(node, "value"):
            leaves[path] = _maybe_scale_output_embedding_array(path, node.value, lm_head_effective_scale)
            return
        if hasattr(node, "items"):
            for key, value in node.items():
                key = str(key)
                child_path = key if path == "" else f"{path}.{key}"
                visit(child_path, value)
            return
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            leaves[path] = _maybe_scale_output_embedding_array(path, node, lm_head_effective_scale)
            return

    visit("", tree)
    return leaves


def _get_first_nested_state_item(tree, candidate_paths):
    for path in candidate_paths:
        try:
            return _get_nested_state_item(tree, path)
        except (AttributeError, KeyError, TypeError):
            continue
    raise KeyError(f"Could not resolve any of the candidate paths: {candidate_paths}")


def _get_adam_nu_and_count(opt_state, path, expected_ndim):
    candidate_paths = (path, ("model", *path))
    for moment_state in _find_moment_states(opt_state):
        try:
            nu = _state_leaf_to_array(
                _get_first_nested_state_item(
                    _get_state_component(moment_state, "nu"),
                    candidate_paths,
                )
            )
            count = _state_leaf_to_array(
                _get_state_component(moment_state, "count")
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        # Masked optimizer states contain placeholders at excluded paths.
        if nu.ndim == expected_ndim:
            return nu, count
    return None


def get_effective_learning_rate_metrics(
    opt_state,
    learning_rate,
    lm_head_learning_rate,
    beta2,
    eps,
):
    """Effective-LR percentiles from Adam's bias-corrected second moment."""
    metrics = {}
    beta2 = jnp.asarray(beta2, dtype=jnp.float32)
    eps = jnp.asarray(eps, dtype=jnp.float32)

    def bias_correct(nu, count):
        count = jnp.asarray(count, dtype=jnp.float32)
        correction = 1.0 - jnp.power(beta2, count)
        return nu / correction

    out_ln_state = _get_adam_nu_and_count(
        opt_state, ("out_ln", "scale"), expected_ndim=1
    )
    if out_ln_state is not None:
        nu, count = out_ln_state
        v_hat = bias_correct(nu, count)
        eff_lr = jnp.asarray(learning_rate, dtype=jnp.float32) / (
            jnp.sqrt(jnp.maximum(v_hat, 0.0)) + eps
        )
        metrics["eff_lr/out_ln_scale/p01"] = jnp.percentile(eff_lr, 1)
        metrics["eff_lr/out_ln_scale/p50"] = jnp.percentile(eff_lr, 50)
        metrics["eff_lr/out_ln_scale/p99"] = jnp.percentile(eff_lr, 99)
        metrics["v_hat/out_ln_scale/p01"] = jnp.percentile(v_hat, 1)
        metrics["v_hat/out_ln_scale/p50"] = jnp.percentile(v_hat, 50)


    lm_head_state = _get_adam_nu_and_count(
        opt_state, ("token_embed_out", "embedding"), expected_ndim=2
    )
    if lm_head_state is not None:
        nu, count = lm_head_state
        v_hat = bias_correct(nu, count)
        v_row = jnp.mean(v_hat, axis=-1)
        eff_lr_row = jnp.asarray(lm_head_learning_rate, dtype=jnp.float32) / (
            jnp.sqrt(jnp.maximum(v_row, 0.0)) + eps
        )
        metrics["eff_lr/lm_head/p01"] = jnp.percentile(eff_lr_row, 1)
        metrics["eff_lr/lm_head/p50"] = jnp.percentile(eff_lr_row, 50)
        metrics["eff_lr/lm_head/p90"] = jnp.percentile(eff_lr_row, 90)
        metrics["v_hat/lm_head/p01"] = jnp.percentile(v_hat, 1)
        metrics["v_hat/lm_head/p50"] = jnp.percentile(v_hat, 50)

    return metrics


def _sanitize_metric_name(name: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or "unnamed"


def get_layer_moment_norms(opt_state):
    adam_state = _find_moment_state(opt_state)
    if adam_state is None:
        return {}

    metrics = {}
    # Helper to traverse the tree with a specific prefix
    def log_tree(tree, prefix):
        def visit(path, node):
            # Case 1: Param leaf (nnx.Param or similar wrapper with .value)
            if hasattr(node, "value"):      
                metrics[f"{prefix}/{path}"] = float(get_l2_norm(node.value))
                return
            # Case 2: State or dict-like object (recurse)
            if hasattr(node, "items"):      
                for key, value in node.items():
                    key = str(key)
                    new_path = key if path == "" else f"{path}.{key}"
                    visit(new_path, value)
                return
            # Case 3: raw array leaf
            if isinstance(node, (jnp.ndarray, np.ndarray)):
                metrics[f"{prefix}/{path}"] = float(get_l2_norm(node))
                return
        
        visit("", tree)
    
    # Process First Moment (mu)
    mu = _get_state_component(adam_state, "mu")
    metrics['moment1_norm/global'] = float(get_l2_norm(mu))
    log_tree(mu, 'moment1_norm')

    # Process Second Moment (nu)
    nu = _get_state_component(adam_state, "nu")
    metrics['moment2_norm/global'] = float(get_l2_norm(nu))
    log_tree(nu, 'moment2_norm')

    return metrics


def _collect_array_leaves(tree):
    leaves = {}

    def visit(path, node):
        if hasattr(node, "value"):
            leaves[path] = jnp.asarray(node.value, dtype=jnp.float32)
            return
        if hasattr(node, "items"):
            for key, value in node.items():
                key = str(key)
                child_path = key if path == "" else f"{path}.{key}"
                visit(child_path, value)
            return
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            leaves[path] = jnp.asarray(node, dtype=jnp.float32)
            return

    visit("", tree)
    return leaves


def get_layer_second_moment_rms_metrics(grads, opt_state, eps: float = 1e-30):
    """Per-leaf RMS diagnostics of normalized updates: sqrt(mean(g^2 / v))."""
    moment_state = _find_moment_state(opt_state)
    if moment_state is None:
        return {}

    second_moment = _get_state_component(moment_state, "nu")
    grad_leaves = _collect_array_leaves(grads)
    moment_leaves = _collect_array_leaves(second_moment)
    common_paths = sorted(set(grad_leaves) & set(moment_leaves))

    if not common_paths:
        return {}

    metrics = {}
    eps_arr = jnp.asarray(eps, dtype=jnp.float32)
    whitened_vals = []

    for path in common_paths:
        grad = grad_leaves[path]
        moment = moment_leaves[path]
        if grad.shape != moment.shape:
            continue

        whitened_rms = jnp.sqrt(jnp.mean(jnp.square(grad) / jnp.maximum(moment, eps_arr)))

        metrics[f"rms_grad_sq_over_second_moment/{path}"] = float(whitened_rms)
        whitened_vals.append(float(whitened_rms))

    if whitened_vals:
        metrics["rms_grad_sq_over_second_moment/global_mean"] = float(np.mean(whitened_vals))

    return metrics


def get_output_embedding_group_metrics(opt_state, token_groups, eps: float):
    """Group-averaged output embedding diagnostics over selected token rows."""
    if not token_groups:
        return {}

    try:
        output_embedding = _state_leaf_to_array(
            _get_nested_state_item(opt_state.model, ("token_embed_out", "embedding"))
        )
    except (AttributeError, KeyError, TypeError):
        return {}

    lm_head_effective_scale = _get_lm_head_effective_scale(opt_state.model, dtype=jnp.float32)
    if lm_head_effective_scale is not None:
        output_embedding = output_embedding * lm_head_effective_scale

    if output_embedding.ndim != 2:
        return {}

    moment_state = _find_moment_state(opt_state)
    first_moment = None
    second_moment = None
    sqrt_second_moment = None
    adam_ratio = None
    if moment_state is not None:
        try:
            first_moment = _state_leaf_to_array(
                _get_first_nested_state_item(
                    _get_state_component(moment_state, "mu"),
                    (
                        ("token_embed_out", "embedding"),
                        ("model", "token_embed_out", "embedding"),
                    ),
                )
            )
            second_moment = _state_leaf_to_array(
                _get_first_nested_state_item(
                    _get_state_component(moment_state, "nu"),
                    (
                        ("token_embed_out", "embedding"),
                        ("model", "token_embed_out", "embedding"),
                    ),
                )
            )
        except (AttributeError, KeyError, TypeError):
            first_moment = None
            second_moment = None

    if (
        first_moment is not None
        and second_moment is not None
        and first_moment.shape == output_embedding.shape
        and second_moment.shape == output_embedding.shape
    ):
        sqrt_second_moment = jnp.sqrt(jnp.maximum(second_moment, 0.0))
        adam_ratio = first_moment / (sqrt_second_moment + jnp.asarray(eps, dtype=jnp.float32))

    vocab_size = int(output_embedding.shape[0])
    metrics = {}
    group_rows_by_name = {}
    group_token_ids_by_name = {}

    for group in token_groups:
        token_ids = np.asarray(group["token_ids"], dtype=np.int32)
        valid_token_ids = token_ids[(token_ids >= 0) & (token_ids < vocab_size)]
        if valid_token_ids.size == 0:
            continue

        metric_group_name = _sanitize_metric_name(group["name"])
        group_rows = output_embedding[valid_token_ids]
        group_rows_by_name[metric_group_name] = group_rows
        group_token_ids_by_name[metric_group_name] = valid_token_ids

        row_l2_norm = jnp.linalg.norm(group_rows, axis=-1)
        row_rms = row_l2_norm / jnp.sqrt(jnp.asarray(group_rows.shape[-1], dtype=jnp.float32))

        metrics[f"output_embedding_groups/{metric_group_name}/row_l2_norm_mean"] = jnp.mean(row_l2_norm)
        metrics[f"output_embedding_groups/{metric_group_name}/row_rms_mean"] = jnp.mean(row_rms)
        if sqrt_second_moment is not None and adam_ratio is not None:
            group_sqrt_second_moment = sqrt_second_moment[valid_token_ids]
            group_adam_ratio = adam_ratio[valid_token_ids]
            row_sqrt_second_moment_mean = jnp.mean(group_sqrt_second_moment, axis=-1)
            row_sqrt_second_moment_max = jnp.max(group_sqrt_second_moment, axis=-1)
            row_adam_ratio_mean = jnp.mean(group_adam_ratio, axis=-1)
            metrics[f"output_embedding_groups/{metric_group_name}/sqrt_second_moment_mean"] = jnp.mean(
                row_sqrt_second_moment_mean
            )
            metrics[f"output_embedding_groups/{metric_group_name}/sqrt_second_moment_max"] = jnp.mean(
                row_sqrt_second_moment_max
            )
            metrics[f"output_embedding_groups/{metric_group_name}/adam_ratio_mean"] = jnp.mean(row_adam_ratio_mean)

    most_frequent_rows = group_rows_by_name.get("most_frequent")
    least_frequent_rows = group_rows_by_name.get("least_frequent")
    least_frequent_token_ids = group_token_ids_by_name.get("least_frequent")
    if most_frequent_rows is not None and least_frequent_rows is not None and least_frequent_token_ids is not None:
        most_frequent_mean = jnp.mean(most_frequent_rows, axis=0)
        most_frequent_mean_norm = jnp.linalg.norm(most_frequent_mean)
        least_frequent_norms = jnp.linalg.norm(least_frequent_rows, axis=-1)
        cosine_denoms = jnp.maximum(most_frequent_mean_norm * least_frequent_norms, jnp.asarray(eps, dtype=jnp.float32))
        cosine_sims = jnp.sum(least_frequent_rows * most_frequent_mean[None, :], axis=-1) / cosine_denoms

        for token_id, cosine_sim in zip(least_frequent_token_ids.tolist(), cosine_sims):
            metrics[
                f"output_embedding_groups/most_frequent_to_least_frequent/cosine_similarity/token_{int(token_id)}"
            ] = cosine_sim
        metrics["output_embedding_groups/most_frequent_to_least_frequent/cosine_similarity_mean"] = jnp.mean(
            cosine_sims
        )

    return metrics


def get_lm_head_oblique_target_metrics(model_state):
    try:
        log_target_rms = _state_leaf_to_array(
            _get_nested_state_item(model_state, ("lm_head_oblique_target_rms_log",))
        )
    except (AttributeError, KeyError, TypeError):
        return {}

    log_target_rms = jnp.asarray(log_target_rms, dtype=jnp.float32)
    target_rms = jnp.exp(log_target_rms)
    return {
        "lm_head_oblique/log_target_rms": float(jnp.squeeze(log_target_rms)),
        "lm_head_oblique/target_rms": float(jnp.squeeze(target_rms)),
    }


def compute_qkv_stats(qkv_dict):
    stats = {}
    for i, (q, k, v) in qkv_dict.items():
        # Using simple mean/std here. Could add min/max/norm if needed.
        stats[f'q/layer_{i}/mean'] = q.mean()
        stats[f'k/layer_{i}/mean'] = k.mean()
        stats[f'v/layer_{i}/mean'] = v.mean()
        stats[f'q/layer_{i}/std'] = q.std()
        stats[f'k/layer_{i}/std'] = k.std()
        stats[f'v/layer_{i}/std'] = v.std()
        stats[f'q/layer_{i}/max'] = q.max()
        stats[f'k/layer_{i}/max'] = k.max()
        stats[f'v/layer_{i}/max'] = v.max()
        stats[f'q/layer_{i}/fro_norm'] = get_l2_norm(q)
        stats[f'k/layer_{i}/fro_norm'] = get_l2_norm(k)
        stats[f'v/layer_{i}/fro_norm'] = get_l2_norm(v)
    return stats


@partial(jax.jit, static_argnames=('model_graphdef', 'vocab_size'))
def _per_token_loss_sum_and_count(model_state, model_graphdef, x, vocab_size):
    """Per-target-token loss sum [V] and count [V] for a single batch."""
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x, return_qkv=False).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)

    # remove wraparound last position
    losses = losses[:, :-1]
    targets = y[:, :-1]
    # both [B·(T-1)]
    losses_flat = losses.reshape(-1)
    targets_flat = targets.reshape(-1)

    # scatter-add each (target_id, loss) pair into per-token bins
    loss_sum = jnp.zeros(vocab_size, dtype=jnp.float32).at[targets_flat].add(losses_flat)
    count = jnp.zeros(vocab_size, dtype=jnp.int32).at[targets_flat].add(1)
    return loss_sum, count


@partial(jax.jit, static_argnames=('model_graphdef',))
def _per_batch_grad_W_U(model_state, model_graphdef, x):
    """[V, D] gradient of summed CE loss w.r.t. W_U for a single batch."""
    def loss_fn(state):
        model = nnx.merge(model_graphdef, state)
        y = jnp.roll(x, -1, axis=1)
        logits = model(x, return_qkv=False).astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        return losses[:, :-1].sum()

    grads = jax.grad(loss_fn)(model_state)
    g_W_U = _get_nested_state_item(grads, ("token_embed_out", "embedding"))
    return _state_leaf_to_array(g_W_U)  # [V, D]


def _compute_adam_update_W_U(model_state, opt_state, lr_at_step, eps, weight_decay):
    """Returns (update_W_U, m_W_U, v_W_U) — AdamW update plus the stored
    first/second moments that produced it."""
    W_U = _state_leaf_to_array(_get_nested_state_item(model_state, ("token_embed_out", "embedding")))
    adam = _find_moment_state(opt_state)
    m_W_U = _state_leaf_to_array(_get_nested_state_item(adam["mu"], ("token_embed_out", "embedding")))
    v_W_U = _state_leaf_to_array(_get_nested_state_item(adam["nu"], ("token_embed_out", "embedding")))
    update_W_U = -lr_at_step * m_W_U / (jnp.sqrt(v_W_U) + eps)
    update_W_U -= lr_at_step * weight_decay * W_U
    return update_W_U, m_W_U, v_W_U


def _dist_stats(x):
    """Return (median, p90, p99, max, mean) as Python floats for a [V] array."""
    return (
        float(jnp.median(x)),
        float(jnp.quantile(x, 0.9)),
        float(jnp.quantile(x, 0.99)),
        float(jnp.max(x)),
        float(jnp.mean(x)),
    )


def _fmt_width(spec):
    return len(f"{0:{spec}}")


def _print_ranked_tables(
    metric_name, sort_key_arr, k, count_total, raw_mean_loss, column_specs,
):
    """Print Top-k and Bottom-k tables sorted by sort_key_arr (Top descending,
    Bottom ascending). Each row shows rank | token_id | count | <column_specs...> |
    mean_loss (— when count==0). column_specs is an ordered list of
    (header, fmt_spec, [V]_array)."""
    widths = [_fmt_width(spec) for _, spec, _ in column_specs]
    for label, ascending in [("Top", False), ("Bottom", True)]:
        idx = jnp.argsort(sort_key_arr if ascending else -sort_key_arr)[:k]
        idx_list = [int(x) for x in idx.tolist()]
        counts = [int(x) for x in count_total[idx].tolist()]
        ml_vals = [float(x) for x in raw_mean_loss[idx].tolist()]
        col_vals = [
            [float(x) for x in arr[idx].tolist()] for _, _, arr in column_specs
        ]

        print(f"\n{label}-{k} tokens by {metric_name}:")
        header = f"  {'rank':>4} {'token_id':>9} {'count':>6}"
        for (name, _, _), w in zip(column_specs, widths):
            header += f" {name:>{w}}"
        header += f" {'mean_loss':>10}"
        print(header)

        for rank in range(k):
            row = f"  {rank+1:>4d} {idx_list[rank]:>9d} {counts[rank]:>6d}"
            for col_idx, (_, spec, _) in enumerate(column_specs):
                row += f" {col_vals[col_idx][rank]:{spec}}"
            ml_str = f"{ml_vals[rank]:>10.4f}" if counts[rank] > 0 else f"{'—':>10}"
            row += f" {ml_str}"
            print(row)


def compute_spike_token_diagnostics(
    model_state, model_graphdef, ds_valid, step: int,
    top_k: int = 20, min_count: int = 5,
    update_W_U=None, m_W_U=None, v_W_U=None, save_dir=None,
):
    """Rank target tokens by mean eval loss; for the top-k tokens, show
    W_U row norms and ||∂L/∂W_U[i,:]|| alongside the vocab-wide distributions.
    Returns wandb-loggable metrics + prints summary."""
    W_U = _state_leaf_to_array(
        _get_nested_state_item(model_state, ("token_embed_out", "embedding"))
    )
    vocab_size, D = int(W_U.shape[0]), int(W_U.shape[1])

    # Single eval pass: accumulate per-token loss sum, count, the abs-grad
    # (used only as the |grad row mean| ranking key — keeps Ting's no-cancellation
    # property), and the SIGNED grad sum (used for grad_signed_row_norms — the
    # apples-to-apples comparison column for upd_rn / m_rn).
    loss_sum_total = jnp.zeros(vocab_size, dtype=jnp.float32)
    count_total = jnp.zeros(vocab_size, dtype=jnp.int32)
    grad_sum_total = jnp.zeros((vocab_size, D), dtype=jnp.float32)
    grad_signed_sum_total = jnp.zeros((vocab_size, D), dtype=jnp.float32)
    for i in range(len(ds_valid)):
        batch = ds_valid[i]
        ls, cnt = _per_token_loss_sum_and_count(
            model_state, model_graphdef, batch, vocab_size
        )
        loss_sum_total = loss_sum_total + ls
        count_total = count_total + cnt
        g_batch = _per_batch_grad_W_U(model_state, model_graphdef, batch)
        grad_sum_total = grad_sum_total + jnp.abs(g_batch)
        grad_signed_sum_total = grad_signed_sum_total + g_batch

    # only rank tokens seen >= min_count times (else one-shot rare tokens dominate)
    mean_loss = jnp.where(
        count_total >= min_count,
        loss_sum_total / jnp.maximum(count_total, 1),
        -jnp.inf,
    )

    # vocab-wide distributions: W_U row norms and per-batch-avg signed-grad row norms
    row_norms = jnp.linalg.norm(W_U, axis=-1)                                            # [V]
    grad_signed_row_norms = jnp.linalg.norm(grad_signed_sum_total / len(ds_valid), axis=-1)  # [V]

    median_rn, p90, p99, max_rn, _ = _dist_stats(row_norms)
    grad_median, grad_p90, grad_p99, grad_max, grad_mean = _dist_stats(grad_signed_row_norms)

    eligible = int(jnp.sum(count_total >= min_count))
    k = min(top_k, eligible) if eligible > 0 else 0
    if k == 0:
        print(f"\n=== Spike analysis @ step {step}: no tokens with count >= {min_count} in eval ===")
        return {}

    top_ids = jnp.argsort(-mean_loss)[:k]
    top_ids_list = [int(x) for x in top_ids.tolist()]
    counts_top = [int(x) for x in count_total[top_ids].tolist()]
    mean_loss_top = [float(x) for x in mean_loss[top_ids].tolist()]
    row_norms_top = [float(x) for x in row_norms[top_ids].tolist()]
    grad_rn_top = [float(x) for x in grad_signed_row_norms[top_ids].tolist()]

    print(f"\n=== Spike analysis @ step {step} ===")
    print(
        f"W_U row_norm dist: median={median_rn:.3f}  p90={p90:.3f}  p99={p99:.3f}  max={max_rn:.3f}"
        f"   (eligible tokens: {eligible}/{vocab_size}, min_count={min_count})"
    )
    print(
        f"signed grad row_norm dist: median={grad_median:.3e}  p90={grad_p90:.3e}  p99={grad_p99:.3e}  max={grad_max:.3e}"
    )
    print(f"Top-{k} target tokens by mean eval loss:")
    print(
        f"  {'rank':>4} {'token_id':>9} {'count':>6} {'mean_loss':>10}"
        f" {'row_norm':>10} {'rn/median':>10}"
        f" {'grad_signed_rn':>15} {'g_rn/median':>12}"
    )
    for rank in range(k):
        print(
            f"  {rank+1:>4d} {top_ids_list[rank]:>9d} {counts_top[rank]:>6d}"
            f" {mean_loss_top[rank]:>10.4f} {row_norms_top[rank]:>10.4f}"
            f" {row_norms_top[rank] / max(median_rn, 1e-9):>10.3f}"
            f" {grad_rn_top[rank]:>15.3e}"
            f" {grad_rn_top[rank] / max(grad_median, 1e-12):>12.3f}"
        )

    grad_row_mean = jnp.mean(grad_sum_total / len(ds_valid), axis=-1)
    top_k_grad = min(top_k, vocab_size)
    raw_mean_loss = loss_sum_total / jnp.maximum(count_total, 1)

    _print_ranked_tables(
        metric_name="|grad row mean|",
        sort_key_arr=grad_row_mean,
        k=top_k_grad,
        count_total=count_total,
        raw_mean_loss=raw_mean_loss,
        column_specs=[
            ("grad_mean", ">12.3e", grad_row_mean),
            ("grad_signed_rn", ">15.3e", grad_signed_row_norms),
            ("g_rn/median", ">12.3f", grad_signed_row_norms / max(grad_median, 1e-12)),
            ("row_norm", ">10.4f", row_norms),
            ("rn/median", ">10.3f", row_norms / max(median_rn, 1e-9)),
        ],
    )

    counts_f32 = count_total.astype(jnp.float32)
    c_min = int(jnp.min(counts_f32))
    c_q1 = int(jnp.quantile(counts_f32, 0.25))
    c_med = int(jnp.quantile(counts_f32, 0.50))
    c_q3 = int(jnp.quantile(counts_f32, 0.75))
    c_max = int(jnp.max(counts_f32))
    nonzero_mask = count_total > 0
    n_nonzero = int(jnp.sum(nonzero_mask))
    counts_nz = counts_f32[nonzero_mask] if n_nonzero > 0 else counts_f32
    nz_min = int(jnp.min(counts_nz))
    nz_q1 = int(jnp.quantile(counts_nz, 0.25))
    nz_med = int(jnp.quantile(counts_nz, 0.50))
    nz_q3 = int(jnp.quantile(counts_nz, 0.75))
    nz_max = int(jnp.max(counts_nz))
    print(
        f"\nTarget-count 5-number summary (all {vocab_size} tokens):"
        f" min={c_min:_}  Q1={c_q1:_}  median={c_med:_}  Q3={c_q3:_}  max={c_max:_}"
    )
    print(
        f"Target-count 5-number summary ({n_nonzero} tokens with count>0):"
        f" min={nz_min:_}  Q1={nz_q1:_}  median={nz_med:_}  Q3={nz_q3:_}  max={nz_max:_}"
    )

    if update_W_U is not None:
        update_row_norm = jnp.linalg.norm(update_W_U, axis=-1)
        update_row_abs_mean = jnp.mean(jnp.abs(update_W_U), axis=-1)
        u_rn_med, u_rn_p90, u_rn_p99, u_rn_max, _ = _dist_stats(update_row_norm)
        u_am_med, u_am_p90, u_am_p99, u_am_max, _ = _dist_stats(update_row_abs_mean)
        print(
            f"\nupdate row_norm dist:    median={u_rn_med:.3e}  p90={u_rn_p90:.3e}  p99={u_rn_p99:.3e}  max={u_rn_max:.3e}"
        )
        print(
            f"update mean(|row|) dist: median={u_am_med:.3e}  p90={u_am_p90:.3e}  p99={u_am_p99:.3e}  max={u_am_max:.3e}"
        )

        column_specs = [
            ("upd_abs_mean", ">13.3e", update_row_abs_mean),
            ("upd_rn", ">12.3e", update_row_norm),
        ]
        if m_W_U is not None and v_W_U is not None:
            m_row_norm = jnp.linalg.norm(m_W_U, axis=-1)
            sqrtv_row_norm = jnp.linalg.norm(jnp.sqrt(v_W_U), axis=-1)
            column_specs += [
                ("m_rn", ">12.3e", m_row_norm),
                ("sqrtv_rn", ">12.3e", sqrtv_row_norm),
            ]
        column_specs += [
            ("grad_signed_rn", ">15.3e", grad_signed_row_norms),
        ]

        _print_ranked_tables(
            metric_name="mean(|update_row|)",
            sort_key_arr=update_row_abs_mean,
            k=top_k_grad,
            count_total=count_total,
            raw_mean_loss=raw_mean_loss,
            column_specs=column_specs,
        )

        if save_dir is not None and v_W_U is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            sqrtv_mean = jnp.mean(jnp.sqrt(v_W_U), axis=-1)
            mask = count_total > 0
            counts_np = np.array(count_total[mask])
            sqrtv_np = np.array(sqrtv_mean[mask])
            plt.figure(figsize=(8, 6))
            plt.scatter(counts_np, sqrtv_np, s=3, alpha=0.35)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('target count')
            plt.ylabel('mean(√v[v, :]) over D features')
            plt.title(f'mean(√v) vs target count @ step {step}  (count>0 only)')
            plt.tight_layout()
            plot_path = os.path.join(str(save_dir), 'sqrtv_vs_count.png')
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"Saved mean(√v) vs count scatter to: {plot_path}")

        if save_dir is not None and m_W_U is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            m_abs_mean = jnp.mean(jnp.abs(m_W_U), axis=-1)
            mask = count_total > 0
            counts_np = np.array(count_total[mask])
            m_np = np.array(m_abs_mean[mask])
            plt.figure(figsize=(8, 6))
            plt.scatter(counts_np, m_np, s=3, alpha=0.35)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('target count')
            plt.ylabel('mean(|m[v, :]|) over D features')
            plt.title(f'mean(|m|) vs target count @ step {step}  (count>0 only)')
            plt.tight_layout()
            plot_path_m = os.path.join(str(save_dir), 'm_vs_count.png')
            plt.savefig(plot_path_m, dpi=120)
            plt.close()
            print(f"Saved mean(|m|) vs count scatter to: {plot_path_m}")

    metrics = {
        "spike_analysis/row_norm/median": median_rn,
        "spike_analysis/row_norm/p90": p90,
        "spike_analysis/row_norm/p99": p99,
        "spike_analysis/row_norm/max": max_rn,
        "spike_analysis/grad_row_norm/mean": grad_mean,
        "spike_analysis/grad_row_norm/median": grad_median,
        "spike_analysis/grad_row_norm/p90": grad_p90,
        "spike_analysis/grad_row_norm/p99": grad_p99,
        "spike_analysis/grad_row_norm/max": grad_max,
        "spike_analysis/eligible_token_count": eligible,
    }
    for rank in range(k):
        prefix = f"spike_analysis/top_{rank+1:02d}"
        metrics[f"{prefix}/token_id"] = top_ids_list[rank]
        metrics[f"{prefix}/mean_loss"] = mean_loss_top[rank]
        metrics[f"{prefix}/count"] = counts_top[rank]
        metrics[f"{prefix}/row_norm"] = row_norms_top[rank]
        metrics[f"{prefix}/row_norm_over_median"] = row_norms_top[rank] / max(median_rn, 1e-9)
        metrics[f"{prefix}/grad_row_norm"] = grad_rn_top[rank]
        metrics[f"{prefix}/grad_row_norm_over_median"] = grad_rn_top[rank] / max(grad_median, 1e-12)
    return metrics
