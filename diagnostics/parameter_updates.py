"""Parameter-update diagnostics kept separate from the model architecture.

These metrics compare a model immediately before and after an optimizer step
while holding the captured operator inputs fixed. They are optional and are
only evaluated when the corresponding diagnostic logging flag is enabled.
"""

import jax
import jax.numpy as jnp
from flax import nnx


_UPDATE_METRIC_EPS = 1e-30
_SPECTRAL_POWER_ITERATIONS = 5


def _frobenius_norm(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(x)))


def _safe_ratio(numerator, denominator):
    return numerator / jnp.maximum(
        denominator, jnp.asarray(_UPDATE_METRIC_EPS, dtype=jnp.float32)
    )


def _spectral_norm_estimate(matrix, iterations: int = _SPECTRAL_POWER_ITERATIONS):
    """Deterministic two-probe power-iteration estimate of a matrix spectral norm."""
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    probe_index = jnp.arange(matrix.shape[1], dtype=jnp.float32) + 1.0
    probes = jnp.stack(
        (
            jnp.sin(probe_index * 0.734375) + jnp.cos(probe_index * 1.234375),
            jnp.sin(probe_index * 1.6171875) - jnp.cos(probe_index * 0.4453125),
        ),
        axis=1,
    )

    def normalize_columns(vectors):
        norms = jnp.sqrt(jnp.sum(jnp.square(vectors), axis=0, keepdims=True))
        return vectors / jnp.maximum(norms, _UPDATE_METRIC_EPS)

    vectors = normalize_columns(probes)

    def power_step(_, current):
        return normalize_columns(matrix.T @ (matrix @ current))

    vectors = jax.lax.fori_loop(0, iterations, power_step, vectors)
    estimates = jnp.sqrt(jnp.sum(jnp.square(matrix @ vectors), axis=0))
    return jnp.max(estimates)


def _linear_output_fro_from_grams(operator_input, matrix):
    """Computes ||XW||_F without materializing XW."""
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    flattened_input = jnp.asarray(operator_input, dtype=jnp.float32).reshape(
        -1, matrix.shape[0]
    )
    input_gram = flattened_input.T @ flattened_input
    weight_gram = matrix @ matrix.T
    squared_norm = jnp.sum(input_gram * weight_gram)
    return jnp.sqrt(jnp.maximum(squared_norm, 0.0))


def _add_matrix_update_metrics(
    metrics,
    path,
    old_weight,
    new_weight,
    operator_input,
    old_output_norm,
    delta_output=None,
    old_matrix=None,
    new_matrix=None,
    delta_output_norm=None,
):
    old_weight = jnp.asarray(old_weight, dtype=jnp.float32)
    new_weight = jnp.asarray(new_weight, dtype=jnp.float32)
    delta_weight = new_weight - old_weight
    delta_weight_norm = _frobenius_norm(delta_weight)
    prefix = f"parameter_update/{path}"
    metrics[f"{prefix}/delta_param_fro"] = delta_weight_norm
    metrics[f"{prefix}/delta_param_rel_fro"] = _safe_ratio(
        delta_weight_norm, _frobenius_norm(old_weight)
    )
    metrics[f"{prefix}/input_fro"] = _frobenius_norm(operator_input)
    if delta_output_norm is None:
        delta_output_norm = _frobenius_norm(delta_output)
    metrics[f"{prefix}/delta_output_fro"] = delta_output_norm
    metrics[f"{prefix}/delta_output_rel_fro"] = _safe_ratio(
        delta_output_norm, old_output_norm
    )

    old_matrix = old_weight if old_matrix is None else jnp.asarray(old_matrix, dtype=jnp.float32)
    new_matrix = new_weight if new_matrix is None else jnp.asarray(new_matrix, dtype=jnp.float32)
    delta_matrix = new_matrix - old_matrix
    metrics[f"{prefix}/delta_param_rel_spectral_estimate"] = _safe_ratio(
        _spectral_norm_estimate(delta_matrix),
        _spectral_norm_estimate(old_matrix),
    )


def _add_scale_update_metrics(metrics, path, old_scale, new_scale, capture):
    old_scale = jnp.asarray(old_scale, dtype=jnp.float32)
    new_scale = jnp.asarray(new_scale, dtype=jnp.float32)
    delta_scale = new_scale - old_scale
    operator_input = jnp.asarray(capture["input"], dtype=jnp.float32)
    delta_output = operator_input * delta_scale
    prefix = f"parameter_update/{path}"
    delta_scale_norm = _frobenius_norm(delta_scale)
    metrics[f"{prefix}/delta_param_fro"] = delta_scale_norm
    metrics[f"{prefix}/delta_param_rel_fro"] = _safe_ratio(
        delta_scale_norm, _frobenius_norm(old_scale)
    )
    metrics[f"{prefix}/input_fro"] = _frobenius_norm(operator_input)
    metrics[f"{prefix}/raw_input_fro"] = capture["raw_input_fro"]
    delta_output_norm = _frobenius_norm(delta_output)
    metrics[f"{prefix}/delta_output_fro"] = delta_output_norm
    metrics[f"{prefix}/delta_output_rel_fro"] = _safe_ratio(
        delta_output_norm, capture["output_fro"]
    )
    metrics[f"{prefix}/delta_param_rel_spectral"] = _safe_ratio(
        jnp.max(jnp.abs(delta_scale)),
        jnp.max(jnp.abs(old_scale)),
    )


def compute_parameter_update_metrics(old_model, new_model, captures):
    """Computes per-operator update metrics on captured, fixed inputs."""
    metrics = {}

    old_embedding = old_model.token_embed_in.embedding.value
    new_embedding = new_model.token_embed_in.embedding.value
    capture = captures["token_embed_in.embedding"]
    delta_embedding = jnp.asarray(new_embedding, dtype=jnp.float32) - jnp.asarray(
        old_embedding, dtype=jnp.float32
    )
    token_ids = capture["input"]
    _add_matrix_update_metrics(
        metrics,
        "token_embed_in.embedding",
        old_embedding,
        new_embedding,
        jnp.ones(token_ids.shape, dtype=jnp.float32),
        capture["output_fro"],
        jnp.take(delta_embedding, token_ids, axis=0),
    )

    for i, (old_block, new_block) in enumerate(zip(old_model.blocks, new_model.blocks)):
        block_prefix = f"blocks.{i}"

        for norm_name in ("ln1", "ln2"):
            old_norm = getattr(old_block, norm_name)
            new_norm = getattr(new_block, norm_name)
            if getattr(old_norm, "scale", None) is not None:
                path = f"{block_prefix}.{norm_name}.scale"
                _add_scale_update_metrics(
                    metrics,
                    path,
                    old_norm.scale.value,
                    new_norm.scale.value,
                    captures[path],
                )

        old_attn = old_block.attn
        new_attn = new_block.attn
        attn_prefix = f"{block_prefix}.attn"

        path = f"{attn_prefix}.qkv_proj.kernel"
        capture = captures[path]
        old_weight = old_attn.qkv_proj.kernel.value
        new_weight = new_attn.qkv_proj.kernel.value
        delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(
            old_weight, dtype=jnp.float32
        )
        old_matrix = jnp.transpose(
            jnp.asarray(old_weight, dtype=jnp.float32), (2, 0, 1, 3)
        ).reshape(old_weight.shape[2], -1)
        new_matrix = jnp.transpose(
            jnp.asarray(new_weight, dtype=jnp.float32), (2, 0, 1, 3)
        ).reshape(new_weight.shape[2], -1)
        _add_matrix_update_metrics(
            metrics,
            path,
            old_weight,
            new_weight,
            capture["input"],
            capture["output_fro"],
            jnp.einsum("BTd,SNdH->SBTNH", capture["input"], delta_weight),
            old_matrix,
            new_matrix,
        )

        if old_attn.elementwise_attn_output_gate:
            path = f"{attn_prefix}.gate_proj.kernel"
            capture = captures[path]
            old_weight = old_attn.gate_proj.kernel.value
            new_weight = new_attn.gate_proj.kernel.value
            delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(
                old_weight, dtype=jnp.float32
            )
            old_matrix = jnp.transpose(
                jnp.asarray(old_weight, dtype=jnp.float32), (1, 0, 2)
            ).reshape(old_weight.shape[1], -1)
            new_matrix = jnp.transpose(
                jnp.asarray(new_weight, dtype=jnp.float32), (1, 0, 2)
            ).reshape(new_weight.shape[1], -1)
            _add_matrix_update_metrics(
                metrics,
                path,
                old_weight,
                new_weight,
                capture["input"],
                capture["output_fro"],
                jnp.einsum("BTd,NdH->BTNH", capture["input"], delta_weight),
                old_matrix,
                new_matrix,
            )

        for norm_name in ("query_norm", "key_norm"):
            old_norm = getattr(old_attn, norm_name)
            new_norm = getattr(new_attn, norm_name)
            if isinstance(old_norm, nnx.RMSNorm) and getattr(old_norm, "scale", None) is not None:
                path = f"{attn_prefix}.{norm_name}.scale"
                _add_scale_update_metrics(
                    metrics,
                    path,
                    old_norm.scale.value,
                    new_norm.scale.value,
                    captures[path],
                )

        path = f"{attn_prefix}.out_proj.kernel"
        capture = captures[path]
        old_weight = old_attn.out_proj.kernel.value
        new_weight = new_attn.out_proj.kernel.value
        delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(
            old_weight, dtype=jnp.float32
        )
        _add_matrix_update_metrics(
            metrics,
            path,
            old_weight,
            new_weight,
            capture["input"],
            capture["output_fro"],
            jnp.einsum("BTnh,nhD->BTD", capture["input"], delta_weight),
            jnp.asarray(old_weight, dtype=jnp.float32).reshape(-1, old_weight.shape[-1]),
            jnp.asarray(new_weight, dtype=jnp.float32).reshape(-1, new_weight.shape[-1]),
        )

        for projection_name in ("up_proj", "down_proj"):
            old_projection = getattr(old_block.mlp, projection_name)
            new_projection = getattr(new_block.mlp, projection_name)
            path = f"{block_prefix}.mlp.{projection_name}.kernel"
            capture = captures[path]
            old_weight = old_projection.kernel.value
            new_weight = new_projection.kernel.value
            delta_weight = jnp.asarray(new_weight, dtype=jnp.float32) - jnp.asarray(
                old_weight, dtype=jnp.float32
            )
            _add_matrix_update_metrics(
                metrics,
                path,
                old_weight,
                new_weight,
                capture["input"],
                capture["output_fro"],
                jnp.asarray(capture["input"], dtype=jnp.float32) @ delta_weight,
            )

    if getattr(old_model.out_ln, "scale", None) is not None:
        path = "out_ln.scale"
        _add_scale_update_metrics(
            metrics,
            path,
            old_model.out_ln.scale.value,
            new_model.out_ln.scale.value,
            captures[path],
        )

    if old_model.lm_head_oblique_learn_target_rms:
        path = "lm_head_oblique_target_rms_log"
        capture = captures[path]
        old_log_scale = jnp.asarray(
            old_model.lm_head_oblique_target_rms_log.value, dtype=jnp.float32
        )
        new_log_scale = jnp.asarray(
            new_model.lm_head_oblique_target_rms_log.value, dtype=jnp.float32
        )
        old_scale = jnp.exp(old_log_scale)
        new_scale = jnp.exp(new_log_scale)
        delta_param = new_log_scale - old_log_scale
        delta_output = jnp.asarray(capture["input"], dtype=jnp.float32) * (
            new_scale - old_scale
        )
        prefix = f"parameter_update/{path}"
        metrics[f"{prefix}/delta_param_fro"] = _frobenius_norm(delta_param)
        metrics[f"{prefix}/delta_param_rel_fro"] = _safe_ratio(
            _frobenius_norm(delta_param), _frobenius_norm(old_log_scale)
        )
        metrics[f"{prefix}/input_fro"] = _frobenius_norm(capture["input"])
        metrics[f"{prefix}/delta_output_fro"] = _frobenius_norm(delta_output)
        metrics[f"{prefix}/delta_output_rel_fro"] = _safe_ratio(
            _frobenius_norm(delta_output), capture["output_fro"]
        )
        metrics[f"{prefix}/delta_param_rel_spectral"] = _safe_ratio(
            jnp.abs(new_scale - old_scale), jnp.abs(old_scale)
        )

    path = "token_embed_out.embedding"
    capture = captures[path]
    old_embedding = old_model.token_embed_out.embedding.value
    new_embedding = new_model.token_embed_out.embedding.value
    old_matrix = jnp.asarray(old_embedding, dtype=jnp.float32).T
    new_matrix = jnp.asarray(new_embedding, dtype=jnp.float32).T
    delta_matrix = new_matrix - old_matrix
    old_output_norm = _linear_output_fro_from_grams(capture["input"], old_matrix)
    delta_output_norm = _linear_output_fro_from_grams(capture["input"], delta_matrix)
    _add_matrix_update_metrics(
        metrics,
        path,
        old_embedding,
        new_embedding,
        capture["input"],
        old_output_norm,
        old_matrix=old_matrix,
        new_matrix=new_matrix,
        delta_output_norm=delta_output_norm,
    )

    return jax.tree.map(jax.lax.stop_gradient, metrics)
