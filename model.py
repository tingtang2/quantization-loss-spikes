import math
import warnings
import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
from omegaconf.dictconfig import DictConfig
from rope import apply_rope
from omegaconf import ListConfig


def _frobenius_norm(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(x)))


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _lm_head_dot_with_fp32_input_grad(query, embedding, precision):
    """Mixed-precision LM-head dot with an FP32 input-gradient matmul."""
    return jnp.dot(query, embedding.T, precision=precision)


def _lm_head_dot_with_fp32_input_grad_fwd(query, embedding, precision):
    logits = jnp.dot(query, embedding.T, precision=precision)
    return logits, (query, embedding)


def _lm_head_dot_with_fp32_input_grad_bwd(precision, residuals, grad_logits):
    query, embedding = residuals

    # This is the only intentionally promoted part of the LM-head backward
    # pass. With a vocabulary-sharded head, any reduction of these partial
    # input gradients is consequently also FP32.
    grad_query = jnp.dot(
        grad_logits.astype(jnp.float32),
        embedding.astype(jnp.float32),
        precision=precision,
    )

    # Preserve the ordinary mixed-precision weight-gradient computation.
    flat_grad_logits = grad_logits.reshape((-1, grad_logits.shape[-1]))
    flat_query = query.reshape((-1, query.shape[-1]))
    grad_embedding = jnp.dot(flat_grad_logits.T, flat_query, precision=precision)
    return grad_query, grad_embedding


_lm_head_dot_with_fp32_input_grad.defvjp(
    _lm_head_dot_with_fp32_input_grad_fwd,
    _lm_head_dot_with_fp32_input_grad_bwd,
)


def _rmsnorm_operator_input(norm: nnx.RMSNorm, x):
    """Returns the normalized activation to which RMSNorm's scale is applied."""
    x_f32 = jnp.asarray(x, dtype=jnp.float32)
    mean_square = jnp.mean(jnp.square(x_f32), axis=norm.reduction_axes, keepdims=True)
    return x_f32 * jax.lax.rsqrt(mean_square + jnp.asarray(norm.epsilon, dtype=jnp.float32))


def _apply_rmsnorm_with_capture(norm, x, captures, path):
    y = norm(x)
    if captures is not None and getattr(norm, "scale", None) is not None:
        captures[path] = {
            "input": _rmsnorm_operator_input(norm, x),
            "raw_input_fro": _frobenius_norm(x),
            "output_fro": _frobenius_norm(y),
        }
    return y


def _symmetric_uniform(bound: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def _megatron_init_stds(c: DictConfig) -> tuple[float, float] | None:
    """Return the input and residual projection stds for Megatron init."""
    if not bool(getattr(c, "megatron_init", False)):
        return None

    init_base_std = float(getattr(c, "init_base_std", 0.01))
    if init_base_std <= 0.0:
        raise ValueError(f"Expected `model.init_base_std` to be positive, got {init_base_std}.")

    residual_std = init_base_std / math.sqrt(2.0 * c.L)
    return init_base_std, residual_std


def resolve_out_ln_use_scale(c: DictConfig) -> bool:
    """Resolves the final RMSNorm scale override, defaulting to the global setting."""
    out_ln_use_scale = getattr(c, "out_ln_use_scale", None)
    if out_ln_use_scale is None:
        out_ln_use_scale = getattr(c, "rmsnorm_use_scale", False)
    return bool(out_ln_use_scale)


class TransformerDecoder(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        lm_head_dtype = getattr(c, "lm_head_dtype", c.activ_dtype)
        lm_head_param_dtype = getattr(c, "lm_head_param_dtype", jnp.float32)
        self.lm_head_is_fp32 = jnp.dtype(lm_head_dtype) == jnp.dtype(jnp.float32)
        self.lm_head_matmul_precision = (
            jax.lax.Precision.HIGHEST if self.lm_head_is_fp32 else None
        )
        self.lm_head_input_grad_fp32 = bool(getattr(c, "lm_head_input_grad_fp32", False))
        self.final_hidden_mean_centering = bool(getattr(c, "final_hidden_mean_centering", False))
        self.final_hidden_mean_centering_coeff = float(getattr(c, "alpha", 1.0))
        self.lm_head_oblique_learn_target_rms = bool(getattr(c, "lm_head_oblique_learn_target_rms", False))
        self.lm_head_oblique_initial_target_rms_from_random_init = bool(
            getattr(c, "lm_head_oblique_initial_target_rms_from_random_init", False)
        )
        self.lm_head_oblique_optimizer_type = str(getattr(c, "lm_head_oblique_optimizer_type", "row_oblique")).lower()
        megatron_stds = _megatron_init_stds(c)
        input_embedding_init = {} if megatron_stds is None else {
            "embedding_init": jax.nn.initializers.normal(megatron_stds[0])
        }
        # Lingua's Megatron path leaves its untied LM head at PyTorch Linear's
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) initialization.
        output_embedding_init = {} if megatron_stds is None else {
            "embedding_init": _symmetric_uniform(c.D ** -0.5)
        }
        self.token_embed_in = nnx.Embed(
            num_embeddings=c.V,
            features=c.D,
            dtype=c.activ_dtype,
            **input_embedding_init,
            rngs=rngs,
        )
        self.token_embed_out = nnx.Embed(
            num_embeddings=c.V,
            features=c.D,
            dtype=lm_head_dtype,
            param_dtype=lm_head_param_dtype,
            **output_embedding_init,
            rngs=rngs,
        )
        if self.lm_head_oblique_learn_target_rms:
            if self.lm_head_oblique_initial_target_rms_from_random_init:
                output_embedding = jnp.asarray(self.token_embed_out.embedding.value, dtype=jnp.float32)
                if self.lm_head_oblique_optimizer_type == "column_oblique":
                    col_rms = jnp.sqrt(jnp.mean(jnp.square(output_embedding), axis=0))
                    initial_target_rms = jnp.mean(col_rms)
                else:
                    row_rms = jnp.sqrt(jnp.mean(jnp.square(output_embedding), axis=1))
                    initial_target_rms = jnp.mean(row_rms)
            else:
                initial_target_rms = float(getattr(c, "lm_head_oblique_initial_target_rms", 1.0))
                if initial_target_rms <= 0.0:
                    raise ValueError(
                        f"Expected `model.lm_head_oblique_initial_target_rms` to be positive, got {initial_target_rms}."
                    )
            self.lm_head_oblique_target_rms_log = nnx.Param(
                jnp.asarray(jnp.log(initial_target_rms), dtype=jnp.float32)
            )
        self.blocks = nnx.List(TransformerBlock(c, rngs, layer_idx=i) for i in range(c.L))
        self.out_ln = nnx.RMSNorm(
            c.D,
            use_scale=resolve_out_ln_use_scale(c),
            dtype=lm_head_dtype,
            rngs=rngs,
        )
        
    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        return_update_inputs: bool = False,
        skip_output_logits: bool = False,
        return_final_norm_input: bool = False,
        return_final_norm_input_with_logits: bool = False,
    ): # [B, S]
        qkv_outputs = {}
        update_inputs = {} if return_update_inputs else None
        # token embedding
        h = self.token_embed_in(x) # [B, T, D]
        if update_inputs is not None:
            update_inputs["token_embed_in.embedding"] = {
                "input": x,
                "output_fro": _frobenius_norm(h),
            }

        # transformer blocks
        for i, block in enumerate(self.blocks):
            if return_qkv:
                h, qkv = block(
                    h,
                    attention_mask=attention_mask,
                    return_qkv=True,
                    update_inputs=update_inputs,
                    metric_prefix=f"blocks.{i}",
                )
                qkv_outputs[i] = qkv
            else:
                h = block(
                    h,
                    attention_mask=attention_mask,
                    update_inputs=update_inputs,
                    metric_prefix=f"blocks.{i}",
                )

        if return_final_norm_input:
            if return_qkv or return_update_inputs or skip_output_logits or return_final_norm_input_with_logits:
                raise ValueError(
                    "return_final_norm_input cannot be combined with return_qkv, "
                    "return_update_inputs, skip_output_logits, or return_final_norm_input_with_logits."
                )
            return h

        if return_final_norm_input_with_logits and (return_qkv or return_update_inputs or skip_output_logits):
            raise ValueError(
                "return_final_norm_input_with_logits cannot be combined with return_qkv, "
                "return_update_inputs, or skip_output_logits."
            )
        final_norm_input = h if return_final_norm_input_with_logits else None

        # project back to vocabulary
        h = _apply_rmsnorm_with_capture(self.out_ln, h, update_inputs, "out_ln.scale")
        if self.final_hidden_mean_centering:
            h = h - self.final_hidden_mean_centering_coeff * jnp.mean(h, axis=-1, keepdims=True)
        if self.lm_head_oblique_learn_target_rms:
            unscaled_h = h
            target_rms = jnp.exp(jnp.asarray(self.lm_head_oblique_target_rms_log.value, dtype=h.dtype))
            h = h * target_rms
            if update_inputs is not None:
                update_inputs["lm_head_oblique_target_rms_log"] = {
                    "input": unscaled_h,
                    "output_fro": _frobenius_norm(h),
                }
        if update_inputs is not None:
            update_inputs["token_embed_out.embedding"] = {"input": h}

        if skip_output_logits:
            if not return_update_inputs:
                raise ValueError("skip_output_logits requires return_update_inputs=True.")
            if return_qkv:
                return h, qkv_outputs, update_inputs
            return h, update_inputs

        if self.lm_head_input_grad_fp32:
            query, embedding = self.token_embed_out.promote_dtype(
                (h, self.token_embed_out.embedding.value),
                dtype=self.token_embed_out.dtype,
            )
            logits = _lm_head_dot_with_fp32_input_grad(
                query,
                embedding,
                self.lm_head_matmul_precision,
            )
        elif self.lm_head_is_fp32:
            query, embedding = self.token_embed_out.promote_dtype(
                (h, self.token_embed_out.embedding.value),
                dtype=self.token_embed_out.dtype,
            )
            logits = jnp.dot(
                query,
                embedding.T,
                precision=self.lm_head_matmul_precision,
            )
        else:
            logits = self.token_embed_out.attend(h)
        if update_inputs is not None:
            update_inputs["token_embed_out.embedding"]["output_fro"] = _frobenius_norm(logits)

        if return_qkv and return_update_inputs:
            return logits, qkv_outputs, update_inputs
        if return_qkv:
            return logits, qkv_outputs
        if return_update_inputs:
            return logits, update_inputs
        if return_final_norm_input_with_logits:
            return logits, final_norm_input

        return logits


def center_output_embeddings(model: TransformerDecoder):
    embeddings = model.token_embed_out.embedding.value
    mean = jnp.mean(embeddings.astype(jnp.float32), axis=0, keepdims=True).astype(embeddings.dtype)
    model.token_embed_out.embedding.value = embeddings - mean


def amplify_output_embedding_mean(model: TransformerDecoder, coefficient):
    """Adds a scaled vocabulary mean to every LM-head embedding row."""
    embeddings = model.token_embed_out.embedding.value
    mean = jnp.mean(embeddings.astype(jnp.float32), axis=0, keepdims=True)
    scaled_mean = (jnp.asarray(coefficient, dtype=jnp.float32) * mean).astype(
        embeddings.dtype
    )
    model.token_embed_out.embedding.value = embeddings + scaled_mean


def get_average_output_embedding_row_rms(model: TransformerDecoder):
    embeddings = jnp.asarray(model.token_embed_out.embedding.value, dtype=jnp.float32)
    row_rms = jnp.sqrt(jnp.mean(jnp.square(embeddings), axis=1))
    return float(jnp.mean(row_rms))


def get_average_output_embedding_col_rms(model: TransformerDecoder):
    embeddings = jnp.asarray(model.token_embed_out.embedding.value, dtype=jnp.float32)
    col_rms = jnp.sqrt(jnp.mean(jnp.square(embeddings), axis=0))
    return float(jnp.mean(col_rms))


def row_normalize_output_embeddings(model: TransformerDecoder, target_rms: float = 1.0, eps: float = 1e-8):
    embeddings = model.token_embed_out.embedding.value
    embeddings_f32 = embeddings.astype(jnp.float32)
    row_norms = jnp.linalg.norm(embeddings_f32, axis=1, keepdims=True)
    safe_row_norms = jnp.maximum(row_norms, jnp.asarray(eps, dtype=jnp.float32))
    target_norm = jnp.asarray(target_rms, dtype=jnp.float32) * jnp.sqrt(
        jnp.asarray(embeddings.shape[1], dtype=jnp.float32)
    )
    model.token_embed_out.embedding.value = (embeddings_f32 * (target_norm / safe_row_norms)).astype(
        embeddings.dtype
    )


def column_normalize_output_embeddings(model: TransformerDecoder, target_rms: float = 1.0, eps: float = 1e-8):
    embeddings = model.token_embed_out.embedding.value
    embeddings_f32 = embeddings.astype(jnp.float32)
    col_norms = jnp.linalg.norm(embeddings_f32, axis=0, keepdims=True)
    safe_col_norms = jnp.maximum(col_norms, jnp.asarray(eps, dtype=jnp.float32))
    target_norm = jnp.asarray(target_rms, dtype=jnp.float32) * jnp.sqrt(
        jnp.asarray(embeddings.shape[0], dtype=jnp.float32)
    )
    model.token_embed_out.embedding.value = (embeddings_f32 * (target_norm / safe_col_norms)).astype(
        embeddings.dtype
    )


class TransformerBlock(nnx.Module):
    def __init__(self, c: DictConfig, rngs: nnx.Rngs, layer_idx: int):
        rmsnorm_use_scale = bool(getattr(c, "rmsnorm_use_scale", False))
        self.ln1 = nnx.RMSNorm(c.D, use_scale=rmsnorm_use_scale, dtype=c.activ_dtype, rngs=rngs)
        self.ln2 = nnx.RMSNorm(c.D, use_scale=rmsnorm_use_scale, dtype=c.activ_dtype, rngs=rngs)
        
        qk_config = c.use_qk_norm
        
        if isinstance(qk_config, (list, tuple, ListConfig)):
            use_qk_norm = layer_idx in qk_config
        else:
            use_qk_norm = bool(qk_config)
        
        self.attn = MultiHeadAttention(c, rngs, use_qk_norm=use_qk_norm)
        self.mlp = MLP(c, rngs)
        
    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        update_inputs=None,
        metric_prefix: str = "",
    ): # [B, T, D]
        ln1_out = _apply_rmsnorm_with_capture(self.ln1, x, update_inputs, f"{metric_prefix}.ln1.scale")
        if return_qkv:
            attn_out, qkv = self.attn(
                ln1_out,
                attention_mask=attention_mask,
                return_qkv=True,
                update_inputs=update_inputs,
                metric_prefix=f"{metric_prefix}.attn",
            )
            x = x + attn_out
        else:
            x = x + self.attn(
                ln1_out,
                attention_mask=attention_mask,
                update_inputs=update_inputs,
                metric_prefix=f"{metric_prefix}.attn",
            )
        
        ln2_out = _apply_rmsnorm_with_capture(self.ln2, x, update_inputs, f"{metric_prefix}.ln2.scale")
        x = x + self.mlp(ln2_out, update_inputs=update_inputs, metric_prefix=f"{metric_prefix}.mlp")

        if return_qkv:
            return x, qkv

        return x


class MultiHeadAttention(nnx.Module):
    """Causal attention layer."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs, use_qk_norm: bool = None):
        rmsnorm_use_scale = bool(getattr(c, "rmsnorm_use_scale", False))
        qk_norm_use_scale = rmsnorm_use_scale and not bool(
            getattr(c, "qk_norm_disable_scale", False)
        )
        megatron_stds = _megatron_init_stds(c)
        qkv_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[0])
        }
        out_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[1])
        }
        self.qkv_proj = nnx.Einsum(
            'BTd,SNdH->SBTNH',
            (3, c.N, c.D, c.H),
            dtype=c.activ_dtype,
            **qkv_init,
            rngs=rngs,
        )
        self.out_proj = nnx.Einsum(
            'BTnh,nhD->BTD',
            (c.N, c.H, c.D),
            dtype=c.activ_dtype,
            **out_init,
            rngs=rngs,
        )
        self.elementwise_attn_output_gate = bool(getattr(c, "elementwise_attn_output_gate", False))
        if self.elementwise_attn_output_gate:
            self.gate_proj = nnx.Einsum(
                'BTd,NdH->BTNH',
                (c.N, c.D, c.H),
                dtype=c.activ_dtype,
                **qkv_init,
                rngs=rngs,
            )
        
        if use_qk_norm is None:
            use_qk_norm = c.use_qk_norm
        
        self.query_norm = nnx.RMSNorm(c.H, use_scale=qk_norm_use_scale, dtype=c.activ_dtype, rngs=rngs) if use_qk_norm else nnx.identity
        self.key_norm = nnx.RMSNorm(c.H, use_scale=qk_norm_use_scale, dtype=c.activ_dtype, rngs=rngs) if use_qk_norm else nnx.identity
        platform = jax.devices()[0].platform
        if c.use_flash_attn and platform == 'tpu' and (c.H % 128 != 0):
            warnings.warn('cannot use flash attention because `model.H` is not a multiple of 128.')
        self.use_tpu_flash_attn = bool(c.use_flash_attn and platform == 'tpu' and (c.H % 128 == 0))
        # cuDNN's causal attention is the fused/FlashAttention-style path for
        # supported CUDA shapes. It cannot consume the arbitrary padding mask
        # handled by the general fallback below.
        self.use_gpu_cudnn_attn = bool(c.use_flash_attn and platform == 'gpu')
        self.attention = (
            partial(tpu_causal_flash_attention)
            if self.use_tpu_flash_attn
            else partial(jax.nn.dot_product_attention, is_causal=True, implementation='cudnn')
            if self.use_gpu_cudnn_attn
            else partial(jax.nn.dot_product_attention, is_causal=False)
        )
        self.fallback_attention = partial(jax.nn.dot_product_attention, is_causal=False)

    def __call__(
        self,
        x,
        attention_mask: jax.Array | None = None,
        return_qkv: bool = False,
        update_inputs=None,
        metric_prefix: str = "",
    ): # [B, T, D]
        B, T, D = x.shape

        # input projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv # [B, T, N, H]
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.qkv_proj.kernel"] = {
                "input": x,
                "output_fro": _frobenius_norm(qkv),
            }
        gate_score = self.gate_proj(x) if self.elementwise_attn_output_gate else None
        if update_inputs is not None and gate_score is not None:
            update_inputs[f"{metric_prefix}.gate_proj.kernel"] = {
                "input": x,
                "output_fro": _frobenius_norm(gate_score),
            }
        if return_qkv:
            raw_qkv = (q, k, v)

        # qk-norm
        if isinstance(self.query_norm, nnx.RMSNorm):
            q = _apply_rmsnorm_with_capture(
                self.query_norm, q, update_inputs, f"{metric_prefix}.query_norm.scale"
            )
        else:
            q = self.query_norm(q)
        if isinstance(self.key_norm, nnx.RMSNorm):
            k = _apply_rmsnorm_with_capture(
                self.key_norm, k, update_inputs, f"{metric_prefix}.key_norm.scale"
            )
        else:
            k = self.key_norm(k)

        # position embedding
        position = jnp.arange(T)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

        # attention
        if self.use_tpu_flash_attn or (self.use_gpu_cudnn_attn and attention_mask is None):
            out = self.attention(q, k, v) # [B, T, N, H]
        else:
             # 1. Create causal mask (allows attending to past)
            # Shape [1, 1, T, T]
            causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_)).reshape(1, 1, T, T)
            
            # 2. Create padding mask (if provided)
            if attention_mask is not None:
                # attention_mask is [B, T]. Reshape to [B, 1, 1, T] (for broadcasting)
                # This mask is 1 (True) for real tokens and 0 (False) for padding
                padding_mask = attention_mask.astype(jnp.bool_).reshape(B, 1, 1, T)
                
                # 3. Combine masks: Both must be True to attend.
                # causal_mask broadcasts to [B, 1, T, T]
                # padding_mask broadcasts to [B, 1, T, T]
                # combined_mask shape [B, 1, T, T]
                combined_mask = jnp.logical_and(causal_mask, padding_mask)
            else:
                combined_mask = causal_mask
            
            # The mask will be broadcast by dot_product_attention to [B, N, T, T]
            out = self.fallback_attention(q, k, v, mask=combined_mask) # [B, T, N, H]

        if self.elementwise_attn_output_gate:
            out = out * jax.nn.sigmoid(gate_score)

        # output projection followed by contraction back to original dims
        out_proj_input = out
        out = self.out_proj(out_proj_input) # [B, T, D]
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.out_proj.kernel"] = {
                "input": out_proj_input,
                "output_fro": _frobenius_norm(out),
            }
        if return_qkv:
            return out, raw_qkv
        return out


def tpu_causal_flash_attention(q, k, v):
    """
    TPU Flash Attention.
    https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py
    https://github.com/AI-Hypercomputer/maxtext/blob/9ea52118535e970096c164460dbbfa478d157066/MaxText/layers/attentions.py#L562
    """
    B, T, N, H = q.shape
    assert H >= 128, 'TPU flash attention reqruies head dim. to be a multiple of 128'

    # scale query
    q /= jnp.sqrt(H)

    # kernel block sizes
    # https://github.com/AI-Hypercomputer/maxtext/blob/afcdf47f8b7c1e1864fa81012a873590c5408122/MaxText/configs/base.yml#L644
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=512,
        block_kv=512,
        block_kv_compute=min(H, 256),
        block_q_dkv=512,
        block_kv_dkv=512,
        block_kv_dkv_compute=min(H, 256),
        block_q_dq=512,
        block_kv_dq=512,
    )

    mesh = jax.sharding.get_abstract_mesh()
    sharding = P('data', None, 'model', None)
    @partial(shard_map, mesh=mesh, in_specs=(sharding, sharding, sharding), out_specs=sharding, check_rep=False)
    def attention(q, k, v):
        _, _, n, _ = q.shape
        causal_mask = splash_attention_mask.CausalMask(shape=(T, T))
        multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(causal_mask,) * n)
        splash_kernel = splash_attention_kernel.make_splash_mha(mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
        out = jax.vmap(splash_kernel)(
            q.swapaxes(1, 2),
            k.swapaxes(1, 2),
            v.swapaxes(1, 2)
        ).swapaxes(1, 2) # [B, T, N, H]
        return out

    return attention(q, k, v)


class MLP(nnx.Module):
    """Multilayer perceptron."""
    def __init__(self, c: DictConfig, rngs: nnx.Rngs):
        megatron_stds = _megatron_init_stds(c)
        up_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[0])
        }
        down_init = {} if megatron_stds is None else {
            "kernel_init": jax.nn.initializers.normal(megatron_stds[1])
        }
        self.up_proj = nnx.Linear(
            in_features=c.D,
            out_features=c.F,
            use_bias=False,
            dtype=c.activ_dtype,
            **up_init,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=c.F,
            out_features=c.D,
            use_bias=False,
            dtype=c.activ_dtype,
            **down_init,
            rngs=rngs,
        )
        
    def __call__(self, x, update_inputs=None, metric_prefix: str = ""): # [B, T, D]
        up_output = self.up_proj(x)
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.up_proj.kernel"] = {
                "input": x,
                "output_fro": _frobenius_norm(up_output),
            }
        h = jax.nn.gelu(up_output) # [B, T, F]
        output = self.down_proj(h) # [B, T, D]
        if update_inputs is not None:
            update_inputs[f"{metric_prefix}.down_proj.kernel"] = {
                "input": h,
                "output_fro": _frobenius_norm(output),
            }
        return output


def create_sharded_model(c: DictConfig, key):
    seed = int(jax.random.randint(key, [1], 0, 1_000_000)[0])
    data_param_sharding = bool(getattr(c, 'data_param_sharding', True))

    @nnx.jit
    def initialize_sharded_model():
        rngs = nnx.Rngs(seed)
        model = TransformerDecoder(c, rngs=rngs) # unsharded at this moment
        state = nnx.state(model) # the model's state, a pure pytree

        def add_sharding(path, v):
            # GPT2M easily fits replicated on a GB200.  Keeping this alternate
            # layout makes it possible to benchmark conventional data parallel
            # training against the existing data-axis parameter sharding.
            if not data_param_sharding:
                return jax.lax.with_sharding_constraint(v, P())

            key = jax.tree_util.keystr(path, simple=True, separator='/')
            pspec = None
            if key == 'token_embed_in/embedding': pspec = P('data', 'model')
            if 'up_proj' in key: pspec = P('data', 'model')
            if 'down_proj' in key: pspec = P('model', 'data')
            if 'qkv_proj' in key: pspec = P(None, 'model', 'data', None)
            if 'gate_proj' in key: pspec = P('model', 'data', None)
            if 'out_proj' in key: pspec = P('model', None, 'data')
            if key == 'token_embed_out/embedding': pspec = P('model', 'data')
            if pspec is None:
                return v
            return jax.lax.with_sharding_constraint(v, pspec)
        state = jax.tree.map_with_path(add_sharding, state)
        nnx.update(model, state) # the model is sharded now
        
        return model

    model = initialize_sharded_model()

    return model
