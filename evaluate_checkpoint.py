import heapq
import hydra
import io
import jax
import jax.numpy as jnp
import math
import numpy as np
import optax
import orbax.checkpoint as ocp
import os
import time
from dataclasses import asdict
from etils import epath
from orbax.checkpoint._src.path import gcs_utils, step as step_lib
from omegaconf import DictConfig, OmegaConf
from flax import nnx
from functools import partial
import json

from analysis_records import (
    AnalysisReport,
    AnalysisSettingsRecord,
    AppliedSwapRecord,
    IterativeContextSwapResult,
    IterativeModeResult,
    IterativeStepRecord,
    TokenEventRecord,
    TopBatchRecord,
    TopEventRecord,
    TopLogitRecord,
    SwapAnalysisRecord,
)
from configs import resolver_setup
import data
import model as model_lib
import utils


class _StandardNameFormatHNS(step_lib._StandardNameFormat):
    """Fixes GCS HNS listing when step_prefix is None."""

    def _glob_step_paths(self, base_path: epath.PathLike) -> list[epath.Path]:
        base_path = epath.Path(base_path)
        if gcs_utils.is_hierarchical_namespace_enabled(base_path):
            bucket_name, path_prefix = gcs_utils.parse_gcs_path(base_path)
            bucket = gcs_utils.get_bucket(bucket_name)
            result = bucket.list_blobs(
                prefix=path_prefix,
                delimiter='/',
                include_folders_as_prefixes=True,
            )
            for _ in result.pages:
                pass
            step_prefix = self.step_prefix or ''
            return [
                epath.Path(f'gs://{bucket_name}/{folder}')
                for folder in result.prefixes
                if folder.startswith(os.path.join(path_prefix, step_prefix))
            ]
        return list(
            epath.Path(base_path).glob(
                f'{step_lib.step_prefix_with_underscore(self.step_prefix)}*'
            )
        )


@partial(jax.jit, static_argnames=('model_graphdef',))
def batch_loss_stats(model_state, model_graphdef, x):
    """Returns batch mean loss, per-example mean loss, and per-token loss [B, T-1]."""
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    token_losses = token_losses[:, :-1]
    per_example_loss = token_losses.mean(axis=1)
    batch_mean_loss = per_example_loss.mean()
    return batch_mean_loss, per_example_loss, token_losses


@partial(jax.jit, static_argnames=('model_graphdef',))
def forward_logits(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    return model(x).astype(jnp.float32)


def forward_logits_mesh_safe(model_state, model_graphdef, x, data_axis_size: int):
    """Pads batch dimension so it is divisible by mesh data axis, then slices back."""
    batch_size = int(x.shape[0])
    if batch_size % data_axis_size == 0:
        return forward_logits(model_state, model_graphdef, x)

    pad = data_axis_size - (batch_size % data_axis_size)
    x_padded = jnp.concatenate([x, jnp.repeat(x[-1:, :], pad, axis=0)], axis=0)
    logits_padded = forward_logits(model_state, model_graphdef, x_padded)
    return logits_padded[:batch_size]


def _cfg_get(c: DictConfig, path: str, default):
    node = c
    for part in path.split('.'):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def _decode_token(tokenizer, token_id: int):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(token_id)]).replace('\n', '\\n')
    except Exception:
        return None


def _decode_span(tokenizer, token_ids):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(t) for t in token_ids]).replace('\n', '\\n')
    except Exception:
        return None


def _push_topk(heap, item, k):
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _top_token_candidates(token_losses_np: np.ndarray, k: int):
    if k <= 0:
        return []
    flat = token_losses_np.reshape(-1)
    if flat.size == 0:
        return []
    kk = min(k, flat.size)
    idx = np.argpartition(flat, -kk)[-kk:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    t_minus_1 = token_losses_np.shape[1]
    out = []
    for i in idx.tolist():
        loss = float(flat[i])
        ex_idx = int(i // t_minus_1)
        pos = int(i % t_minus_1)
        out.append((loss, ex_idx, pos))
    return out


def _topk_logits(logits_vec: np.ndarray, top_k: int):
    top_k = max(1, min(int(top_k), logits_vec.shape[-1]))
    idx = np.argpartition(logits_vec, -top_k)[-top_k:]
    idx = idx[np.argsort(logits_vec[idx])[::-1]]
    return [(int(i), float(logits_vec[i])) for i in idx.tolist()]


def _get_tokenizer(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as e:
        print(f"Tokenizer unavailable ({name}): {e}. Continuing with token IDs only.")
        return None


def _numpy_dtype_from_name(name: str):
    normalized = str(name).lower()
    if normalized in ('float16', 'fp16', 'f16'):
        return np.float16
    if normalized in ('float32', 'fp32', 'f32'):
        return np.float32
    raise ValueError(f'Unsupported analysis.full_logits_dtype={name!r}. Use float16 or float32.')


def _loss_at_position_from_logits(logits_vec: np.ndarray, target_id: int) -> float:
    log_probs = np.asarray(jax.device_get(jax.nn.log_softmax(jnp.asarray(logits_vec))), dtype=np.float32)
    return float(-log_probs[int(target_id)])


def _eval_single_event_loss(
    model_state,
    model_graphdef,
    seq: np.ndarray,
    pos: int,
    target_id: int,
    data_axis_size: int,
):
    logits_vec = np.asarray(
        jax.device_get(
            forward_logits_mesh_safe(
                model_state,
                model_graphdef,
                jnp.asarray(seq[None, :], dtype=jnp.int32),
                data_axis_size=data_axis_size,
            )
        ),
        dtype=np.float32,
    )[0, pos]
    return _loss_at_position_from_logits(logits_vec, target_id), logits_vec


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c)

    run_name = c.run_name if c.run_name else 'picodo_run'
    split = str(_cfg_get(c, 'analysis.split', 'valid')).lower()
    top_batches = int(_cfg_get(c, 'analysis.top_batches', 10))
    top_token_events = int(_cfg_get(c, 'analysis.top_token_events', 50))
    context_window = int(_cfg_get(c, 'analysis.context_window', 32))
    top_logits_k = int(_cfg_get(c, 'analysis.top_logits_k', 10))
    tokenizer_name = str(_cfg_get(c, 'analysis.tokenizer_name', 'gpt2'))
    include_next_train_batch_analysis = bool(_cfg_get(c, 'analysis.include_next_train_batch_analysis', True))
    next_train_top_token_events = int(_cfg_get(c, 'analysis.next_train_top_token_events', top_token_events))
    save_full_logits = bool(_cfg_get(c, 'analysis.save_full_logits', False))
    full_logits_dtype_name = str(_cfg_get(c, 'analysis.full_logits_dtype', 'float16'))
    iterative_swap_enabled = bool(_cfg_get(c, 'analysis.iterative_context_swap_enabled', False))
    iterative_swap_token_id = int(_cfg_get(c, 'analysis.iterative_context_swap_token_id', 50256))
    iterative_swap_alternate_token_id = int(_cfg_get(c, 'analysis.iterative_context_swap_alternate_token_id', 0))
    iterative_swap_scope = str(_cfg_get(c, 'analysis.iterative_context_swap_scope', 'window')).lower()
    iterative_swap_mode = str(_cfg_get(c, 'analysis.iterative_context_swap_mode', 'cumulative')).lower()
    swap_relative_positions = [int(x) for x in _cfg_get(c, 'analysis.swap_relative_positions', [])]
    swap_replacement_token_ids = [int(x) for x in _cfg_get(c, 'analysis.swap_replacement_token_ids', [])]
    full_logits_dtype = _numpy_dtype_from_name(full_logits_dtype_name)

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    print("---------------------")

    if swap_relative_positions and len(swap_relative_positions) != len(swap_replacement_token_ids):
        raise ValueError(
            'analysis.swap_relative_positions and analysis.swap_replacement_token_ids must have same length.'
        )

    if any(rel > 0 for rel in swap_relative_positions):
        raise ValueError('Only non-positive relative positions are supported (must edit context, not target/future).')
    if iterative_swap_scope not in ('window', 'prefix'):
        raise ValueError("analysis.iterative_context_swap_scope must be one of: 'window', 'prefix'.")
    if iterative_swap_mode not in ('cumulative', 'non_cumulative'):
        raise ValueError(
            "analysis.iterative_context_swap_mode must be one of: 'cumulative', 'non_cumulative'."
        )

    jax.distributed.initialize()
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print(f"Sharding mesh: {mesh.shape} (data, model)")

    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    utils.sync_lm_head_oblique_model_config(c)
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    print(f"Model vocab size rounded to: {c.model.V}")

    print('Initializing model...')
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)

    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')

    print('Loading datasets...')
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (
            n_params['n_param_nonembed'] + n_params['n_param_embed']
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

    print(f"Loaded {len(ds_train)} training batches.")
    print(f"Loaded {len(ds_valid)} validation batches.")

    print('Initializing abstract optimizer structure for restore...')
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = utils.build_learning_rate_schedule(
        c.opt, c.opt.peak_lr, warmup_steps, num_opt_steps
    )
    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        c.opt.b2,
        eps=float(_cfg_get(c, 'opt.eps', 1e-8)),
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    tx, _ = utils.maybe_partition_rmsnorm_scale_sgd(model, c.opt, tx, lr_schedule)
    if c.opt.clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(c.opt.clip_by_global_norm), tx)
    optimizer = nnx.ModelAndOptimizer(model, tx)
    _, opt_state = nnx.split(optimizer)
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
    if gcp_bucket:
        if not gcp_bucket.startswith('gs://'):
            gcp_bucket = f'gs://{gcp_bucket}'
        ckpt_dir = f'{gcp_bucket.rstrip("/")}/{run_name}'
        print(f'Checkpoint source: {ckpt_dir}')
    else:
        ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)
        print(f'Checkpoint source: {ckpt_dir}')

    step_prefix = getattr(c.checkpoint, 'step_prefix', None)
    if gcp_bucket:
        name_format = _StandardNameFormatHNS(step_prefix=step_prefix)
        mngr_options = ocp.CheckpointManagerOptions(create=False, step_name_format=name_format)
    else:
        mngr_options = ocp.CheckpointManagerOptions(create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=mngr_options)

    step_to_load = (
        c.checkpoint.start_step
        if c.checkpoint.start_step is not None
        else ckpt_mngr.latest_step()
    )
    if step_to_load is None:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        return

    print(f"Restoring checkpoint from step {step_to_load}...")
    restored_data = ckpt_mngr.restore(
        step_to_load,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_opt_state),
            training_metadata=ocp.args.JsonRestore(),
        ),
    )
    opt_state = restored_data['state']
    training_metadata = restored_data.get('training_metadata', {})
    next_train_step = training_metadata.get('next_step')
    if next_train_step is None:
        next_train_step = int(step_to_load) + 1
    else:
        next_train_step = int(next_train_step)
    ckpt_mngr.close()
    model_state = opt_state.model
    print('Checkpoint restored successfully.')
    print(f'next_train_step={next_train_step}')

    if gcp_bucket:
        default_save_dir = f"{ckpt_dir.rstrip('/')}/evaluation_metrics/step_{step_to_load}"
    else:
        local_base = c.checkpoint.workdir if c.checkpoint.workdir else os.getcwd()
        default_save_dir = os.path.join(local_base, 'evaluation_metrics', run_name, f'step_{step_to_load}')
    save_dir = str(_cfg_get(c, 'analysis.save_dir', default_save_dir))
    save_dir_path = epath.Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    tokenizer = _get_tokenizer(tokenizer_name)

    if split == 'train':
        dataset = ds_train
    else:
        split = 'valid'
        dataset = ds_valid

    data_axis_size = int(mesh.shape['data'])

    print(
        f"Running eval-only analysis on {split} split with {len(dataset)} batches "
        f"(top_batches={top_batches}, top_token_events={top_token_events})"
    )

    top_batch_heap = []
    top_token_heap = []

    t0 = time.time()
    for batch_idx in range(len(dataset)):
        batch = dataset[batch_idx]
        batch_mean_loss, _, token_losses = batch_loss_stats(model_state, model_graphdef, batch)

        batch_mean_loss_f = float(jax.device_get(batch_mean_loss))
        token_losses_np = np.asarray(jax.device_get(token_losses), dtype=np.float32)
        max_token_loss_f = float(token_losses_np.max()) if token_losses_np.size > 0 else float('-inf')

        _push_topk(top_batch_heap, (batch_mean_loss_f, batch_idx, max_token_loss_f), top_batches)

        for loss, ex_idx, pos in _top_token_candidates(token_losses_np, top_token_events):
            _push_topk(top_token_heap, (loss, batch_idx, ex_idx, pos), top_token_events)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataset):
            elapsed = time.time() - t0
            print(f"Processed {batch_idx + 1}/{len(dataset)} batches in {elapsed:.1f}s")

    top_batches_sorted = sorted(top_batch_heap, key=lambda x: x[0], reverse=True)
    top_tokens_sorted = sorted(top_token_heap, key=lambda x: x[0], reverse=True)

    top_batch_records = []
    for rank, (mean_loss, batch_idx, max_token_loss) in enumerate(top_batches_sorted, start=1):
        top_batch_records.append(
            TopBatchRecord(
                rank=int(rank),
                batch_index=int(batch_idx),
                mean_loss=float(mean_loss),
                max_token_loss=float(max_token_loss),
            )
        )

    token_event_records = []
    full_logits_before = []
    full_logits_after = []
    full_logits_before_rank = []
    full_logits_after_rank = []
    full_logits_before_batch_idx = []
    full_logits_after_batch_idx = []
    full_logits_before_example_idx = []
    full_logits_after_example_idx = []
    full_logits_before_pos = []
    full_logits_after_pos = []
    for rank, (loss, batch_idx, ex_idx, pos) in enumerate(top_tokens_sorted, start=1):
        batch_np = np.asarray(jax.device_get(dataset[batch_idx]), dtype=np.int32)
        seq = batch_np[ex_idx]

        if pos + 1 >= seq.shape[0]:
            print(f'pos is either invalid or last token in sequence: {pos}')
            continue

        target_id = int(seq[pos + 1])
        input_token_id = int(seq[pos])

        base_loss, logits_np = _eval_single_event_loss(
            model_state=model_state,
            model_graphdef=model_graphdef,
            seq=seq,
            pos=pos,
            target_id=target_id,
            data_axis_size=data_axis_size,
        )

        top_logits = []
        for token_id, logit in _topk_logits(logits_np, top_logits_k):
            top_logits.append(
                TopLogitRecord(
                    token_id=int(token_id),
                    logit=float(logit),
                    token=_decode_token(tokenizer, token_id),
                )
            )

        context_start = max(0, pos - context_window + 1)
        context_ids = seq[context_start:pos + 1].tolist()

        record = TokenEventRecord(
            rank=int(rank),
            loss=float(loss),
            recomputed_loss=float(base_loss),
            batch_index=int(batch_idx),
            example_index=int(ex_idx),
            position=int(pos),
            input_token_id=int(input_token_id),
            input_token=_decode_token(tokenizer, input_token_id),
            target_token_id=int(target_id),
            target_token=_decode_token(tokenizer, target_id),
            context_start=int(context_start),
            context_ids=[int(x) for x in context_ids],
            context_text=_decode_span(tokenizer, context_ids),
            top_logits=top_logits,
        )

        if save_full_logits:
            full_logits_before.append(logits_np.astype(full_logits_dtype))
            full_logits_before_rank.append(int(rank))
            full_logits_before_batch_idx.append(int(batch_idx))
            full_logits_before_example_idx.append(int(ex_idx))
            full_logits_before_pos.append(int(pos))

        if swap_relative_positions:
            edited_seq = seq.copy()
            applied_swaps = []
            for rel, new_token in zip(swap_relative_positions, swap_replacement_token_ids):
                token_pos = pos + rel
                if token_pos < 0 or token_pos >= edited_seq.shape[0]:
                    print(f'swap pos is invalid: {token_pos}')
                    continue
                old_token = int(edited_seq[token_pos])
                edited_seq[token_pos] = int(new_token)
                applied_swaps.append(
                    AppliedSwapRecord(
                        relative_position=int(rel),
                        absolute_position=int(token_pos),
                        old_token_id=int(old_token),
                        old_token=_decode_token(tokenizer, old_token),
                        new_token_id=int(new_token),
                        new_token=_decode_token(tokenizer, int(new_token)),
                    )
                )

            if applied_swaps:
                edited_logits_np = np.asarray(
                    _eval_single_event_loss(
                        model_state=model_state,
                        model_graphdef=model_graphdef,
                        seq=edited_seq,
                        pos=pos,
                        target_id=target_id,
                        data_axis_size=data_axis_size,
                    )[1]
                )
                edited_loss = _loss_at_position_from_logits(edited_logits_np, target_id)

                edited_top_logits = []
                for token_id, logit in _topk_logits(edited_logits_np, top_logits_k):
                    edited_top_logits.append(
                        TopLogitRecord(
                            token_id=int(token_id),
                            logit=float(logit),
                            token=_decode_token(tokenizer, token_id),
                        )
                    )

                record.swap_analysis = SwapAnalysisRecord(
                    applied_swaps=applied_swaps,
                    loss_after_swap=float(edited_loss),
                    loss_delta=float(edited_loss - base_loss),
                    target_logit_before=float(logits_np[target_id]),
                    target_logit_after=float(edited_logits_np[target_id]),
                    target_logit_delta=float(edited_logits_np[target_id] - logits_np[target_id]),
                    top_logits_after_swap=edited_top_logits,
                )
                if save_full_logits:
                    full_logits_after.append(edited_logits_np.astype(full_logits_dtype))
                    full_logits_after_rank.append(int(rank))
                    full_logits_after_batch_idx.append(int(batch_idx))
                    full_logits_after_example_idx.append(int(ex_idx))
                    full_logits_after_pos.append(int(pos))

        token_event_records.append(record)

    next_train_batch_analysis = None
    if include_next_train_batch_analysis:
        if 0 <= next_train_step < len(ds_train):
            train_batch = ds_train[next_train_step]
            train_batch_mean_loss, train_per_example_loss, train_token_losses = batch_loss_stats(
                model_state, model_graphdef, train_batch
            )
            train_batch_np = np.asarray(jax.device_get(train_batch), dtype=np.int32)
            train_token_losses_np = np.asarray(jax.device_get(train_token_losses), dtype=np.float32)
            train_top_candidates = _top_token_candidates(train_token_losses_np, next_train_top_token_events)

            train_top_event_records = []
            for rank, (loss, ex_idx, pos) in enumerate(train_top_candidates, start=1):
                seq = train_batch_np[ex_idx]
                if pos + 1 >= seq.shape[0]:
                    continue
                target_id = int(seq[pos + 1])
                input_token_id = int(seq[pos])
                base_loss, logits_np = _eval_single_event_loss(
                    model_state=model_state,
                    model_graphdef=model_graphdef,
                    seq=seq,
                    pos=pos,
                    target_id=target_id,
                    data_axis_size=data_axis_size,
                )
                top_logits = []
                for token_id, logit in _topk_logits(logits_np, top_logits_k):
                    top_logits.append(
                        TopLogitRecord(
                            token_id=int(token_id),
                            logit=float(logit),
                            token=_decode_token(tokenizer, token_id),
                        )
                    )
                context_start = max(0, pos - context_window + 1)
                context_ids = seq[context_start:pos + 1].tolist()
                train_top_event_records.append(
                    TokenEventRecord(
                        rank=int(rank),
                        loss=float(loss),
                        recomputed_loss=float(base_loss),
                        batch_index=int(next_train_step),
                        example_index=int(ex_idx),
                        position=int(pos),
                        input_token_id=int(input_token_id),
                        input_token=_decode_token(tokenizer, input_token_id),
                        target_token_id=int(target_id),
                        target_token=_decode_token(tokenizer, target_id),
                        context_start=int(context_start),
                        context_ids=[int(x) for x in context_ids],
                        context_text=_decode_span(tokenizer, context_ids),
                        top_logits=top_logits,
                    )
                )

            next_train_batch_analysis = {
                'train_step': int(next_train_step),
                'batch_shape': [int(x) for x in train_batch_np.shape],
                'batch_mean_loss': float(jax.device_get(train_batch_mean_loss)),
                'batch_med_example_loss': float(jax.device_get(jnp.median(train_per_example_loss))),
                'batch_lower_90th_mean_loss': float(
                    jax.device_get(utils.compute_lower_90th_percentile_mean(train_token_losses))
                ),
                'batch_max_token_loss': float(train_token_losses_np.max()) if train_token_losses_np.size else float('-inf'),
                'top_token_events': [asdict(x) for x in train_top_event_records],
            }
        else:
            next_train_batch_analysis = {
                'train_step': int(next_train_step),
                'error': f'next_train_step out of range for ds_train len={len(ds_train)}',
            }

    iterative_context_swap_result = None
    if iterative_swap_enabled and top_tokens_sorted:
        top_loss, top_batch_idx, top_ex_idx, top_pos = top_tokens_sorted[0]
        top_batch_np = np.asarray(jax.device_get(dataset[top_batch_idx]), dtype=np.int32)
        top_seq = top_batch_np[top_ex_idx].copy()
        top_target_id = int(top_seq[top_pos + 1])

        if iterative_swap_scope == 'prefix':
            context_positions = list(range(0, top_pos + 1))
        else:
            ctx_start = max(0, top_pos - context_window + 1)
            context_positions = list(range(ctx_start, top_pos + 1))

        base_event_loss, _ = _eval_single_event_loss(
            model_state=model_state,
            model_graphdef=model_graphdef,
            seq=top_seq,
            pos=top_pos,
            target_id=top_target_id,
            data_axis_size=data_axis_size,
        )

        def _run_iterative_mode(mode_name: str):
            edited_seq_local = top_seq.copy()
            prev_loss_local = base_event_loss
            steps_local = []
            for i, token_pos in enumerate(context_positions, start=1):
                if mode_name == 'cumulative':
                    candidate_seq = edited_seq_local
                else:
                    candidate_seq = top_seq.copy()

                old_token_id = int(candidate_seq[token_pos])
                new_token_id = (
                    iterative_swap_token_id
                    if old_token_id != iterative_swap_token_id
                    else iterative_swap_alternate_token_id
                )
                candidate_seq[token_pos] = new_token_id
                new_loss, _ = _eval_single_event_loss(
                    model_state=model_state,
                    model_graphdef=model_graphdef,
                    seq=candidate_seq,
                    pos=top_pos,
                    target_id=top_target_id,
                    data_axis_size=data_axis_size,
                )
                step_record = IterativeStepRecord(
                    step=int(i),
                    token_position=int(token_pos),
                    old_token_id=int(old_token_id),
                    old_token=_decode_token(tokenizer, old_token_id),
                    new_token_id=int(new_token_id),
                    new_token=_decode_token(tokenizer, int(new_token_id)),
                    loss_before_step=float(prev_loss_local if mode_name == 'cumulative' else base_event_loss),
                    loss_after_step=float(new_loss),
                    loss_delta_vs_prev=float(new_loss - prev_loss_local),
                    loss_delta_vs_base=float(new_loss - base_event_loss),
                    went_down_vs_prev=bool(new_loss <= prev_loss_local),
                    went_down_vs_base=bool(new_loss <= base_event_loss),
                )
                steps_local.append(step_record)
                if mode_name == 'cumulative':
                    edited_seq_local = candidate_seq
                prev_loss_local = new_loss

            monotonic_nonincreasing_local = all(s.went_down_vs_prev for s in steps_local)
            always_down_vs_base_local = all(s.went_down_vs_base for s in steps_local)
            return IterativeModeResult(
                mode=mode_name,
                final_loss=float(prev_loss_local),
                monotonic_nonincreasing_loss=bool(monotonic_nonincreasing_local),
                always_down_vs_base_loss=bool(always_down_vs_base_local),
                steps=steps_local,
            )

        cumulative_result = _run_iterative_mode('cumulative')
        non_cumulative_result = _run_iterative_mode('non_cumulative')
        selected_mode_result = cumulative_result if iterative_swap_mode == 'cumulative' else non_cumulative_result
        iterative_context_swap_result = IterativeContextSwapResult(
            enabled=True,
            scope=iterative_swap_scope,
            mode=iterative_swap_mode,
            replacement_token_id=int(iterative_swap_token_id),
            alternate_replacement_token_id=int(iterative_swap_alternate_token_id),
            top_event=TopEventRecord(
                rank=1,
                loss=float(top_loss),
                batch_index=int(top_batch_idx),
                example_index=int(top_ex_idx),
                position=int(top_pos),
                target_token_id=int(top_target_id),
                target_token=_decode_token(tokenizer, int(top_target_id)),
            ),
            context_positions=[int(p) for p in context_positions],
            base_loss=float(base_event_loss),
            final_loss=float(selected_mode_result.final_loss),
            monotonic_nonincreasing_loss=bool(selected_mode_result.monotonic_nonincreasing_loss),
            always_down_vs_base_loss=bool(selected_mode_result.always_down_vs_base_loss),
            steps=selected_mode_result.steps,
            modes={
                'cumulative': cumulative_result,
                'non_cumulative': non_cumulative_result,
            },
        )

    report = AnalysisReport(
        run_name=run_name,
        checkpoint_step=int(step_to_load),
        checkpoint_dir=ckpt_dir,
        split=split,
        num_batches=int(len(dataset)),
        analysis=AnalysisSettingsRecord(
            top_batches=int(top_batches),
            top_token_events=int(top_token_events),
            context_window=int(context_window),
            top_logits_k=int(top_logits_k),
            tokenizer_name=tokenizer_name,
            save_full_logits=bool(save_full_logits),
            full_logits_dtype=full_logits_dtype_name,
            iterative_context_swap_enabled=bool(iterative_swap_enabled),
            iterative_context_swap_token_id=int(iterative_swap_token_id),
            iterative_context_swap_alternate_token_id=int(iterative_swap_alternate_token_id),
            iterative_context_swap_scope=iterative_swap_scope,
            iterative_context_swap_mode=iterative_swap_mode,
            swap_relative_positions=[int(x) for x in swap_relative_positions],
            swap_replacement_token_ids=[int(x) for x in swap_replacement_token_ids],
        ),
        top_batches_by_mean_loss=top_batch_records,
        top_token_events=token_event_records,
        iterative_context_swap_top_event=iterative_context_swap_result,
    )

    report_path = save_dir_path / f'loss_trigger_analysis_step_{step_to_load}_{split}.json'
    with report_path.open('w') as f:
        report_payload = asdict(report)
        report_payload['analysis']['include_next_train_batch_analysis'] = bool(include_next_train_batch_analysis)
        report_payload['analysis']['next_train_top_token_events'] = int(next_train_top_token_events)
        report_payload['analysis']['next_train_step'] = int(next_train_step)
        if iterative_context_swap_result is None:
            report_payload.pop('iterative_context_swap_top_event', None)
        if next_train_batch_analysis is not None:
            report_payload['next_train_batch_analysis'] = next_train_batch_analysis
        json.dump(report_payload, f, indent=2)

    full_logits_path = None
    if save_full_logits:
        full_logits_path = save_dir_path / f'loss_trigger_full_logits_step_{step_to_load}_{split}.npz'
        npz_payload = {
            'before_logits': np.stack(full_logits_before, axis=0)
            if full_logits_before
            else np.zeros((0, c.model.V), dtype=full_logits_dtype),
            'before_rank': np.asarray(full_logits_before_rank, dtype=np.int32),
            'before_batch_index': np.asarray(full_logits_before_batch_idx, dtype=np.int32),
            'before_example_index': np.asarray(full_logits_before_example_idx, dtype=np.int32),
            'before_position': np.asarray(full_logits_before_pos, dtype=np.int32),
            'after_logits': np.stack(full_logits_after, axis=0)
            if full_logits_after
            else np.zeros((0, c.model.V), dtype=full_logits_dtype),
            'after_rank': np.asarray(full_logits_after_rank, dtype=np.int32),
            'after_batch_index': np.asarray(full_logits_after_batch_idx, dtype=np.int32),
            'after_example_index': np.asarray(full_logits_after_example_idx, dtype=np.int32),
            'after_position': np.asarray(full_logits_after_pos, dtype=np.int32),
        }
        with io.BytesIO() as buf:
            np.savez_compressed(buf, **npz_payload)
            with full_logits_path.open('wb') as f:
                f.write(buf.getvalue())

    print('\nTop batches (mean loss):')
    for row in top_batch_records[: min(10, len(top_batch_records))]:
        print(
            f"  rank={row.rank:<3} batch={row.batch_index:<6} "
            f"mean_loss={row.mean_loss:.6f} max_token_loss={row.max_token_loss:.6f}"
        )

    print('\nTop token events:')
    for row in token_event_records[: min(10, len(token_event_records))]:
        print(
            f"  rank={row.rank:<3} loss={row.loss:.6f} "
            f"batch={row.batch_index:<6} ex={row.example_index:<4} pos={row.position:<4} "
            f"input={row.input_token_id} target={row.target_token_id}"
        )
        if row.context_text:
            print(f"    context: {row.context_text!r}")

    print(f"\nSaved analysis report to: {report_path}")
    if full_logits_path is not None:
        print(f"Saved full-logits artifact to: {full_logits_path}")
    if next_train_batch_analysis is not None:
        if 'error' in next_train_batch_analysis:
            print(f"Next-train-batch analysis skipped: {next_train_batch_analysis['error']}")
        else:
            print(
                "Next-train-batch analysis: "
                f"step={next_train_batch_analysis['train_step']} "
                f"mean_loss={next_train_batch_analysis['batch_mean_loss']:.6f} "
                f"max_token_loss={next_train_batch_analysis['batch_max_token_loss']:.6f}"
            )
    if iterative_context_swap_result is not None:
        cumulative_always_down = iterative_context_swap_result.modes['cumulative'].monotonic_nonincreasing_loss
        non_cumulative_always_down = iterative_context_swap_result.modes['non_cumulative'].always_down_vs_base_loss
        print(f"Iterative swap check (cumulative): loss always went down = {cumulative_always_down}")
        print(f"Iterative swap check (non_cumulative): loss always went down = {non_cumulative_always_down}")


if __name__ == '__main__':
    main()
