"""Checkpoint setup, storage, and diagnostic artifact helpers."""

import gc
import os
from dataclasses import dataclass

import jax
import orbax.checkpoint as ocp
from etils import epath
from jax.experimental import multihost_utils
from orbax.checkpoint.checkpoint_managers import preservation_policy
from orbax.checkpoint._src.path import gcs_utils, step as step_lib

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


@dataclass(frozen=True)
class CheckpointSetup:
    opt_state: object
    manager: object | None
    directory: str | None
    start_step: int


def _resolve_diagnostics_dir(ckpt_dir):
    if not ckpt_dir:
        return None
    diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    return diagnostics_dir


def _save_eval_diagnostics(
    diagnostics_dir,
    step,
    train_raw_loss,
    eval_raw_loss,
    train_logit_gaps,
    train_target_gaps,
    eval_logits,
):
    if not diagnostics_dir:
        return
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'train_raw_losses_step_{step}.npy', data=train_raw_loss
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{step}.npy', data=eval_raw_loss
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'train_mean_logit_gaps_step_{step}.npy', data=train_logit_gaps
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'train_target_logit_gaps_step_{step}.npy', data=train_target_gaps
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'eval_logits_step_{step}.npy', data=eval_logits
    )


def _save_checkpoint(ckpt_mngr, step, opt_state):
    ckpt_mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(opt_state),
            training_metadata=ocp.args.JsonSave({
                'step': step,
                'next_step': step + 1,
            }),
        ),
        force=True,
    )


def _should_checkpoint(c, step):
    if not c.checkpoint.turn_on:
        return False
    checkpoint_steps = getattr(c.checkpoint, 'checkpoint_steps', None)
    if checkpoint_steps is not None:
        return step in checkpoint_steps
    return step % c.checkpoint.checkpoint_every_steps == 0


def setup_checkpointing(c, opt_state):
    start_step = 0
    ckpt_dir = None
    ckpt_mngr = None
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    if c.checkpoint.turn_on:
        run_name = c.run_name if c.run_name else 'picodo_run'

        # Use GCP bucket if specified, otherwise use local workdir
        gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
        if gcp_bucket:
            # Format: gs://bucket-name/path
            if not gcp_bucket.startswith('gs://'):
                gcp_bucket = f'gs://{gcp_bucket}'
            ckpt_dir = os.path.join(gcp_bucket, run_name)
            if jax.process_index() == 0:
                print(f'Checkpoints will be saved to GCS bucket: {ckpt_dir}')
        else:
            ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)

        step_prefix = getattr(c.checkpoint, 'step_prefix', None)

        # Base checkpoint manager options
        mngr_options_kwargs = {
            'create': True,
            'preservation_policy': preservation_policy.LatestN(c.checkpoint.max_to_keep)
        }

        if gcp_bucket:
            mngr_options_kwargs['step_name_format'] = _StandardNameFormatHNS(
                step_prefix=step_prefix
            )

        # Add multihost settings if running on multiple hosts
        is_multihost = jax.process_count() > 1
        if is_multihost:
            mngr_options_kwargs['enable_async_checkpointing'] = True
            # Orbax API changed across versions:
            # newer versions expose options under ocp.options.*,
            # older versions use ocp.multiprocessing.*.
            mp_options_cls = None
            if hasattr(ocp, 'options') and hasattr(ocp.options, 'MultiprocessingOptions'):
                mp_options_cls = ocp.options.MultiprocessingOptions
            elif hasattr(ocp, 'multiprocessing') and hasattr(ocp.multiprocessing, 'MultiprocessingOptions'):
                mp_options_cls = ocp.multiprocessing.MultiprocessingOptions

            if mp_options_cls is not None:
                mngr_options_kwargs['multiprocessing_options'] = mp_options_cls(
                    primary_host=0,
                    active_processes=set(range(jax.process_count()))
                )
                # Newer Orbax does not allow create=True when active_processes is set.
                mngr_options_kwargs['create'] = False
            if jax.process_index() == 0:
                print(f'Multihost checkpointing enabled with {jax.process_count()} processes')

        # If create=False (required for some multihost Orbax versions),
        # ensure root directory exists before CheckpointManager construction.
        if not mngr_options_kwargs.get('create', True):
            if jax.process_index() == 0:
                epath.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            if is_multihost:
                multihost_utils.sync_global_devices('ckpt_root_dir_ready')

        mngr_options = ocp.CheckpointManagerOptions(**mngr_options_kwargs)

        ckpt_mngr = ocp.CheckpointManager(
            ckpt_dir,
            options=mngr_options
        )
        
        print(f'Checking for existing checkpoints in: {ckpt_dir}')
        latest_step = c.checkpoint.start_step if c.checkpoint.start_step != None else ckpt_mngr.latest_step()

        if latest_step is not None:
            print(f'Restoring checkpoint from step {latest_step} in {ckpt_dir}...')
            
            restored_data = ckpt_mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_opt_state),
                    training_metadata=ocp.args.JsonRestore(),
                ),
            )
            opt_state = restored_data['state']
            training_metadata = restored_data.get('training_metadata', {})
            meta_step = training_metadata.get('step')
            meta_next_step = training_metadata.get('next_step')
            if meta_next_step is not None:
                start_step = meta_next_step
            elif meta_step is not None:
                if meta_step == latest_step + 1:
                    start_step = meta_step
                elif meta_step == latest_step:
                    start_step = meta_step + 1
                else:
                    print(
                        'Warning: checkpoint metadata step does not match checkpoint id '
                        f'(metadata step={meta_step}, checkpoint id={latest_step}). '
                        'Falling back to resume from checkpoint id + 1.'
                    )
                    start_step = latest_step + 1
            else:
                print(
                    'Warning: checkpoint metadata missing step/next_step; '
                    'falling back to resume from checkpoint id + 1.'
                )
                start_step = latest_step + 1
            del restored_data
            gc.collect()
            print(f'Successfully restored checkpoint. Resuming from step {start_step}.')
        else:
            print('No checkpoint found. Starting from scratch.')
    del abstract_opt_state
    gc.collect()
    return CheckpointSetup(
        opt_state=opt_state,
        manager=ckpt_mngr,
        directory=ckpt_dir,
        start_step=start_step,
    )


