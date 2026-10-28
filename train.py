import math
import os

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf.dictconfig import DictConfig

import data, utils
import model as model_lib
from training.checkpoints import (
    _resolve_diagnostics_dir,
    setup_checkpointing,
)
from training.loop import TrainingContext, run_training
from training.optimizers import build_optimizer


def train_and_evaluate(c: DictConfig):
    if not c.use_single_vm_among_multiple:
        # init distributed env if using multiple vms
        jax.distributed.initialize()
    
    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print('sharding mesh:', ', '.join(f'{k}={v}' for k, v in mesh.shape.items()))
    multihost_batch_sharding = bool(getattr(c, 'multihost_batch_sharding', False))
    data_param_sharding = bool(getattr(c.model, 'data_param_sharding', True))
    conventional_data_parallel = not data_param_sharding
    if conventional_data_parallel and c.num_tp_devices != 1:
        raise ValueError(
            'model.data_param_sharding=false currently requires num_tp_devices=1.'
        )
    if conventional_data_parallel and jax.process_index() == 0:
        print(
            'conventional data parallel enabled: parameters are replicated and '
            'only the batch is sharded over the data axis'
        )
    if multihost_batch_sharding and jax.process_count() > 1:
        local_batch_size = c.opt.batch_size // jax.process_count()
        if c.opt.batch_size % jax.process_count() != 0:
            raise ValueError(
                "opt.batch_size must divide evenly across JAX processes when "
                "multihost_batch_sharding=true."
            )
        if jax.process_index() == 0:
            print(
                "multihost batch sharding enabled: "
                f"global_batch={c.opt.batch_size}, per_host_batch={local_batch_size}, "
                f"per_device_batch={c.opt.batch_size // jax.device_count()}"
            )

    profiling_cfg = getattr(c, 'profiling', None)
    profiling_enabled = bool(getattr(profiling_cfg, 'enabled', False))
    profiling_start_step = int(getattr(profiling_cfg, 'start_step', 5))
    profiling_num_steps = int(getattr(profiling_cfg, 'num_steps', 10))
    profiling_log_dir = str(getattr(profiling_cfg, 'log_dir', '') or '')
    profiling_create_perfetto_trace = bool(
        getattr(profiling_cfg, 'create_perfetto_trace', True)
    )
    if profiling_enabled:
        if jax.process_count() != 1:
            raise ValueError(
                'JAX profiling is currently supported only for a single-host run; '
                'each host would otherwise try to write a trace to the same directory.'
            )
        if profiling_start_step < 0 or profiling_num_steps <= 0:
            raise ValueError(
                'profiling.start_step must be non-negative and profiling.num_steps '
                'must be positive.'
            )
        if not profiling_log_dir:
            profiling_log_dir = os.environ.get('PICODO_PROFILE_LOG_DIR', '')
        if not profiling_log_dir:
            profiling_log_dir = os.path.abspath('jax-profile')
        profiling_log_dir = os.path.abspath(profiling_log_dir)
        os.makedirs(profiling_log_dir, exist_ok=True)
        print(
            'JAX profiling enabled: '
            f'steps=[{profiling_start_step}, '
            f'{profiling_start_step + profiling_num_steps}), '
            f'log_dir={profiling_log_dir}'
        )

    # model
    print('initializing model...')
    utils.sync_lm_head_oblique_model_config(c)
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count()) # round V up to enable sharding
    model = model_lib.create_sharded_model(c.model, key_model)
    utils.validate_row_oblique_lm_head_options(c.opt)
    lm_head_optimizer_type = utils.get_lm_head_optimizer_type(c.opt)
    resolved_lm_head_target_rms = None
    if lm_head_optimizer_type in {"row_oblique", "column_oblique"}:
        learn_target_rms = utils.lm_head_uses_learned_target_rms(c.opt)
        target_rms_from_random_init = utils.lm_head_target_rms_from_random_init(c.opt)
        if learn_target_rms:
            resolved_lm_head_target_rms = utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
        elif target_rms_from_random_init:
            resolved_lm_head_target_rms = (
                float(model_lib.get_average_output_embedding_row_rms(model))
                if lm_head_optimizer_type == "row_oblique"
                else float(model_lib.get_average_output_embedding_col_rms(model))
            )
        else:
            resolved_lm_head_target_rms = utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
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

    # get num. model parameters
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')
    
    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.batch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train)
    optimizer_setup = build_optimizer(
        c, model, num_opt_steps, resolved_lm_head_target_rms
    )
    opt_graphdef = optimizer_setup.opt_graphdef
    opt_state = optimizer_setup.opt_state
    lr_schedule = optimizer_setup.lr_schedule
    lm_head_tx_lr_schedule = optimizer_setup.lm_head_tx_lr_schedule
    lm_head_metric_lr_schedule = optimizer_setup.lm_head_metric_lr_schedule
    b2_schedule = optimizer_setup.b2_schedule
    tokens_per_opt_step = optimizer_setup.tokens_per_opt_step
    del model

    # set up checkpointing
    checkpoint_setup = setup_checkpointing(c, opt_state)
    opt_state = checkpoint_setup.opt_state
    ckpt_mngr = checkpoint_setup.manager
    ckpt_dir = checkpoint_setup.directory
    start_step = checkpoint_setup.start_step

    if c.diagnostics.save_raw_losses:
        diagnostics_dir = _resolve_diagnostics_dir(ckpt_dir)
        if diagnostics_dir:
            utils.save_to_numpy(save_dir=diagnostics_dir, name='val_dataset', data=ds_valid)
            utils.save_to_numpy(save_dir=diagnostics_dir, name='train_dataset', data=ds_train[:c.diagnostics.end_step])

    run_training(
        TrainingContext(
            c=c,
            n_params=n_params,
            ds_train=ds_train,
            ds_valid=ds_valid,
            mesh=mesh,
            model_graphdef=model_graphdef,
            opt_graphdef=opt_graphdef,
            opt_state=opt_state,
            num_opt_steps=num_opt_steps,
            start_step=start_step,
            tokens_per_opt_step=tokens_per_opt_step,
            lr_schedule=lr_schedule,
            lm_head_tx_lr_schedule=lm_head_tx_lr_schedule,
            lm_head_metric_lr_schedule=lm_head_metric_lr_schedule,
            b2_schedule=b2_schedule,
            ckpt_dir=ckpt_dir,
            ckpt_mngr=ckpt_mngr,
            multihost_batch_sharding=multihost_batch_sharding,
            conventional_data_parallel=conventional_data_parallel,
            profiling_enabled=profiling_enabled,
            profiling_start_step=profiling_start_step,
            profiling_num_steps=profiling_num_steps,
            profiling_log_dir=profiling_log_dir,
            profiling_create_perfetto_trace=profiling_create_perfetto_trace,
        )
    )
