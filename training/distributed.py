"""Batch placement helpers for single- and multi-host JAX runs."""

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def _maybe_shard_multihost_batch(
    batch,
    mesh,
    enabled: bool,
    force_data_sharding: bool = False,
):
    """Turns a batch into a global array sharded on the mesh data axis.

    The lazy dataset deliberately returns NumPy arrays.  In a multi-controller
    run, handing that full global batch to every controller leaves placement
    ambiguous and can replicate large intermediates.  With this opt-in path,
    each host supplies its distinct contiguous batch slice and JAX assembles a
    single global array whose leading dimension is sharded over the data mesh.

    ``force_data_sharding`` additionally covers a single-host conventional data
    parallel run: its host-resident global batch is explicitly split over the
    data axis, rather than leaving input placement to compiler inference.
    """
    if not enabled and not force_data_sharding:
        return batch

    batch = np.asarray(batch)
    if batch.ndim < 1:
        raise ValueError(f"Expected a batched array, got shape {batch.shape}.")
    if jax.process_count() == 1:
        if not force_data_sharding:
            return batch
        sharding = NamedSharding(mesh, P('data', *([None] * (batch.ndim - 1))))
        global_batch = jax.device_put(batch, sharding)
    else:
        if batch.shape[0] % jax.process_count() != 0:
            raise ValueError(
                "Global batch size must divide evenly across JAX processes when "
                f"multihost_batch_sharding=true; got batch={batch.shape[0]} and "
                f"processes={jax.process_count()}."
            )

        local_batch_size = batch.shape[0] // jax.process_count()
        start = jax.process_index() * local_batch_size
        local_batch = batch[start:start + local_batch_size]
        sharding = NamedSharding(mesh, P('data', *([None] * (batch.ndim - 1))))
        global_batch = jax.make_array_from_process_local_data(
            sharding, local_batch, global_shape=batch.shape
        )

    if global_batch.shape != batch.shape:
        raise ValueError(
            f"Data-sharded batch assembly changed shape from {batch.shape} to "
            f"{global_batch.shape}."
        )
    if not getattr(_maybe_shard_multihost_batch, '_layout_logged', False):
        local_shapes = [tuple(shard.data.shape) for shard in global_batch.addressable_shards]
        print(
            "data-sharded batch placement: "
            f"process={jax.process_index()}, global_shape={global_batch.shape}, "
            f"addressable_shard_shapes={local_shapes}"
        )
        _maybe_shard_multihost_batch._layout_logged = True
    return global_batch

