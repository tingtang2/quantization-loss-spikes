import os
import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding


class LazyBatchDataset:
    def __init__(self, data, indices):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices),) + tuple(data.shape[1:])
        self.dtype = data.dtype
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return np.asarray(self.data[self.indices[index]])

    def __array__(self, dtype=None):
        array = np.asarray(self[:])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array


def _load_ds_eager(key, mesh, ds_path, seq_len, batch_size, n_tokens_valid, n_tokens_train=None):

    # get dataset size
    print('getting dataset size...')
    ds_path = os.path.expanduser(ds_path)
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(data)
    n_seq_dataset = n_tokens_dataset // seq_len

    # if n_tokens_train is None, use full dataset
    if n_tokens_train is not None: assert n_tokens_train + n_tokens_valid <= n_tokens_dataset
    if n_tokens_train is None: n_tokens_train = n_tokens_dataset - n_tokens_valid

    # get num. of train. and valid. batches
    n_batch_train = n_tokens_train // (batch_size * seq_len)
    n_batch_valid = n_tokens_valid // (batch_size * seq_len)
    n_batch = n_batch_train + n_batch_valid

    # memmap data
    print('reading data...')
    data = np.memmap(ds_path, dtype=np.uint16, shape=[n_batch, batch_size, seq_len], mode='r')
    
    # load data onto jax devices, sharded across batch dimension
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'data', 'model'))
    callback = lambda index: data[index]
    data = jax.make_array_from_callback(data.shape, sharding, callback)

    # shuffle batches
    print('shuffling data...')
    data = jax.random.permutation(key, data, axis=0)

    # split data
    print('splitting data...')
    data_train = data[:n_batch_train]
    data_valid = data[n_batch_train:]
    
    return data_train, data_valid


def load_ds(key, mesh, ds_path, seq_len, batch_size, n_tokens_valid, n_tokens_train=None, lazy=True):

    if not lazy:
        return _load_ds_eager(key, mesh, ds_path, seq_len, batch_size, n_tokens_valid, n_tokens_train)

    # get dataset size
    print('getting dataset size...')
    ds_path = os.path.expanduser(ds_path)
    tokens = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(tokens)
    print(f'number of tokens in dataset: {n_tokens_dataset}')

    # if n_tokens_train is None, use full dataset
    if n_tokens_train is not None: assert n_tokens_train + n_tokens_valid <= n_tokens_dataset
    if n_tokens_train is None: n_tokens_train = n_tokens_dataset - n_tokens_valid

    # get num. of train. and valid. batches
    n_batch_train = n_tokens_train // (batch_size * seq_len)
    n_batch_valid = n_tokens_valid // (batch_size * seq_len)
    n_batch = n_batch_train + n_batch_valid

    # memmap data
    print('reading data lazily...')
    data = np.memmap(ds_path, dtype=np.uint16, shape=[n_batch, batch_size, seq_len], mode='r')

    # shuffle batch indices instead of shuffling the token array
    print('shuffling batch indices...')
    indices = jax.random.permutation(key, jnp.arange(n_batch, dtype=jnp.int32), axis=0)
    indices = np.asarray(jax.device_get(indices), dtype=np.int64)

    # split data
    print('splitting indices...')
    data_train = LazyBatchDataset(data, indices[:n_batch_train])
    data_valid = LazyBatchDataset(data, indices[n_batch_train:])

    return data_train, data_valid
