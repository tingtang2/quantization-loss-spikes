import os
import fire
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars; disable_progress_bars()
from typing import Optional, Literal


def load_data_shard(file):
    # https://github.com/KellerJordan/modded-nanogpt/blob/a202a3a0ca99d69bb7f847e5337c7c6e0890fd92/train_gpt.py#L411
    header = np.fromfile(file, dtype=np.int32, count=256) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with Path(file).open("rb", buffering=0) as f:
        tokens = np.empty(num_tokens, dtype=np.uint16) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def get_num_tokens(file):
    header = np.fromfile(file, dtype=np.int32, count=256)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2])


def download_dataset(
        dataset: Literal['fineweb', 'finewebedu'] = 'fineweb',
        num_chunks: Optional[int] = None,
        full_fineweb100b: bool = False,
        stream_write: bool = False,
        shard_dir: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        keep_shards: bool = False,
    ):
    """download dataset, save it as a np.memmap binary file"""

    # full_fineweb100b is only available for fineweb.
    if full_fineweb100b and dataset != 'fineweb':
        raise ValueError("full_fineweb100b=True is only supported when dataset='fineweb'.")

    repo_id = f'kjj0/{dataset}10B-gpt2'
    default_chunks = 103 if dataset == 'fineweb' else 99
    output_name = f'{dataset}_gpt2.bin'

    if full_fineweb100b:
        repo_id = 'kjj0/fineweb100B-gpt2'
        default_chunks = 1030
        output_name = 'fineweb100B_gpt2.bin'

    # get num. chunks
    # each chunk is 100M tokens
    if num_chunks is None:
        num_chunks = default_chunks

    # save to disk
    out_dir = os.path.expanduser('~/datasets')
    out_path = f'{out_dir}/{output_name}'
    os.makedirs(out_dir, exist_ok=True)

    if stream_write:
        if shard_dir is None:
            shard_dir = '/dev/shm/fineweb_shards' if os.path.isdir('/dev/shm') else '/tmp/fineweb_shards'
        if hf_cache_dir is None:
            hf_cache_dir = '/dev/shm/hf_cache' if os.path.isdir('/dev/shm') else None

        shard_dir = os.path.expanduser(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)
        if hf_cache_dir is not None:
            hf_cache_dir = os.path.expanduser(hf_cache_dir)
            os.makedirs(hf_cache_dir, exist_ok=True)

        print('downloading + streaming to output...')
        with Path(out_path).open('wb') as fout:
            for i in tqdm(range(1, num_chunks + 1)):
                download_kwargs = dict(
                    repo_id=repo_id,
                    filename=f'{dataset}_train_{i:06}.bin',
                    repo_type='dataset',
                    local_dir=shard_dir,
                )
                if hf_cache_dir is not None:
                    download_kwargs['cache_dir'] = hf_cache_dir
                shard_path = hf_hub_download(**download_kwargs)
                shard = load_data_shard(shard_path)
                shard.tofile(fout)
                if not keep_shards and os.path.exists(shard_path):
                    os.remove(shard_path)
    else:
        # download all shard paths first
        print('downloading...')
        shard_paths = []
        for i in tqdm(range(1, num_chunks+1)):
            download_kwargs = dict(
                repo_id=repo_id,
                filename=f'{dataset}_train_{i:06}.bin',
                repo_type='dataset',
            )
            if hf_cache_dir is not None:
                download_kwargs['cache_dir'] = os.path.expanduser(hf_cache_dir)
            shard_path = hf_hub_download(**download_kwargs)
            shard_paths.append(shard_path)

        print('saving...')
        n_tokens = sum(get_num_tokens(path) for path in shard_paths)
        out = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=[n_tokens])
        i = 0
        for shard_path in tqdm(shard_paths):
            shard = load_data_shard(shard_path)
            out[i:i+len(shard)] = shard
            i += len(shard)
        out.flush()


if __name__ == '__main__':
    fire.Fire(download_dataset)
