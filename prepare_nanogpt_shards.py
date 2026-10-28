#!/usr/bin/env python3
"""Flatten NanoGPT uint16 shards into one headerless PicoDo token file.

PicoDo's data loader uses ``np.memmap(..., dtype=np.uint16)`` and therefore
cannot read NanoGPT shard headers. This tool validates each input shard, strips
its 1 KiB header, and concatenates its token payload into one raw uint16 file.

The default includes only ``fineweb_train_*.bin``. PicoDo creates its own
validation split, so the NanoGPT validation shard is normally excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO


HEADER_BYTES = 256 * 4
MAGIC = 20240520
VERSION = 1
COPY_BUFFER_BYTES = 64 * 1024 * 1024
TRAIN_SHARD_RE = re.compile(r"fineweb_train_(\d{6})\.bin$")


@dataclass(frozen=True)
class Shard:
    path: Path
    token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing fineweb_train_*.bin NanoGPT shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination headerless uint16 .bin file for PicoDo",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Prepend fineweb_val_000000.bin (normally leave this disabled)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous interrupted conversion for the same paths and options",
    )
    return parser.parse_args()


def read_shard(path: Path) -> Shard:
    with path.open("rb") as file:
        header = file.read(HEADER_BYTES)
    if len(header) != HEADER_BYTES:
        raise ValueError(f"{path}: file is smaller than a NanoGPT header")
    magic, version, token_count = struct.unpack_from("<3i", header)
    if magic != MAGIC or version != VERSION or token_count <= 0:
        raise ValueError(f"{path}: unsupported or invalid NanoGPT header")
    expected_size = HEADER_BYTES + 2 * token_count
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {actual_size}")
    return Shard(path=path, token_count=token_count)


def discover_shards(input_dir: Path, include_validation: bool) -> list[Shard]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    matches: list[tuple[int, Path]] = []
    for path in input_dir.glob("fineweb_train_*.bin"):
        match = TRAIN_SHARD_RE.fullmatch(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    matches.sort()
    if not matches:
        raise ValueError(f"No fineweb_train_*.bin shards found in {input_dir}")

    indices = [index for index, _ in matches]
    expected_indices = list(range(1, len(indices) + 1))
    if indices != expected_indices:
        raise ValueError(
            "Training shard indices must be contiguous and start at 000001; "
            f"found {indices[0]:06d} through {indices[-1]:06d}."
        )

    shards: list[Shard] = []
    if include_validation:
        validation_path = input_dir / "fineweb_val_000000.bin"
        if not validation_path.is_file():
            raise ValueError(f"Requested validation shard is missing: {validation_path}")
        shards.append(read_shard(validation_path))
    shards.extend(read_shard(path) for _, path in matches)
    return shards


def write_json_atomically(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary_path, path)


def copy_payload(source: BinaryIO, destination: BinaryIO, payload_bytes: int) -> None:
    source.seek(HEADER_BYTES)
    remaining = payload_bytes
    while remaining:
        chunk = source.read(min(COPY_BUFFER_BYTES, remaining))
        if not chunk:
            raise OSError("Unexpected end of shard while copying token payload")
        destination.write(chunk)
        remaining -= len(chunk)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shards = discover_shards(input_dir, args.include_validation)

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    state_path = output_path.with_suffix(output_path.suffix + ".partial.json")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")

    if output_path.exists():
        raise FileExistsError(
            f"Destination already exists: {output_path}. Refusing to overwrite it."
        )
    if (partial_path.exists() or state_path.exists()) and not args.resume:
        raise FileExistsError(
            f"Found an interrupted conversion ({partial_path.name} or {state_path.name}). "
            "Re-run with --resume or inspect/remove those files."
        )

    completed_shards = 0
    bytes_written = 0
    if args.resume:
        if not (partial_path.is_file() and state_path.is_file()):
            if partial_path.exists() or state_path.exists():
                raise FileNotFoundError("Both partial data and its resume-state file are required")
        else:
            state = json.loads(state_path.read_text())
            expected_state = {
                "input_dir": str(input_dir),
                "output": str(output_path),
                "include_validation": args.include_validation,
            }
            if any(state.get(key) != value for key, value in expected_state.items()):
                raise ValueError("Resume state belongs to different input, output, or validation options")
            completed_shards = int(state["completed_shards"])
            bytes_written = int(state["bytes_written"])
            if not 0 <= completed_shards <= len(shards):
                raise ValueError("Resume state has an invalid completed-shard count")
            if partial_path.stat().st_size < bytes_written:
                raise ValueError("Partial output is smaller than the checkpointed byte count")

    mode = "r+b" if partial_path.exists() else "w+b"
    with partial_path.open(mode) as destination:
        # Discard any uncheckpointed bytes from a process that was interrupted
        # midway through copying one source shard.
        destination.truncate(bytes_written)
        destination.seek(bytes_written)
        for index, shard in enumerate(shards[completed_shards:], start=completed_shards):
            payload_bytes = 2 * shard.token_count
            print(
                f"[{index + 1}/{len(shards)}] {shard.path.name}: "
                f"{shard.token_count:,} tokens",
                flush=True,
            )
            with shard.path.open("rb") as source:
                copy_payload(source, destination, payload_bytes)
            destination.flush()
            os.fsync(destination.fileno())
            bytes_written += payload_bytes
            completed_shards = index + 1
            write_json_atomically(
                state_path,
                {
                    "input_dir": str(input_dir),
                    "output": str(output_path),
                    "include_validation": args.include_validation,
                    "completed_shards": completed_shards,
                    "bytes_written": bytes_written,
                },
            )

    expected_bytes = sum(2 * shard.token_count for shard in shards)
    if bytes_written != expected_bytes or partial_path.stat().st_size != expected_bytes:
        raise RuntimeError("Output size does not match the validated source-shard payload sizes")

    metadata = {
        "format": "headerless little-endian uint16 token stream",
        "consumer": "picodo.data.load_ds",
        "input_dir": str(input_dir),
        "include_validation": args.include_validation,
        "num_source_shards": len(shards),
        "total_tokens": bytes_written // 2,
        "total_bytes": bytes_written,
        "source_shards": [asdict(shard) | {"path": str(shard.path)} for shard in shards],
    }
    write_json_atomically(metadata_path, metadata)
    os.replace(partial_path, output_path)
    state_path.unlink(missing_ok=True)
    print(f"Completed {bytes_written // 2:,} tokens: {output_path}", flush=True)


if __name__ == "__main__":
    main()
