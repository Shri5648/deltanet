from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import RepositoryNotFoundError


def _open_stream(
    dataset_name: str,
    split: str,
    *,
    seed: int,
    shuffle_buffer_size: int,
    hf_token: str | None,
):
    """Open SlimPajama stream with compatibility fallbacks for older datasets versions."""

    attempts = [
        {"path": dataset_name, "name": None},
        {"path": dataset_name, "name": "default"},
    ]

    last_error = None
    for attempt in attempts:
        try:
            stream = load_dataset(
                attempt["path"],
                name=attempt["name"],
                split=split,
                streaming=True,
                token=hf_token,
            )
            return stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        except (DatasetNotFoundError, RepositoryNotFoundError, FileNotFoundError) as exc:
            last_error = exc

    raise RuntimeError(
        "Could not open SlimPajama from Hugging Face. "
        "Tried dataset ids/configs: "
        f"{[(a['path'], a['name']) for a in attempts]}. "
        "If this dataset exists but still fails, upgrade dependencies and retry: "
        "`pip install -U datasets huggingface_hub hf-xet`, then optionally run "
        "`huggingface-cli login` and set `HF_TOKEN` if your environment requires auth."
    ) from last_error


def build_slimpajama_byte_splits(
    output_dir: str,
    *,
    seed: int = 1337,
    total_tokens: int = 70_000_000,
    val_tokens: int = 10_000_000,
    test_tokens: int = 10_000_000,
    dataset_name: str = "cerebras/SlimPajama-627B",
    split: str = "train",
    shuffle_buffer_size: int = 10_000,
    hf_token: str | None = None,
) -> Dict[str, Path]:
    """Stream SlimPajama once and persist byte-level train/val/test splits.

    The function makes a single pseudorandom pass over the streaming dataset,
    writing exactly `total_tokens` bytes split across train/val/test buffers.
    """

    if val_tokens + test_tokens >= total_tokens:
        raise ValueError("val_tokens + test_tokens must be smaller than total_tokens")

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    test_path = out_dir / "test.bin"

    if meta_path.exists() and train_path.exists() and val_path.exists() and test_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("total_tokens") == total_tokens
            and meta.get("val_tokens") == val_tokens
            and meta.get("test_tokens") == test_tokens
            and meta.get("seed") == seed
            and meta.get("dataset_name") == dataset_name
            and meta.get("split") == split
        ):
            return {"train": train_path, "val": val_path, "test": test_path, "meta": meta_path}

    train_tokens = total_tokens - val_tokens - test_tokens
    capacities = {"val": val_tokens, "test": test_tokens, "train": train_tokens}
    order = ("val", "test", "train")

    buffers = {name: np.empty(cap, dtype=np.uint8) for name, cap in capacities.items()}
    write_pos = {name: 0 for name in capacities}

    stream = _open_stream(
        dataset_name,
        split,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        hf_token=hf_token,
    )

    active_idx = 0

    for sample in stream:
        text = sample.get("text")
        if not text:
            continue

        doc = (text + "\n").encode("utf-8", errors="ignore")
        offset = 0

        while offset < len(doc):
            if active_idx >= len(order):
                break

            active_split = order[active_idx]
            remaining = capacities[active_split] - write_pos[active_split]
            if remaining == 0:
                active_idx += 1
                continue

            take = min(remaining, len(doc) - offset)
            start = write_pos[active_split]
            end = start + take
            buffers[active_split][start:end] = np.frombuffer(doc[offset : offset + take], dtype=np.uint8)
            write_pos[active_split] = end
            offset += take

        if active_idx >= len(order):
            break

    if any(write_pos[name] < capacities[name] for name in capacities):
        got = {k: write_pos[k] for k in capacities}
        raise RuntimeError(
            f"Streaming ended before reaching token budget. Expected {capacities}, got {got}."
        )

    buffers["train"].tofile(train_path)
    buffers["val"].tofile(val_path)
    buffers["test"].tofile(test_path)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "split": split,
                "seed": seed,
                "total_tokens": total_tokens,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
                "test_tokens": test_tokens,
                "dtype": "uint8",
            },
            f,
            indent=2,
        )

    return {"train": train_path, "val": val_path, "test": test_path, "meta": meta_path}


def load_memmap_splits(output_dir: str) -> Tuple[np.memmap, np.memmap, np.memmap]:
    out_dir = Path(output_dir)
    train = np.memmap(out_dir / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(out_dir / "val.bin", dtype=np.uint8, mode="r")
    test = np.memmap(out_dir / "test.bin", dtype=np.uint8, mode="r")
    return train, val, test
