from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


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
) -> Dict[str, Path]:
    """Stream SlimPajama once and persist byte-level train/val/test splits.

    The function makes a single pseudorandom pass over the streaming dataset,
    writing exactly `total_tokens` bytes split across train/val/test buffers.
    """

    if val_tokens + test_tokens >= total_tokens:
        raise ValueError("val_tokens + test_tokens must be smaller than total_tokens")

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

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install the `datasets` package to stream SlimPajama: `pip install datasets`."
        ) from exc

    train_tokens = total_tokens - val_tokens - test_tokens
    capacities = {"val": val_tokens, "test": test_tokens, "train": train_tokens}
    order = ("val", "test", "train")

    buffers = {name: np.empty(cap, dtype=np.uint8) for name, cap in capacities.items()}
    write_pos = {name: 0 for name in capacities}

    stream = load_dataset(dataset_name, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    active_idx = 0
    active_split = order[active_idx]

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
