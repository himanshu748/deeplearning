"""Deterministic train/validation/test splitting helpers."""
from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0:
        raise ValueError("train_ratio must be greater than 0")
    if val_ratio <= 0:
        raise ValueError("val_ratio must be greater than 0")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")


def split_pairs(
    pairs: Sequence[T],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Return deterministic non-empty train, validation, and test splits."""
    validate_split_ratios(train_ratio, val_ratio)
    if len(pairs) < 3:
        raise ValueError("At least 3 image-mask pairs are required for train/val/test splits")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    train_count = max(1, int(round(len(shuffled) * train_ratio)))
    val_count = max(1, int(round(len(shuffled) * val_ratio)))

    if train_count + val_count >= len(shuffled):
        overflow = train_count + val_count - len(shuffled) + 1
        if train_count >= val_count and train_count > 1:
            train_count -= overflow
        elif val_count > 1:
            val_count -= overflow

    if train_count <= 0 or val_count <= 0 or train_count + val_count >= len(shuffled):
        raise ValueError("Could not create non-empty train/val/test splits from the provided pairs")

    train_pairs = shuffled[:train_count]
    val_pairs = shuffled[train_count: train_count + val_count]
    test_pairs = shuffled[train_count + val_count:]
    return train_pairs, val_pairs, test_pairs
