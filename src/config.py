import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .splits import validate_split_ratios


@dataclass
class Config:
    seed: int = 42
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stop_patience: int = 10
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    device: object = field(init=False, default=None)
    use_amp: bool = field(init=False, default=False)

    def __post_init__(self):
        self.validate()

    def initialize_runtime(self) -> None:
        torch = require_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        validate_split_ratios(self.train_ratio, self.val_ratio)
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be zero or greater")
        if self.lr <= 0:
            raise ValueError("lr must be greater than 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be zero or greater")
        if self.dice_weight < 0:
            raise ValueError("dice_weight must be zero or greater")
        if self.bce_weight < 0:
            raise ValueError("bce_weight must be zero or greater")
        if self.dice_weight + self.bce_weight <= 0:
            raise ValueError("dice_weight + bce_weight must be greater than 0")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for training. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return torch
