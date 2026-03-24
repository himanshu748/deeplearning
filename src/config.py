import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


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
    device: torch.device = field(init=False)
    use_amp: bool = field(init=False)

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
