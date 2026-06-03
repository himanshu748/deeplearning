from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import DiceBCELoss
from .metrics import compute_metrics


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def _run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None):
    if len(loader) == 0:
        raise ValueError("DataLoader is empty; cannot run an epoch")

    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    keys = ["iou", "dice", "accuracy", "precision", "recall"]
    sums = {k: 0.0 for k in keys}
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, masks in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                m = compute_metrics(outputs, masks)
                for k in sums:
                    sums[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in sums.items()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    architecture: str,
    device: torch.device,
    use_amp: bool = False,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    patience: int = 10,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    checkpoint_dir: Path = Path("checkpoints"),
):
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")

    model = model.to(device)
    criterion = DiceBCELoss(dice_weight, bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=use_amp)
    early_stop = EarlyStopping(patience=patience)

    best_iou = float("-inf")
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "lr": []}
    ckpt_path = checkpoint_dir / f"{architecture}_best.pth"

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs} | LR: {current_lr:.2e}")

        train_loss, train_m = _run_epoch(
            model, train_loader, criterion, device, use_amp, optimizer, scaler
        )
        val_loss, val_m = _run_epoch(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_m["iou"])
        history["val_iou"].append(val_m["iou"])
        history["lr"].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f} | IoU: {train_m['iou']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | IoU: {val_m['iou']:.4f} | Dice: {val_m['dice']:.4f}")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_iou": best_iou,
                    "architecture": architecture,
                },
                ckpt_path,
            )
            print(f"  >>> New best IoU: {best_iou:.4f} -- saved")

        if early_stop(val_m["iou"]):
            print(f"\nEarly stopping at epoch {epoch}. Best IoU: {best_iou:.4f}")
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model from epoch {ckpt['epoch']} (IoU={ckpt['best_iou']:.4f})")
    return model, history, best_iou


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False, dice_w=0.5, bce_w=0.5):
    criterion = DiceBCELoss(dice_w, bce_w)
    return _run_epoch(model, loader, criterion, device, use_amp)
