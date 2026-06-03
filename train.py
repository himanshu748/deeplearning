#!/usr/bin/env python3
"""CLI entry point for training ForestSight AI models."""
import argparse
from pathlib import Path

import kagglehub
import pandas as pd
import torch

from src.config import Config, seed_everything
from src.dataset import create_dataloaders, load_pairs
from src.models import MODEL_REGISTRY, build_model
from src.trainer import evaluate, train_model


MODEL_CHOICES = tuple(sorted(MODEL_REGISTRY.values()))


def main():
    parser = argparse.ArgumentParser(description="ForestSight AI -- Train forest segmentation models")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=None, help="Square input image size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for checkpoints/results")
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_CHOICES,
                        help="Architectures to train: unet, unet_attention, deeplabv3plus")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset. Downloads from Kaggle if not provided.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = Path(args.checkpoint_dir)
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg.validate()
    seed_everything(cfg.seed)
    print(f"Device: {cfg.device} | AMP: {cfg.use_amp}")

    if args.data_dir:
        dataset_path = Path(args.data_dir)
    else:
        print("Downloading dataset from Kaggle...")
        dataset_path = Path(kagglehub.dataset_download("quadeer15sh/augmented-forest-segmentation"))
    print(f"Dataset: {dataset_path}")

    pairs = load_pairs(dataset_path)
    print(f"Total pairs: {len(pairs)}")

    train_loader, val_loader, test_loader, *_ = create_dataloaders(
        pairs,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    archs = args.models or list(MODEL_REGISTRY.values())
    name_for = {v: k for k, v in MODEL_REGISTRY.items()}
    test_results = {}

    for arch in archs:
        label = name_for.get(arch, arch)
        print(f"\n{'='*60}\n  Training: {label}\n{'='*60}")
        model = build_model(arch)
        model, history, best_iou = train_model(
            model, train_loader, val_loader,
            architecture=arch, device=cfg.device, use_amp=cfg.use_amp,
            lr=cfg.lr, weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            patience=cfg.early_stop_patience,
            dice_weight=cfg.dice_weight, bce_weight=cfg.bce_weight,
            checkpoint_dir=cfg.checkpoint_dir,
        )
        test_loss, test_m = evaluate(model, test_loader, cfg.device, cfg.use_amp)
        test_results[label] = {"Loss": test_loss, **{k.capitalize(): v for k, v in test_m.items()}}
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(test_results).T.round(4)
    print(f"\n{'='*60}\n  TEST RESULTS\n{'='*60}")
    print(df.to_string())
    best = df["Iou"].idxmax()
    print(f"\nBest: {best} (IoU={df.loc[best, 'Iou']:.4f})")

    df.to_csv(cfg.checkpoint_dir / "results.csv")
    print(f"Results saved to {cfg.checkpoint_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
