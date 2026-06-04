#!/usr/bin/env python3
"""CLI entry point for training ForestSight AI models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config, seed_everything
from src.dataset import load_pairs
from src.models import MODEL_REGISTRY
from src.splits import split_pairs


MODEL_CHOICES = tuple(sorted(MODEL_REGISTRY.values()))


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, dataset pairing, and deterministic splits without training.",
    )
    return parser


def build_config(args: argparse.Namespace) -> Config:
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

    cfg.validate()
    return cfg


def resolve_dataset_path(data_dir: str | None) -> Path:
    if data_dir:
        dataset_path = Path(data_dir).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        return dataset_path

    print("Downloading dataset from Kaggle...")
    kagglehub = require_kagglehub()
    return Path(kagglehub.dataset_download("quadeer15sh/augmented-forest-segmentation"))


def summarize_dataset(pairs: list[tuple[Path, Path]], cfg: Config) -> dict[str, int]:
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    return {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
    }


def selected_models(models: list[str] | None) -> list[str]:
    return list(models or MODEL_CHOICES)


def print_dry_run_summary(
    dataset_path: Path,
    pairs: list[tuple[Path, Path]],
    cfg: Config,
    models: list[str] | None = None,
) -> None:
    split_summary = summarize_dataset(pairs, cfg)
    model_names = selected_models(models)
    print("\nDry run complete")
    print(f"- dataset: {dataset_path}")
    print(f"- total pairs: {split_summary['total_pairs']}")
    print(
        "- split: "
        f"train={split_summary['train_pairs']}, "
        f"val={split_summary['val_pairs']}, "
        f"test={split_summary['test_pairs']}"
    )
    print(f"- image size: {cfg.image_size}")
    print(f"- batch size: {cfg.batch_size}")
    print(f"- models: {', '.join(model_names)}")


def require_kagglehub():
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kagglehub is required when --data-dir is omitted. "
            "Install dependencies with `pip install -r requirements.txt`, "
            "or pass --data-dir to use a local dataset."
        ) from exc
    return kagglehub


def require_training_stack():
    try:
        import pandas as pd
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pandas and PyTorch are required for training. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    from src.dataset import create_dataloaders
    from src.models import build_model
    from src.trainer import evaluate, train_model

    return pd, torch, create_dataloaders, build_model, evaluate, train_model


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = build_config(args)

    dataset_path = resolve_dataset_path(args.data_dir)
    print(f"Dataset: {dataset_path}")

    pairs = load_pairs(dataset_path)
    print(f"Total pairs: {len(pairs)}")

    if args.dry_run:
        print_dry_run_summary(dataset_path, pairs, cfg, args.models)
        return 0

    cfg.initialize_runtime()
    seed_everything(cfg.seed)
    print(f"Device: {cfg.device} | AMP: {cfg.use_amp}")

    pd, torch, create_dataloaders, build_model, evaluate, train_model = require_training_stack()

    train_loader, val_loader, test_loader, *_ = create_dataloaders(
        pairs,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    archs = selected_models(args.models)
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
    return 0


def cli_entry(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_entry())
