# ForestSight AI -- Forest Detection from Satellite Imagery

Binary semantic segmentation of aerial/satellite images to detect forest cover using deep learning.

This is a reproducible training scaffold: dataset discovery, deterministic train/validation/test splits, model training, evaluation, and visualization helpers live in small modules instead of only in the notebook.

## Models

| Model | Encoder | Key Feature |
|-------|---------|-------------|
| U-Net | EfficientNet-B4 | Strong baseline |
| Attention U-Net | EfficientNet-B4 + scSE | Boundary-focused attention |
| DeepLabV3+ | ResNet-101 + ASPP | Multi-scale context |

## Quick Start

```bash
pip install -r requirements.txt
python train.py --data-dir /path/to/forest-dataset --dry-run
python train.py --data-dir /path/to/forest-dataset --epochs 5 --models unet
```

The CLI is local-first by default. If `--data-dir` is omitted, it refuses to continue unless you explicitly add `--download-data` to fetch the Kaggle dataset with `kagglehub`. Configure Kaggle credentials outside the repo; never commit `kaggle.json`.

## CLI Options

```
--epochs          Number of training epochs (default: 50)
--batch-size      Batch size (default: 16)
--lr              Learning rate (default: 1e-4)
--image-size      Input image size (default: 256)
--num-workers     DataLoader worker processes (default: 4)
--checkpoint-dir  Directory for checkpoints/results (default: checkpoints)
--models          Architectures to train: unet, unet_attention, deeplabv3plus
--data-dir        Path to local dataset
--download-data   Download Kaggle dataset if --data-dir is omitted
--seed            Random seed (default: 42)
--dry-run         Validate config, dataset pairing/readability, and splits without training
--split-manifest  Optional JSON path for the deterministic train/val/test split manifest
```

## Verification

These checks run without downloading the dataset:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/deeplearning-pycache python3 -m compileall train.py src tests
python3 -m unittest discover -s tests
```

The lightweight tests cover split determinism, configuration validation, CLI dry-run helpers, import-safe model validation, local dataset image/mask pairing, and paired-file readability checks. They do not download Kaggle data or train a model.

Use `--dry-run` before any GPU run to confirm the local dataset is discoverable, paired files are readable and non-empty, and splits produce non-empty train/validation/test sets. With `--data-dir`, this path does not build models, create DataLoaders, initialize Torch runtime state, or create checkpoint directories:

```bash
python train.py --data-dir /path/to/forest-dataset --dry-run
```

To save the exact deterministic split assignment for review or repeat runs:

```bash
python train.py --data-dir /path/to/forest-dataset --dry-run --split-manifest splits.json
```

## Docker

```bash
docker build -t forestsight .
docker run --gpus all -v /path/to/dataset:/data forestsight --data-dir /data --epochs 5 --models unet
```

## Project Structure

```
├── src/
│   ├── config.py      # Dataclass configuration
│   ├── dataset.py     # Dataset, augmentation, loaders
│   ├── models.py      # Model factory
│   ├── losses.py      # Dice+BCE loss
│   ├── metrics.py     # IoU, Dice, Accuracy, Precision, Recall
│   ├── splits.py      # Deterministic train/val/test split helpers
│   ├── trainer.py     # Training engine (AMP, early stopping)
│   └── visualize.py   # Plotting utilities
├── train.py           # CLI entry point
├── tests/             # Lightweight tests for pure helpers
├── forest_detection.ipynb  # Interactive notebook
├── SRS.md             # Software Requirements Specification
├── Dockerfile         # Container support
└── requirements.txt
```

## Dataset

[Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation) (Kaggle). Download it explicitly with `--download-data`, or provide a local copy with `--data-dir`.

Expected local layout can be either explicit `images/` and `masks/` directories, or compatible image/mask directories whose files share stems such as `sample.png`, `sample_mask.png`, or `sample_sat_01.png` / `sample_mask_01.png`.
