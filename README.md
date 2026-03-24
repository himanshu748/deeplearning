# ForestSight AI -- Forest Detection from Satellite Imagery

Binary semantic segmentation of aerial/satellite images to detect forest cover using deep learning.

## Models

| Model | Encoder | Key Feature |
|-------|---------|-------------|
| U-Net | EfficientNet-B4 | Strong baseline |
| Attention U-Net | EfficientNet-B4 + scSE | Boundary-focused attention |
| DeepLabV3+ | ResNet-101 + ASPP | Multi-scale context |

## Quick Start

```bash
pip install -r requirements.txt
python train.py --epochs 50 --models unet unet_attention deeplabv3plus
```

## CLI Options

```
--epochs        Number of training epochs (default: 50)
--batch-size    Batch size (default: 16)
--lr            Learning rate (default: 1e-4)
--image-size    Input image size (default: 256)
--models        Architectures to train: unet, unet_attention, deeplabv3plus
--data-dir      Path to local dataset (auto-downloads from Kaggle if omitted)
--seed          Random seed (default: 42)
```

## Docker

```bash
docker build -t forestsight .
docker run --gpus all forestsight --epochs 50
```

## Project Structure

```
├── src/
│   ├── config.py      # Dataclass configuration
│   ├── dataset.py     # Dataset, augmentation, loaders
│   ├── models.py      # Model factory
│   ├── losses.py      # Dice+BCE loss
│   ├── metrics.py     # IoU, Dice, Accuracy, Precision, Recall
│   ├── trainer.py     # Training engine (AMP, early stopping)
│   └── visualize.py   # Plotting utilities
├── train.py           # CLI entry point
├── forest_detection.ipynb  # Interactive notebook
├── SRS.md             # Software Requirements Specification
├── Dockerfile         # Container support
└── requirements.txt
```

## Dataset

[Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation) (Kaggle). Auto-downloaded via `kagglehub`.
