import os
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .splits import split_pairs


def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class ForestDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)
        return image, mask


def find_data_dirs(base_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {base_path}")

    img_dir, mask_dir = None, None
    for root, _, _ in os.walk(base_path):
        name = os.path.basename(root).lower()
        if name in ("images", "image", "img", "forest images"):
            img_dir = Path(root)
        elif name in ("masks", "mask", "labels", "label", "forest masks"):
            mask_dir = Path(root)
    if img_dir is None or mask_dir is None:
        subdirs = sorted(
            [d for d in base_path.rglob("*") if d.is_dir() and any(d.iterdir())]
        )
        for d in subdirs:
            sample = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.tif"))
            if sample:
                if img_dir is None:
                    img_dir = d
                elif mask_dir is None:
                    mask_dir = d
                    break
    return img_dir, mask_dir


def get_mask_path(img_path: Path, mask_dir: Path) -> Optional[Path]:
    stem = img_path.stem
    # Direct match
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Common mask naming conventions
    for pattern in [f"{stem}_mask", f"{stem}_label", f"mask_{stem}"]:
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            candidate = mask_dir / f"{pattern}{ext}"
            if candidate.exists():
                return candidate
    # Handle sat->mask pattern: "10452_sat_08" -> "10452_mask_08"
    if "_sat_" in stem:
        mask_stem = stem.replace("_sat_", "_mask_")
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = mask_dir / f"{mask_stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_pairs(dataset_path: Path) -> List[Tuple[Path, Path]]:
    img_dir, mask_dir = find_data_dirs(dataset_path)
    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not find image/mask dirs in {dataset_path}. "
            f"Found img_dir={img_dir}, mask_dir={mask_dir}"
        )
    image_files = sorted(
        f for f in img_dir.glob("*")
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    pairs = []
    for img_path in image_files:
        mask_path = get_mask_path(img_path, mask_dir)
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No valid image-mask pairs found under {dataset_path}")
    return pairs


def create_dataloaders(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, list, list, list]:
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_ds = ForestDataset(train_pairs, transform=get_train_transforms(image_size))
    val_ds = ForestDataset(val_pairs, transform=get_val_transforms(image_size))
    test_ds = ForestDataset(test_pairs, transform=get_val_transforms(image_size))

    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, train_pairs, val_pairs, test_pairs
