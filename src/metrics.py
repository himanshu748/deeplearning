import torch


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()
    tn = ((1 - preds_bin) * (1 - targets)).sum()
    eps = 1e-6
    return {
        "iou": ((tp + eps) / (tp + fp + fn + eps)).item(),
        "dice": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item(),
        "accuracy": ((tp + tn) / (tp + tn + fp + fn)).item(),
        "precision": ((tp + eps) / (tp + fp + eps)).item(),
        "recall": ((tp + eps) / (tp + fn + eps)).item(),
    }
