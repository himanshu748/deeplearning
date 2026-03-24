import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor.cpu() * s + m).clamp(0, 1)


def plot_training_curves(histories: dict, save_path: str = None):
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, h) in enumerate(histories.items()):
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0, col].plot(ep, h["train_loss"], label="Train", linewidth=2)
        axes[0, col].plot(ep, h["val_loss"], label="Val", linewidth=2)
        axes[0, col].set_title(f"{name} -- Loss", fontweight="bold")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(ep, h["train_iou"], label="Train", linewidth=2)
        axes[1, col].plot(ep, h["val_iou"], label="Val", linewidth=2)
        axes[1, col].set_title(f"{name} -- IoU", fontweight="bold")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bars(test_results: dict, save_path: str = None):
    metrics = ["IoU", "Dice", "Accuracy", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, m) in enumerate(test_results.items()):
        vals = [m[k] for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- Test Set", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(model, loader, device, n_samples: int = 4, save_path: str = None):
    model.eval()
    images, masks = next(iter(loader))
    with torch.no_grad():
        preds = torch.sigmoid(model(images.to(device))).cpu()

    n = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    fig.suptitle("Predictions", fontsize=16, fontweight="bold")
    if n == 1:
        axes = axes.reshape(1, 4)

    for i in range(n):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        gt = masks[i, 0].numpy()
        pr = (preds[i, 0] > 0.5).float().numpy()
        overlay = img.copy()
        overlay[pr > 0.5] = overlay[pr > 0.5] * 0.5 + np.array([0, 0.8, 0]) * 0.5

        for ax in axes[i]:
            ax.axis("off")
        axes[i, 0].imshow(img); axes[i, 0].set_title("Input")
        axes[i, 1].imshow(gt, cmap="Greens"); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pr, cmap="Greens"); axes[i, 2].set_title("Prediction")
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("Overlay")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(model, loader, device, save_path: str = None):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            out = model(images.to(device))
            p = (torch.sigmoid(out) > 0.5).float().cpu().view(-1)
            all_preds.append(p)
            all_targets.append(masks.view(-1))

    y_pred = torch.cat(all_preds).numpy().astype(int)
    y_true = torch.cat(all_targets).numpy().astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")
    labels = ["Non-Forest", "Forest"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title("Counts"); axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Greens", ax=axes[1],
                xticklabels=labels, yticklabels=labels)
    axes[1].set_title("Normalized"); axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
