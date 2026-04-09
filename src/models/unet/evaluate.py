"""
Evaluate trained U-Net on the test set.

Produces:
  - results/unet.json          — metrics in comparison.py format
  - results/unet_pred_*.png    — side-by-side visualisations (RGB | GT | Pred)
"""

import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.unet.dataset import MangroveDataset, get_dataloaders
from src.models.unet.model import build_unet
from src.evaluation.metrics import compute_metrics, print_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "models", "unet_best.pt")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SPLITS_DIR = os.path.join(REPO_ROOT, "data", "splits")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained U-Net from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint saved by ``train.py``.
    device : torch.device
        Target device (cpu / cuda).

    Returns
    -------
    tuple[torch.nn.Module, dict]
        ``(model, checkpoint)`` — the checkpoint dict carries training
        metadata (epoch, config, etc.).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model with the same architecture used during training
    config = checkpoint.get("config", {})
    model = build_unet(
        encoder_name=config.get("encoder_name", "resnet18"),
        encoder_weights=None,  # weights come from the checkpoint
        in_channels=config.get("in_channels", 6),
        classes=config.get("classes", 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict_patches(model, loader, device):
    """Run inference on every batch in *loader*.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net in eval mode.
    loader : torch.utils.data.DataLoader
        DataLoader yielding ``(images, masks)`` batches.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(pred_patches, gt_patches)`` — each element is a
        ``(256, 256)`` uint8 array with values 0 or 1.
    """
    pred_patches = []
    gt_patches = []

    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            with autocast():
                logits = model(images.to(device))

            preds = (
                torch.sigmoid(logits).squeeze(1).cpu().numpy() > 0.5
            ).astype(np.uint8)
            gts = masks.squeeze(1).cpu().numpy().astype(np.uint8)

            for i in range(preds.shape[0]):
                pred_patches.append(preds[i])  # (256, 256)
                gt_patches.append(gts[i])

    return pred_patches, gt_patches


def save_visualizations(model, test_dataset, device, num_samples: int = 5):
    """Save side-by-side prediction visualisations for *num_samples* patches.

    Each saved figure has three panels:
        Sentinel-2 RGB  |  Ground Truth  |  Prediction

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net in eval mode.
    test_dataset : MangroveDataset
        Test split dataset (needed to access ``img_paths``).
    device : torch.device
        Target device.
    num_samples : int
        Number of visualisations to save.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = len(test_dataset)
    indices = np.linspace(0, n - 1, num_samples, dtype=int)

    model.eval()
    for count, idx in enumerate(indices):
        img_tensor, mask_tensor = test_dataset[idx]

        # --- raw (unnormalised) image for RGB display ---
        raw_img = np.load(test_dataset.img_paths[idx]).astype(np.float32)
        # Sentinel-2 bands: B2(0), B3(1), B4(2) → RGB = bands [2, 1, 0]
        rgb = raw_img[[2, 1, 0]].transpose(1, 2, 0)  # (256, 256, 3)
        rgb = np.clip(rgb, 0, 0.3) / 0.3              # contrast stretch → [0, 1]

        # --- prediction ---
        with torch.no_grad():
            with autocast():
                logit = model(img_tensor.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logit).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        gt = mask_tensor.squeeze().numpy().astype(np.uint8)

        # --- plot ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(rgb)
        axes[0].set_title("Sentinel-2 RGB")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="Greens", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="Greens", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"unet_pred_{count}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def save_results_json(metrics: dict, training_info: dict):
    """Write evaluation results to ``results/unet.json``.

    The JSON schema matches what ``comparison.py`` expects so that the
    three-model comparison table can be built automatically.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_metrics`.
    training_info : dict
        Dictionary with training metadata (from checkpoint and/or
        ``training_summary.json``).
    """
    # Resolve training time from multiple possible sources
    training_time = training_info.get("training_time_sec", 0)
    if training_time == 0:
        config = training_info.get("config", {})
        training_time = config.get("training_time_sec", 0)

    result = {
        "model": "unet",
        "encoder": training_info.get("config", {}).get("encoder_name", "resnet18"),
        "best_epoch": training_info.get("epoch", None),
        "training_time_sec": training_time,
        "test_metrics": {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "iou": round(metrics["iou"], 4),
            "f1": round(metrics["f1"], 4),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "unet.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load model -------------------------------------------------------
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    model, checkpoint = load_model(CHECKPOINT_PATH, device)

    # ---- try to load training summary (has training_time_sec) -------------
    summary_path = os.path.join(REPO_ROOT, "models", "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            training_info = json.load(f)
        print(f"Loaded training summary from {summary_path}")
    else:
        training_info = checkpoint
        print("No training_summary.json found; using checkpoint metadata.")

    # ---- dataloaders -------------------------------------------------------
    print("\nBuilding test dataloader ...")
    _, _, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    # Build a standalone test dataset for visualisation (need img_paths)
    test_txt = os.path.join(SPLITS_DIR, "test.txt")
    norm_stats = os.path.join(SPLITS_DIR, "norm_stats.json")
    test_dataset = MangroveDataset(test_txt, norm_stats, augment=False)
    print(f"Test patches: {len(test_dataset)}")

    # ---- predict -----------------------------------------------------------
    print("\nRunning inference on test set ...")
    pred_patches, gt_patches = predict_patches(model, test_loader, device)

    # ---- metrics -----------------------------------------------------------
    print("\nComputing metrics ...")
    all_preds = np.concatenate([p.ravel() for p in pred_patches])
    all_gts = np.concatenate([g.ravel() for g in gt_patches])
    metrics = compute_metrics(all_preds, all_gts)
    print_metrics(metrics)

    # ---- visualisations ----------------------------------------------------
    print("\nSaving prediction visualisations ...")
    save_visualizations(model, test_dataset, device, num_samples=5)

    # ---- results JSON ------------------------------------------------------
    print("\nSaving results JSON ...")
    save_results_json(metrics, training_info)

    print("\nDone! Results at results/unet.json")
