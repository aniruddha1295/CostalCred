"""
U-Net training loop for mangrove segmentation.

Uses mixed-precision training, gradient accumulation, and early stopping
on validation IoU. Designed to run on 4 GB VRAM (RTX 3050) with
batch_size=8 and 256x256 patches.
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.unet.dataset import MangroveDataset, compute_pos_weight, get_dataloaders
from src.models.unet.model import build_unet, count_parameters
from src.evaluation.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "unet_best.pt")
SPLITS_DIR = os.path.join(REPO_ROOT, "data", "splits")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, accumulation_steps=1
) -> float:
    """Run one training epoch with mixed-precision and gradient accumulation."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader), desc="  Train", leave=False)
    for batch_idx, (images, masks) in pbar:
        try:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                logits = model(images)
                loss = criterion(logits, masks) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(
                    f"\n  [WARNING] OOM on batch {batch_idx}, skipping. "
                    "Consider reducing batch_size or patch_size."
                )
                continue
            raise

    return running_loss / max(num_batches, 1)


def validate(model, loader, criterion, device) -> tuple:
    """Validate the model and return (avg_loss, metrics_dict)."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds = []
    all_gts = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)

            running_loss += loss.item()
            num_batches += 1

            # Predictions: sigmoid → threshold → uint8 numpy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
            gts = masks.cpu().numpy().astype(np.uint8)

            all_preds.append(preds)
            all_gts.append(gts)

    # Concatenate all patches into single arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    metrics = compute_metrics(all_preds, all_gts)
    avg_loss = running_loss / max(num_batches, 1)

    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(config=None) -> dict:
    """Train the U-Net model with the given configuration."""
    default_config = {
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 50,
        "patience": 7,
        "accumulation_steps": 1,
        "num_workers": 0,
        "encoder_name": "resnet18",
        "encoder_weights": "imagenet",
    }
    if config is not None:
        default_config.update(config)
    config = default_config

    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)} ({device})")
    else:
        print(f"Device: {device} (no GPU detected — training will be slow)")

    # ---- Data -------------------------------------------------------------
    train_loader, val_loader, _test_loader = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ---- Positive-class weight --------------------------------------------
    train_split_file = os.path.join(SPLITS_DIR, "train.txt")
    pos_weight = compute_pos_weight(train_split_file)
    print(f"Positive-class weight: {pos_weight:.2f}")

    # ---- Model ------------------------------------------------------------
    model = build_unet(
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
    )
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # ---- Loss, optimizer, scheduler ---------------------------------------
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Training loop ----------------------------------------------------
    best_iou = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    epochs = config["epochs"]
    patience = config["patience"]

    print(f"\nStarting training for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=config["accumulation_steps"],
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_iou = val_metrics["iou"]

        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_iou": best_iou,
                    "config": config,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> New best model saved (IoU: {best_iou:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time:.1f}s  |  Best IoU: {best_iou:.4f} (epoch {best_epoch})")

    return {
        "model": "unet",
        "encoder": config["encoder_name"],
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "training_time_sec": round(training_time, 1),
        "config": config,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for mangrove segmentation")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: 7)")
    parser.add_argument("--accumulation-steps", type=int, default=None, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: 0)")
    args = parser.parse_args()

    # Build config from CLI overrides
    cli_config = {}
    if args.batch_size is not None:
        cli_config["batch_size"] = args.batch_size
    if args.lr is not None:
        cli_config["lr"] = args.lr
    if args.epochs is not None:
        cli_config["epochs"] = args.epochs
    if args.patience is not None:
        cli_config["patience"] = args.patience
    if args.accumulation_steps is not None:
        cli_config["accumulation_steps"] = args.accumulation_steps
    if args.num_workers is not None:
        cli_config["num_workers"] = args.num_workers

    results = train(cli_config if cli_config else None)

    # Save training summary for evaluate.py
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    # Convert config values to JSON-serializable types
    summary = {k: v for k, v in results.items() if k != "config"}
    summary["config"] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                         for k, v in results.get("config", {}).items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to {summary_path}")

    print("\n--- Training Summary ---")
    for key, value in results.items():
        if key != "config":
            print(f"  {key}: {value}")

    print("\nRun evaluate.py next to get test metrics")
