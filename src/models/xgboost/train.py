"""
Train an XGBoost pixel classifier for mangrove segmentation.

Memory-efficient version: streams through ALL patches, sampling a fixed
number of pixels per patch from BOTH classes.  Stays under ~1 GB RAM.

Usage:
    python src/models/xgboost/train.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import time
import json
import gc

import numpy as np
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.xgboost.features import extract_features, FEATURE_NAMES
from src.evaluation.metrics import compute_metrics_from_patches, print_metrics


def _load_split(split_path: str):
    """Return list of patch paths from a split file."""
    with open(split_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def _mask_path(img_path: str) -> str:
    """Derive the mask path from an image path."""
    directory = os.path.dirname(img_path)
    basename = os.path.basename(img_path).replace("img_", "mask_")
    return os.path.join(directory, basename)


def _count_class_distribution(patch_paths: list):
    """First pass: count class distribution from masks only (no images loaded).

    Returns (total_pos, total_neg).
    """
    total_pos = 0
    total_neg = 0
    n = len(patch_paths)

    for i, path in enumerate(patch_paths):
        mask = np.load(_mask_path(path))
        pos = int(mask.sum())
        total_pos += pos
        total_neg += mask.size - pos
        del mask

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print("    Counted %d/%d patches (pos so far: %d)" % (i + 1, n, total_pos))

    return total_pos, total_neg


def _stream_collect_pixels(patch_paths: list, target_total: int = 200_000,
                           seed: int = 42):
    """Second pass: stream through ALL patches, collecting pixels.

    Strategy:
      - Sample a FIXED number of pixels per patch from BOTH classes
      - This caps memory at ~target_total pixels regardless of patch count

    Returns X (n, 10), y (n,).
    """
    rng = np.random.RandomState(seed)
    n = len(patch_paths)
    # Sample a FIXED number of pixels per patch from BOTH classes
    pos_per_patch = max(target_total // (2 * n), 2)
    neg_per_patch = max(target_total // (2 * n), 2)

    print("    Streaming %d patches, ~%d pos + ~%d neg pixels/patch..." % (
        n, pos_per_patch, neg_per_patch))

    chunks = []
    label_chunks = []

    for i, path in enumerate(patch_paths):
        img = np.load(path)
        mask = np.load(_mask_path(path))
        feats = extract_features(img)  # (65536, 10)
        labels = mask.ravel()

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        # Sample from BOTH classes
        if len(pos_idx) > 0:
            n_sample_pos = min(pos_per_patch, len(pos_idx))
            sampled_pos = rng.choice(pos_idx, size=n_sample_pos, replace=False)
            chunks.append(feats[sampled_pos])
            label_chunks.append(np.ones(n_sample_pos, dtype=np.int32))

        if len(neg_idx) > 0:
            n_sample_neg = min(neg_per_patch, len(neg_idx))
            sampled_neg = rng.choice(neg_idx, size=n_sample_neg, replace=False)
            chunks.append(feats[sampled_neg])
            label_chunks.append(np.zeros(n_sample_neg, dtype=np.int32))

        del img, mask, feats, labels

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print("    Processed %d/%d patches" % (i + 1, n))

    X = np.concatenate(chunks, axis=0)
    y = np.concatenate(label_chunks, axis=0)
    del chunks, label_chunks
    gc.collect()

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("    Final dataset: %d pixels (%d pos, %d neg)" % (
        len(y), int(y.sum()), len(y) - int(y.sum())))

    return X, y


def _stream_collect_val_pixels(patch_paths: list, target: int = 20_000,
                               seed: int = 123):
    """Collect a small validation pixel sample for early stopping eval_set.

    Samples a fixed budget from BOTH classes per patch to cap memory.
    """
    rng = np.random.RandomState(seed)
    n = len(patch_paths)
    pos_per_patch = max(target // (2 * n), 2)
    neg_per_patch = max(target // (2 * n), 2)

    chunks = []
    label_chunks = []

    for i, path in enumerate(patch_paths):
        img = np.load(path)
        mask = np.load(_mask_path(path))
        feats = extract_features(img)
        labels = mask.ravel()

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        if len(pos_idx) > 0:
            n_sample_pos = min(pos_per_patch, len(pos_idx))
            sampled_pos = rng.choice(pos_idx, size=n_sample_pos, replace=False)
            chunks.append(feats[sampled_pos])
            label_chunks.append(np.ones(n_sample_pos, dtype=np.int32))

        if len(neg_idx) > 0:
            n_sample_neg = min(neg_per_patch, len(neg_idx))
            sampled_neg = rng.choice(neg_idx, size=n_sample_neg, replace=False)
            chunks.append(feats[sampled_neg])
            label_chunks.append(np.zeros(n_sample_neg, dtype=np.int32))

        del img, mask, feats, labels

        if (i + 1) % 200 == 0 or (i + 1) == n:
            print("    Val sampling: %d/%d patches" % (i + 1, n))

    X = np.concatenate(chunks, axis=0)
    y = np.concatenate(label_chunks, axis=0)
    del chunks, label_chunks
    gc.collect()

    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("    Val pixels: %d (%d pos, %d neg)" % (
        len(y), int(y.sum()), len(y) - int(y.sum())))
    return X, y


def _eval_on_split(model, patch_paths: list, label: str = ""):
    """Evaluate the model patch-by-patch (low memory) and return metrics."""
    pred_patches = []
    gt_patches = []
    total = len(patch_paths)

    for i, path in enumerate(patch_paths):
        img = np.load(path)
        mask = np.load(_mask_path(path))
        feats = extract_features(img)
        preds = model.predict(feats).reshape(mask.shape)
        pred_patches.append(preds)
        gt_patches.append(mask)
        del img, feats

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print("    %s: evaluated %d/%d patches..." % (label, i + 1, total))

    return compute_metrics_from_patches(pred_patches, gt_patches)


def main():
    print("=" * 60)
    print("  XGBoost Mangrove Pixel Classifier -- Training")
    print("  (Streaming version: ALL patches, sampled pixels)")
    print("=" * 60)

    train_split = "data/splits/train.txt"
    val_split = "data/splits/val.txt"

    train_paths = _load_split(train_split)
    val_paths = _load_split(val_split)
    print("Train patches: %d" % len(train_paths))
    print("Val patches:   %d" % len(val_paths))

    # 1. First pass: count class distribution (masks only, no images)
    print("")
    print("[1/6] Counting class distribution (masks only)...")
    total_pos, total_neg = _count_class_distribution(train_paths)
    scale_pos_weight = total_neg / max(total_pos, 1)
    print("  Total positive pixels: %d" % total_pos)
    print("  Total negative pixels: %d" % total_neg)
    print("  scale_pos_weight = %.2f" % scale_pos_weight)

    # 2. Second pass: collect training samples (stream, low memory)
    print("")
    print("[2/6] Streaming training pixels from ALL %d patches..." % len(train_paths))
    X_train, y_train = _stream_collect_pixels(train_paths, target_total=200_000)

    # 3. Collect validation pixels for early stopping
    print("")
    print("[3/6] Streaming validation pixels from %d patches..." % len(val_paths))
    X_val, y_val = _stream_collect_val_pixels(val_paths, target=20_000)

    # 4. Train XGBoost
    print("")
    print("[4/6] Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=10,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    training_time = time.time() - train_start
    n_estimators_used = model.best_iteration + 1 if model.best_iteration else model.n_estimators
    print("")
    print("  Training time: %.1fs" % training_time)
    print("  Best iteration: %d" % n_estimators_used)

    # Free training data
    del X_train, y_train, X_val, y_val
    gc.collect()

    # 5. Save model
    os.makedirs("models", exist_ok=True)
    model.save_model("models/xgboost_model.json")
    print("")
    print("[5/6] Model saved to models/xgboost_model.json")

    # 6. Feature importance plot
    os.makedirs("results", exist_ok=True)
    importances = model.feature_importances_
    importance_dict = {
        name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)
    }

    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_NAMES[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#2e86ab",
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost -- Mangrove Classifier Feature Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()
    print("  Feature importance plot -> results/feature_importance.png")

    # 7. Evaluate on val set (patch-by-patch, low memory)
    print("")
    print("[6/6] Evaluating on validation set (patch-by-patch)...")
    val_metrics = _eval_on_split(model, val_paths, label="Val")
    print_metrics(val_metrics)

    # Save intermediate info for evaluate.py
    info = {
        "training_time_sec": round(training_time, 2),
        "n_estimators_used": n_estimators_used,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "feature_importance": importance_dict,
        "val_metrics": {
            "precision": round(val_metrics["precision"], 4),
            "recall": round(val_metrics["recall"], 4),
            "iou": round(val_metrics["iou"], 4),
            "f1": round(val_metrics["f1"], 4),
        },
    }
    with open("results/xgboost_train_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("")
    print("  Training info saved to results/xgboost_train_info.json")
    print("  Done! Run evaluate.py next for test-set metrics.")


if __name__ == "__main__":
    main()
