"""
Evaluate the trained XGBoost mangrove classifier on the test set.

Usage:
    python src/models/xgboost/evaluate.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import json

import numpy as np
import xgboost as xgb

from src.models.xgboost.features import extract_features, FEATURE_NAMES
from src.evaluation.metrics import (
    compute_metrics_from_patches,
    print_metrics,
)


def _load_split(split_path: str):
    with open(split_path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def _mask_path(img_path: str) -> str:
    directory = os.path.dirname(img_path)
    basename = os.path.basename(img_path).replace("img_", "mask_")
    return os.path.join(directory, basename)


def _eval_patches(model, patch_paths: list):
    """Predict + collect metrics patch-by-patch (low memory)."""
    pred_patches = []
    gt_patches = []

    for i, path in enumerate(patch_paths):
        img = np.load(path)
        mask = np.load(_mask_path(path))
        feats = extract_features(img)
        preds = model.predict(feats).reshape(mask.shape)
        pred_patches.append(preds)
        gt_patches.append(mask)

        if (i + 1) % 20 == 0 or (i + 1) == len(patch_paths):
            print("  Processed %d/%d patches" % (i + 1, len(patch_paths)))

    return compute_metrics_from_patches(pred_patches, gt_patches)


def main():
    print("=" * 60)
    print("  XGBoost Mangrove Pixel Classifier -- Evaluation")
    print("=" * 60)

    model_path = "models/xgboost_model.json"
    if not os.path.exists(model_path):
        print("ERROR: Model not found at %s. Run train.py first." % model_path)
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Loaded model from %s" % model_path)

    # Test set
    test_paths = _load_split("data/splits/test.txt")
    print("Test patches: %d" % len(test_paths))
    print("")
    print("Evaluating on test set...")
    test_metrics = _eval_patches(model, test_paths)

    print("")
    print("--- Test Set ---")
    print_metrics(test_metrics)

    # Val set (for the JSON output)
    val_paths = _load_split("data/splits/val.txt")
    print("Val patches: %d" % len(val_paths))
    print("Evaluating on validation set...")
    val_metrics = _eval_patches(model, val_paths)

    print("")
    print("--- Validation Set ---")
    print_metrics(val_metrics)

    # Load training info saved by train.py
    train_info_path = "results/xgboost_train_info.json"
    if os.path.exists(train_info_path):
        with open(train_info_path) as f:
            train_info = json.load(f)
    else:
        train_info = {
            "training_time_sec": -1,
            "n_estimators_used": -1,
            "scale_pos_weight": -1,
            "feature_importance": {},
        }

    # Write results/xgboost.json
    results = {
        "model": "xgboost",
        "test_metrics": {
            "precision": round(test_metrics["precision"], 4),
            "recall": round(test_metrics["recall"], 4),
            "iou": round(test_metrics["iou"], 4),
            "f1": round(test_metrics["f1"], 4),
        },
        "val_metrics": {
            "precision": round(val_metrics["precision"], 4),
            "recall": round(val_metrics["recall"], 4),
            "iou": round(val_metrics["iou"], 4),
            "f1": round(val_metrics["f1"], 4),
        },
        "training_time_sec": train_info["training_time_sec"],
        "n_estimators_used": train_info["n_estimators_used"],
        "scale_pos_weight": train_info["scale_pos_weight"],
        "feature_importance": train_info["feature_importance"],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/xgboost.json", "w") as f:
        json.dump(results, f, indent=2)

    print("")
    print("Results saved to results/xgboost.json")
    print("Done!")


if __name__ == "__main__":
    main()
