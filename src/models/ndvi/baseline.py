"""
NDVI Threshold Baseline for Mangrove Segmentation
==================================================

Rule-based baseline: no training, just threshold tuning on the
validation set. NDVI = (B8 - B4) / (B8 + B4 + 1e-8). Pixels above
the threshold are predicted as mangrove.

Band layout in patches (6, 256, 256):
  [0]=B2  [1]=B3  [2]=B4(Red)  [3]=B8(NIR)  [4]=B11  [5]=B12

Usage:
    python src/models/ndvi/baseline.py
"""

import sys
import os
import json
import time

# Allow running as script from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics_from_patches, print_metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

SPLITS_DIR = os.path.join(PROJECT_ROOT, 'data', 'splits')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'ndvi.json')

THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

B4_IDX = 2   # Red
B8_IDX = 3   # NIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(split_name: str):
    """Load image and mask patches for a given split (train/val/test).

    Returns (images, masks) where each is a list of np.ndarray.
    """
    split_file = os.path.join(SPLITS_DIR, f'{split_name}.txt')
    with open(split_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    images = []
    masks = []
    skipped = 0

    for rel_path in tqdm(lines, desc=f'Loading {split_name}'):
        img_path = os.path.join(PROJECT_ROOT, rel_path)
        # Derive mask path: replace img_ with mask_
        mask_path = img_path.replace('img_', 'mask_')

        if not os.path.exists(img_path):
            skipped += 1
            continue
        if not os.path.exists(mask_path):
            skipped += 1
            continue

        images.append(np.load(img_path))
        masks.append(np.load(mask_path))

    if skipped > 0:
        print(f'  Warning: skipped {skipped} missing patches in {split_name}')

    print(f'  Loaded {len(images)} patches for {split_name}')
    return images, masks


def compute_ndvi(img: np.ndarray) -> np.ndarray:
    """Compute NDVI from a (6, H, W) patch."""
    nir = img[B8_IDX].astype(np.float64)
    red = img[B4_IDX].astype(np.float64)
    return (nir - red) / (nir + red + 1e-8)


def predict_with_threshold(images: list, threshold: float) -> list:
    """Return list of binary prediction arrays for a given threshold."""
    preds = []
    for img in images:
        ndvi = compute_ndvi(img)
        pred = (ndvi > threshold).astype(np.uint8)
        preds.append(pred)
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()

    # --- Load data ---
    print('\n=== NDVI Threshold Baseline ===\n')
    print('Loading validation set...')
    val_images, val_masks = load_split('val')
    print('Loading test set...')
    test_images, test_masks = load_split('test')

    if len(val_images) == 0:
        print('ERROR: No validation patches loaded. Check data/splits/val.txt')
        sys.exit(1)
    if len(test_images) == 0:
        print('ERROR: No test patches loaded. Check data/splits/test.txt')
        sys.exit(1)

    # --- Threshold search on validation set ---
    print(f'\nSearching {len(THRESHOLDS)} thresholds on validation set...\n')
    threshold_search = {}
    best_threshold = None
    best_iou = -1.0

    for thr in tqdm(THRESHOLDS, desc='Threshold search'):
        preds = predict_with_threshold(val_images, thr)
        metrics = compute_metrics_from_patches(preds, val_masks)
        threshold_search[str(thr)] = {'iou': metrics['iou']}

        print(f'  threshold={thr:.2f}  IoU={metrics["iou"]:.4f}  '
              f'P={metrics["precision"]:.4f}  R={metrics["recall"]:.4f}')

        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            best_threshold = thr

    print(f'\nBest threshold: {best_threshold} (val IoU={best_iou:.4f})')

    # --- Evaluate best threshold on val and test ---
    print('\nEvaluating best threshold on validation set...')
    val_preds = predict_with_threshold(val_images, best_threshold)
    val_metrics = compute_metrics_from_patches(val_preds, val_masks)

    print('\nValidation metrics:')
    print_metrics(val_metrics)

    print('\nEvaluating best threshold on test set...')
    test_preds = predict_with_threshold(test_images, best_threshold)
    test_metrics = compute_metrics_from_patches(test_preds, test_masks)

    print('\nTest metrics:')
    print_metrics(test_metrics)

    elapsed = time.time() - start

    # --- Save results ---
    results = {
        'model': 'ndvi_threshold',
        'best_threshold': best_threshold,
        'val_metrics': {
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'iou': val_metrics['iou'],
            'f1': val_metrics['f1'],
        },
        'test_metrics': {
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'iou': test_metrics['iou'],
            'f1': test_metrics['f1'],
        },
        'training_time_sec': 0,
        'threshold_search': threshold_search,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'\nResults saved to {RESULTS_PATH}')
    print(f'Total runtime: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
