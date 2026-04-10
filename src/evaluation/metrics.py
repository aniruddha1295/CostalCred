"""
Shared evaluation metrics for binary mangrove segmentation.

All three models (NDVI baseline, XGBoost, U-Net) use these same
functions so that results are directly comparable.

We deliberately omit plain accuracy: mangroves cover ~2% of pixels,
so a model predicting all-zero achieves 98% accuracy. Instead we
report Precision, Recall, IoU (Jaccard), and F1 — metrics that
are meaningful under severe class imbalance.
"""

import numpy as np


def _confusion_counts(prediction: np.ndarray, ground_truth: np.ndarray):
    """Return TP, FP, FN, TN counts for binary arrays.

    Parameters
    ----------
    prediction : np.ndarray
        Binary prediction array (0 or 1).
    ground_truth : np.ndarray
        Binary ground-truth array (0 or 1).

    Returns
    -------
    tp, fp, fn, tn : int
    """
    pred = prediction.astype(bool).ravel()
    gt = ground_truth.astype(bool).ravel()

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))

    return tp, fp, fn, tn


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute pixel-level binary segmentation metrics.

    We avoid plain accuracy because mangroves are ~2%% of pixels —
    a trivial all-negative model would score 98%% accuracy.  Instead
    we report Precision, Recall, IoU, and F1, which require the model
    to actually find the positive (mangrove) class.

    Parameters
    ----------
    prediction : np.ndarray
        Binary prediction array with values 0 (not mangrove) and
        1 (mangrove).  Any shape; will be flattened internally.
    ground_truth : np.ndarray
        Binary ground-truth array with the same shape as *prediction*.

    Returns
    -------
    dict
        Keys: ``"precision"``, ``"recall"``, ``"iou"``, ``"f1"``,
        ``"tp"``, ``"fp"``, ``"fn"``, ``"tn"``.
        All values are Python floats (safe for ``json.dump``).
    """
    tp, fp, fn, tn = _confusion_counts(prediction, ground_truth)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def compute_metrics_from_patches(
    pred_patches: list, gt_patches: list
) -> dict:
    """Compute *global* metrics over a list of patch pairs.

    Rather than averaging per-patch metrics (which biases toward
    patches with few positive pixels), we concatenate all patches
    into one large array and compute metrics once.  This gives a
    single, unbiased set of numbers for the whole test set.

    Parameters
    ----------
    pred_patches : list[np.ndarray]
        List of binary prediction arrays (one per patch).
    gt_patches : list[np.ndarray]
        Corresponding list of binary ground-truth arrays.

    Returns
    -------
    dict
        Same format as :func:`compute_metrics`.

    Raises
    ------
    ValueError
        If the two lists have different lengths.
    """
    if len(pred_patches) != len(gt_patches):
        raise ValueError(
            f"Mismatched patch counts: {len(pred_patches)} predictions "
            f"vs {len(gt_patches)} ground-truth patches."
        )

    all_preds = np.concatenate([p.ravel() for p in pred_patches])
    all_gts = np.concatenate([g.ravel() for g in gt_patches])

    return compute_metrics(all_preds, all_gts)


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dictionary to the console.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_metrics` or
        :func:`compute_metrics_from_patches`.
    """
    print("=" * 40)
    print("  Binary Segmentation Metrics")
    print("=" * 40)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print("-" * 40)
    print(f"  TP: {metrics['tp']:.0f}   FP: {metrics['fp']:.0f}")
    print(f"  FN: {metrics['fn']:.0f}   TN: {metrics['tn']:.0f}")
    print("=" * 40)
