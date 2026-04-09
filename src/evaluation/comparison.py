"""
Build the 3-model comparison table and grouped bar chart.

Reads per-model result JSONs from ``results/`` and produces:

- ``results/comparison_table.csv``  — CSV of the comparison DataFrame
- ``results/comparison_table.png``  — Grouped bar chart (4 metrics x 3 models)
- Console-printed table with IoU highlighted as the primary metric

Usage::

    python src/evaluation/comparison.py

No arguments — all paths are relative to the repository root.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

MODEL_NAMES = ["ndvi", "xgboost", "unet"]
DISPLAY_NAMES = {"ndvi": "NDVI Threshold", "xgboost": "XGBoost", "unet": "U-Net"}
METRICS = ["precision", "recall", "iou", "f1"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "iou": "IoU",
    "f1": "F1",
}
PRIMARY_METRIC = "iou"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_result(model_name: str) -> dict | None:
    """Load a model's result JSON, returning *None* on failure."""
    path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    if not os.path.isfile(path):
        print(f"[WARNING] {path} not found — skipping {model_name}")
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARNING] Failed to read {path}: {exc} — skipping {model_name}")
        return None


def _extract_row(model_name: str, result: dict) -> dict:
    """Pull the fields we need into a flat dict for the DataFrame."""
    metrics = result.get("test_metrics", {})
    return {
        "Model": DISPLAY_NAMES.get(model_name, model_name),
        "Precision": metrics.get("precision", float("nan")),
        "Recall": metrics.get("recall", float("nan")),
        "IoU": metrics.get("iou", float("nan")),
        "F1": metrics.get("f1", float("nan")),
        "Training Time (s)": result.get("training_time_sec", float("nan")),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def build_comparison_table() -> pd.DataFrame:
    """Load all available model results and return a comparison DataFrame."""
    rows: list[dict] = []
    for name in MODEL_NAMES:
        result = _load_result(name)
        if result is not None:
            rows.append(_extract_row(name, result))

    if not rows:
        print("[ERROR] No model result files found in results/. Nothing to compare.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values("IoU", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-site breakdown (optional)
# ---------------------------------------------------------------------------


def build_per_site_table() -> pd.DataFrame | None:
    """If any result JSON contains ``test_metrics_per_site``, build a table."""
    rows: list[dict] = []
    for name in MODEL_NAMES:
        result = _load_result(name)
        if result is None:
            continue
        per_site = result.get("test_metrics_per_site")
        if not per_site:
            continue
        for site, metrics in per_site.items():
            rows.append(
                {
                    "Model": DISPLAY_NAMES.get(name, name),
                    "Site": site,
                    "Precision": metrics.get("precision", float("nan")),
                    "Recall": metrics.get("recall", float("nan")),
                    "IoU": metrics.get("iou", float("nan")),
                    "F1": metrics.get("f1", float("nan")),
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["Model", "IoU"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_comparison(df: pd.DataFrame, save_path: str) -> None:
    """Create a grouped bar chart comparing metric scores across models."""
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=len(METRICS))

    metric_cols = ["Precision", "Recall", "IoU", "F1"]
    plot_df = df.melt(
        id_vars="Model",
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_plot = sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette,
        edgecolor="0.3",
        ax=ax,
    )

    # Annotate each bar with its value
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt="%.3f", fontsize=7, padding=2)

    # Mark IoU as the primary metric in the legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        f"{lbl} (primary)" if lbl == "IoU" else lbl for lbl in labels
    ]
    ax.legend(handles, labels, title="Metric", loc="lower right", fontsize=8)

    ax.set_title("Model Comparison \u2014 Mangrove Segmentation", fontsize=13, weight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)  # room for bar labels
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Chart saved to {save_path}")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_table(df: pd.DataFrame) -> None:
    """Print the comparison table with IoU highlighted."""
    print()
    print("=" * 72)
    print("  Model Comparison — Mangrove Segmentation")
    print("  (sorted by IoU descending; IoU is the primary metric)")
    print("=" * 72)

    # Header
    header = (
        f"  {'Model':<18} {'Precision':>10} {'Recall':>10} "
        f"{'IoU*':>10} {'F1':>10} {'Time (s)':>10}"
    )
    print(header)
    print("  " + "-" * 68)

    for _, row in df.iterrows():
        line = (
            f"  {row['Model']:<18} "
            f"{row['Precision']:>10.4f} "
            f"{row['Recall']:>10.4f} "
            f"{row['IoU']:>10.4f} "
            f"{row['F1']:>10.4f} "
            f"{row['Training Time (s)']:>10.1f}"
        )
        print(line)

    print("  " + "-" * 68)
    print("  * IoU (Intersection over Union) is the primary ranking metric.")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Overall comparison ---
    df = build_comparison_table()
    print_table(df)

    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV saved to {csv_path}")

    png_path = os.path.join(RESULTS_DIR, "comparison_table.png")
    plot_comparison(df, png_path)

    # --- Per-site breakdown (if available) ---
    site_df = build_per_site_table()
    if site_df is not None:
        print()
        print("Per-Site Breakdown")
        print("-" * 72)
        print(site_df.to_string(index=False))

        site_csv = os.path.join(RESULTS_DIR, "comparison_per_site.csv")
        site_df.to_csv(site_csv, index=False)
        print(f"\n[INFO] Per-site CSV saved to {site_csv}")
    else:
        print("[INFO] No per-site breakdowns found in result JSONs.")


if __name__ == "__main__":
    main()
