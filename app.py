"""
CoastalCred -- Streamlit Dashboard
Blue Carbon MRV System for Mangrove Ecosystems
"""

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import glob as glob_mod

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CoastalCred",
    page_icon="\U0001F30A",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
PATCHES_DIR = os.path.join(DATA_DIR, "patches")

# Theme colours
COLOR_OCEAN = "#0077B6"
COLOR_MANGROVE = "#2D6A4F"
COLOR_LIGHT = "#90E0EF"
COLOR_ACCENT = "#40916C"
COLOR_WARN = "#E76F51"
PALETTE = [COLOR_OCEAN, COLOR_MANGROVE, COLOR_LIGHT, COLOR_ACCENT, COLOR_WARN]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_patch(patch_path):
    """Load a .npy patch (H, W, C) and its mask."""
    img = np.load(patch_path)
    mask_path = patch_path.replace("_img.npy", "_mask.npy")
    mask = np.load(mask_path) if os.path.exists(mask_path) else None
    return img, mask


def metric_card(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## CoastalCred")
    st.markdown("**Blue Carbon MRV**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Model Comparison", "XGBoost Analysis", "Data Explorer", "Satellite Imagery"],
    )
    st.markdown("---")
    st.caption("Sem VI Mini Project -- RCOEM Dept. of Data Science")

# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("CoastalCred -- Blue Carbon MRV Dashboard")
    st.markdown(
        "A blockchain-based blue carbon registry and MRV system focused on "
        "mangrove ecosystems in India. We measure mangrove cover change via "
        "satellite imagery and ML, then convert hectares to carbon credits "
        "using IPCC Tier 1 methodology."
    )

    st.markdown("---")

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Patches", "7,153")
    with c2:
        metric_card("Sites", "3")
    with c3:
        metric_card("Models Trained", "2 / 3")

    st.markdown("---")

    # Data split breakdown
    st.subheader("Data Split Breakdown")
    split_df = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Test"],
            "Patches": [6374, 708, 71],
        }
    )
    fig_split = go.Figure()
    colours = [COLOR_OCEAN, COLOR_MANGROVE, COLOR_WARN]
    cumulative = 0
    for i, row in split_df.iterrows():
        fig_split.add_trace(
            go.Bar(
                y=["Dataset"],
                x=[row["Patches"]],
                name=f"{row['Split']} ({row['Patches']})",
                orientation="h",
                marker_color=colours[i],
                text=[row["Patches"]],
                textposition="inside",
            )
        )
    fig_split.update_layout(
        barmode="stack",
        height=150,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_title="Number of Patches",
    )
    st.plotly_chart(fig_split, use_container_width=True)

    # Sites
    st.subheader("Study Sites")
    site_data = {
        "Site": ["Sundarbans", "Gulf of Kutch", "Pichavaram"],
        "State": ["West Bengal", "Gujarat", "Tamil Nadu"],
        "Role": ["Train", "Train", "Test (unseen)"],
        "Patches": [5815, 1267, 71],
    }
    st.table(pd.DataFrame(site_data))

# ---------------------------------------------------------------------------
# Page 2: Model Comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")

    ndvi = load_json(os.path.join(RESULTS_DIR, "ndvi.json"))
    xgb = load_json(os.path.join(RESULTS_DIR, "xgboost.json"))
    unet = load_json(os.path.join(RESULTS_DIR, "unet.json"))

    models = {}
    if ndvi:
        models["NDVI Threshold"] = ndvi
    if xgb:
        models["XGBoost"] = xgb
    if unet:
        models["U-Net"] = unet

    if not models:
        st.warning("No result JSON files found in results/.")
        st.stop()

    if not unet:
        st.info("U-Net results not yet available -- awaiting U-Net training.")

    metrics_list = ["precision", "recall", "iou", "f1"]

    # Build dataframe for grouped bar chart
    rows = []
    for name, data in models.items():
        for split in ["val_metrics", "test_metrics"]:
            split_label = "Validation" if "val" in split else "Test"
            if split in data:
                for m in metrics_list:
                    rows.append(
                        {
                            "Model": name,
                            "Split": split_label,
                            "Metric": m.upper() if m != "f1" else "F1",
                            "Value": round(data[split].get(m, 0), 4),
                        }
                    )

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        facet_col="Split",
        color_discrete_sequence=[COLOR_OCEAN, COLOR_MANGROVE, COLOR_ACCENT],
        text_auto=".3f",
    )
    fig.update_layout(height=450, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side tables
    st.subheader("Detailed Metrics")
    col_v, col_t = st.columns(2)

    def make_table(split_key, label):
        tbl = {"Metric": [m.upper() if m != "f1" else "F1" for m in metrics_list]}
        for name, data in models.items():
            if split_key in data:
                tbl[name] = [round(data[split_key].get(m, 0), 4) for m in metrics_list]
        return pd.DataFrame(tbl)

    with col_v:
        st.markdown("**Validation Set**")
        st.table(make_table("val_metrics", "Validation"))
    with col_t:
        st.markdown("**Test Set (Pichavaram -- unseen)**")
        st.table(make_table("test_metrics", "Test"))

    # Generalization gap
    st.subheader("Generalization Gap (Val IoU - Test IoU)")
    gap_rows = []
    for name, data in models.items():
        val_iou = data.get("val_metrics", {}).get("iou", 0)
        test_iou = data.get("test_metrics", {}).get("iou", 0)
        gap_rows.append(
            {"Model": name, "Val IoU": round(val_iou, 4), "Test IoU": round(test_iou, 4), "Gap": round(val_iou - test_iou, 4)}
        )
    gap_df = pd.DataFrame(gap_rows)
    st.table(gap_df)

    st.info(
        "Test metrics are significantly lower because the test site (Pichavaram, Tamil Nadu) "
        "is geographically and ecologically distinct from the training sites (Sundarbans, Gulf of Kutch). "
        "This site-level split honestly measures generalisation to unseen mangrove ecosystems."
    )

# ---------------------------------------------------------------------------
# Page 3: XGBoost Analysis
# ---------------------------------------------------------------------------
elif page == "XGBoost Analysis":
    st.title("XGBoost Analysis")

    xgb = load_json(os.path.join(RESULTS_DIR, "xgboost.json"))
    xgb_info = load_json(os.path.join(RESULTS_DIR, "xgboost_train_info.json"))

    if not xgb and not xgb_info:
        st.warning("No XGBoost results found.")
        st.stop()

    source = xgb or xgb_info
    fi = source.get("feature_importance", {})

    # Feature importance bar chart
    st.subheader("Feature Importance")
    fi_df = (
        pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        .sort_values("Importance", ascending=True)
    )
    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#CAD2C5", "#2D6A4F", "#1B4332"],
        text_auto=".3f",
    )
    fig_fi.update_layout(height=400, showlegend=False, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig_fi, use_container_width=True)

    # Feature importance image
    fi_img_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    if os.path.exists(fi_img_path):
        with st.expander("Feature Importance (saved image)"):
            st.image(fi_img_path, use_container_width=True)

    # Training details
    st.subheader("Training Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Training Time", f"{source.get('training_time_sec', 'N/A')} sec")
    with c2:
        st.metric("Estimators Used", source.get("n_estimators_used", "N/A"))
    with c3:
        st.metric("scale_pos_weight", source.get("scale_pos_weight", "N/A"))

    # Sorted table
    st.subheader("Feature Importance Table")
    fi_table = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_table.index = fi_table.index + 1
    st.table(fi_table)

# ---------------------------------------------------------------------------
# Page 4: Data Explorer
# ---------------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("Data Explorer")

    # Normalization stats
    norm = load_json(os.path.join(SPLITS_DIR, "norm_stats.json"))
    band_names = ["B2", "B3", "B4", "B8", "B11", "B12"]

    if norm:
        st.subheader("Normalization Statistics (Training Set)")
        norm_df = pd.DataFrame(
            {"Band": band_names, "Mean": [round(v, 6) for v in norm["mean"]], "Std": [round(v, 6) for v in norm["std"]]}
        )
        st.table(norm_df)

        # Bar chart with error bars
        fig_norm = go.Figure()
        fig_norm.add_trace(
            go.Bar(
                x=band_names,
                y=norm["mean"],
                error_y=dict(type="data", array=norm["std"], visible=True),
                marker_color=COLOR_OCEAN,
                name="Mean +/- Std",
            )
        )
        fig_norm.update_layout(
            height=350,
            yaxis_title="Reflectance",
            xaxis_title="Band",
            title="Per-Band Mean with Std Error Bars",
        )
        st.plotly_chart(fig_norm, use_container_width=True)
    else:
        st.warning("Normalization stats not found at data/splits/norm_stats.json")

    # Patch counts per site -- pie chart
    st.subheader("Patch Distribution by Site")
    site_counts = {"Sundarbans": 5815, "Gulf of Kutch": 1267, "Pichavaram": 71}
    fig_pie = px.pie(
        names=list(site_counts.keys()),
        values=list(site_counts.values()),
        color_discrete_sequence=[COLOR_OCEAN, COLOR_MANGROVE, COLOR_ACCENT],
        hole=0.35,
    )
    fig_pie.update_traces(textinfo="label+value+percent")
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Patch viewer
    st.subheader("Patch Viewer")
    if os.path.exists(PATCHES_DIR):
        # Collect all _img.npy files
        all_patches = []
        for site_dir in sorted(os.listdir(PATCHES_DIR)):
            site_path = os.path.join(PATCHES_DIR, site_dir)
            if os.path.isdir(site_path):
                for f in os.listdir(site_path):
                    if f.endswith("_img.npy"):
                        all_patches.append(os.path.join(site_path, f))

        if all_patches:
            st.write(f"Found **{len(all_patches)}** patches across all sites.")
            if st.button("Load Random Patch"):
                idx = np.random.randint(0, len(all_patches))
                patch_path = all_patches[idx]
                img, mask = load_patch(patch_path)

                st.caption(f"Patch: `{os.path.relpath(patch_path, BASE_DIR)}`")

                fig_patch, axes = plt.subplots(1, 3, figsize=(14, 4))

                # RGB composite (B4=idx2, B3=idx1, B2=idx0)
                rgb = img[:, :, [2, 1, 0]].copy()
                # Normalise to 0-1 for display
                for ch in range(3):
                    cmin, cmax = np.percentile(rgb[:, :, ch], [2, 98])
                    rgb[:, :, ch] = np.clip((rgb[:, :, ch] - cmin) / (cmax - cmin + 1e-8), 0, 1)
                axes[0].imshow(rgb)
                axes[0].set_title("RGB (B4, B3, B2)")
                axes[0].axis("off")

                # NDVI: (B8 - B4) / (B8 + B4 + 1e-8) -- B8 is index 3, B4 is index 2
                b8 = img[:, :, 3].astype(float)
                b4 = img[:, :, 2].astype(float)
                ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
                im_ndvi = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
                axes[1].set_title("NDVI")
                axes[1].axis("off")
                plt.colorbar(im_ndvi, ax=axes[1], fraction=0.046, pad=0.04)

                # Ground truth mask
                if mask is not None:
                    axes[2].imshow(mask.squeeze(), cmap="Greens", vmin=0, vmax=1)
                    axes[2].set_title("Ground Truth Mask")
                else:
                    axes[2].text(0.5, 0.5, "No mask", ha="center", va="center", transform=axes[2].transAxes)
                    axes[2].set_title("Ground Truth Mask")
                axes[2].axis("off")

                plt.tight_layout()
                st.pyplot(fig_patch)
                plt.close(fig_patch)
        else:
            st.info("No .npy patch files found in data/patches/.")
    else:
        st.info("Patches directory not found. Run the data pipeline first.")

# ---------------------------------------------------------------------------
# Page 5: Satellite Imagery
# ---------------------------------------------------------------------------
elif page == "Satellite Imagery":
    st.title("Satellite Imagery & Alignment Checks")

    # Preview image
    preview_path = os.path.join(RESULTS_DIR, "sundarbans_2024_preview.png")
    if os.path.exists(preview_path):
        st.subheader("Sundarbans 2024 -- Sentinel-2 RGB Preview")
        st.image(preview_path, use_container_width=True)
        st.markdown("---")

    # Alignment check selector
    st.subheader("Alignment Check: Sentinel-2 vs GMW Mask")
    sites = ["sundarbans", "gulf_of_kutch", "pichavaram"]
    years = ["2020", "2024"]

    col_s, col_y = st.columns(2)
    with col_s:
        site = st.selectbox("Site", sites, format_func=lambda s: s.replace("_", " ").title())
    with col_y:
        year = st.selectbox("Year", years)

    img_name = f"alignment_check_{site}_{year}.png"
    img_path = os.path.join(RESULTS_DIR, img_name)
    if os.path.exists(img_path):
        st.image(img_path, caption=f"{site.replace('_', ' ').title()} -- {year}", use_container_width=True)
    else:
        st.warning(f"Image not found: {img_name}")

    # Show all alignment images at once (optional)
    with st.expander("View All Alignment Checks"):
        for s in sites:
            for y in years:
                p = os.path.join(RESULTS_DIR, f"alignment_check_{s}_{y}.png")
                if os.path.exists(p):
                    st.image(p, caption=f"{s.replace('_', ' ').title()} -- {y}", use_container_width=True)
