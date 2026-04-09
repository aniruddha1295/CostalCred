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
import tempfile

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import torch
    import segmentation_models_pytorch as smp
    HAS_UNET = True
except ImportError:
    HAS_UNET = False

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
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

@st.cache_data(ttl=60)
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=60)
def load_carbon_predictions():
    """Load pre-computed carbon predictions from results/carbon_predictions.json."""
    path = os.path.join(RESULTS_DIR, "carbon_predictions.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=60)
def load_patch(patch_path):
    """Load a .npy patch (H, W, C) and its mask."""
    img = np.load(patch_path)
    mask_path = patch_path.replace("_img.npy", "_mask.npy")
    mask = np.load(mask_path) if os.path.exists(mask_path) else None
    return img, mask


def metric_card(label, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


# ---------------------------------------------------------------------------
# Carbon Prediction helpers
# ---------------------------------------------------------------------------

SITE_DISPLAY = {
    "sundarbans": "Sundarbans (West Bengal)",
    "gulf_of_kutch": "Gulf of Kutch (Gujarat)",
    "pichavaram": "Pichavaram (Tamil Nadu)",
}

BIOMASS_DENSITY = 230.0
CARBON_FRACTION = 0.47
CO2_TO_C_RATIO = 44.0 / 12.0
ANNUAL_SEQUESTRATION = 7.0


def extract_features_inline(img):
    """Extract 10 per-pixel features from a (6, H, W) patch."""
    B2, B3, B4, B8, B11, B12 = img[0], img[1], img[2], img[3], img[4], img[5]
    ndvi = (B8 - B4) / (B8 + B4 + 1e-8)
    evi = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0)
    ndwi = (B3 - B8) / (B3 + B8 + 1e-8)
    savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)
    h, w = B2.shape
    features = np.stack([B2, B3, B4, B8, B11, B12, ndvi, evi, ndwi, savi], axis=0)
    features = features.reshape(10, h * w).T
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def compute_carbon(mask, pixel_size_m=10.0):
    """IPCC Tier 1 carbon stock from a binary mask."""
    pixel_area_ha = (pixel_size_m ** 2) / 10000.0
    hectares = float(np.sum(mask > 0)) * pixel_area_ha
    stock = hectares * BIOMASS_DENSITY * CARBON_FRACTION * CO2_TO_C_RATIO
    return hectares, stock


def patchify(data, patch_size=256):
    """Cut a (C, H, W) array into non-overlapping patches."""
    C, H, W = data.shape
    patches = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patches.append(data[:, y:y+patch_size, x:x+patch_size])
    return patches


@st.cache_data(ttl=60)
def load_site_patches(site, year):
    """Load all img and mask patches for a site+year from data/patches/."""
    patch_dir = os.path.join(PATCHES_DIR, f"{site}_{year}")
    imgs, masks = [], []
    if not os.path.isdir(patch_dir):
        return imgs, masks
    img_files = sorted([f for f in os.listdir(patch_dir) if f.startswith("img_") and f.endswith(".npy")])
    for fname in img_files:
        img = np.load(os.path.join(patch_dir, fname))
        imgs.append(img)
        mask_fname = fname.replace("img_", "mask_")
        mask_path = os.path.join(patch_dir, mask_fname)
        if os.path.exists(mask_path):
            masks.append(np.load(mask_path))
        else:
            masks.append(np.zeros(img.shape[1:], dtype=np.uint8))
    return imgs, masks


@st.cache_data(ttl=60)
def load_xgb_model():
    """Load XGBoost model from disk."""
    model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    if not os.path.exists(model_path):
        return None
    if not HAS_XGB:
        return None
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def run_ndvi_prediction(patches, threshold):
    """Run NDVI threshold prediction on a list of (6, H, W) patches."""
    preds = []
    for img in patches:
        ndvi = (img[3] - img[2]) / (img[3] + img[2] + 1e-8)
        pred = (ndvi > threshold).astype(np.uint8)
        preds.append(pred)
    return preds


def run_xgb_prediction(patches, model, progress_bar=None):
    """Run XGBoost prediction on a list of (6, H, W) patches."""
    preds = []
    total = len(patches)
    for i, img in enumerate(patches):
        feats = extract_features_inline(img)
        pred_flat = model.predict(feats)
        pred = pred_flat.reshape(img.shape[1], img.shape[2]).astype(np.uint8)
        preds.append(pred)
        if progress_bar is not None:
            progress_bar.progress((i + 1) / total)
    return preds


def load_unet_model():
    """Load trained U-Net model from checkpoint."""
    checkpoint_path = os.path.join(MODELS_DIR, "unet_best.pt")
    if not os.path.exists(checkpoint_path):
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        model = smp.Unet(
            encoder_name=config.get("encoder_name", "resnet18"),
            encoder_weights=None,
            in_channels=config.get("in_channels", 6),
            classes=config.get("classes", 1),
            decoder_use_batchnorm=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading U-Net: {e}")
        return None

def run_unet_prediction(patches, model, device, progress=None):
    """Run U-Net prediction on patches. Returns list of binary mask arrays."""
    import json as _json
    norm_path = os.path.join(SPLITS_DIR, "norm_stats.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            stats = _json.load(f)
        mean = np.array(stats["mean"]).reshape(6, 1, 1)
        std = np.array(stats["std"]).reshape(6, 1, 1)
    else:
        mean, std = 0.0, 1.0

    predictions = []
    batch_size = 8
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        imgs = []
        for p in batch:
            img = p.astype(np.float32)
            img = (img - mean) / (std + 1e-8)
            imgs.append(img)
        imgs_tensor = torch.from_numpy(np.stack(imgs)).float().to(device)
        with torch.no_grad():
            logits = model(imgs_tensor)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
        for p in preds:
            predictions.append(p)
        if progress:
            progress.progress(min((i + batch_size) / len(patches), 1.0))
    return predictions


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## CoastalCred")
    st.markdown("**Blue Carbon MRV**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Carbon Prediction", "Overview", "Model Comparison", "XGBoost Analysis", "Data Explorer", "Satellite Imagery"],
    )
    st.markdown("---")
    st.caption("Sem VI Mini Project -- RCOEM Dept. of Data Science")

# ---------------------------------------------------------------------------
# Page 0: Carbon Prediction
# ---------------------------------------------------------------------------
if page == "Carbon Prediction":
    st.header("Carbon Credit Prediction")
    st.markdown("Predict mangrove cover and estimate carbon credits using IPCC Tier 1 methodology")

    mode = st.radio("Mode", ["Pre-loaded Sites", "Upload GeoTIFF"], horizontal=True)

    if mode == "Pre-loaded Sites":
        col1, col2 = st.columns(2)
        with col1:
            site = st.selectbox("Select Site", ["sundarbans", "gulf_of_kutch", "pichavaram"],
                                format_func=lambda s: SITE_DISPLAY.get(s, s))
        with col2:
            model_options = ["NDVI Threshold", "XGBoost"]
            # Show U-Net if model file exists OR pre-computed predictions contain unet data
            predictions_precheck = load_carbon_predictions()
            has_unet_predictions = predictions_precheck is not None and "unet" in predictions_precheck
            if has_unet_predictions or (HAS_UNET and os.path.exists(os.path.join(MODELS_DIR, "unet_best.pt"))):
                model_options.append("U-Net")
            model_choice = st.selectbox("Select Model", model_options)

        predict_button = st.button("Predict Carbon Credits", type="primary")

        if predict_button:
            predictions = predictions_precheck if predictions_precheck is not None else load_carbon_predictions()
            if predictions is None:
                st.error("Pre-computed predictions not found. Run `python src/carbon/precompute_predictions.py` first.")
                st.stop()

            model_key = "ndvi" if model_choice == "NDVI Threshold" else ("unet" if model_choice == "U-Net" else "xgboost")
            site_data = predictions.get(model_key, {}).get(site, {})

            if not site_data or "2020" not in site_data or "2024" not in site_data:
                st.error(f"No pre-computed data for {SITE_DISPLAY[site]} with {model_choice}. Re-run precompute script.")
                st.stop()

            baseline = site_data["2020"]
            current = site_data["2024"]
            flux = site_data.get("carbon_flux", {})

            baseline_ha = flux.get("baseline_ha_2020", baseline["predicted_hectares"])
            current_ha = flux.get("current_ha_2024", current["predicted_hectares"])
            delta_ha = flux.get("delta_ha", current_ha - baseline_ha)
            total_flux = flux.get("total_flux_tco2e_4yr", delta_ha * ANNUAL_SEQUESTRATION * 4)
            baseline_stock = baseline["predicted_stock_tco2e"]
            current_stock = current["predicted_stock_tco2e"]

            site_display = SITE_DISPLAY.get(site, site)

            st.success("Prediction loaded instantly from cache!")

            # Row 1 -- Metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Baseline Area (2020)", f"{baseline_ha:.1f} ha")
            with c2:
                st.metric("Current Area (2024)", f"{current_ha:.1f} ha")
            with c3:
                st.metric("Area Change", f"{delta_ha:+.1f} ha")
            with c4:
                st.metric("Carbon Credits (4yr)", f"{total_flux:+,.0f} tCO\u2082e")

            st.markdown("---")

            # Row 2 -- Side-by-side visualizations from saved samples
            st.subheader("Sample Patch Predictions")
            sample_dir_2020 = os.path.join(RESULTS_DIR, "samples", model_key, f"{site}_2020")
            sample_dir_2024 = os.path.join(RESULTS_DIR, "samples", model_key, f"{site}_2024")

            has_2020_samples = os.path.isdir(sample_dir_2020)
            has_2024_samples = os.path.isdir(sample_dir_2024)

            if has_2020_samples or has_2024_samples:
                # Count available samples
                n_samples_2020 = len([f for f in os.listdir(sample_dir_2020) if f.startswith("rgb_")]) if has_2020_samples else 0
                n_samples_2024 = len([f for f in os.listdir(sample_dir_2024) if f.startswith("rgb_")]) if has_2024_samples else 0
                n_vis = min(3, max(n_samples_2020, n_samples_2024))

                if n_vis > 0:
                    fig, axes = plt.subplots(n_vis, 2, figsize=(12, 5 * n_vis))
                    if n_vis == 1:
                        axes = axes.reshape(1, 2)

                    for row in range(n_vis):
                        # 2020 column
                        if row < n_samples_2020:
                            rgb_2020 = np.load(os.path.join(sample_dir_2020, f"rgb_{row}.npy"))
                            pred_2020 = np.load(os.path.join(sample_dir_2020, f"pred_{row}.npy"))
                            axes[row, 0].imshow(rgb_2020)
                            mask_overlay = np.zeros((*pred_2020.shape, 4))
                            mask_overlay[pred_2020 == 1] = [0, 1, 0, 0.35]
                            axes[row, 0].imshow(mask_overlay)
                            axes[row, 0].set_title(f"2020 -- Sample {row+1}")
                        else:
                            axes[row, 0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[row, 0].transAxes)
                            axes[row, 0].set_title("2020 -- N/A")
                        axes[row, 0].axis('off')

                        # 2024 column
                        if row < n_samples_2024:
                            rgb_2024 = np.load(os.path.join(sample_dir_2024, f"rgb_{row}.npy"))
                            pred_2024 = np.load(os.path.join(sample_dir_2024, f"pred_{row}.npy"))
                            axes[row, 1].imshow(rgb_2024)
                            mask_overlay = np.zeros((*pred_2024.shape, 4))
                            mask_overlay[pred_2024 == 1] = [0, 1, 0, 0.35]
                            axes[row, 1].imshow(mask_overlay)
                            axes[row, 1].set_title(f"2024 -- Sample {row+1}")
                        else:
                            axes[row, 1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[row, 1].transAxes)
                            axes[row, 1].set_title("2024 -- N/A")
                        axes[row, 1].axis('off')

                    fig.suptitle(f"{site_display} -- Mangrove Predictions ({model_choice})", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No sample visualizations saved. Re-run precompute script to generate them.")

            st.markdown("---")

            # Row 3 -- Detailed Carbon Report table
            st.subheader("Carbon Credit Report")
            report_data = {
                "Metric": [
                    "Mangrove Area (2020)", "Mangrove Area (2024)", "Area Change",
                    "Carbon Stock (2020)", "Carbon Stock (2024)",
                    "Annual Sequestration", "Total Carbon Credits (4yr)",
                    "Total Patches (2020)", "Total Patches (2024)",
                    "Mangrove Fraction (2024)",
                ],
                "Value": [
                    f"{baseline_ha:.2f} ha", f"{current_ha:.2f} ha", f"{delta_ha:+.2f} ha",
                    f"{baseline_stock:,.0f} tCO\u2082e", f"{current_stock:,.0f} tCO\u2082e",
                    f"{delta_ha * 7.0:+,.0f} tCO\u2082e/yr", f"{total_flux:+,.0f} tCO\u2082e",
                    str(baseline.get("total_patches", "N/A")),
                    str(current.get("total_patches", "N/A")),
                    f"{current.get('mangrove_fraction', 0) * 100:.2f}%",
                ]
            }
            st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

            # Row 4 -- IPCC methodology explanation
            with st.expander("IPCC Tier 1 Methodology"):
                st.markdown("""
                **Constants (IPCC 2013 Wetlands Supplement):**
                - Biomass density: 230 t/ha (dry matter)
                - Carbon fraction: 0.47
                - CO\u2082:C ratio: 44/12 \u2248 3.667
                - Annual sequestration: 7.0 tCO\u2082e/ha/year

                **Stock formula:** hectares \u00d7 230 \u00d7 0.47 \u00d7 3.667 = tCO\u2082e

                **Flux formula:** (current_ha \u2212 baseline_ha) \u00d7 7.0 \u00d7 years = tCO\u2082e credits
                """)

    else:
        # Upload GeoTIFF mode
        if not HAS_RASTERIO:
            st.error("rasterio is not installed. Install it with: pip install rasterio")
            st.stop()

        st.info("Upload two Sentinel-2 GeoTIFF composites (6 bands: B2, B3, B4, B8, B11, B12)")
        col1, col2 = st.columns(2)
        with col1:
            baseline_file = st.file_uploader("Baseline Year GeoTIFF", type=["tif", "tiff"], key="baseline")
        with col2:
            current_file = st.file_uploader("Current Year GeoTIFF", type=["tif", "tiff"], key="current")

        years_between = st.number_input("Years between images", min_value=1, max_value=20, value=4)

        upload_model_options = ["NDVI Threshold", "XGBoost"]
        if HAS_UNET and os.path.exists(os.path.join(MODELS_DIR, "unet_best.pt")):
            upload_model_options.append("U-Net")
        model_choice_upload = st.selectbox("Select Model", upload_model_options, key="upload_model")

        predict_upload = st.button("Predict Carbon Credits", type="primary", key="predict_upload")

        if predict_upload:
            if baseline_file is None or current_file is None:
                st.error("Please upload both baseline and current year GeoTIFFs.")
                st.stop()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save uploaded files
                baseline_path = os.path.join(tmpdir, "baseline.tif")
                current_path = os.path.join(tmpdir, "current.tif")
                with open(baseline_path, "wb") as f:
                    f.write(baseline_file.getbuffer())
                with open(current_path, "wb") as f:
                    f.write(current_file.getbuffer())

                # Read with rasterio
                with st.spinner("Reading GeoTIFFs..."):
                    with rasterio.open(baseline_path) as src:
                        baseline_data = src.read().astype(np.float32) / 10000.0
                        pixel_size = src.res[0]
                    with rasterio.open(current_path) as src:
                        current_data = src.read().astype(np.float32) / 10000.0

                # Cut into patches
                patches_bl = patchify(baseline_data)
                patches_cur = patchify(current_data)

                if not patches_bl and not patches_cur:
                    st.error("Images are too small to extract 256x256 patches.")
                    st.stop()

                total_patches = len(patches_bl) + len(patches_cur)

                # Run model
                if model_choice_upload == "NDVI Threshold":
                    ndvi_results = load_json(os.path.join(RESULTS_DIR, "ndvi.json"))
                    if ndvi_results is None:
                        st.error("results/ndvi.json not found. Run NDVI evaluation first.")
                        st.stop()
                    threshold = ndvi_results.get("best_threshold", 0.3)
                    with st.spinner(f"Running NDVI threshold ({threshold:.2f}) on {total_patches} patches..."):
                        preds_bl = run_ndvi_prediction(patches_bl, threshold)
                        preds_cur = run_ndvi_prediction(patches_cur, threshold)
                elif model_choice_upload == "XGBoost":
                    if not HAS_XGB:
                        st.error("XGBoost is not installed. Install it with: pip install xgboost")
                        st.stop()
                    xgb_model = load_xgb_model()
                    if xgb_model is None:
                        st.error("models/xgboost_model.json not found. Train XGBoost first.")
                        st.stop()
                    progress = st.progress(0)
                    with st.spinner(f"Running XGBoost on {total_patches} patches..."):
                        preds_bl = run_xgb_prediction(patches_bl, xgb_model, progress)
                        preds_cur = run_xgb_prediction(patches_cur, xgb_model, progress)
                elif model_choice_upload == "U-Net":
                    result = load_unet_model()
                    if result is None:
                        st.error("U-Net model not found. Train it first.")
                        st.stop()
                    unet_model, device = result
                    progress = st.progress(0, text="Running U-Net prediction (baseline)...")
                    preds_bl = run_unet_prediction(patches_bl, unet_model, device, progress)
                    progress = st.progress(0, text="Running U-Net prediction (current)...")
                    preds_cur = run_unet_prediction(patches_cur, unet_model, device, progress)

                st.success("Prediction complete!")

                # Stitch and compute carbon
                all_bl = np.concatenate([p.flatten() for p in preds_bl]) if preds_bl else np.array([])
                all_cur = np.concatenate([p.flatten() for p in preds_cur]) if preds_cur else np.array([])

                baseline_ha, baseline_stock = compute_carbon(all_bl, pixel_size_m=pixel_size) if len(all_bl) > 0 else (0.0, 0.0)
                current_ha, current_stock = compute_carbon(all_cur, pixel_size_m=pixel_size) if len(all_cur) > 0 else (0.0, 0.0)
                delta_ha = current_ha - baseline_ha
                total_flux = delta_ha * ANNUAL_SEQUESTRATION * years_between

                # Metric cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Baseline Area", f"{baseline_ha:.2f} ha")
                with c2:
                    st.metric("Current Area", f"{current_ha:.2f} ha")
                with c3:
                    st.metric("Area Change", f"{delta_ha:+.2f} ha")
                with c4:
                    st.metric("Carbon Credits", f"{total_flux:+,.0f} tCO\u2082e")

                st.markdown("---")

                # Visualizations
                st.subheader("Sample Patch Predictions")
                rng = np.random.RandomState(42)
                n_vis_bl = min(3, len(patches_bl))
                n_vis_cur = min(3, len(patches_cur))
                n_vis = max(n_vis_bl, n_vis_cur)

                if n_vis > 0:
                    idx_bl = rng.choice(len(patches_bl), size=n_vis_bl, replace=False) if n_vis_bl > 0 else []
                    idx_cur = rng.choice(len(patches_cur), size=n_vis_cur, replace=False) if n_vis_cur > 0 else []

                    fig, axes = plt.subplots(n_vis, 2, figsize=(12, 5 * n_vis))
                    if n_vis == 1:
                        axes = axes.reshape(1, 2)
                    for row in range(n_vis):
                        if row < n_vis_bl:
                            img_bl = patches_bl[idx_bl[row]]
                            rgb_bl = np.clip(img_bl[[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)
                            axes[row, 0].imshow(rgb_bl)
                            axes[row, 0].imshow(preds_bl[idx_bl[row]], alpha=0.3, cmap='Greens')
                            axes[row, 0].set_title(f"Baseline -- Patch {row+1}")
                        else:
                            axes[row, 0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[row, 0].transAxes)
                        axes[row, 0].axis('off')
                        if row < n_vis_cur:
                            img_cur = patches_cur[idx_cur[row]]
                            rgb_cur = np.clip(img_cur[[2, 1, 0]].transpose(1, 2, 0) / 0.3, 0, 1)
                            axes[row, 1].imshow(rgb_cur)
                            axes[row, 1].imshow(preds_cur[idx_cur[row]], alpha=0.3, cmap='Greens')
                            axes[row, 1].set_title(f"Current -- Patch {row+1}")
                        else:
                            axes[row, 1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[row, 1].transAxes)
                        axes[row, 1].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.markdown("---")

                # Carbon report table
                st.subheader("Carbon Credit Report")
                report_data = {
                    "Metric": ["Mangrove Area (Baseline)", "Mangrove Area (Current)", "Area Change",
                                "Carbon Stock (Baseline)", "Carbon Stock (Current)",
                                "Annual Sequestration", f"Total Carbon Credits ({years_between}yr)"],
                    "Value": [f"{baseline_ha:.2f} ha", f"{current_ha:.2f} ha", f"{delta_ha:+.2f} ha",
                              f"{baseline_stock:,.0f} tCO\u2082e", f"{current_stock:,.0f} tCO\u2082e",
                              f"{delta_ha * 7.0:+,.0f} tCO\u2082e/yr", f"{total_flux:+,.0f} tCO\u2082e"]
                }
                st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

                with st.expander("IPCC Tier 1 Methodology"):
                    st.markdown("""
                    **Constants (IPCC 2013 Wetlands Supplement):**
                    - Biomass density: 230 t/ha (dry matter)
                    - Carbon fraction: 0.47
                    - CO\u2082:C ratio: 44/12 \u2248 3.667
                    - Annual sequestration: 7.0 tCO\u2082e/ha/year

                    **Stock formula:** hectares \u00d7 230 \u00d7 0.47 \u00d7 3.667 = tCO\u2082e

                    **Flux formula:** (current_ha \u2212 baseline_ha) \u00d7 7.0 \u00d7 years = tCO\u2082e credits
                    """)

# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------
elif page == "Overview":
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
        _trained = sum(1 for f in ["ndvi.json", "xgboost.json", "unet.json"] if os.path.exists(os.path.join(RESULTS_DIR, f)))
        metric_card("Models Trained", f"{_trained} / 3")

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
