"""
Per-pixel spectral feature extraction for the XGBoost mangrove classifier.

Extracts 10 features per pixel from a 6-band Sentinel-2 patch:
  6 raw bands  +  NDVI, EVI, NDWI, SAVI

Band order in patches: [0]=B2, [1]=B3, [2]=B4, [3]=B8, [4]=B11, [5]=B12
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np


FEATURE_NAMES = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "NDVI", "EVI", "NDWI", "SAVI",
]


def extract_features(img: np.ndarray) -> np.ndarray:
    """Convert a 6-band patch into a per-pixel feature matrix.

    Parameters
    ----------
    img : np.ndarray
        Shape ``(6, 256, 256)``, float32, values in [0, 1].

    Returns
    -------
    np.ndarray
        Shape ``(65536, 10)`` -- one row per pixel, 10 features.
    """
    # Unpack bands -- each is (256, 256)
    B2 = img[0]
    B3 = img[1]
    B4 = img[2]
    B8 = img[3]
    B11 = img[4]
    B12 = img[5]

    # Vegetation / water indices
    ndvi = (B8 - B4) / (B8 + B4 + 1e-8)
    evi = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0)
    ndwi = (B3 - B8) / (B3 + B8 + 1e-8)
    savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)

    # Stack into (H*W, 10)
    h, w = B2.shape
    n_pixels = h * w
    features = np.stack(
        [B2, B3, B4, B8, B11, B12, ndvi, evi, ndwi, savi], axis=0
    )  # (10, H, W)
    features = features.reshape(10, n_pixels).T  # (n_pixels, 10)

    # Replace any NaN / Inf from index computation with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features.astype(np.float32)
