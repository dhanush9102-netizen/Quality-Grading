"""GLCM texture metrics and LBP pattern analysis."""

from __future__ import annotations

import cv2
import numpy as np


def _compute_glcm_numpy(
    gray_roi: np.ndarray, num_levels: int = 16
) -> dict[str, float]:
    """Compute GLCM metrics using numpy fallback when skimage is unavailable."""
    if gray_roi.size == 0 or np.max(gray_roi) == 0:
        return {
            "glcm_homogeneity": 1.0,
            "glcm_contrast": 0.0,
            "glcm_energy": 1.0,
            "glcm_correlation": 1.0,
        }

    # Quantize grayscale image to num_levels
    quantized = (gray_roi.astype(np.float32) / 256.0 * num_levels).astype(np.int32)
    quantized = np.clip(quantized, 0, num_levels - 1)

    h, w = quantized.shape
    glcm = np.zeros((num_levels, num_levels), dtype=np.float64)

    # Accumulate horizontal, vertical, and diagonal offsets
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    count = 0

    for dy, dx in offsets:
        # Source region
        y_start = max(0, -dy)
        y_end = min(h, h - dy)
        x_start = max(0, -dx)
        x_end = min(w, w - dx)

        src = quantized[y_start:y_end, x_start:x_end]
        dst = quantized[y_start + dy : y_end + dy, x_start + dx : x_end + dx]

        # Ignore pairs where both are background (level 0) if desired, or include all
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                glcm[src[i, j], dst[i, j]] += 1.0
                glcm[dst[i, j], src[i, j]] += 1.0
                count += 2

    if count == 0:
        return {
            "glcm_homogeneity": 1.0,
            "glcm_contrast": 0.0,
            "glcm_energy": 1.0,
            "glcm_correlation": 1.0,
        }

    glcm_norm = glcm / np.sum(glcm)

    i_indices, j_indices = np.ogrid[:num_levels, :num_levels]

    # Homogeneity
    homogeneity = np.sum(glcm_norm / (1.0 + (i_indices - j_indices) ** 2))

    # Contrast
    contrast = np.sum(glcm_norm * ((i_indices - j_indices) ** 2))

    # Energy
    energy = np.sum(glcm_norm**2)

    # Correlation
    i_mean = np.sum(i_indices * np.sum(glcm_norm, axis=1, keepdims=True))
    j_mean = np.sum(j_indices * np.sum(glcm_norm, axis=0, keepdims=True))
    i_var = np.sum(((i_indices - i_mean) ** 2) * np.sum(glcm_norm, axis=1, keepdims=True))
    j_var = np.sum(((j_indices - j_mean) ** 2) * np.sum(glcm_norm, axis=0, keepdims=True))

    denom = np.sqrt(i_var * j_var)
    if denom < 1e-9:
        correlation = 1.0
    else:
        correlation = np.sum(
            glcm_norm * (i_indices - i_mean) * (j_indices - j_mean)
        ) / denom

    return {
        "glcm_homogeneity": float(homogeneity),
        "glcm_contrast": float(contrast),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation),
    }


def _compute_lbp_numpy(gray_roi: np.ndarray) -> float:
    """Compute 8-neighbor Local Binary Pattern uniformity metric."""
    if gray_roi.shape[0] < 3 or gray_roi.shape[1] < 3:
        return 1.0

    h, w = gray_roi.shape
    center = gray_roi[1 : h - 1, 1 : w - 1].astype(np.int16)

    # 8 neighbors
    neighbors = [
        gray_roi[0 : h - 2, 0 : w - 2],
        gray_roi[0 : h - 2, 1 : w - 1],
        gray_roi[0 : h - 2, 2 : w],
        gray_roi[1 : h - 1, 2 : w],
        gray_roi[2 : h, 2 : w],
        gray_roi[2 : h, 1 : w - 1],
        gray_roi[2 : h, 0 : w - 2],
        gray_roi[1 : h - 1, 0 : w - 2],
    ]

    lbp_code = np.zeros(center.shape, dtype=np.uint8)
    for bit_idx, neighbor in enumerate(neighbors):
        lbp_code |= ((neighbor.astype(np.int16) >= center).astype(np.uint8)) << bit_idx

    counts = np.bincount(lbp_code.ravel(), minlength=256).astype(np.float64)
    total = np.sum(counts)
    if total == 0:
        return 1.0

    probs = counts / total
    # Uniformity: sum of squared probabilities
    uniformity = np.sum(probs**2)
    return float(uniformity)


def extract_texture_features(
    image_rgb: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Extract GLCM texture metrics and LBP uniformity over the foreground region.

    Args:
        image_rgb: Input RGB image (H, W, 3).
        mask: Binary foreground mask (H, W).

    Returns:
        Dictionary containing GLCM homogeneity, contrast, energy, correlation, and LBP uniformity.
    """
    fg_mask = mask > 0
    if not np.any(fg_mask):
        return {
            "glcm_homogeneity": 1.0,
            "glcm_contrast": 0.0,
            "glcm_energy": 1.0,
            "glcm_correlation": 1.0,
            "lbp_uniformity": 1.0,
        }

    # Convert to grayscale uint8
    if image_rgb.dtype == np.uint8:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        rgb_uint8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    # Crop to bounding box of the foreground mask for speed
    y_indices, x_indices = np.where(fg_mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    gray_roi = gray[y_min : y_max + 1, x_min : x_max + 1]
    mask_roi = fg_mask[y_min : y_max + 1, x_min : x_max + 1]

    # Zero out background in ROI
    masked_roi = np.where(mask_roi, gray_roi, 0)

    # Try using scikit-image if available, otherwise fall back to pure numpy implementation
    try:
        from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

        quantized = (masked_roi.astype(np.float32) / 256.0 * 16).astype(np.uint8)
        glcm = graycomatrix(
            quantized,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16,
            symmetric=True,
            normed=True,
        )
        glcm_res = {
            "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
            "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
            "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
            "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        }

        lbp = local_binary_pattern(masked_roi, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
        lbp_uniformity = float(np.sum((lbp_hist / (np.sum(lbp_hist) + 1e-9)) ** 2))
        glcm_res["lbp_uniformity"] = lbp_uniformity
        return glcm_res

    except ImportError:
        glcm_res = _compute_glcm_numpy(masked_roi, num_levels=16)
        glcm_res["lbp_uniformity"] = _compute_lbp_numpy(masked_roi)
        return glcm_res
