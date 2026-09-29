"""Main feature extraction pipeline combining physical, spectral, color, and texture modules."""

from __future__ import annotations

import numpy as np

from agrigrade.core.enums import ProduceClass, family_for
from agrigrade.core.errors import (
    FeatureExtractionError,
    ProduceNotFoundError,
)
from agrigrade.features.color import extract_color_features
from agrigrade.features.geometry import extract_geometry_features
from agrigrade.features.schema import FEATURE_NAMES, FeatureVector
from agrigrade.features.spectral import extract_spectral_features
from agrigrade.features.texture import extract_texture_features


def extract_produce_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    crop_type: ProduceClass | str,
    pixel_to_mm_ratio: float,
) -> dict[str, float]:
    """Extract class-conditioned multi-spectral, physical, color, and texture features.

    Args:
        image_rgb: Input RGB image (H, W, 3) as uint8 or float [0, 1].
        mask: Binary foreground mask (H, W) where pixels > 0 indicate produce.
        crop_type: Produce class as predicted by segmentation stage.
        pixel_to_mm_ratio: Calibration scale ratio (mm / pixel).

    Returns:
        Ordered dictionary of feature names and float values matching canonical schema.

    Raises:
        ProduceNotFoundError: If mask contains no positive foreground pixels.
        CalibrationError: If pixel_to_mm_ratio is invalid or <= 0.
        FeatureExtractionError: If image/mask dimensions are incompatible or extraction fails.
    """
    if not isinstance(image_rgb, np.ndarray) or not isinstance(mask, np.ndarray):
        raise FeatureExtractionError("image_rgb and mask must be numpy arrays.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise FeatureExtractionError(
            f"Expected 3-channel RGB image (H, W, 3), got shape {image_rgb.shape}."
        )

    if mask.ndim != 2:
        raise FeatureExtractionError(
            f"Expected 2D binary mask (H, W), got shape {mask.shape}."
        )

    if image_rgb.shape[:2] != mask.shape[:2]:
        raise FeatureExtractionError(
            f"Image dimensions {image_rgb.shape[:2]} and mask dimensions {mask.shape[:2]} do not match."
        )

    fg_mask = mask > 0
    if not np.any(fg_mask):
        raise ProduceNotFoundError(
            "Foreground mask is empty or contains no positive pixels."
        )

    # 1. Physical Geometry Metrics
    geom_features = extract_geometry_features(mask, pixel_to_mm_ratio)

    # 2. Color Space Statistics
    color_features = extract_color_features(image_rgb, mask)

    # 3. Class-Conditioned Spectral Features
    family = family_for(crop_type)

    if image_rgb.dtype == np.uint8:
        rgb_float = image_rgb.astype(np.float32) / 255.0
    else:
        rgb_float = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)

    fg_r = rgb_float[:, :, 0][fg_mask]
    fg_g = rgb_float[:, :, 1][fg_mask]
    fg_b = rgb_float[:, :, 2][fg_mask]
    fg_l = np.full_like(fg_r, color_features["color_l_mean"])

    spectral_features = extract_spectral_features(
        fg_r, fg_g, fg_b, fg_l, family
    )

    # 4. GLCM & LBP Texture Metrics
    texture_features = extract_texture_features(image_rgb, mask)

    # Combine all feature dictionaries
    combined: dict[str, float] = {}
    combined.update(geom_features)
    combined.update(spectral_features)
    combined.update(color_features)
    combined.update(texture_features)

    # Validate against canonical feature schema
    vector = FeatureVector(features=combined)
    return vector.to_dict()
