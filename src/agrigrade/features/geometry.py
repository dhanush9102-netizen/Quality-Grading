"""Physical size, area, and volume calibration using on-screen reference markers."""

from __future__ import annotations

import math
import numpy as np

from agrigrade.core.errors import CalibrationError, ProduceNotFoundError


def extract_geometry_features(
    mask: np.ndarray, pixel_to_mm_ratio: float
) -> dict[str, float]:
    """Calculate camera-distance-invariant geometric and physical metrics.

    Args:
        mask: Binary foreground mask (H, W) where values > 0 indicate produce.
        pixel_to_mm_ratio: Calibration scale ratio (mm / pixel). Must be > 0.

    Returns:
        Dictionary containing `true_area_cm2`, `equivalent_diameter_mm`, and `estimated_volume_cm3`.

    Raises:
        ProduceNotFoundError: If mask contains no foreground pixels.
        CalibrationError: If pixel_to_mm_ratio is missing, <= 0, or NaN/Inf.
    """
    if (
        pixel_to_mm_ratio is None
        or not isinstance(pixel_to_mm_ratio, (int, float))
        or math.isnan(pixel_to_mm_ratio)
        or math.isinf(pixel_to_mm_ratio)
        or pixel_to_mm_ratio <= 0
    ):
        raise CalibrationError(
            f"Invalid pixel_to_mm_ratio: {pixel_to_mm_ratio}. Must be a positive finite float."
        )

    fg_pixels = np.sum(mask > 0)
    if fg_pixels == 0:
        raise ProduceNotFoundError(
            "Foreground mask is empty or contains no positive pixels."
        )

    # True area in cm^2 (README section 3C)
    # True Area (cm^2) = Pixel Area * (Scale Ratio mm/pixel)^2 * 10^-2
    true_area_cm2 = float(fg_pixels * (pixel_to_mm_ratio**2) / 100.0)

    # Equivalent diameter in mm (diameter of circle with same area)
    equivalent_diameter_mm = float(2.0 * np.sqrt(fg_pixels / np.pi) * pixel_to_mm_ratio)

    # Estimated volume in cm^3 (spherical approximation based on equivalent radius in cm)
    radius_cm = (equivalent_diameter_mm / 2.0) / 10.0
    estimated_volume_cm3 = float((4.0 / 3.0) * np.pi * (radius_cm**3))

    return {
        "true_area_cm2": true_area_cm2,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "estimated_volume_cm3": estimated_volume_cm3,
    }
