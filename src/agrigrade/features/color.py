"""Color space analysis and statistics over foreground masks."""

from __future__ import annotations

import cv2
import numpy as np


def extract_color_features(
    image_rgb: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Extract RGB and CIELAB color statistics over the foreground mask.

    Args:
        image_rgb: Input image in RGB format (H, W, 3) as uint8 or float [0, 1].
        mask: Binary foreground mask (H, W) where pixels > 0 indicate object.

    Returns:
        Dictionary containing color means and std values for RGB and CIELAB channels.
    """
    fg_mask = mask > 0
    if not np.any(fg_mask):
        return {
            "color_r_mean": 0.0,
            "color_g_mean": 0.0,
            "color_b_mean": 0.0,
            "color_l_mean": 0.0,
            "color_a_mean": 0.0,
            "color_b_lab_mean": 0.0,
            "color_l_std": 0.0,
            "color_a_std": 0.0,
        }

    # Normalize image_rgb float to [0, 1] if needed
    if image_rgb.dtype == np.uint8:
        rgb_float = image_rgb.astype(np.float32) / 255.0
        rgb_uint8 = image_rgb
    else:
        rgb_float = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
        rgb_uint8 = (rgb_float * 255.0).astype(np.uint8)

    fg_r = rgb_float[:, :, 0][fg_mask]
    fg_g = rgb_float[:, :, 1][fg_mask]
    fg_b = rgb_float[:, :, 2][fg_mask]

    # Convert to CIELAB space
    lab_img = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_l = lab_img[:, :, 0][fg_mask]
    fg_a = lab_img[:, :, 1][fg_mask]
    fg_b_lab = lab_img[:, :, 2][fg_mask]

    return {
        "color_r_mean": float(np.mean(fg_r)),
        "color_g_mean": float(np.mean(fg_g)),
        "color_b_mean": float(np.mean(fg_b)),
        "color_l_mean": float(np.mean(fg_l)),
        "color_a_mean": float(np.mean(fg_a)),
        "color_b_lab_mean": float(np.mean(fg_b_lab)),
        "color_l_std": float(np.std(fg_l)),
        "color_a_std": float(np.std(fg_a)),
    }
