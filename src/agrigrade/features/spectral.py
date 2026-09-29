"""Spectral index calculations and class-conditioned spectral feature extraction."""

from __future__ import annotations

import numpy as np

from agrigrade.core.enums import ProduceFamily


def compute_vari(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Compute Visible Automatically Resistant Index (VARI).

    VARI = (G - R) / (G + R - B + eps)
    """
    return (g - r) / (g + r - b + eps)


def compute_ndti(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Compute Normalized Difference Red-Green Index (NDTI).

    NDTI = (R - G) / (R + G + eps)
    """
    return (r - g) / (r + g + eps)


def compute_yi(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Compute Yellowness Index (YI).

    YI = (R + G - 2*B) / (R + G + B + eps)
    """
    return (r + g - 2.0 * b) / (r + g + b + eps)


def compute_exb(
    r: np.ndarray, g: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Compute Excess Blue Index (ExB).

    ExB = 2*B - R - G
    """
    return 2.0 * b - r - g


def extract_spectral_features(
    fg_r: np.ndarray,
    fg_g: np.ndarray,
    fg_b: np.ndarray,
    fg_l: np.ndarray,
    family: ProduceFamily,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Extract class-conditioned spectral features for foreground pixels.

    Args:
        fg_r: Red channel values of foreground pixels (float [0, 1]).
        fg_g: Green channel values of foreground pixels (float [0, 1]).
        fg_b: Blue channel values of foreground pixels (float [0, 1]).
        fg_l: Lightness channel L* values of foreground pixels.
        family: The ProduceFamily determined by the router.
        eps: Epsilon for numerical stability.

    Returns:
        Dictionary containing `spectral_primary`, `spectral_secondary`, and `defect_index`.
    """
    if len(fg_r) == 0:
        return {
            "spectral_primary": 0.0,
            "spectral_secondary": 0.0,
            "defect_index": 0.0,
        }

    if family == ProduceFamily.RED_SMOOTH:
        ndti = compute_ndti(fg_r, fg_g, fg_b, eps)
        vari = compute_vari(fg_r, fg_g, fg_b, eps)
        spectral_primary = float(np.mean(ndti))
        spectral_secondary = float(np.mean(vari))
        defect_index = float(np.std(ndti))

    elif family == ProduceFamily.YELLOW_GREEN:
        yi = compute_yi(fg_r, fg_g, fg_b, eps)
        vari = compute_vari(fg_r, fg_g, fg_b, eps)
        spectral_primary = float(np.mean(yi))
        spectral_secondary = float(np.mean(vari))
        # Defect index: fraction of pixels with low yellowness index (spotting/decay)
        defect_index = float(np.sum(yi < 0.1) / len(yi))

    elif family == ProduceFamily.BROWN_TEXTURED:
        exb = compute_exb(fg_r, fg_g, fg_b)
        spectral_primary = float(np.mean(exb))
        spectral_secondary = float(np.mean(fg_l))
        defect_index = float(np.std(fg_l))

    else:
        # UNKNOWN or fallback family
        vari = compute_vari(fg_r, fg_g, fg_b, eps)
        ndti = compute_ndti(fg_r, fg_g, fg_b, eps)
        spectral_primary = float(np.mean(vari))
        spectral_secondary = float(np.mean(ndti))
        defect_index = float(np.std(vari))

    return {
        "spectral_primary": spectral_primary,
        "spectral_secondary": spectral_secondary,
        "defect_index": defect_index,
    }
