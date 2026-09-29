"""Contract tests for feature extraction pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from agrigrade.core.enums import ProduceClass, ProduceFamily
from agrigrade.core.errors import CalibrationError, ProduceNotFoundError
from agrigrade.features import (
    FEATURE_NAMES,
    FeatureVector,
    extract_color_features,
    extract_geometry_features,
    extract_produce_features,
    extract_spectral_features,
    extract_texture_features,
    resolve_family,
)


class _RaisesContext:
    def __init__(self, expected_exception: type[BaseException]) -> None:
        self.expected_exception = expected_exception

    def __enter__(self) -> _RaisesContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if exc_type is None:
            raise AssertionError(f"Expected exception {self.expected_exception.__name__} was not raised.")
        return issubclass(exc_type, self.expected_exception)


def approx(val: float, abs_tol: float = 1e-3) -> Any:
    if pytest is not None:
        return pytest.approx(val, abs=abs_tol)

    class ApproxVal:
        def __eq__(self, other: object) -> bool:
            if isinstance(other, (int, float)):
                return abs(val - float(other)) <= abs_tol
            return False

    return ApproxVal()


def raises(expected_exception: type[BaseException]) -> Any:
    if pytest is not None:
        return pytest.raises(expected_exception)
    return _RaisesContext(expected_exception)


def test_family_router() -> None:
    """Test ProduceClass to ProduceFamily routing."""
    assert resolve_family(ProduceClass.APPLE) == ProduceFamily.RED_SMOOTH
    assert resolve_family(ProduceClass.BANANA) == ProduceFamily.YELLOW_GREEN
    assert resolve_family(ProduceClass.KIWI) == ProduceFamily.BROWN_TEXTURED
    assert resolve_family("unknown_fruit") == ProduceFamily.UNKNOWN


def test_spectral_indices_red_smooth() -> None:
    """Test spectral indices for Red/Smooth family (NDTI & VARI)."""
    fg_r = np.array([0.8, 0.7, 0.9], dtype=np.float32)
    fg_g = np.array([0.2, 0.3, 0.1], dtype=np.float32)
    fg_b = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    fg_l = np.array([50.0, 50.0, 50.0], dtype=np.float32)

    res = extract_spectral_features(fg_r, fg_g, fg_b, fg_l, ProduceFamily.RED_SMOOTH)
    assert "spectral_primary" in res
    assert "spectral_secondary" in res
    assert "defect_index" in res
    assert res["spectral_primary"] > 0  # NDTI (R > G) should be positive


def test_spectral_indices_yellow_green() -> None:
    """Test spectral indices for Yellow/Green family (YI & VARI)."""
    fg_r = np.array([0.8, 0.7, 0.9], dtype=np.float32)
    fg_g = np.array([0.8, 0.7, 0.9], dtype=np.float32)
    fg_b = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    fg_l = np.array([70.0, 70.0, 70.0], dtype=np.float32)

    res = extract_spectral_features(fg_r, fg_g, fg_b, fg_l, ProduceFamily.YELLOW_GREEN)
    assert res["spectral_primary"] > 0  # YI should be positive for high R+G low B


def test_geometry_calibration() -> None:
    """Test physical geometric area and equivalent diameter calculations."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1  # 60x60 = 3600 pixels
    ratio = 0.5  # 0.5 mm / pixel

    geom = extract_geometry_features(mask, ratio)
    # Area (cm^2) = 3600 * (0.5)^2 / 100 = 9.0 cm^2
    assert geom["true_area_cm2"] == approx(9.0)
    assert geom["equivalent_diameter_mm"] > 0
    assert geom["estimated_volume_cm3"] > 0


def test_geometry_error_handling() -> None:
    """Test geometry error handling for empty mask or invalid ratio."""
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    with raises(ProduceNotFoundError):
        extract_geometry_features(empty_mask, 0.5)

    valid_mask = np.ones((10, 10), dtype=np.uint8)
    with raises(CalibrationError):
        extract_geometry_features(valid_mask, -1.0)

    with raises(CalibrationError):
        extract_geometry_features(valid_mask, 0.0)


def test_extract_produce_features_pipeline() -> None:
    """Test full extract_produce_features pipeline produces valid vector matching schema."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [200, 50, 50]  # Red produce
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    features = extract_produce_features(
        image_rgb=img,
        mask=mask,
        crop_type=ProduceClass.APPLE,
        pixel_to_mm_ratio=0.5,
    )

    assert isinstance(features, dict)
    assert len(features) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in features
        assert isinstance(features[name], float)

    # Validate container
    vector = FeatureVector(features=features)
    assert len(vector.to_list()) == len(FEATURE_NAMES)
