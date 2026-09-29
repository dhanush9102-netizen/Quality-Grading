"""Canonical feature names, schema definition, and FeatureVector container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agrigrade.core.errors import SchemaMismatchError

#: Canonical, position-sensitive list of feature names emitted by feature-extraction
#: and consumed by model inference.
FEATURE_NAMES: tuple[str, ...] = (
    "true_area_cm2",
    "equivalent_diameter_mm",
    "estimated_volume_cm3",
    "spectral_primary",
    "spectral_secondary",
    "defect_index",
    "color_r_mean",
    "color_g_mean",
    "color_b_mean",
    "color_l_mean",
    "color_a_mean",
    "color_b_lab_mean",
    "color_l_std",
    "color_a_std",
    "glcm_homogeneity",
    "glcm_contrast",
    "glcm_energy",
    "glcm_correlation",
    "lbp_uniformity",
)


@dataclass(frozen=True)
class FeatureVector:
    """Immutable container holding a validated feature vector."""

    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate feature vector schema."""
        validate_feature_dict(self.features)

    def to_dict(self) -> dict[str, float]:
        """Return features as a dictionary in canonical order."""
        return {name: float(self.features[name]) for name in FEATURE_NAMES}

    def to_list(self) -> list[float]:
        """Return features as an ordered float list matching FEATURE_NAMES."""
        return [float(self.features[name]) for name in FEATURE_NAMES]

    def __getitem__(self, key: str) -> float:
        """Access feature value by name."""
        return float(self.features[key])


def validate_feature_dict(features: dict[str, Any]) -> None:
    """Validate that a feature dictionary contains all canonical feature names.

    Raises:
        SchemaMismatchError: If any feature name is missing or extra features exist.
    """
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise SchemaMismatchError(
            f"Feature dictionary missing required feature(s): {missing}"
        )
