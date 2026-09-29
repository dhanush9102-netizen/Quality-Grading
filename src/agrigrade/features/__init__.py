"""agrigrade.features — Multi-spectral, physical, color, and texture feature extraction engine."""

from agrigrade.features.color import extract_color_features
from agrigrade.features.geometry import extract_geometry_features
from agrigrade.features.pipeline import extract_produce_features
from agrigrade.features.router import get_family_recipe, resolve_family
from agrigrade.features.schema import FEATURE_NAMES, FeatureVector
from agrigrade.features.spectral import extract_spectral_features
from agrigrade.features.texture import extract_texture_features

__all__ = [
    "FEATURE_NAMES",
    "FeatureVector",
    "extract_color_features",
    "extract_geometry_features",
    "extract_produce_features",
    "extract_spectral_features",
    "extract_texture_features",
    "get_family_recipe",
    "resolve_family",
]
