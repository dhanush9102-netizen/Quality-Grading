"""Dynamic Feature Router mapping produce classes to colour family feature recipes."""

from __future__ import annotations

from typing import Any

from agrigrade.core.enums import ProduceClass, ProduceFamily, family_for

#: Family feature recipe descriptions as defined in README Section 3B.
FAMILY_RECIPES: dict[ProduceFamily, dict[str, str]] = {
    ProduceFamily.RED_SMOOTH: {
        "primary_spectral": "NDTI (Red-Green Index)",
        "secondary_spectral": "VARI (Vegetation Index)",
        "defect_signature": "Standard deviation of NDTI over foreground mask",
        "description": "Quantifies lycopene/anthocyanin accumulation and chlorophyll decay for red/smooth produce.",
    },
    ProduceFamily.YELLOW_GREEN: {
        "primary_spectral": "YI (Yellowness Index)",
        "secondary_spectral": "VARI (Chlorophyll Decay)",
        "defect_signature": "Ratio of pixels with YI < 0.1 (decay / spotting)",
        "description": "Tracks carotenoid progression and spotting in bananas, mangoes, citrus, and papayas.",
    },
    ProduceFamily.BROWN_TEXTURED: {
        "primary_spectral": "ExB (Excess Blue / Mold)",
        "secondary_spectral": "L* Lightness Mean",
        "defect_signature": "Standard deviation of L* channel (soft rot variance)",
        "description": "Isolates surface mold via Excess Blue and detects soft rot via lightness variance on textured produce.",
    },
    ProduceFamily.UNKNOWN: {
        "primary_spectral": "VARI",
        "secondary_spectral": "NDTI",
        "defect_signature": "Standard deviation of VARI",
        "description": "Fallback recipe for unrecognised produce classes.",
    },
}


def resolve_family(produce_class: ProduceClass | str) -> ProduceFamily:
    """Resolve the ProduceFamily for a given ProduceClass or class string.

    Args:
        produce_class: ProduceClass enum or string name.

    Returns:
        ProduceFamily corresponding to the produce class.
    """
    return family_for(produce_class)


def get_family_recipe(family: ProduceFamily | str) -> dict[str, Any]:
    """Retrieve feature extraction recipe details for a given ProduceFamily.

    Args:
        family: ProduceFamily enum or string.

    Returns:
        Dictionary describing the spectral indices and defect signatures for the family.
    """
    try:
        fam = ProduceFamily(str(family).strip().lower())
    except ValueError:
        fam = ProduceFamily.UNKNOWN
    return FAMILY_RECIPES.get(fam, FAMILY_RECIPES[ProduceFamily.UNKNOWN])
