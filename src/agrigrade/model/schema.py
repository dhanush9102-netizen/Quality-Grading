"""Canonical feature order and units for the Random Forest.

The Random Forest is positional: a trained estimator has no idea what column 7
means. The order declared here is therefore part of the model contract in the
same way the grade ladder is, and reordering these features invalidates every
serialized artifact.

The vector is deliberately fixed-length for all produce families. The feature
stage routes a class to a family recipe, but every recipe emits the same slots,
so one ensemble covers Kiwi and Apple alike. What differs between families is
the *meaning* of the polymorphic slots, which :mod:`agrigrade.model.schema`
records per family so explanations read correctly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from agrigrade.core.enums import ProduceFamily

#: Bumped when slots are added. Artifacts record the version they were trained
#: against and refuse to load against a newer schema.
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One column of the feature vector.

    Attributes:
        name: Column name as emitted by ``agrigrade.features``.
        unit: Physical unit, or ``""`` for dimensionless ratios.
        description: Human-readable meaning, family independent.
        polymorphic: True when the numeric meaning shifts between families, in
            which case ``semantics`` explains the value per family.
        semantics: Per-family interpretation of a polymorphic slot.
    """

    name: str
    unit: str
    description: str
    polymorphic: bool = False
    semantics: Mapping[ProduceFamily, str] = field(default_factory=dict)

    def label(self, family: ProduceFamily) -> str:
        """Return the label to show for this slot under ``family``."""
        if self.polymorphic:
            return self.semantics.get(family, f"{self.name} (unspecified family)")
        return f"{self.name} [{self.unit}]" if self.unit else self.name


#: Declared in vector order. Append only.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- Geometric, README section 3C -------------------------------------
    FeatureSpec("true_area_cm2", "cm^2", "Foreground area after reference-marker scaling."),
    FeatureSpec("equivalent_diameter_mm", "mm", "Circle of equal area to the foreground mask."),
    FeatureSpec("volume_cm3", "cm^3", "Spheroidal volume estimate from the equivalent diameter."),
    FeatureSpec("reference_scale_mm_per_px", "mm/px", "Scale ratio recovered from the on-screen marker."),
    FeatureSpec(
        "calibration_confidence",
        "",
        "Detector confidence that the reference marker was found and measured.",
    ),
    # --- Polymorphic spectral slot, README section 3A / 3B ---------------
    FeatureSpec(
        "spectral_primary",
        "",
        "Family-selected primary spectral index.",
        polymorphic=True,
        semantics={
            ProduceFamily.RED_SMOOTH: "NDTI, anthocyanin and lycopene accumulation",
            ProduceFamily.YELLOW_GREEN: "YI, carotenoid progression during ripening",
            ProduceFamily.BROWN_TEXTURED: "ExB, grey fungal growth on dark skin",
            ProduceFamily.UNKNOWN: "family unknown, index not interpretable",
        },
    ),
    FeatureSpec(
        "spectral_secondary",
        "",
        "Family-selected secondary spectral index.",
        polymorphic=True,
        semantics={
            ProduceFamily.RED_SMOOTH: "VARI, chlorophyll density",
            ProduceFamily.YELLOW_GREEN: "VARI, chlorophyll decay with age",
            ProduceFamily.BROWN_TEXTURED: "L* lightness, rot darkening",
            ProduceFamily.UNKNOWN: "family unknown, index not interpretable",
        },
    ),
    FeatureSpec(
        "defect_index",
        "",
        "Family-selected defect severity.",
        polymorphic=True,
        semantics={
            ProduceFamily.RED_SMOOTH: "NDTI standard deviation, dark spotting",
            ProduceFamily.YELLOW_GREEN: "fraction of pixels below the YI ripe threshold",
            ProduceFamily.BROWN_TEXTURED: "L* standard deviation, soft rot",
            ProduceFamily.UNKNOWN: "family unknown, index not interpretable",
        },
    ),
    # --- Colour space statistics ----------------------------------------
    FeatureSpec("mean_l", "L*", "Mean lightness over the foreground mask."),
    FeatureSpec("mean_a", "a*", "Mean green-red opponent axis."),
    FeatureSpec("mean_b", "b*", "Mean blue-yellow opponent axis."),
    FeatureSpec("lab_l_std", "L*", "Lightness dispersion, high for blotchy rot."),
    # --- Texture, README section 3B --------------------------------------
    FeatureSpec("glcm_homogeneity", "", "GLCM local uniformity, low for rough skin."),
    FeatureSpec("glcm_contrast", "", "GLCM local variation."),
    FeatureSpec("lbp_entropy", "bits", "Local binary pattern spread."),
    FeatureSpec("spot_ratio", "", "Contour area fraction of dark surface spots."),
)


class FeatureSchema:
    """Ordered, immutable description of the model's input vector."""

    def __init__(self, specs: Sequence[FeatureSpec] = FEATURE_SPECS, version: int = SCHEMA_VERSION) -> None:
        if not specs:
            raise ValueError("feature schema must not be empty")
        names = [spec.name for spec in specs]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate feature names in schema: {sorted(duplicates)}")
        self._specs: tuple[FeatureSpec, ...] = tuple(specs)
        self._index: dict[str, int] = {spec.name: position for position, spec in enumerate(self._specs)}
        self.version = version

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        return self._specs

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def __iter__(self) -> Iterable[FeatureSpec]:
        return iter(self._specs)

    def spec(self, name: str) -> FeatureSpec:
        try:
            return self._specs[self._index[name]]
        except KeyError:
            raise KeyError(f"unknown feature {name!r}; expected one of {list(self.names)}") from None

    def index(self, name: str) -> int:
        return self._index[name]

    def indices(self, names: Sequence[str]) -> list[int]:
        return [self.index(name) for name in names]

    def label(self, name: str, family: ProduceFamily) -> str:
        return self.spec(name).label(family)


#: The schema used when ``agrigrade.core.feature_schema`` is not available yet.
DEFAULT_SCHEMA = FeatureSchema()
