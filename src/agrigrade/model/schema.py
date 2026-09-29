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

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agrigrade.core.enums import ProduceFamily
from agrigrade.core.errors import SchemaMismatchError

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


#: Declared in vector order. Append only once an artifact has been trained
#: against it; a reorder invalidates every serialized model.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- Geometric, README section 3C -------------------------------------
    FeatureSpec("true_area_cm2", "cm^2", "Foreground area after reference-marker scaling."),
    FeatureSpec("equivalent_diameter_mm", "mm", "Circle of equal area to the foreground mask."),
    FeatureSpec(
        "volume_cm3",
        "cm^3",
        "Spheroidal volume estimate from the equivalent diameter.",
    ),
    FeatureSpec(
        "reference_scale_mm_per_px",
        "mm/px",
        "Scale ratio recovered from the on-screen marker.",
    ),
    FeatureSpec(
        "calibration_confidence",
        "",
        "Detector confidence that the reference marker was found and measured.",
    ),
    # --- Class conditioning ---------------------------------------------
    # `spectral_primary` carries opposite polarity between families: NDTI rises
    # with ripeness on red produce, ExB rises with mould on brown produce. A
    # pooled ensemble given only the value cannot learn both, so the family is
    # supplied as an explicit one-hot. Without these slots the forest scores
    # near chance while looking correctly trained.
    FeatureSpec("family_red_smooth", "", "One-hot: the sample is a red/smooth class."),
    FeatureSpec("family_yellow_green", "", "One-hot: the sample is a yellow/green class."),
    FeatureSpec("family_brown_textured", "", "One-hot: the sample is a brown/textured class."),
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

    def __init__(
        self,
        specs: Sequence[FeatureSpec] = FEATURE_SPECS,
        version: int = SCHEMA_VERSION,
    ) -> None:
        if not specs:
            raise ValueError("feature schema must not be empty")
        names = [spec.name for spec in specs]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate feature names in schema: {sorted(duplicates)}")
        self._specs: tuple[FeatureSpec, ...] = tuple(specs)
        self._index: dict[str, int] = {
            spec.name: position for position, spec in enumerate(self._specs)
        }
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

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self._specs)

    def spec(self, name: str) -> FeatureSpec:
        try:
            return self._specs[self._index[name]]
        except KeyError:
            raise KeyError(
                f"unknown feature {name!r}; expected one of {list(self.names)}"
            ) from None

    def index(self, name: str) -> int:
        return self._index[name]

    def indices(self, names: Sequence[str]) -> list[int]:
        return [self.index(name) for name in names]

    def label(self, name: str, family: ProduceFamily) -> str:
        return self.spec(name).label(family)

    # --- validation and coercion ----------------------------------------

    def validate(self, values: Mapping[str, Any]) -> Mapping[str, float]:
        """Return ``values`` as a validated float mapping.

        Rejects a mapping that is missing a slot, carries an unknown slot, or
        holds a non-finite or non-numeric value. Padding and truncating are both
        silent scoring bugs, so neither is offered.

        Raises:
            SchemaMismatchError: on any deviation from the declared schema.
        """
        missing = [name for name in self.names if name not in values]
        if missing:
            raise SchemaMismatchError(f"missing {len(missing)} feature(s): {missing}")
        unknown = sorted(set(values) - set(self.names))
        if unknown:
            raise SchemaMismatchError(f"unknown feature(s) present: {unknown}")

        coerced: dict[str, float] = {}
        for spec in self._specs:
            raw = values[spec.name]
            try:
                number = float(raw)
            except (TypeError, ValueError):
                raise SchemaMismatchError(
                    f"feature {spec.name!r} is not numeric: {raw!r}"
                ) from None
            if number != number or number in (float("inf"), float("-inf")):
                raise SchemaMismatchError(f"feature {spec.name!r} is not finite: {raw!r}")
            coerced[spec.name] = number
        return coerced

    def to_vector(self, values: Mapping[str, Any]) -> list[float]:
        """Validate ``values`` and project them into vector order."""
        validated = self.validate(values)
        return [validated[name] for name in self.names]

    def to_matrix(self, rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        return [self.to_vector(row) for row in rows]

    def from_vector(self, vector: Sequence[float]) -> dict[str, float]:
        """Rebuild a mapping from a positional vector, checking the length."""
        if len(vector) != len(self._specs):
            raise SchemaMismatchError(f"expected {len(self._specs)} features, got {len(vector)}")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise SchemaMismatchError(f"vector is not numeric: {exc}") from exc
        return dict(zip(self.names, values, strict=True))

    def to_frame(self) -> list[dict[str, str]]:
        """Tabular description, used by the API's ``/schema`` route."""
        return [
            {
                "index": str(position),
                "name": spec.name,
                "unit": spec.unit,
                "description": spec.description,
                "polymorphic": str(spec.polymorphic).lower(),
            }
            for position, spec in enumerate(self._specs)
        ]


#: The schema used when ``agrigrade.core.feature_schema`` is not available yet.
DEFAULT_SCHEMA = FeatureSchema()


def load_schema() -> FeatureSchema:
    """Return the canonical schema, preferring ``agrigrade.core``.

    The shared contract on ``main`` is the source of truth. Until
    ``agrigrade.core.feature_schema`` merges, fall back to the local declaration
    so the branch is runnable in isolation. When it lands, the fallback is
    deleted and this function returns the core schema unchanged.
    """
    try:
        from agrigrade.core import feature_schema as core_schema  # type: ignore[attr-defined]
    except ImportError:
        return DEFAULT_SCHEMA
    return FeatureSchema(core_schema.FEATURE_SPECS, version=core_schema.SCHEMA_VERSION)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    loaded = load_schema()
    print(f"schema v{loaded.version}, {len(loaded)} features")
    print(json.dumps(loaded.to_frame(), indent=2))
