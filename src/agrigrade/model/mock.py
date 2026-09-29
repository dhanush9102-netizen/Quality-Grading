"""Labelled synthetic feature data.

The feature stage lives on the ``feature-extraction`` branch and is not merged,
so the model is developed and demonstrated against synthetic data. The generator
is not a random walk: it encodes the defect physics from the spec so the
resulting dataset is learnable and the numbers are plausible.

Three properties matter for it to be useful as a stand-in:

* **Family conditioning.** A Kiwi and an Apple populate the same slots with
  different distributions, mirroring the class-conditioned routing in spec
  section 3B. A model tuned on one family is visibly wrong on the other.
* **A learnable label rule.** Quality is assigned from a latent score built
  from the defect drivers plus noise, so a fitted ensemble reaches high accuracy
  for a stated reason. A uniformly random label would make every metric
  meaningless.
* **Reproducibility.** All randomness flows from one seeded generator, so a run
  is repeatable and two branches can be compared on identical data.

Per-family slot ranges follow the routing table:

===============  ====================  ====================================
Family           spectral_primary      Defect driver
===============  ====================  ====================================
red_smooth       NDTI ~ 0.15..0.80     NDTI dispersion, dark spotting
yellow_green     YI  ~ 0.05..0.62      sub-threshold pixel fraction
brown_textured   ExB ~ -0.20..0.10     L* dispersion, soft rot
===============  ====================  ====================================
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from agrigrade.core.enums import GRADE_ORDER, ProduceClass, ProduceFamily, QualityGrade
from agrigrade.model.schema import DEFAULT_SCHEMA, FeatureSchema

#: Representative produce class per family, used when labelling a family.
FAMILY_REPRESENTATIVE: dict[ProduceFamily, ProduceClass] = {
    ProduceFamily.RED_SMOOTH: ProduceClass.APPLE,
    ProduceFamily.YELLOW_GREEN: ProduceClass.BANANA,
    ProduceFamily.BROWN_TEXTURED: ProduceClass.KIWI,
    ProduceFamily.UNKNOWN: ProduceClass.UNKNOWN,
}

#: Per-family sampling ranges, named so the distribution can be read off the
#: spec. Gaussian draws are centred on the range with sigma scaled by
#: ``noise`` * width, so the stated bounds are two to three sigma out.
FAMILY_RANGES: dict[ProduceFamily, dict[str, tuple[float, float]]] = {
    ProduceFamily.RED_SMOOTH: {
        "spectral_primary": (0.15, 0.80),
        "spectral_secondary": (-0.45, 0.05),
        "defect_index": (0.01, 0.16),
        "mean_l": (30.0, 68.0),
        "mean_a": (25.0, 55.0),
        "mean_b": (15.0, 45.0),
        "lab_l_std": (2.0, 14.0),
        "glcm_homogeneity": (0.45, 0.92),
        "glcm_contrast": (0.05, 0.45),
        "lbp_entropy": (3.0, 6.2),
        "spot_ratio": (0.005, 0.09),
    },
    ProduceFamily.YELLOW_GREEN: {
        "spectral_primary": (0.05, 0.62),
        "spectral_secondary": (-0.35, 0.30),
        "defect_index": (0.01, 0.42),
        "mean_l": (45.0, 88.0),
        "mean_a": (-8.0, 18.0),
        "mean_b": (30.0, 75.0),
        "lab_l_std": (3.0, 16.0),
        "glcm_homogeneity": (0.40, 0.88),
        "glcm_contrast": (0.08, 0.55),
        "lbp_entropy": (3.2, 6.4),
        "spot_ratio": (0.01, 0.22),
    },
    ProduceFamily.BROWN_TEXTURED: {
        "spectral_primary": (-0.20, 0.10),
        "spectral_secondary": (25.0, 62.0),
        "defect_index": (1.5, 16.0),
        "mean_l": (18.0, 52.0),
        "mean_a": (0.0, 16.0),
        "mean_b": (5.0, 30.0),
        "lab_l_std": (2.0, 18.0),
        "glcm_homogeneity": (0.25, 0.70),
        "glcm_contrast": (0.15, 0.75),
        "lbp_entropy": (4.0, 7.2),
        "spot_ratio": (0.01, 0.18),
    },
}

#: Which slots carry defect signal, and in which direction. ``+1`` means a
#: higher normalised value is evidence of better quality, ``-1`` means worse.
#: Everything not listed here is a distractor: informative in a real dataset
#: only by correlation, and drawn independently here.
CHANNELS: dict[ProduceFamily, dict[str, float]] = {
    ProduceFamily.RED_SMOOTH: {
        "spectral_primary": 1.0,  # NDTI up, anthocyanin accumulation
        "defect_index": -1.0,  # NDTI dispersion, dark spotting
        "lab_l_std": -0.7,  # blotchy surface
        "spot_ratio": -1.0,  # dark spot contour area
        "glcm_homogeneity": 0.5,  # smooth unblemished skin
    },
    ProduceFamily.YELLOW_GREEN: {
        "spectral_primary": 1.0,  # YI up, carotenoid progression
        "defect_index": -1.0,  # sub-threshold pixel fraction
        "spot_ratio": -1.0,  # black spot contour
        "mean_l": 0.6,  # ripe fruit is brighter
        "mean_b": 0.4,  # warm yellow cast
    },
    ProduceFamily.BROWN_TEXTURED: {
        "spectral_primary": -1.0,  # ExB up means grey fungal growth
        "defect_index": -1.0,  # L* dispersion, soft rot
        "lab_l_std": -0.8,  # uneven rot
        "glcm_homogeneity": 0.5,  # intact hairy skin
    },
}

#: How far a channel travels across the quality range, as a fraction of its
#: normalised span. The grade bands are wide relative to this, so adjacent
#: grades genuinely overlap and the confusion matrix has a realistic diagonal
#: bleed instead of a clean 1.00.
CHANNEL_SWING = 0.45

#: Gaussian noise added to each channel, in normalised units. This is the
#: signal-to-noise ratio a real feature stage has to beat; at 0.035 a fitted
#: ensemble lands near 0.83 on held-out data, the rest being band overlap and
#: label noise rather than model capacity.
CHANNEL_NOISE = 0.035

#: Spread of the distractor slots, centred mid-range.
DISTRACTOR_NOISE = 0.18

#: Quality bands per grade, in latent-quality units. Adjacent bands touch, so
#: grades near a boundary are genuinely ambiguous.
QUALITY_BANDS: dict[QualityGrade, tuple[float, float]] = {
    QualityGrade.GRADE_A: (0.62, 1.00),
    QualityGrade.GRADE_B: (0.34, 0.62),
    QualityGrade.GRADE_C: (0.10, 0.34),
    QualityGrade.REJECT: (0.00, 0.10),
}

#: Probability that a label is flipped to an adjacent grade, modelling
#: annotator disagreement and borderline cases. Caps attainable accuracy just
#: under 100% so the confusion matrix has something to show.
LABEL_NOISE = 0.05

#: One-hot column per family. Carried as ordinary features because the pooled
#: ensemble needs them: the spectral slots have opposite polarity between
#: families and are unreadable without knowing which recipe produced them.
FAMILY_FLAGS: dict[ProduceFamily, tuple[str, ...]] = {
    ProduceFamily.RED_SMOOTH: ("family_red_smooth",),
    ProduceFamily.YELLOW_GREEN: ("family_yellow_green",),
    ProduceFamily.BROWN_TEXTURED: ("family_brown_textured",),
    ProduceFamily.UNKNOWN: (),
}

#: Every flag column, so an absent family is emitted as an explicit 0.0 rather
#: than omitted. The schema is strict on purpose.
FAMILY_FLAG_NAMES: tuple[str, ...] = (
    "family_red_smooth",
    "family_yellow_green",
    "family_brown_textured",
)

#: Ranges for the family-independent slots. These carry no grade signal: a
#: grading model must not use them, and the XAI output showing near-zero
#: importance for them is a result, not a gap.
GLOBAL_RANGES: dict[str, tuple[float, float]] = {
    "true_area_cm2": (12.0, 48.0),
    "equivalent_diameter_mm": (39.0, 78.0),
    "volume_cm3": (31.0, 249.0),
    "reference_scale_mm_per_px": (0.15, 0.45),
    "calibration_confidence": (0.72, 0.99),
}


# --- containers ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MockSample:
    """One synthetic observation."""

    values: dict[str, float]
    grade: QualityGrade
    family: ProduceFamily
    produce_class: ProduceClass
    latent_quality: float

    def as_mapping(self) -> Mapping[str, float]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class MockDataset:
    """An ordered batch of samples plus the feature order used to build them."""

    samples: tuple[MockSample, ...]
    schema: FeatureSchema = DEFAULT_SCHEMA

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[MockSample]:
        return iter(self.samples)

    @property
    def families(self) -> tuple[ProduceFamily, ...]:
        seen: list[ProduceFamily] = []
        for sample in self.samples:
            if sample.family not in seen:
                seen.append(sample.family)
        return tuple(seen)

    @property
    def grade_counts(self) -> dict[QualityGrade, int]:
        counts = dict.fromkeys(GRADE_ORDER, 0)
        for sample in self.samples:
            counts[sample.grade] += 1
        return counts

    def split(self, test_fraction: float = 0.25) -> tuple[MockDataset, MockDataset]:
        """Split by sample index, stratified per grade, without shuffling."""
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")

        per_grade: dict[QualityGrade, list[MockSample]] = {grade: [] for grade in GRADE_ORDER}
        for sample in self.samples:
            per_grade[sample.grade].append(sample)

        train: list[MockSample] = []
        test: list[MockSample] = []
        for grade in GRADE_ORDER:
            group = per_grade[grade]
            cut = max(1, round(len(group) * test_fraction))
            test.extend(group[:cut])
            train.extend(group[cut:])
        return MockDataset(tuple(train), self.schema), MockDataset(tuple(test), self.schema)

    def to_matrix(self) -> np.ndarray:
        return np.asarray([self.schema.to_vector(s.values) for s in self.samples], dtype=np.float64)

    def to_labels(self) -> list[QualityGrade]:
        return [sample.grade for sample in self.samples]


# --- generation ---------------------------------------------------------


def _draw(rng: np.random.Generator, low: float, high: float, noise: float) -> float:
    """Draw from a family range, centred mid-range with sigma scaled by width."""
    return float(rng.normal((low + high) / 2.0, (high - low) * noise))


def _denormalise(family: ProduceFamily, name: str, unit: float) -> float:
    low, high = FAMILY_RANGES[family][name]
    return float(low + min(1.0, max(0.0, unit)) * (high - low))


def _draw_quality(rng: np.random.Generator, grade: QualityGrade) -> float:
    low, high = QUALITY_BANDS[grade]
    return float(rng.uniform(low, high))


def _flip_label(rng: np.random.Generator, grade: QualityGrade) -> QualityGrade:
    """Move a label one step along the ladder, for annotator disagreement."""
    position = GRADE_ORDER.index(grade)
    step = 1 if rng.random() < 0.5 else -1
    target = min(len(GRADE_ORDER) - 1, max(0, position + step))
    return GRADE_ORDER[target]


def _generate_values(
    rng: np.random.Generator,
    family: ProduceFamily,
    quality: float,
    noise: float,
) -> dict[str, float]:
    """Draw every family slot for a sample of the given latent quality.

    Channels move monotonically with quality; distractors are independent of it.
    """
    ranges = FAMILY_RANGES[family]
    channels = CHANNELS[family]
    values: dict[str, float] = {}

    for name, (low, high) in ranges.items():
        if name in channels:
            sign = channels[name]
            unit = 0.5 + CHANNEL_SWING * sign * (quality - 0.5)
            unit += rng.normal(0.0, CHANNEL_NOISE * (1.0 + noise))
            values[name] = _denormalise(family, name, unit)
        else:
            values[name] = _draw(rng, low, high, DISTRACTOR_NOISE * (1.0 + noise))

    for name, (low, high) in GLOBAL_RANGES.items():
        values[name] = _draw(rng, low, high, DISTRACTOR_NOISE * (1.0 + noise))
    for flag in FAMILY_FLAG_NAMES:
        values[flag] = 1.0 if flag in FAMILY_FLAGS[family] else 0.0
    return values


def make_sample(
    rng: np.random.Generator,
    family: ProduceFamily,
    grade: QualityGrade | None = None,
    quality: float | None = None,
    noise: float = 0.22,
) -> MockSample:
    """Draw one synthetic observation for ``family``.

    Supply ``grade`` or ``quality`` to target a specific band; with neither, the
    grade is drawn from the nominal class balance.
    """
    if family not in FAMILY_RANGES:
        raise ValueError(f"no synthetic distribution for family {family!r}")
    if (grade is None) == (quality is None):
        raise ValueError("supply exactly one of grade or quality")
    if grade is not None and quality is not None:
        raise ValueError("supply exactly one of grade or quality")

    if quality is None:
        target = grade if grade is not None else QualityGrade.GRADE_B
        quality = _draw_quality(rng, target)
    label = grade if grade is not None else _band_for(quality)
    if rng.random() < LABEL_NOISE:
        label = _flip_label(rng, label)

    return MockSample(
        values=_generate_values(rng, family, quality, noise),
        grade=label,
        family=family,
        produce_class=FAMILY_REPRESENTATIVE[family],
        latent_quality=quality,
    )


def _band_for(quality: float) -> QualityGrade:
    for grade in GRADE_ORDER:
        low, high = QUALITY_BANDS[grade]
        if low <= quality <= high:
            return grade
    return QualityGrade.REJECT


def make_dataset(
    n_per_grade: int = 60,
    families: Sequence[ProduceFamily] | None = None,
    seed: int = 7,
    noise: float = 0.22,
) -> MockDataset:
    """Build a dataset with exactly ``n_per_grade`` samples of each grade.

    Grades are assigned first and the features generated to match, so the class
    balance is exact rather than an artefact of the sampling. Families rotate
    within each grade, so no family dominates a class.
    """
    if n_per_grade < 1:
        raise ValueError("n_per_grade must be at least 1")
    families = list(families) if families is not None else list(FAMILY_RANGES)
    unknown = [f for f in families if f not in FAMILY_RANGES]
    if unknown:
        raise ValueError(f"no synthetic distribution for families {unknown}")

    rng = np.random.default_rng(seed)
    samples: list[MockSample] = []
    for grade in GRADE_ORDER:
        for position in range(n_per_grade):
            family = families[position % len(families)]
            samples.append(make_sample(rng, family, grade=grade, noise=noise))
    return MockDataset(tuple(samples))


def single_frame(
    family: ProduceFamily = ProduceFamily.BROWN_TEXTURED,
    grade: QualityGrade = QualityGrade.GRADE_B,
    seed: int = 3,
) -> MockSample:
    """One observation, for exercising the inference and XAI paths."""
    return make_sample(np.random.default_rng(seed), family, grade=grade)


def describe(dataset: MockDataset) -> str:
    """Human-readable class balance table."""
    counts = dataset.grade_counts
    width = max(len(str(grade)) for grade in GRADE_ORDER)
    lines = [f"{len(dataset)} samples across {len(dataset.families)} families"]
    for grade in GRADE_ORDER:
        share = 100.0 * counts[grade] / max(1, len(dataset))
        bar = "#" * round(share / 2.5)
        lines.append(f"  {grade!s:>{width}}  {counts[grade]:>5}  {share:5.1f}%  {bar}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    built = make_dataset(n_per_grade=40, seed=7)
    print(describe(built))
    print()
    example = single_frame()
    print(f"{example.produce_class} / {example.family} -> {example.grade}")
    print(f"latent quality {example.latent_quality:.3f}")
    for name, value in example.values.items():
        print(f"  {name:28s} {value: .4f}")
