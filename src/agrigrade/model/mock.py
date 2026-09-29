"""Labelled synthetic feature data.

The feature stage lives on the ``feature-extraction`` branch and is not merged,
so the model is developed and demonstrated against synthetic data. The generator
below is not a random walk: it encodes the defect physics from the spec so the
resulting dataset is learnable and the numbers are plausible.

Two properties matter for it to be useful as a stand-in:

* **Family conditioning.** A Kiwi and an Apple produce the same slots with
  different distributions, mirroring the class-conditioned routing the spec
  describes. A model trained only on one family is visibly wrong on the other.
* **A learnable label rule.** Quality is assigned from a latent score built
  from the defect drivers plus noise, so a fitted ensemble reaches high accuracy
  for a reason. A uniformly random label would make every metric meaningless.

All randomness flows from a single seeded generator, so a run is reproducible
and two branches can be compared on identical data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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


@dataclass(frozen=True, slots=True)
class MockSample:
    """One synthetic observation."""

    values: dict[str, float]
    grade: QualityGrade
    family: ProduceFamily
    produce_class: ProduceClass
    latent_quality: float

    def as_mapping(self) -> dict[str, float]:
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
            cut = max(1, int(round(len(group) * test_fraction)))
            test.extend(group[:cut])
            train.extend(group[cut:])
        return MockDataset(tuple(train), self.schema), MockDataset(tuple(test), self.schema)

    def to_matrix(self) -> np.ndarray:
        return np.asarray([self.schema.to_vector(s.values) for s in self.samples], dtype=np.float64)

    def to_labels(self) -> list[QualityGrade]:
        return [sample.grade for sample in self.samples]
