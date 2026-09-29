"""Explainable AI for a grading decision.

The spec promises farmers and buyers a feature breakdown explaining *why* a batch
received its grade. Two levels are provided:

*Global.* Mean decrease in impurity per feature, which says what the model
relies on across a dataset. Cheap, and the fastest way to catch a feature stage
emitting a dead column.

*Local.* Per-sample attribution for one decision, by single-feature occlusion:
each slot in turn is replaced by the training-set baseline and the change in the
winning grade's probability is recorded. A slot that was pushing the sample
towards Grade A shows a positive contribution when removed, and a slot that was
holding it back shows a negative one.

Why occlusion rather than SHAP: ``shap`` is not installed on the edge runtime,
and TreeSHAP is a compile-time dependency of a Python package we would otherwise
not ship. Occlusion needs only ``predict_proba``, runs in a single batched
forward pass, and produces a number a grower can read. Its limitation is
stated plainly below, because it is a real one.

Limitation: occlusion contributions are *not* additive. They are measured
one-at-a-time against the full context, so for correlated features they
overlap and the sum can exceed or undershoot the total shift away from the
baseline. Shapley values would fix the attribution, not the additivity, and
would cost 2^n evaluations. The ranking is what is reliable; the arithmetic is
not a decomposition and must not be presented as one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agrigrade.core.enums import ProduceFamily
from agrigrade.model.confidence import build_report
from agrigrade.model.schema import FeatureSchema, FeatureSpec

#: Contributed features returned unless a caller asks for fewer.
DEFAULT_TOP_K = 5


class ExplainsDecisions(Protocol):
    """The slice of the grader this module needs.

    Declared as a protocol so the attribution layer depends on behaviour rather
    than on :class:`~agrigrade.model.rf.QualityGrader`, which keeps the two
    modules independent and lets a stub stand in for a real ensemble.
    """

    schema: FeatureSchema

    @property
    def feature_importances_(self) -> np.ndarray: ...

    def baseline(self) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Class probabilities with columns in ``GRADE_ORDER``."""


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """One row of the global importance table."""

    rank: int
    name: str
    label: str
    importance: float
    cumulative: float

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "name": self.name,
            "label": self.label,
            "importance": round(self.importance, 6),
            "cumulative": round(self.cumulative, 4),
        }


@dataclass(frozen=True, slots=True)
class Attribution:
    """How one slot moved the winning grade's probability."""

    name: str
    label: str
    value: float
    baseline: float
    contribution: float
    unit: str

    @property
    def direction(self) -> str:
        if abs(self.contribution) < 1e-6:
            return "neutral"
        return "raised" if self.contribution > 0 else "suppressed"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "value": round(self.value, 4),
            "baseline": round(self.baseline, 4),
            "unit": self.unit,
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class Explanation:
    """A full local explanation, ready to serialise for the API."""

    grade: str
    confidence_pct: float
    family: ProduceFamily
    attributions: tuple[Attribution, ...]
    drivers: tuple[Attribution, ...]
    total_shift: float

    def as_dict(self) -> dict[str, object]:
        return {
            "grade": self.grade,
            "confidence_pct": round(self.confidence_pct, 2),
            "family": str(self.family),
            "total_shift": round(self.total_shift, 4),
            "drivers": [a.as_dict() for a in self.drivers],
            "attributions": [a.as_dict() for a in self.attributions],
        }

    def render(self) -> str:
        """Aligned text breakdown, for logs and the terminal demo."""
        width = max((len(a.name) for a in self.attributions), default=4)
        lines = [f"{self.grade} @ {self.confidence_pct:.1f}%   family {self.family}"]
        for attribution in self.attributions:
            marker = "+" if attribution.contribution > 0 else "-"
            magnitude = abs(attribution.contribution)
            bar = "#" * round(magnitude * 120)
            unit = f" {attribution.unit}" if attribution.unit else ""
            lines.append(
                f"  {attribution.name:>{width}}  {attribution.value:9.3f}{unit:<6}"
                f"  {marker}{magnitude:.4f} {bar}"
            )
        return "\n".join(lines)


def _plain_label(spec: FeatureSpec) -> str:
    """Unit form of a label, for tables that span more than one family."""
    return f"{spec.name} [{spec.unit}]" if spec.unit else str(spec.name)


def global_importance(
    grader: ExplainsDecisions,
    family: ProduceFamily | None = None,
    top_k: int | None = None,
) -> list[FeatureImportance]:
    """Rank features by mean decrease in impurity.

    Args:
        grader: A fitted ensemble satisfying :class:`ExplainsDecisions`.
        family: When given, slots are labelled using this family's reading of
            the polymorphic spectral slots. A global table spans families, so
            the label is a presentation choice and defaults to the unit form.
        top_k: Keep only the top N rows. ``cumulative`` is always computed
            against the full distribution first, so truncation does not change
            the percentages.

    Returns:
        Importances sorted by descending contribution.
    """
    schema = grader.schema
    importances = np.asarray(grader.feature_importances_, dtype=np.float64)
    order = np.argsort(importances)[::-1]
    total = float(importances.sum()) or 1.0

    rows: list[FeatureImportance] = []
    running = 0.0
    for rank, position in enumerate(order, start=1):
        running += float(importances[position]) / total
        spec = schema.specs[int(position)]
        label = spec.label(family) if family is not None else _plain_label(spec)
        rows.append(
            FeatureImportance(
                rank=rank,
                name=spec.name,
                label=label,
                importance=float(importances[position]),
                cumulative=running,
            )
        )
    return rows[:top_k] if top_k else rows


def explain(
    grader: ExplainsDecisions,
    values: Mapping[str, float],
    family: ProduceFamily = ProduceFamily.UNKNOWN,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Explanation:
    """Explain one decision by single-feature occlusion.

    Args:
        grader: A fitted ensemble satisfying :class:`ExplainsDecisions`.
        values: Feature mapping, validated against the grader's schema.
        family: Produce family, used to label the polymorphic spectral slots in
            the grader's own words. Passing the wrong family mislabels the
            explanation without affecting the numbers.
        top_k: How many rows to mark as the headline drivers.

    Returns:
        An :class:`Explanation`, with attributions sorted by descending
        absolute contribution.

    All perturbations are evaluated in one batched call, so the cost is a single
    forward pass over ``n_features`` rows rather than one per feature.
    """
    schema = grader.schema
    vector = np.asarray(schema.to_vector(values), dtype=np.float64)
    baseline = np.asarray(grader.baseline(), dtype=np.float64)
    if baseline.shape != vector.shape:
        raise ValueError(f"baseline has {baseline.shape[0]} features, sample has {vector.shape[0]}")

    probabilities = np.asarray(grader.predict_proba(vector), dtype=np.float64)[0]
    report = build_report(probabilities)
    winning = int(probabilities.argmax())

    perturbed = np.repeat(vector[np.newaxis, :], len(schema), axis=0)
    for position in range(len(schema)):
        perturbed[position, position] = baseline[position]
    shifted = np.asarray(grader.predict_proba(perturbed), dtype=np.float64)
    contributions = probabilities[winning] - shifted[:, winning]

    attributions = tuple(
        Attribution(
            name=schema.names[position],
            label=schema.label(schema.names[position], family),
            value=float(vector[position]),
            baseline=float(baseline[position]),
            contribution=float(contributions[position]),
            unit=schema.specs[position].unit,
        )
        for position in np.argsort(np.abs(contributions))[::-1]
    )

    return Explanation(
        grade=str(report.grade),
        confidence_pct=report.confidence_pct,
        family=family,
        attributions=attributions,
        drivers=attributions[:top_k],
        # Distance from the baseline sample, not a sum of the contributions.
        # The contributions overlap by construction, so summing them would
        # report a number this method does not support.
        total_shift=float(
            probabilities[winning] - float(grader.predict_proba(baseline)[0][winning])
        ),
    )


def render_global(rows: Sequence[FeatureImportance], limit: int = 10) -> str:
    """Aligned text table of global importances."""
    if not rows:
        return "no fitted model"
    width = max(len(row.name) for row in rows)
    lines = [f"{'#':>2}  {'feature':<{width}}  importance  cumulative"]
    for row in rows[:limit]:
        lines.append(
            f"{row.rank:>2}  {row.name:<{width}}  {row.importance:10.5f}  {row.cumulative:8.1%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    from agrigrade.core.enums import QualityGrade, family_for
    from agrigrade.model.mock import make_dataset, single_frame
    from agrigrade.model.rf import QualityGrader

    built = make_dataset(n_per_grade=60, seed=11)
    train, _test = built.split(0.25)
    grader = QualityGrader(train.schema, n_estimators=100, random_state=7)
    grader.fit(train.to_matrix(), train.to_labels())

    print("global feature importance (tree impurity, all families pooled)")
    print(render_global(global_importance(grader)))
    print()

    print("same table read as brown produce, where the spectral slots change meaning")
    print(render_global(global_importance(grader, family=ProduceFamily.BROWN_TEXTURED), limit=5))
    print()

    frame = single_frame(ProduceFamily.BROWN_TEXTURED, grade=QualityGrade.GRADE_B)
    print(f"local explanation, {frame.produce_class} true grade {frame.grade}")
    print(explain(grader, frame.values, family_for(frame.produce_class)).render())
    print()
    print("payload shape the API returns")
    payload = explain(grader, frame.values, family_for(frame.produce_class)).as_dict()
    print(json.dumps({k: payload[k] for k in ("grade", "confidence_pct", "family")}, indent=2))
    print(f"  keys: {sorted(payload)}")
