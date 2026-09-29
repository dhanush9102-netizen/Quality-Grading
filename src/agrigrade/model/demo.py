"""Staged end-to-end walkthrough.

    PYTHONPATH=src python -m agrigrade.model

Runs the whole pipeline over mock captures and prints the result of every stage,
so the branch can be reviewed as a sequence of claims rather than as a library.
Each section is numbered and states what it is supposed to demonstrate, so a
failing section points at the stage that broke.

Stages:

1. Schema, the column contract the ensemble is positional about
2. Mock data, family-conditioned and class balanced
3. Fit, with the out-of-bag score
4. Held-out evaluation, ordinal penalty and calibration
5. Inference on a single frame, with the confidence gate
6. XAI, global importances and a per-sample attribution
7. Persistence, a round-trip through disk with a mismatch guard
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from agrigrade.core.enums import GRADE_ORDER, ProduceFamily, QualityGrade, family_for
from agrigrade.core.errors import SchemaMismatchError
from agrigrade.model import artifacts
from agrigrade.model.confidence import build_report, confidence_for, ensemble_soft_vote
from agrigrade.model.evaluate import ClassificationReport, evaluate
from agrigrade.model.mock import CHANNELS, MockDataset, describe, make_dataset, single_frame
from agrigrade.model.rf import QualityGrader
from agrigrade.model.schema import FeatureSchema, load_schema
from agrigrade.model.xai import explain, global_importance, render_global

#: Seed shared by every stage, so a run is reproducible end to end.
SEED = 7


def _rule(title: str, claim: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print(f"  claim: {claim}")
    print("=" * 78)


def stage_schema() -> FeatureSchema:
    _rule(
        "1. SCHEMA",
        "the ensemble is positional, so the column order is part of the contract",
    )
    schema = load_schema()
    try:
        from agrigrade.core import feature_schema  # type: ignore[attr-defined]  # noqa: F401

        source = "agrigrade.core.feature_schema"
    except ImportError:
        source = "local declaration, core has not merged"

    print(f"schema v{schema.version}, {len(schema)} features, source: {source}")
    print()
    for position, spec in enumerate(schema):
        mark = "*" if spec.polymorphic else " "
        print(f" {position:2d} {mark} {spec.name:26s} {spec.description}")
    print()
    print("slots marked * are polymorphic: same column, different meaning per family")
    for family in ProduceFamily:
        if family is ProduceFamily.UNKNOWN:
            continue
        print(f"  {family!s:16s} spectral_primary = {schema.label('spectral_primary', family)}")
    return schema


def stage_data() -> tuple[MockDataset, MockDataset]:
    _rule(
        "2. MOCK DATA",
        "family conditioned and class balanced, so the metrics below mean something",
    )
    dataset = make_dataset(n_per_grade=100, seed=SEED)
    train, test = dataset.split(0.25)
    print(describe(train))
    print()
    print("defect channels per family, and the direction quality moves them")
    for family, channels in CHANNELS.items():
        flags = "  ".join(f"{name} {'+' if sign > 0 else '-'}" for name, sign in channels.items())
        print(f"  {family!s:16s} {flags}")
    print()
    print(f"train {len(train)} / held-out {len(test)}")
    return train, test


def stage_fit(train: MockDataset) -> QualityGrader:
    _rule("3. FIT", "100 trees, out-of-bag score reported rather than assumed")
    grader = QualityGrader(train.schema, n_estimators=100, random_state=SEED)
    report = grader.fit(train.to_matrix(), train.to_labels())
    print(report.summary())
    print()
    baseline = grader.baseline()
    shown = list(zip(train.schema.names[:3], baseline[:3], strict=True))
    print(f"baseline captured for attribution, {len(baseline)} features, first 3:")
    for name, value in shown:
        print(f"  {name:26s} {value: .4f}")
    return grader


def stage_evaluation(grader: QualityGrader, test: MockDataset) -> ClassificationReport:
    _rule(
        "4. HELD-OUT EVALUATION",
        "accuracy, ordinal penalty and whether the percentages can be believed",
    )
    _grades, probabilities = grader.predict(test.to_matrix())
    report = evaluate(test.to_labels(), probabilities)
    print(report.summary())
    print()
    print("confusion, rows are truth. mass is concentrated on the diagonal, which")
    print("is the expected failure mode: adjacent grades are the ambiguous ones.")
    print(report.confusion_table())
    print()
    print(report.reliability_table())
    print()
    admitted = [build_report(p).is_confident for p in probabilities]
    subset = evaluate(
        [t for t, ok in zip(test.to_labels(), admitted, strict=True) if ok],
        probabilities[np.array(admitted)],
    )
    print(
        f"the confidence gate admits {sum(admitted)}/{len(test)} samples at "
        f"{subset.accuracy:.4f} accuracy against {report.accuracy:.4f} overall"
    )
    return report


def stage_inference(grader: QualityGrader) -> None:
    _rule(
        "5. INFERENCE",
        "one frame in, a grade and a defensible confidence out",
    )
    for family, target in (
        (ProduceFamily.RED_SMOOTH, QualityGrade.GRADE_A),
        (ProduceFamily.YELLOW_GREEN, QualityGrade.GRADE_C),
        (ProduceFamily.BROWN_TEXTURED, QualityGrade.REJECT),
    ):
        frame = single_frame(family, grade=target, seed=SEED)
        row = grader.predict_proba(frame.values)[0]
        report = build_report(row)
        literal_grade, literal_pct = confidence_for(row)
        print(f"{frame.produce_class!s:8s} true {target!s:8s} -> {report.summary()}")
        agree = literal_grade is report.grade and abs(literal_pct - report.confidence_pct) < 1e-9
        print(f"         spec 4A gives {literal_grade!s} @ {literal_pct:.1f}%, agrees: {agree}")

    frame = single_frame(ProduceFamily.BROWN_TEXTURED, grade=QualityGrade.GRADE_B, seed=SEED)
    votes = ensemble_soft_vote(grader.tree_votes(frame.values), len(GRADE_ORDER))[0]
    proba = grader.predict_proba(frame.values)[0]
    print()
    print("soft vote from the N trees, against predict_proba, on the same frame")
    for grade, vote, probability in zip(GRADE_ORDER, votes, proba, strict=True):
        print(f"  {grade!s:8s} {vote:6.3f} {probability:6.3f}  delta {abs(vote - probability):.1e}")


def stage_xai(grader: QualityGrader) -> None:
    _rule(
        "6. XAI",
        "why this grade, in words a grower can act on",
    )
    print("global: what the model relies on across the dataset")
    print(render_global(global_importance(grader), limit=8))
    print()
    frame = single_frame(ProduceFamily.BROWN_TEXTURED, grade=QualityGrade.GRADE_B, seed=SEED)
    family = family_for(frame.produce_class)
    print(f"local: one {frame.produce_class}, true grade {frame.grade}")
    explanation = explain(grader, frame.values, family)
    print(explanation.render())
    print()
    print("the same three drivers, as the API would serialise them")
    for attribution in explanation.drivers:
        print(
            f"  {attribution.direction:>10}  {attribution.label:52s} "
            f"({attribution.value:.3f} vs baseline {attribution.baseline:.3f})"
        )


def stage_persistence(grader: QualityGrader, train: MockDataset, test: MockDataset) -> None:
    _rule(
        "7. PERSISTENCE",
        "a bundle carries its context and refuses to load against a different one",
    )
    expected, probabilities = grader.predict(test.to_matrix())
    report = evaluate(test.to_labels(), probabilities)
    bundle = artifacts.build_bundle(
        grader,
        n_samples=len(train),
        X=train.to_matrix(),
        metrics={"accuracy": report.accuracy, "macro_f1": report.macro_f1},
        notes="MOCK BASELINE - synthetic data, not trained on real produce",
    )

    with tempfile.TemporaryDirectory() as directory:
        path = artifacts.save(bundle, Path(directory) / "demo")
        print(f"wrote {path.name}, {path.stat().st_size / 1024:.0f} KiB")
        print(f"      plus {path.with_suffix('.manifest.json').name}")

        restored = artifacts.load(path)
        same = restored.grader.predict(test.to_matrix())[0] == [str(g) for g in expected]
        print(f"round trip reproduces every held-out grade: {same}")
        stable = artifacts.checksum_of(grader) == artifacts.checksum_of(restored.grader)
        print(f"checksum stable across the round trip: {stable}")

        print()
        print("the guard, exercised by reversing the bundle's feature order")
        tampered = artifacts.ArtifactBundle(
            grader=grader,
            feature_order=tuple(reversed(bundle.feature_order)),
            grade_order=bundle.grade_order,
            schema_version=bundle.schema_version,
            bundle_format=bundle.bundle_format,
            metadata=bundle.metadata,
        )
        try:
            tampered.verify_against()
            print("  NOT RAISED, which would be the bug")
        except SchemaMismatchError as exc:
            print(f"  SchemaMismatchError: {str(exc).splitlines()[0][:72]}")
        print("  without this, a reordered vector scores silently instead of failing")


def main() -> int:
    print("AgriGrade AI - Random Forest quality grader")
    print("mock feature data, no trained-on-produce model in this run")

    stage_schema()
    train, test = stage_data()
    grader = stage_fit(train)
    stage_evaluation(grader, test)
    stage_inference(grader)
    stage_xai(grader)
    stage_persistence(grader, train, test)

    print()
    print("=" * 78)
    print("all stages completed")
    print("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    raise SystemExit(main())
