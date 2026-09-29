"""Versioned model artifact persistence.

A serialized estimator is only as safe as the context saved beside it. Column
order, grade order, schema version, hyperparameters and the training data's
distribution all have to travel with the weights, because a model loaded into a
process running a different feature order will not raise: it will score.

:class:`ArtifactBundle` therefore refuses to load when the schema or grade
ladder disagrees with the current process, instead of warning and continuing.

Binary files are excluded from version control by the repository's gitignore, so
this package also carries the checksums needed to detect a truncated download
or a partially written file on a device that lost power mid-save.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from agrigrade.core.enums import GRADE_ORDER, QualityGrade
from agrigrade.core.errors import SchemaMismatchError
from agrigrade.model.rf import QualityGrader
from agrigrade.model.schema import load_schema

#: Artifacts land here, which the repository gitignores.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

#: Bundles are written with this suffix so a half-written file is recognisable.
BUNDLE_SUFFIX = ".agrigrade.joblib"

#: Bumped when the bundle layout changes incompatibly.
BUNDLE_FORMAT = 1

#: Reserved metrics key holding the per-column training distribution.
FEATURE_SUMMARY_KEY = "_feature_summary"


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    """Everything needed to explain where an artifact came from."""

    n_samples: int
    n_features: int
    n_estimators: int
    random_state: int
    oob_score: float | None
    grade_counts: dict[str, int]
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    created_by: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """A fitted grader plus the context needed to use it safely."""

    grader: QualityGrader
    feature_order: tuple[str, ...]
    grade_order: tuple[str, ...]
    schema_version: int
    bundle_format: int
    metadata: TrainingMetadata

    @property
    def feature_summary(self) -> dict[str, float]:
        """Min, max and mean per feature, for drift checks on new captures."""
        stored = self.metadata.metrics.get(FEATURE_SUMMARY_KEY, {})
        return {k: float(v) for k, v in stored.items()}

    def drift(self, X: np.ndarray) -> dict[str, float]:  # noqa: N803
        """Per-feature shift of ``X`` against the training distribution.

        Returns the ratio of each observed mean to the training mean, for the
        columns where that is meaningful. A device whose captures arrive with a
        systematically different distribution is the most likely cause of a
        sudden drop in accuracy, and this is how it gets noticed.
        """
        stored = self.metadata.metrics.get(FEATURE_SUMMARY_KEY, {})
        observed = feature_summary(X, self.feature_order)
        shifts: dict[str, float] = {}
        for name in self.feature_order:
            baseline = stored.get(f"{name}.mean")
            if baseline:
                shifts[name] = observed[f"{name}.mean"] / float(baseline)
        return shifts

    def verify_against(self, other: Mapping[str, Any] | None = None) -> None:
        """Raise if this bundle does not match the current process.

        Args:
            other: An alternative context, for checking a bundle against another
                bundle's manifest rather than against the live schema.

        Raises:
            SchemaMismatchError: On any disagreement.
        """
        context = dict(other) if other else _current_context()
        checks = {
            "schema_version": (self.schema_version, context["schema_version"]),
            "feature_order": (list(self.feature_order), list(context["feature_order"])),
            "grade_order": (list(self.grade_order), list(context["grade_order"])),
            "bundle_format": (self.bundle_format, BUNDLE_FORMAT),
        }
        for label, (mine, theirs) in checks.items():
            if mine != theirs:
                raise SchemaMismatchError(
                    f"artifact {label} mismatch: bundle has {mine!r}, "
                    f"this process has {theirs!r}. Retrain against the current schema."
                )

    def manifest(self) -> dict[str, Any]:
        """JSON-serialisable description, for the API and for run logs."""
        return {
            "bundle_format": self.bundle_format,
            "schema_version": self.schema_version,
            "feature_order": list(self.feature_order),
            "grade_order": list(self.grade_order),
            "n_features": len(self.feature_order),
            "n_estimators": self.metadata.n_estimators,
            "n_samples": self.metadata.n_samples,
            "oob_score": self.metadata.oob_score,
            "created_at": self.metadata.created_at,
            "created_by": self.metadata.created_by,
            "checksum": checksum_of(self.grader),
        }


def _current_context() -> dict[str, Any]:
    schema = load_schema()
    return {
        "schema_version": schema.version,
        "feature_order": list(schema.names),
        "grade_order": [str(grade) for grade in GRADE_ORDER],
    }


def checksum_of(grader: QualityGrader) -> str:
    """SHA-256 over the estimator's arrays, not its pickled bytes.

    Hashing the pickle would change with the scikit-learn version, so a file
    could verify as corrupt simply for being moved between environments. Hashing
    the fitted parameters keeps the check meaningful.
    """
    digest = hashlib.sha256()
    estimator = grader.estimator
    digest.update(str(estimator.get_params()).encode())
    for tree in estimator.estimators_:
        digest.update(np.ascontiguousarray(tree.tree_.value).tobytes())
    return digest.hexdigest()


def feature_summary(X: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:  # noqa: N803
    """Min, max and mean per column, for detecting distribution shift."""
    matrix = np.asarray(X, dtype=np.float64)
    summary: dict[str, float] = {}
    for index, name in enumerate(names):
        column = matrix[:, index]
        summary[f"{name}.min"] = float(column.min())
        summary[f"{name}.max"] = float(column.max())
        summary[f"{name}.mean"] = float(column.mean())
    return summary


def build_bundle(
    grader: QualityGrader,
    n_samples: int,
    X: np.ndarray,  # noqa: N803
    metrics: Mapping[str, Any] | None = None,
    notes: str = "",
) -> ArtifactBundle:
    """Package a fitted grader with its training context."""
    if not grader.is_fitted:
        raise ValueError("cannot bundle an unfitted grader")
    schema = grader.schema
    report = grader.fit_report
    combined = dict(metrics or {})
    combined[FEATURE_SUMMARY_KEY] = feature_summary(X, schema.names)

    return ArtifactBundle(
        grader=grader,
        feature_order=schema.names,
        grade_order=tuple(str(grade) for grade in GRADE_ORDER),
        schema_version=schema.version,
        bundle_format=BUNDLE_FORMAT,
        metadata=TrainingMetadata(
            n_samples=n_samples,
            n_features=len(schema),
            n_estimators=grader.n_estimators,
            random_state=grader.random_state,
            oob_score=report.oob_score if report else None,
            grade_counts={
                str(k): v for k, v in (report.train_grade_counts.items() if report else {})
            },
            metrics=combined,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            created_by=f"{platform.node()} python{sys.version_info.major}.{sys.version_info.minor}",
            notes=notes,
        ),
    )


def default_path(name: str = "grader") -> Path:
    return ARTIFACT_DIR / f"{name}{BUNDLE_SUFFIX}"


def save(bundle: ArtifactBundle, path: str | Path | None = None) -> Path:
    """Write a bundle atomically.

    The payload is written to a temporary file in the destination directory and
    then renamed, so a crash mid-write leaves the previous artifact intact rather
    than a truncated one that still looks loadable.
    """
    target = default_path() if path is None else Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as handle:
        temp = Path(handle.name)
    try:
        joblib.dump(bundle, temp, compress=3)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)

    (target.with_suffix(".manifest.json")).write_text(
        json.dumps(bundle.manifest(), indent=2), encoding="utf-8"
    )
    return target


def load(path: str | Path | None = None, *, verify: bool = True) -> ArtifactBundle:
    """Read a bundle and check it against the current process.

    Args:
        path: Artifact path. Defaults to the current name in ``artifacts/``.
        verify: Run the schema and grade-order check. Leave enabled in
            production; disabling it is only useful for inspecting an old bundle
            that no longer matches.

    Returns:
        The loaded :class:`ArtifactBundle`.

    Raises:
        SchemaMismatchError: If verification fails.
    """
    target = default_path() if path is None else Path(path)
    try:
        bundle = joblib.load(target)
    except Exception as exc:
        # A manifest sidecar, a truncated download and a pickle from another
        # joblib version all land here. None of them is worth a stack trace.
        raise SchemaMismatchError(f"{target} could not be read as a model bundle: {exc}") from exc
    if not isinstance(bundle, ArtifactBundle):
        raise SchemaMismatchError(f"{target} is not an {ArtifactBundle.__name__}")
    if verify:
        bundle.verify_against()
    return bundle


def load_grader(path: str | Path | None = None, *, verify: bool = True) -> QualityGrader:
    """Convenience wrapper returning just the estimator."""
    return load(path, verify=verify).grader


def describe_quality(grade: str) -> str:
    """Human-facing meaning of a grade, for the frontend and for logs."""
    return {
        QualityGrade.GRADE_A: "premium, uniform surface, no visible defect",
        QualityGrade.GRADE_B: "marketable, minor cosmetic blemishing",
        QualityGrade.GRADE_C: "processing only, noticeable defect load",
        QualityGrade.REJECT: "not marketable, spoilage or severe mechanical damage",
    }.get(QualityGrade(str(grade)), "unknown grade")


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from agrigrade.model.evaluate import evaluate
    from agrigrade.model.mock import make_dataset

    built = make_dataset(n_per_grade=60, seed=11)
    train, test = built.split(0.25)
    grader = QualityGrader(train.schema, n_estimators=100, random_state=7)
    grader.fit(train.to_matrix(), train.to_labels())
    _grades, probabilities = grader.predict(test.to_matrix())
    report = evaluate(test.to_labels(), probabilities)

    packaged = build_bundle(
        grader,
        n_samples=len(train),
        X=train.to_matrix(),
        metrics={"accuracy": report.accuracy, "macro_f1": report.macro_f1},
        notes="mock-data baseline, not trained on real captures",
    )
    print(json.dumps(packaged.manifest(), indent=2)[:700])
    print()
    for grade in GRADE_ORDER:
        print(f"  {grade!s:<9} {describe_quality(grade)}")
