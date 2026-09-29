"""Random Forest quality classifier.

Wraps ``sklearn.ensemble.RandomForestClassifier`` and adds the three things the
spec cares about and scikit-learn does not give us:

* **Fixed grade order.** The estimator's ``classes_`` is whatever it saw in
  training. :meth:`QualityGrader.predict_proba` always returns columns in
  ``GRADE_ORDER``, reindexing a partial fit so a grade that never appeared in the
  data is a zero column rather than a shifted one.
* **Schema gating.** A vector that does not match the declared feature order is
  rejected outright, because the ensemble is positional and a silent reorder
  scores garbage.
* **Thread pinning.** A forest with N trees is embarrassingly parallel, but the
  edge client and a small container both prefer a bounded thread count.

The confidence and explanation layers live in :mod:`agrigrade.model.confidence`
and :mod:`agrigrade.model.xai`; this module is deliberately just the ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from agrigrade.core.enums import GRADE_ORDER, QualityGrade
from agrigrade.core.errors import ModelNotLoadedError
from agrigrade.model.schema import DEFAULT_SCHEMA, FeatureSchema

#: N = 100 trees, per spec section 4A.
DEFAULT_N_ESTIMATORS = 100

#: Columns of a ``predict_proba`` result, always in this order.
GRADE_INDEX: dict[QualityGrade, int] = {grade: i for i, grade in enumerate(GRADE_ORDER)}


@dataclass(frozen=True, slots=True)
class FitReport:
    """Summary returned by :meth:`QualityGrader.fit`."""

    n_samples: int
    n_features: int
    train_grade_counts: dict[QualityGrade, int]
    oob_score: float | None
    feature_order: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        lines = [
            f"fitted on {self.n_samples} samples x {self.n_features} features",
            "  train balance: "
            + ", ".join(f"{grade}={count}" for grade, count in self.train_grade_counts.items()),
        ]
        if self.oob_score is not None:
            lines.append(f"  out-of-bag accuracy: {self.oob_score:.4f}")
        return "\n".join(lines)


class QualityGrader:
    """Grade a feature vector into a market quality class.

    Args:
        schema: Feature order the ensemble is fitted against. Defaults to the
            branch-local declaration; production should pass the loaded core
            schema.
        n_estimators: Trees in the ensemble.
        max_depth: ``None`` grows trees to full depth, which is the right default
            for tabular data and the spec's 100-tree ensemble.
        n_jobs: Forest parallelism. ``-1`` uses every core.
        random_state: Seed, for reproducible training.
    """

    def __init__(
        self,
        schema: FeatureSchema = DEFAULT_SCHEMA,
        *,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: str | dict[QualityGrade, float] | None = "balanced_subsample",
        n_jobs: int = -1,
        random_state: int = 7,
        max_features: str | float | None = "sqrt",
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        self.schema = schema
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            bootstrap=True,
        )
        self._fitted = False
        self.fit_report: FitReport | None = None
        self._baseline: np.ndarray | None = None

    # --- training -------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,  # noqa: N803 - matches the scikit-learn convention
        y: Sequence[QualityGrade | str],
        *,
        oob: bool = True,
    ) -> FitReport:
        """Fit the ensemble.

        Args:
            X: ``(n_samples, n_features)`` matrix already in schema order.
            y: Grade labels. Strings are coerced through ``QualityGrade``.
            oob: Request an out-of-bag score. Requires ``bootstrap=True`` and
                enough samples per class for a fold to see every grade.

        Returns:
            A :class:`FitReport`.
        """
        matrix = self._check_matrix(X)
        labels = self._check_labels(y, len(matrix))

        # Out-of-bag scoring is only defined when every bootstrap sample can hold
        # at least one of each class; with few samples per class it raises
        # instead of returning a meaningless number.
        use_oob = oob and matrix.shape[0] >= 5 * len(GRADE_ORDER)
        self._estimator.set_params(oob_score=use_oob, bootstrap=True)
        self._estimator.fit(matrix, np.asarray([str(grade) for grade in labels], dtype=object))

        self._fitted = True
        self._baseline = matrix.mean(axis=0)
        counts = dict.fromkeys(GRADE_ORDER, 0)
        for grade in labels:
            counts[grade] += 1
        self.fit_report = FitReport(
            n_samples=matrix.shape[0],
            n_features=matrix.shape[1],
            train_grade_counts=counts,
            oob_score=float(self._estimator.oob_score_) if use_oob else None,
            feature_order=self.schema.names,
        )
        return self.fit_report

    # --- inference ------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray | Mapping[str, Any] | Sequence[Mapping[str, Any]] | Sequence[Sequence[float]],  # noqa: N803
    ) -> np.ndarray:
        """Class probabilities with columns in ``GRADE_ORDER``.

        Accepts a raw matrix, a single feature mapping, a sequence of mappings,
        or a sequence of positional vectors. Mappings are validated against the
        schema, so a mapping is the safe input: it fails loudly instead of
        misaligning.
        """
        matrix = self._as_matrix(X)
        raw = self._estimator.predict_proba(matrix)
        return self._reindex(raw)

    def predict(
        self,
        X: np.ndarray | Mapping[str, Any] | Sequence[Mapping[str, Any]] | Sequence[Sequence[float]],  # noqa: N803
    ) -> tuple[list[QualityGrade], np.ndarray]:
        """Return the argmax grade per row and the full probability matrix."""
        probabilities = self.predict_proba(X)
        grades = [GRADE_ORDER[int(i)] for i in probabilities.argmax(axis=1)]
        return grades, probabilities

    def tree_votes(
        self,
        X: np.ndarray | Mapping[str, Any] | Sequence[Mapping[str, Any]] | Sequence[Sequence[float]],  # noqa: N803
    ) -> np.ndarray:
        """Per-tree hard class indices, shape ``(n_trees, n_samples)``.

        Indices are over ``GRADE_ORDER``, not the estimator's own ``classes_``.
        Exposed so the soft-voting arithmetic in section 4A of the spec can be
        reproduced and checked against :meth:`predict_proba`, rather than
        assumed to be the same thing.
        """
        matrix = self._as_matrix(X)
        # Each tree predicts integer codes into the forest's shared classes_,
        # not grade names, so a code is translated twice: code -> the label the
        # forest was fitted with, then label -> GRADE_ORDER position.
        lookup = [GRADE_INDEX[QualityGrade(str(name))] for name in self._estimator.classes_]
        votes = np.empty((self.n_estimators, matrix.shape[0]), dtype=np.int64)
        for position, tree in enumerate(self._estimator.estimators_):
            votes[position] = [lookup[int(code)] for code in tree.predict(matrix)]
        return votes

    # --- properties -----------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def estimator(self) -> RandomForestClassifier:
        """The underlying scikit-learn estimator.

        Exposed for serialisation, checksum computation and ONNX export,
        which all need the fitted object rather than this wrapper.
        """
        self._require_fitted()
        return self._estimator

    @property
    def classes_(self) -> tuple[QualityGrade, ...]:
        """Grades the ensemble actually saw, ascending along the ladder."""
        self._require_fitted()
        seen = (QualityGrade(c) for c in self._estimator.classes_)
        return tuple(sorted(seen, key=GRADE_ORDER.index))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean decrease in impurity per feature, in schema order."""
        self._require_fitted()
        return np.asarray(self._estimator.feature_importances_, dtype=np.float64)

    def baseline(self) -> np.ndarray:
        """Per-feature mean of the training data, used as the XAI reference.

        Recorded at fit time because a baseline has to come from the data the
        model was trained on; a constant or zero vector would make an
        explanation uninterpretable.
        """
        self._require_fitted()
        assert self._baseline is not None
        return self._baseline.copy()

    # --- internals ------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ModelNotLoadedError("QualityGrader.fit has not been called")

    def _check_matrix(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - sklearn convention
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {matrix.shape}")
        if matrix.shape[1] != len(self.schema):
            raise ValueError(
                f"expected {len(self.schema)} features in schema order, got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("X contains non-finite values")
        return matrix

    def _as_matrix(
        self,
        X: np.ndarray | Mapping[str, Any] | Sequence[Mapping[str, Any]] | Sequence[Sequence[float]],  # noqa: N803
    ) -> np.ndarray:
        self._require_fitted()
        if isinstance(X, np.ndarray):
            return self._check_matrix(X)
        # A bare mapping is a single frame, which is the common case on the edge
        # client and in the API, so accept it rather than forcing a list wrapper.
        if isinstance(X, Mapping):
            return np.asarray([self.schema.to_vector(X)], dtype=np.float64)
        rows = list(X)
        if not rows:
            return np.empty((0, len(self.schema)), dtype=np.float64)
        if isinstance(rows[0], Mapping):
            mappings: list[Mapping[str, Any]] = [row for row in rows if isinstance(row, Mapping)]
            return np.asarray(self.schema.to_matrix(mappings), dtype=np.float64)
        return self._check_matrix(np.asarray(rows, dtype=np.float64))

    @staticmethod
    def _check_labels(y: Sequence[QualityGrade | str], n_samples: int) -> list[QualityGrade]:
        labels = [QualityGrade(str(grade)) for grade in y]
        if len(labels) != n_samples:
            raise ValueError(f"y has {len(labels)} entries but X has {n_samples} rows")
        return labels

    def _reindex(self, raw: np.ndarray) -> np.ndarray:
        """Expand the estimator's ``classes_`` into full ``GRADE_ORDER`` columns."""
        out = np.zeros((raw.shape[0], len(GRADE_ORDER)), dtype=np.float64)
        for position, name in enumerate(self._estimator.classes_):
            out[:, GRADE_INDEX[QualityGrade(name)]] = raw[:, position]
        return out


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from agrigrade.model.mock import describe, make_dataset

    built = make_dataset(n_per_grade=50, seed=11)
    train, test = built.split(0.25)
    grader = QualityGrader(train.schema, n_estimators=100, random_state=7)
    print(describe(train))
    print()
    print(grader.fit(train.to_matrix(), train.to_labels()).summary())
    print(f"classes seen: {[str(g) for g in grader.classes_]}")
    grades, probabilities = grader.predict(test.to_matrix())
    hits = sum(1 for g, truth in zip(grades, test.to_labels(), strict=True) if g == truth)
    print(f"test accuracy: {hits}/{len(test)} = {hits / len(test):.4f}")
