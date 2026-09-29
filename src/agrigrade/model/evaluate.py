"""Classification metrics and calibration for the grader.

Accuracy alone is not a report for a four-class ordinal problem. Grade A and
Grade C being swapped is a worse failure than Grade C and Reject, and a plain
accuracy cannot tell the two apart. So this module reports:

* Per-class precision, recall and F1, plus macro averages. Macro-F1 exposes a
  collapsed minority grade that an accuracy would hide behind the majority.
* A confusion matrix, with off-diagonal mass reported as a distance-weighted
  penalty, so adjacent-grade slips and Grade A to Reject failures do not score
  the same.
* Probability quality: log loss, Brier score, expected calibration error, and a
  reliability table. A forest's softmax is systematically overconfident on a
  dataset this size, and a reported 0.8 that is really 0.65 is worse for a
  grading instrument than a lower number that means what it says.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss

from agrigrade.core.enums import GRADE_ORDER, QualityGrade

#: Penalty applied per grade step, used to weight the confusion matrix.
ADJACENT_PENALTY = 1.0

#: Bin edges for the reliability table.
N_BINS = 10


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Metrics for one evaluation set."""

    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[QualityGrade, dict[str, float]]
    confusion: np.ndarray
    off_diagonal_rate: float
    ordinal_penalty: float
    log_loss: float
    brier: float
    expected_calibration_error: float
    reliability: tuple[tuple[float, float, int], ...]

    def summary(self) -> str:
        lines = [
            f"{self.n_samples} samples   accuracy {self.accuracy:.4f}   "
            f"macro-F1 {self.macro_f1:.4f}",
            f"ordinal penalty {self.ordinal_penalty:.4f}   "
            f"off-diagonal {self.off_diagonal_rate:.1%}   "
            f"log loss {self.log_loss:.4f}   ECE {self.expected_calibration_error:.4f}",
            "",
            f"{'grade':<10} {'prec':>7} {'rec':>7} {'f1':>7} {'n':>6}",
        ]
        for grade in GRADE_ORDER:
            row = self.per_class[grade]
            lines.append(
                f"{grade!s:<10} {row['precision']:7.4f} {row['recall']:7.4f} "
                f"{row['f1']:7.4f} {int(row['support']):6d}"
            )
        return "\n".join(lines)

    def confusion_table(self) -> str:
        """Render the confusion matrix with the row as truth."""
        width = max(len(str(grade)) for grade in GRADE_ORDER)
        header = " " * (width + 2) + "".join(f"{g!s:>9}" for g in GRADE_ORDER)
        lines = [header]
        for truth in GRADE_ORDER:
            row = self.confusion[GRADE_ORDER.index(truth)]
            cells = "".join(f"{int(v):>8}" for v in row)
            lines.append(f"{truth!s:<{width}}  {cells}")
        return "\n".join(lines)

    def reliability_table(self) -> str:
        """Render predicted-probability bins against observed frequency."""
        lines = [f"{'bin':>10} {'n':>6} {'mean conf':>10} {'accuracy':>9} {'gap':>7}"]
        for low, accuracy, count in self.reliability:
            mean_conf = low + 0.5 / N_BINS
            lines.append(
                f"{f'[{low:.1f},{low + 1 / N_BINS:.1f})':>10} {count:6d} "
                f"{mean_conf:10.3f} {accuracy:9.3f} {accuracy - mean_conf:+7.3f}"
            )
        return "\n".join(lines)


def _per_class(
    truth: np.ndarray,
    predicted: np.ndarray,
    n_classes: int,
) -> tuple[dict[QualityGrade, dict[str, float]], np.ndarray]:
    """Precision, recall, F1 and support per class index."""
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)

    per_class: dict[QualityGrade, dict[str, float]] = {}
    for index, grade in enumerate(GRADE_ORDER):
        true_positive = float(confusion[index, index])
        predicted_total = float(confusion[:, index].sum())
        actual_total = float(confusion[index, :].sum())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[grade] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total,
        }
    return per_class, confusion


def _ordinal_penalty(confusion: np.ndarray) -> tuple[float, float]:
    """Distance-weighted off-diagonal mass, and the plain off-diagonal rate.

    An error one grade apart is an ordinary borderline case. An error from Grade A
    to Reject is a mislabelled fruit that would have been sold at a premium, and
    the report should not let the two blur together.
    """
    n = len(GRADE_ORDER)
    total = float(confusion.sum()) or 1.0
    off_diagonal = 0.0
    weighted = 0.0
    for truth in range(n):
        for predicted in range(n):
            if truth == predicted:
                continue
            count = float(confusion[truth, predicted])
            off_diagonal += count
            weighted += count * abs(truth - predicted) * ADJACENT_PENALTY
    return off_diagonal / total, weighted / total


def _brier(probabilities: np.ndarray, truth_index: np.ndarray) -> float:
    """Multiclass Brier score, the mean squared error against a one-hot target."""
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(truth_index)), truth_index] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def _reliability(
    probabilities: np.ndarray,
    truth_index: np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[tuple[float, float, int], ...]:
    """Bin by winning probability, report observed accuracy per bin."""
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == truth_index).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[tuple[float, float, int]] = []
    for index in range(n_bins):
        low, high = float(edges[index]), float(edges[index + 1])
        if index == n_bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else float("nan")
        rows.append((low, accuracy, count))
    return tuple(rows)


def evaluate(
    truth: Sequence[QualityGrade | str],
    probabilities: np.ndarray,
) -> ClassificationReport:
    """Score predictions against truth.

    Args:
        truth: True grades.
        probabilities: ``(n_samples, len(GRADE_ORDER))`` matrix in grade order.

    Returns:
        A :class:`ClassificationReport`.
    """
    n_classes = len(GRADE_ORDER)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != n_classes:
        raise ValueError(f"probabilities must be (n, {n_classes}), got {scores.shape}")
    truth_index = np.array([GRADE_ORDER.index(QualityGrade(str(g))) for g in truth], dtype=np.int64)
    if scores.shape[0] != truth_index.shape[0]:
        raise ValueError(
            f"got {scores.shape[0]} probability rows for {truth_index.shape[0]} labels"
        )

    predicted = scores.argmax(axis=1)
    correct = predicted == truth_index
    per_class, confusion = _per_class(truth_index, predicted, n_classes)
    off_diagonal, weighted = _ordinal_penalty(confusion)

    reliability = _reliability(scores, truth_index)
    populated = [(low, acc) for low, acc, count in reliability if count]
    ece = (
        float(
            sum(
                count / len(truth_index) * abs(acc - (low + 0.5 / N_BINS))
                for low, acc, count in reliability
                if count
            )
        )
        if populated
        else float("nan")
    )

    return ClassificationReport(
        n_samples=int(scores.shape[0]),
        accuracy=float(correct.mean()),
        macro_precision=float(np.mean([row["precision"] for row in per_class.values()])),
        macro_recall=float(np.mean([row["recall"] for row in per_class.values()])),
        macro_f1=float(np.mean([row["f1"] for row in per_class.values()])),
        per_class=per_class,
        confusion=confusion,
        off_diagonal_rate=off_diagonal,
        ordinal_penalty=weighted,
        log_loss=float(log_loss(truth_index, scores, labels=list(range(n_classes)))),
        brier=_brier(scores, truth_index),
        expected_calibration_error=ece,
        reliability=reliability,
    )


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from agrigrade.model.confidence import build_report
    from agrigrade.model.mock import make_dataset
    from agrigrade.model.rf import QualityGrader

    built = make_dataset(n_per_grade=100, seed=21)
    train, test = built.split(0.25)
    grader = QualityGrader(train.schema, n_estimators=100, random_state=0)
    grader.fit(train.to_matrix(), train.to_labels())
    _grades, probabilities = grader.predict(test.to_matrix())

    report = evaluate(test.to_labels(), probabilities)
    print(report.summary())
    print()
    print(report.confusion_table())
    print()
    print("calibration, predicted probability against observed accuracy")
    print(report.reliability_table())

    confident = [i for i, p in enumerate(probabilities) if build_report(p).is_confident]
    if confident:
        subset = evaluate([test.to_labels()[i] for i in confident], probabilities[confident])
        print()
        print(
            f"on the {len(confident)} samples the confidence gate admits: "
            f"accuracy {subset.accuracy:.4f} vs {report.accuracy:.4f} overall"
        )
