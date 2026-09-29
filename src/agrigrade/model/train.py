"""Training entry point.

    python -m agrigrade.model.train --samples-per-grade 200 --save

Generates data, fits the ensemble, reports metrics, writes an artifact. The
default data source is the synthetic generator, so the command runs on a clean
checkout with no dataset present. ``--from-npz`` points it at real captures
instead, and that is the only way the reported numbers mean anything about
produce.

Deliberate refusals, each of which exists because the alternative produces a
number that looks fine and is not:

* ``--min-accuracy`` gates the run. A model below the bar is reported and not
  saved by default, so a bad fit cannot become the artifact a device loads.
* ``--min-oob`` gates on the out-of-bag score, which catches a model that
  memorised a small dataset.
* Saving without a held-out evaluation is refused, since accuracy on the
  training set is not evidence of anything.
* When the data is synthetic, the artifact is stamped as such, so nobody later
  mistakes a mock baseline for a trained-on-produce model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agrigrade.core.enums import GRADE_ORDER, ProduceFamily, QualityGrade
from agrigrade.core.errors import AgriGradeError
from agrigrade.model import artifacts
from agrigrade.model.confidence import build_report
from agrigrade.model.evaluate import ClassificationReport, evaluate
from agrigrade.model.mock import make_dataset
from agrigrade.model.rf import DEFAULT_N_ESTIMATORS, FitReport, QualityGrader
from agrigrade.model.schema import load_schema
from agrigrade.model.xai import global_importance, render_global

#: Refuse to save a model that cannot beat majority class by a usable margin.
DEFAULT_MIN_ACCURACY = 0.55

#: Refuse to save a model whose out-of-bag score suggests memorisation.
DEFAULT_MIN_OOB = 0.45


@dataclass(frozen=True, slots=True)
class Split:
    """Training and held-out matrices with their labels."""

    X_train: np.ndarray
    y_train: list[QualityGrade]
    X_test: np.ndarray
    y_test: list[QualityGrade]
    synthetic: bool
    n_per_grade: int
    families: tuple[ProduceFamily, ...]

    def summary(self) -> str:
        source = "synthetic" if self.synthetic else "capture file"
        return (
            f"{source}: {len(self.y_train)} train / {len(self.y_test)} held-out, "
            f"{len(self.families)} families"
        )


def build_split(args: argparse.Namespace) -> Split:
    """Assemble train and held-out data from the requested source."""
    schema = load_schema()
    if args.from_npz:
        return _split_from_npz(Path(args.from_npz), schema.version, args.test_fraction)

    built = make_dataset(
        n_per_grade=args.samples_per_grade,
        seed=args.seed,
        noise=args.noise,
    )
    train, test = built.split(args.test_fraction)
    return Split(
        X_train=train.to_matrix(),
        y_train=train.to_labels(),
        X_test=test.to_matrix(),
        y_test=test.to_labels(),
        synthetic=True,
        n_per_grade=args.samples_per_grade,
        families=train.families,
    )


def _split_from_npz(path: Path, schema_version: int, test_fraction: float) -> Split:
    """Load ``X``, ``y`` and optionally ``family`` from a capture dump.

    The dump is an ``.npz`` with ``X`` shaped ``(n, n_features)`` in schema
    order, ``y`` holding grade strings, and optionally ``family``. The schema
    version is recorded in the file and checked here, so a dump captured before a
    schema change is rejected instead of scored against the wrong columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"capture dump not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        recorded = int(data["schema_version"]) if "schema_version" in data else schema_version
        if recorded != schema_version:
            raise AgriGradeError(
                f"{path} was captured against schema v{recorded}, "
                f"this process is v{schema_version}. Re-extract features, "
                "do not re-align the columns by hand."
            )
        matrix = np.asarray(data["X"], dtype=np.float64)
        y = [QualityGrade(str(value)) for value in data["y"]]
        families = (
            tuple(ProduceFamily(str(v)) for v in np.unique(data["family"]))
            if "family" in data
            else ()
        )

    if matrix.ndim != 2:
        raise AgriGradeError(f"X must be 2-dimensional, got shape {matrix.shape}")
    if matrix.shape[0] != len(y):
        raise AgriGradeError(f"X has {matrix.shape[0]} rows but y has {len(y)} labels")
    if not np.isfinite(matrix).all():
        raise AgriGradeError("X contains non-finite values")

    # Stratified split by index, matching MockDataset.split.
    per_grade: dict[QualityGrade, list[int]] = {grade: [] for grade in GRADE_ORDER}
    for index, grade in enumerate(y):
        per_grade[grade].append(index)
    test_index: list[int] = []
    train_index: list[int] = []
    for grade in GRADE_ORDER:
        group = per_grade[grade]
        cut = max(1, round(len(group) * test_fraction))
        test_index.extend(group[:cut])
        train_index.extend(group[cut:])

    train_rows = np.array(sorted(train_index), dtype=np.int64)
    test_rows = np.array(sorted(test_index), dtype=np.int64)
    return Split(
        X_train=matrix[train_rows],
        y_train=[y[i] for i in train_rows],
        X_test=matrix[test_rows],
        y_test=[y[i] for i in test_rows],
        synthetic=False,
        n_per_grade=0,
        families=families,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agrigrade.model.train",
        description="Fit the AgriGrade Random Forest quality classifier.",
    )
    source = parser.add_argument_group("data")
    source.add_argument(
        "--from-npz",
        metavar="PATH",
        help="Capture dump with X, y and optional family. Omit to use synthetic data.",
    )
    source.add_argument("--samples-per-grade", type=int, default=150)
    source.add_argument("--test-fraction", type=float, default=0.25)
    source.add_argument("--seed", type=int, default=7)
    source.add_argument(
        "--noise",
        type=float,
        default=0.22,
        help="Extra generator noise, synthetic source only. 0.0 is a clean signal.",
    )
    source.add_argument(
        "--families",
        nargs="*",
        default=None,
        choices=[str(f) for f in ProduceFamily],
        help="Restrict the synthetic generator to these families.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--trees", type=int, default=DEFAULT_N_ESTIMATORS)
    model.add_argument("--max-depth", type=int, default=None)
    model.add_argument("--min-samples-leaf", type=int, default=1)

    gate = parser.add_argument_group("acceptance gates")
    gate.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    gate.add_argument("--min-oob", type=float, default=DEFAULT_MIN_OOB)
    gate.add_argument(
        "--force",
        action="store_true",
        help="Save even if a gate fails. The manifest still records the failure.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--save", action="store_true", help="Write a bundle to artifacts/.")
    output.add_argument("--name", default="grader", help="Artifact name, without suffix.")
    output.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        split = build_split(args)
    except (AgriGradeError, FileNotFoundError, ValueError) as exc:
        print(f"data: {exc}", file=sys.stderr)
        return 2

    if split.synthetic and args.families:
        print("note: --families is ignored unless --from-npz is given", file=sys.stderr)

    schema = load_schema()
    grader = QualityGrader(
        schema,
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )
    fit_report = grader.fit(split.X_train, split.y_train)

    _grades, probabilities = grader.predict(split.X_test)
    report = evaluate(split.y_test, probabilities)

    failures: list[str] = []
    if report.accuracy < args.min_accuracy:
        failures.append(f"accuracy {report.accuracy:.4f} below {args.min_accuracy:.4f}")
    if fit_report.oob_score is not None and fit_report.oob_score < args.min_oob:
        failures.append(f"oob {fit_report.oob_score:.4f} below {args.min_oob:.4f}")

    admitted = [build_report(row).is_confident for row in probabilities]
    confident_accuracy = (
        float(
            np.mean(
                [g == t for g, t, ok in zip(_grades, split.y_test, admitted, strict=True) if ok]
            )
        )
        if any(admitted)
        else float("nan")
    )

    payload = {
        "data": split.summary(),
        "fit": {
            "n_samples": fit_report.n_samples,
            "n_features": fit_report.n_features,
            "oob_score": fit_report.oob_score,
            "grade_counts": {str(k): v for k, v in fit_report.train_grade_counts.items()},
        },
        "held_out": {
            "n_samples": report.n_samples,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "ordinal_penalty": report.ordinal_penalty,
            "log_loss": report.log_loss,
            "expected_calibration_error": report.expected_calibration_error,
            "accuracy_when_confident": confident_accuracy,
            "confident_fraction": float(np.mean(admitted)),
        },
        "gates": {"failed": failures, "forced": bool(args.force)},
        "top_features": [row.as_dict() for row in global_importance(grader)[:8]],
        "synthetic_data": split.synthetic,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_report(payload, report, fit_report, grader, split)

    if failures and not args.force:
        print("\ngates failed, nothing written:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("pass --force to save anyway", file=sys.stderr)
        return 1

    if args.save:
        bundle = artifacts.build_bundle(
            grader,
            n_samples=fit_report.n_samples,
            X=split.X_train,
            metrics={
                "accuracy": report.accuracy,
                "macro_f1": report.macro_f1,
                "log_loss": report.log_loss,
                "expected_calibration_error": report.expected_calibration_error,
            },
            notes=(
                "MOCK BASELINE - synthetic data, not trained on real produce"
                if split.synthetic
                else "trained on captured frames"
            )
            + (f"; gates failed but --force was passed: {failures}" if failures else ""),
        )
        path = artifacts.save(bundle, artifacts.default_path(args.name))
        print(f"\nsaved {path.name} ({path.stat().st_size / 1024:.0f} KiB)")
        print(f"  manifest {path.with_suffix('.manifest.json').name}")
        if bundle.metadata.notes.startswith("MOCK"):
            print("  note: this bundle is stamped as a mock baseline")

    return 0


def _print_report(
    payload: dict[str, Any],
    report: ClassificationReport,
    fit_report: FitReport,
    grader: QualityGrader,
    split: Split,
) -> None:
    held = payload["held_out"]
    print(f"data    {payload['data']}")
    print(f"fit     {fit_report.summary()}")
    print()
    print(f"held-out evaluation, {report.n_samples} samples")
    print(report.summary())
    print()
    print("confusion, rows are truth")
    print(report.confusion_table())
    print()
    print(
        f"confidence gate admits {held['confident_fraction']:.0%} of samples "
        f"at {held['accuracy_when_confident']:.4f} accuracy "
        f"against {held['accuracy']:.4f} overall"
    )
    print()
    print("global feature importance")
    print(render_global(global_importance(grader), limit=8))
    if split.synthetic:
        print()
        print("NOTE: synthetic data. These numbers describe the generator, not produce.")


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    raise SystemExit(main())
