"""Random Forest quality grading for AgriGrade AI.

This package owns stage 2 of the pipeline: it turns a feature vector produced by
``agrigrade.features`` into a market quality grade, a calibrated confidence
percentage and an explainable breakdown.

Layout:

===========================  ==============================================
``schema.py``                canonical feature order + validation
``mock.py``                  labelled synthetic feature data for development
``rf.py``                    ``RandomForestClassifier`` wrapper
``confidence.py``            soft-voting math and confidence reports
``xai.py``                   global importances and per-sample attribution
``evaluate.py``              classification metrics and calibration
``artifacts.py``             versioned model bundle persistence
``train.py``                 training CLI
``demo.py``                   staged end-to-end walkthrough
===========================  ==============================================

Only :mod:`agrigrade.core` is imported, and only for the shared contracts. This
package never imports ``agrigrade.features``: it receives plain mappings and
sequences of floats, so it stays importable on the edge client where the feature
stage is replaced by an ONNX or TFLite graph.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "GRADE_ORDER",
    "GradePrediction",
    "QualityGrader",
    "load_grader",
]

__version__ = "0.1.0"

from agrigrade.core.enums import GRADE_ORDER

# Re-exported at module scope so ``from agrigrade.model import QualityGrader``
# works without importing the whole public surface eagerly. The submodule
# imports below are deferred to first use to keep `import agrigrade.model` cheap
# on the mobile client.
_LAZY = {
    "QualityGrader": "agrigrade.model.rf",
    "GradePrediction": "agrigrade.model.rf",
    "load_grader": "agrigrade.model.artifacts",
}


def __getattr__(name: str) -> object:
    try:
        module_name = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
