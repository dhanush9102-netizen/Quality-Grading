"""Shared contract layer.

Every module in this package is a cross-branch dependency. Anything a
workstream branch needs from another workstream must be expressed here as a
type or a protocol, never as an import of that workstream's internals.

Authored so far:

- :mod:`agrigrade.core.enums`  - produce, family and grade vocabularies
- :mod:`agrigrade.core.errors` - exception hierarchy

Reserved, to be authored when the first consumer lands on ``main``:

- ``types.py``          - ``SegmentationResult``, ``FeatureVector``, ``GradeResult``
- ``feature_schema.py`` - canonical feature names, order and validation
- ``config.py``         - settings and schema versioning
"""

from agrigrade.core.enums import (
    CLASS_TO_FAMILY,
    GRADE_ORDER,
    ProduceClass,
    ProduceFamily,
    QualityGrade,
    family_for,
)
from agrigrade.core.errors import (
    AgriGradeError,
    CalibrationError,
    ContractError,
    FeatureExtractionError,
    ModelNotLoadedError,
    ProduceNotFoundError,
    SchemaMismatchError,
    WorkstreamNotImplementedError,
)

__all__ = [
    "AgriGradeError",
    "CLASS_TO_FAMILY",
    "CalibrationError",
    "ContractError",
    "FeatureExtractionError",
    "GRADE_ORDER",
    "ModelNotLoadedError",
    "ProduceClass",
    "ProduceFamily",
    "ProduceNotFoundError",
    "QualityGrade",
    "SchemaMismatchError",
    "WorkstreamNotImplementedError",
    "family_for",
]
