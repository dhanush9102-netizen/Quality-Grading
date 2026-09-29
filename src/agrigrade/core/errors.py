"""Exception hierarchy for the AgriGrade pipeline."""

from __future__ import annotations


class AgriGradeError(Exception):
    """Base class for every error raised by this package."""


class ContractError(AgriGradeError):
    """A value violates a shared contract in :mod:`agrigrade.core`."""


class ProduceNotFoundError(ContractError):
    """Segmentation returned no usable foreground mask."""


class CalibrationError(ContractError):
    """Reference-marker geometry is missing or implausible."""


class FeatureExtractionError(AgriGradeError):
    """A feature could not be computed for a valid mask."""


class SchemaMismatchError(ContractError):
    """Feature vector does not match the model's expected schema."""


class ModelNotLoadedError(AgriGradeError):
    """Inference was requested before a model artifact was loaded."""


class WorkstreamNotImplementedError(AgriGradeError):
    """A stage that lives on an unmerged feature branch was invoked.

    Raised instead of a bare ``ImportError`` so the API can answer 501 with a
    clear message while the ``feature-extraction`` / ``model`` branches are
    still in flight.
    """
