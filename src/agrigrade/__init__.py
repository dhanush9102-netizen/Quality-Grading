"""AgriGrade AI: class-conditioned produce quality grading.

The package is split so that three workstreams can progress on separate
branches without touching each other's files:

``agrigrade.core``
    Shared contracts (vocabularies, error types, feature schema, settings).
    Owned by ``main``. Treat as an API surface.

``agrigrade.features``
    Spectral, texture and geometric feature extraction. Branch:
    ``feature-extraction``.

``agrigrade.model``
    Random Forest training and inference, confidence, XAI. Branch: ``model``.

``agrigrade.segmentation`` / ``agrigrade.api``
    Shared plumbing consumed by more than one workstream. Owned by ``main``.

The web client lives outside this package in ``frontend/``. Branch:
``frontend``.
"""

__version__ = "0.1.0"
