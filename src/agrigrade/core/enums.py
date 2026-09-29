"""Closed vocabularies shared by every workstream.

These enums are the join point between the three branches: ``feature-extraction``
produces values keyed by these classes, ``model`` consumes them, and ``frontend``
renders them. Adding a member here is a cross-branch change, so treat this file
as an API surface and bump ``schema_version`` in :mod:`agrigrade.core.config`
whenever a breaking edit lands.
"""

from __future__ import annotations

from enum import StrEnum


class ProduceFamily(StrEnum):
    """Colour family that determines which feature recipe is used."""

    RED_SMOOTH = "red_smooth"
    YELLOW_GREEN = "yellow_green"
    BROWN_TEXTURED = "brown_textured"
    UNKNOWN = "unknown"


class ProduceClass(StrEnum):
    """Produce class as predicted by the segmentation stage."""

    APPLE = "apple"
    TOMATO = "tomato"
    RED_GRAPE = "red_grape"
    BANANA = "banana"
    MANGO = "mango"
    CITRUS = "citrus"
    PAPAYA = "papaya"
    KIWI = "kiwi"
    CHIKOO = "chikoo"
    BROWN_PEAR = "brown_pear"
    UNKNOWN = "unknown"


class QualityGrade(StrEnum):
    """Market quality grades emitted by the classifier."""

    GRADE_A = "Grade A"
    GRADE_B = "Grade B"
    GRADE_C = "Grade C"
    REJECT = "Reject"


#: Canonical, order-sensitive grade ladder. The model is trained against exactly
#: this list, so the position of each member is part of the contract.
GRADE_ORDER: tuple[QualityGrade, ...] = (
    QualityGrade.GRADE_A,
    QualityGrade.GRADE_B,
    QualityGrade.GRADE_C,
    QualityGrade.REJECT,
)


#: Routing table from README section 3B: class -> colour family -> recipe.
CLASS_TO_FAMILY: dict[ProduceClass, ProduceFamily] = {
    ProduceClass.APPLE: ProduceFamily.RED_SMOOTH,
    ProduceClass.TOMATO: ProduceFamily.RED_SMOOTH,
    ProduceClass.RED_GRAPE: ProduceFamily.RED_SMOOTH,
    ProduceClass.BANANA: ProduceFamily.YELLOW_GREEN,
    ProduceClass.MANGO: ProduceFamily.YELLOW_GREEN,
    ProduceClass.CITRUS: ProduceFamily.YELLOW_GREEN,
    ProduceClass.PAPAYA: ProduceFamily.YELLOW_GREEN,
    ProduceClass.KIWI: ProduceFamily.BROWN_TEXTURED,
    ProduceClass.CHIKOO: ProduceFamily.BROWN_TEXTURED,
    ProduceClass.BROWN_PEAR: ProduceFamily.BROWN_TEXTURED,
    ProduceClass.UNKNOWN: ProduceFamily.UNKNOWN,
}


def family_for(produce_class: ProduceClass | str) -> ProduceFamily:
    """Return the colour family for ``produce_class``.

    Unrecognised values degrade to :attr:`ProduceFamily.UNKNOWN` rather than
    raising, so a segmentation model trained on a new class can be rolled out
    before the feature router learns about it.
    """
    try:
        produce_class = ProduceClass(str(produce_class).strip().lower())
    except ValueError:
        return ProduceFamily.UNKNOWN
    return CLASS_TO_FAMILY[produce_class]
