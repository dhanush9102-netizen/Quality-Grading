# `agrigrade.model` — owned by branch `model`

Random Forest quality classifier, confidence scoring and XAI. No file in this
directory may be edited from the `feature-extraction` or `frontend` branches.

## Planned modules

| Module | Responsibility |
| --- | --- |
| `schema.py` | Loads `agrigrade.core.feature_schema` and validates incoming vectors. |
| `rf.py` | `RandomForestClassifier` wrapper, N = 100 trees, 4-class output (README §4A). |
| `confidence.py` | Soft-voting average across the ensemble, scaled to a percentage. |
| `xai.py` | Feature-importance breakdown for the top-k drivers of a decision. |
| `train.py` | CLI entry point: dataset loading, fit, evaluation, export. |
| `artifacts/` | Serialized models. Git-ignored; tracked only as `.gitkeep`. |

## Boundary rules

- Consume features by name from `agrigrade.core.feature_schema`. Do not
  re-derive indices inside this package.
- Reject a vector that does not match the schema exactly — raise
  `SchemaMismatchError` rather than padding or truncating.
- Class order is `GRADE_ORDER` from `agrigrade.core.enums`. A model whose
  `classes_` differ is not loadable.
- `predict` must be reachable with a cold-started artifact loader, since the API
  process and the mobile client both load independently.

## Tests

Contract tests live in `tests/contract/` marked `@pytest.mark.model` and must
run against a tiny fixture model so they stay fast.
