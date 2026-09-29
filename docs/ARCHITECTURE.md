# Architecture

## Workstream split

Three teams work in parallel on three branches, all cut from `main`. `main`
owns the shared spine and is the only place cross-branch contracts change.

```
                   main  (shared spine, stable contracts)
                     |
   +-----------------+-----------------+
   |                                   |
feature-extraction                   model
src/agrigrade/features/              src/agrigrade/model/
   |                                   |
   +-----------------+-----------------+
                     |
                 frontend
              frontend/
```

None of the three branches imports another branch's internals. Where they meet,
they meet in `src/agrigrade/core/`.

## Directory map

```
Quality-Grading/
├── README.md                     project specification (source of truth)
├── CONTRIBUTING.md               branch workflow, PR rules, ownership table
├── pyproject.toml                packaging + per-workstream dependency extras
├── .gitignore
├── docs/
│   └── ARCHITECTURE.md           this file
├── src/agrigrade/                python package, src layout
│   ├── core/                     SHARED CONTRACTS - owned by main
│   │   ├── enums.py                produce / family / grade vocabularies
│   │   ├── errors.py               exception hierarchy
│   │   ├── types.py                (reserved) SegmentationResult, FeatureVector, GradeResult
│   │   ├── feature_schema.py       (reserved) canonical feature names + order
│   │   └── config.py               (reserved) settings, schema version
│   ├── features/                 BRANCH feature-extraction
│   ├── model/                    BRANCH model
│   │   └── artifacts/              git-ignored model binaries
│   ├── segmentation/             SHARED - owned by main
│   └── api/                      SHARED - owned by main, consumed by frontend
│       └── routes/
├── frontend/                     BRANCH frontend  (React + TS + Vite)
├── tests/
│   ├── contract/                 per-workstream contract tests
│   └── integration/              cross-workstream, main only
├── scripts/                      shared dev scripts
└── notebooks/                    EDA, kept out of the package
```

## The two contracts that matter

**`core.enums`** — `ProduceClass`, `ProduceFamily`, `QualityGrade`, and the
`CLASS_TO_FAMILY` routing table. Segmentation produces these values, the feature
router consumes them, the model predicts them, the API serialises them and the
frontend renders them.

**`core.feature_schema`** (reserved) — the ordered list of feature names that
`features` emits and `model` consumes. The Random Forest is positional, so this
ordering is as much a part of the contract as the enum. Names may be added with
a schema version bump; renaming or reordering requires retraining.

## Data flow

1. Client sends a still frame plus an optional reference-marker diameter.
2. `segmentation` returns a foreground mask and a `ProduceClass`.
3. `core.enums.family_for` maps the class to a `ProduceFamily`.
4. `features` runs that family's recipe and returns a vector matching
   `core.feature_schema`.
5. `model` returns a `QualityGrade`, a confidence percentage and a feature
   importance breakdown.
6. `api` serialises the three together; `frontend` displays them.

## Dependency isolation

`core` depends on nothing. Each workstream declares its own extra so a checkout
of a single branch installs only what it needs:

| Extra | Contents | Branch |
| --- | --- | --- |
| `features` | numpy, opencv-python-headless, scikit-image | `feature-extraction` |
| `model` | numpy, scikit-learn, joblib, onnx, onnxruntime, shap | `model` |
| `api` | fastapi, uvicorn, python-multipart, pillow | `main` |
| `frontend` | httpx | `frontend` |
| `dev` | pytest, ruff, mypy | all |

## Constraints carried over from the spec

- Illumination invariance: spectral indices are ratios, so no darkroom hardware.
- Class conditioning exists to stop brown produce being scored as rotten red
  produce. The family recipe, not the model, is the defence.
- Edge deployment: inference must be exportable to TFLite/ONNX, so `core` stays
  dependency-free and no stage depends on a Python-only plotting stack.
