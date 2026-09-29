# `agrigrade.features` — owned by branch `feature-extraction`

Class-conditioned feature extraction. No file in this directory may be edited
from the `model` or `frontend` branches.

## Planned modules

| Module | Responsibility |
| --- | --- |
| `spectral.py` | VARI, NDTI, YI, ExB (README §3A). |
| `color.py` | sRGB → L\*a\*b\*, per-channel means and variance over the foreground mask. |
| `texture.py` | GLCM homogeneity / contrast / energy, LBP histogram. |
| `geometry.py` | Reference-marker calibration, equivalent diameter, true area, volume (README §3C). |
| `router.py` | Maps `ProduceClass` → family recipe (README §3B). |
| `pipeline.py` | `extract_produce_features(...)` → validated `FeatureVector`. |

## Boundary rules

- Import vocabularies from `agrigrade.core.enums`; never redefine a grade or
  produce class locally.
- Emit exactly the feature names declared in `agrigrade.core.feature_schema`
  and no others. The model is positional, so an extra or renamed feature is a
  silent scoring bug.
- The output slot is family-agnostic. A Kiwi and an Apple both populate
  `spectral_primary`; what the number *means* is documented per family rather
  than encoded in the column name.
- Heavy dependencies (`opencv-python-headless`, `scikit-image`) belong to the
  `features` extra only.

## Tests

Contract tests live in `tests/contract/` marked `@pytest.mark.features` and
must not require a GPU or a trained model.
