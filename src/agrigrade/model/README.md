# `agrigrade.model` — owned by branch `model`

Random Forest quality classification, confidence scoring, XAI and artifact
persistence. Stage 2 of the pipeline: a feature vector in, a market grade out.

## Status

Runs end to end against **synthetic data**. No model here has been trained on
real produce, and every artifact written is stamped `MOCK BASELINE` in its
manifest notes. The metrics in the demo describe the generator, not fruit.

```bash
PYTHONPATH=src python -m agrigrade.model                      # 7-stage walkthrough
PYTHONPATH=src python -m agrigrade.model.train --save         # fit and persist
PYTHONPATH=src python -m agrigrade.model.train --from-npz cap.npz
PYTHONPATH=src python -m agrigrade.model.<module>             # one stage at a time
```

## Layout

| Module | Responsibility |
| --- | --- |
| `schema.py` | 19 slots in vector order, strict validation, per-family slot labels |
| `mock.py` | Labelled family-conditioned synthetic features |
| `rf.py` | `QualityGrader`: fit, predict, per-tree votes, training baseline |
| `confidence.py` | Soft voting per spec 4A, margin gate, abstention |
| `xai.py` | Global importances, per-sample occlusion attribution |
| `evaluate.py` | Per-class metrics, ordinal penalty, reliability table |
| `artifacts.py` | Versioned bundle save/load, checksum, drift |
| `train.py` | Training CLI with acceptance gates |
| `demo.py` | Staged end-to-end walkthrough |
| `artifacts/` | Git-ignored. Only `.gitkeep` and a local `.gitignore` are tracked. |

## The one change this branch needs from `main`

**`agrigrade.core.feature_schema` must carry the three `family_*` one-hot
slots.** This is not optional.

The spec routes a class to a family recipe, then writes that recipe's result
into a fixed set of slots. `spectral_primary` is NDTI for red produce, YI for
yellow-green, and ExB for brown. NDTI and YI rise with quality; ExB rises with
mould. A single pooled forest given only that value scored **0.43** against a
0.25 chance baseline, and 0.60 even restricted to one family at a time, because
the sign of the relationship depends on which recipe filled the column.

Family is not recoverable from the colour statistics either — `mean_a` and
`mean_l` overlap across families. So the class conditioning has to enter the
model as explicit inputs. With the one-hots, the same fit reaches **0.81** on
held-out data.

Until `core.feature_schema` merges, `schema.py` declares the slots locally and
`load_schema()` prefers the core version when it exists. When it lands, delete
the local `FEATURE_SPECS` and the fallback.

## Contract with the other branches

- **Consumes** feature mappings or matrices from `agrigrade.features`. Never
  imports it. Takes plain floats, so an ONNX or TFLite graph can replace the
  feature stage without touching this package.
- **Depends on** `agrigrade.core` only, for the vocabularies and errors.
- **Provides** to `agrigrade.api` and `frontend`: `QualityGrader.predict_proba`
  (columns in `GRADE_ORDER`), `build_report`, `Explanation.as_dict()`,
  `ArtifactBundle.manifest()`.
- **Must not** import `agrigrade.features` or be imported by the frontend
  directly. The client goes through the API.

## Findings worth carrying forward

- **Confidence is systematically under-stated on this data.** Every reliability
  bin but the last sits 0.14 to 0.23 below its predicted probability, ECE 0.17.
  Do not assume the usual "forests are overconfident" complaint applies without
  measuring on real captures.
- **The entropy gate does not work; the margin gate does.** Swept on held-out
  data, a 0.60 entropy ceiling abstains on 67% of samples while 0.65 to 0.75
  selects *worse* than the ungated baseline — a threshold fitted to noise. A
  0.20 margin abstains on 27% and lifts accuracy from 0.81 to 0.88 on the rest.
  `DEFAULT_MAX_ENTROPY` is set loose on purpose.
- **Occlusion contributions are not additive.** They are measured one at a time
  against full context, so they overlap for correlated slots. The ranking is
  reliable; the sum is not, and `total_shift` reports distance from the baseline
  rather than a sum that would mean nothing. SHAP would fix the attribution but
  not the additivity.
- **Two bugs were found by exercising the code, not by reading it**: `drift()`
  read summary keys from the wrong level of the metrics dict and silently
  returned nothing, and a manifest sidecar passed to `load()` leaked a bare
  `KeyError`. Both are covered by the checks in `demo.py` stage 7.

## Before this ships

1. Train on real captures and replace the synthetic baseline.
2. Recalibrate the confidence gate; treat the current thresholds as arbitrary.
3. Re-tune the acceptance gates in `train.py`; the defaults are placeholders
   chosen to catch an obviously broken fit, not to encode a product bar.
4. Consider an explicit calibration step. A per-family model, or per-family
   thresholds, would also address the one-directional pooled confusion seen in
   the demo.
5. Add `skl2onnx` export and verify parity with the joblib bundle. The
   dependency is not installed here, so the edge path is untested.
