# `agrigrade.segmentation` — shared, owned by `main`

YOLOv8 instance segmentation adapter. Not one of the three workstreams: it is a
pre-existing dependency of `features` (needs the mask and the predicted class)
and of the API (needs the image encoded back to the client for overlays).

## Planned modules

| Module | Responsibility |
| --- | --- |
| `base.py` | `Segmenter` protocol and `SegmentationResult` construction. |
| `yolo.py` | Ultralytics YOLOv8-seg adapter, cached model handle. |
| `weights.py` | Weight resolution and cache directory. |

## Boundary rules

- A new segmentation strategy is added as a new adapter behind `Segmenter`; it
  is never imported directly by `features` or the API.
- If the detector starts emitting a produce class outside `ProduceClass`, the
  enum and `CLASS_TO_FAMILY` on `main` are updated first.
