# `agrigrade.api` — shared, owned by `main`

FastAPI surface consumed by the `frontend` branch. Kept thin on purpose: it
validates input, delegates to the workstreams, and serialises the result. All
grading logic lives in `features` and `model`.

## Planned routes

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness plus which workstreams are available. |
| `GET` | `/api/v1/classes` | `ProduceClass` / `ProduceFamily` / grade vocabularies, so the client mirrors the enum rather than hardcoding it. |
| `POST` | `/api/v1/segment` | Image → mask overlay + predicted class. |
| `POST` | `/api/v1/grade` | Image + optional reference-marker diameter → `GradeResult`. |
| `GET` | `/api/v1/schema` | Feature names, order and units, for client-side display. |

## Boundary rules

- Import workstream symbols lazily inside the handler. `main` must import and
  boot before `feature-extraction` and `model` have merged, and a missing
  workstream returns 501 via `WorkstreamNotImplementedError` — never a 500 and
  never a startup crash.
- Serialise with a versioned envelope. Fields are added, never renamed; the
  frontend branch may lag behind a merge.
- Never log image bytes or reference measurements that could identify a farm's
  throughput.
