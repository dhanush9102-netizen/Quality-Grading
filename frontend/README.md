# `frontend` — owned by branch `frontend`

React + TypeScript + Vite web client. No file in this directory may be edited
from the `feature-extraction` or `model` branches.

## Scope

- Live camera capture via `getUserMedia`, with a still-capture fallback and an
  on-screen reference marker (ArUco tag or coin) for scale calibration.
- Upload / retake / calibration-guidance flow.
- Grade result view: grade, confidence, and the per-feature XAI breakdown
  returned by the API.
- Grade history for a session.

## Boundary rules

- The client knows nothing about OpenCV, Random Forest internals or pixel math.
  It consumes the JSON contract in `src/agrigrade/api/` and nothing else.
- Produce classes and grade strings are mirrored from
  `agrigrade.core.enums`. When a new value lands in that enum, update
  `src/types/grades.ts` in the same PR.
- Camera frames stay in the browser unless the user submits them. The on-device
  claim in the README means no frame is retained server-side by default.

## Planned layout

```
frontend/
  package.json          vite, react, typescript, vitest
  tsconfig.json
  vite.config.ts
  index.html
  .env.example          VITE_API_BASE_URL
  public/               PWA manifest, icons, ArUco reference marker
  src/
    api/                typed client for the FastAPI surface
    components/         CaptureView, CalibrationCard, ResultCard, BreakdownChart
    hooks/              useCamera, useGradeSession
    types/              grades.ts, api.ts
    App.tsx
    main.tsx
```

## Tests

`vitest` + Testing Library, colocated as `*.test.ts(x)`. Mock the API client;
no browser or camera is required to run the suite.
