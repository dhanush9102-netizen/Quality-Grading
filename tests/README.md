# Tests

| Directory | Runs on | Markers |
| --- | --- | --- |
| `contract/` | every workstream branch | `features`, `model`, `frontend` |
| `integration/` | `main` only | `integration` |

- `contract/` holds the tests each branch owns for the boundaries it promises to
  honour — a vector that matches the schema, a router that picks the right
  family, a response body the client can parse.
- `integration/` exercises a full frame through segmentation → features → model →
  API. It needs all three workstreams merged, so it is skipped when a stage is
  absent instead of failing.

Fixtures live in `tests/conftest.py` and are created in-memory. No committed
image fixtures: raw captures are git-ignored, and CI must not depend on a
dataset that only exists on one machine.
