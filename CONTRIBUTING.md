# Contributing

## Branches

Three long-lived workstream branches, all cut from `main`:

| Branch | Owns |
| --- | --- |
| `feature-extraction` | `src/agrigrade/features/`, `@pytest.mark.features` tests |
| `model` | `src/agrigrade/model/`, `@pytest.mark.model` tests |
| `frontend` | `frontend/`, frontend tests |

`main` owns everything else: `src/agrigrade/core/`, `src/agrigrade/segmentation/`,
`src/agrigrade/api/`, `tests/integration/`, `docs/`, `scripts/`, packaging and CI.

## Rules

1. **Stay in your lane.** A workstream branch edits only its own directory. If
   you need something from another branch, it goes in `core/` via a PR to
   `main` first.
2. **Never import across branches.** Not `agrigrade.features` from `model`, not
   `frontend` from Python, not a vendored copy of another branch's file. Shared
   vocabularies and types live in `core/`.
3. **Merge into your branch, not into `main`.** When `main` needs your work,
   open a PR from your branch to `main`. Reviewers check the contract, not the
   internals.
4. **`core/` changes are a separate PR to `main`.** Schema additions need a
   version bump; renames and reorderings need a retrained model.
5. **Keep the extras honest.** New dependencies go in the extra that matches the
   workstream. `core` stays dependency-free.
6. **`main` must boot without the workstreams merged.** The API imports them
   lazily and answers 501 when one is absent.

## Definition of done

- `ruff check .` and `ruff format --check .` clean.
- `mypy src` clean.
- `pytest` green, including the contract tests your branch owns.
- New behaviour documented in the `README.md` of the directory you touched.
- Model artifacts are not committed; re-training is a documented command.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[features,model,api,dev]"

ruff check . && ruff format --check .
mypy src
pytest
```

Frontend, on the `frontend` branch:

```bash
cd frontend
npm install
npm run dev
```
