# CI/CD Pipeline

_Last updated: 2026-06-01_

GitHub Actions workflow: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).
Runs on every push and pull request to `develop` and `main`. Superseded runs on the
same ref are cancelled (`concurrency`), and the job uses least-privilege
`permissions: contents: read`.

## Jobs & gates

### `backend-tests` (Backend · pytest)
Runs on `ubuntu-latest` with service containers:
- **Postgres** — `pgvector/pgvector:pg16` (matches the app's real DB; plain postgres
  lacks the `vector` extension).
- **Redis** — `redis:7-alpine`.

Steps (all **blocking** unless noted):
| Step | Command | Notes |
|------|---------|-------|
| Install uv | `astral-sh/setup-uv@v4` | dependency cache keyed on `backend/uv.lock` |
| Set up Python | `uv python install 3.12` | |
| Install deps | `uv sync` | installs the `dev` group (pytest, black) |
| Lint (black) | `uv run black --check app/ tests/` | **blocking** |
| Run tests | `uv run pytest --tb=short -q` | **blocking** — 2657 tests |

`app` is importable in CI via `pythonpath = ["."]` in `backend/pyproject.toml` — the
`tests/` directory is not a package, so without this pytest puts `tests/` (not the
backend root) on `sys.path` and collection fails with `ModuleNotFoundError: app`.
(The Docker image masks this with `PYTHONPATH=/app`; CI sets none.)

### `frontend` (Frontend · build + test)
Runs on `ubuntu-latest`, Node 20, npm cache keyed on `ui/core-ui/package-lock.json`.

| Step | Command | Notes |
|------|---------|-------|
| Install deps | `npm ci --legacy-peer-deps` | `@angular/material` pins a slightly older `@angular/cdk` than the lockfile resolves |
| Lint | `npm run lint` | **blocking** on errors; ~190 `any`/`console` items are `warn`-level and don't fail |
| Build | `npm run build:ng -- --configuration development` | **blocking** — full type + template check |
| Test | `npm test -- --ci --runInBand` | **blocking** — 130 Jest tests |

The build uses the **development** configuration: the production bundle currently
exceeds the 3 MB `angular.json` budget (three.js + Angular Material), tracked as tech
debt in the [roadmap](../roadmap/command-deck-cognition-next-steps.md).

## Reproducing CI locally

Backend (mirrors CI — confirms `app` imports without `PYTHONPATH`), against the local
Docker Postgres/Redis:

```bash
docker compose exec -T core-backend bash -lc \
  "cd /app && unset PYTHONPATH && uv run pytest -q"
docker compose exec -T core-backend bash -lc \
  "cd /app && uv run black --check app/ tests/"
```

Frontend (from `ui/core-ui`):

```bash
npm ci --legacy-peer-deps
npm run lint
npm run build:ng -- --configuration development
npm test -- --ci
```

## Notes
- Linters are enforced: `black` (backend) and `ng lint` errors (frontend) fail the build.
  Reformat with `uv run black app/ tests/` and `npm run lint:fix` before pushing.
- `gh` is handy for inspecting runs (`gh run list`, `gh run view <id> --log-failed`);
  the public run/job metadata is also available via the GitHub REST API without auth.
