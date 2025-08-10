### CORE Backend (FastAPI)

Run the backend locally with uv and a virtual environment.

- Create and activate the `.venv` (Windows PowerShell):
  - Install deps and create venv in one go: `uv sync`
  - Or create venv explicitly: `uv venv`
  - Activate: `./.venv/Scripts/Activate.ps1`

- Install/update dependencies (inside the venv):
  - `uv sync` (preferred; uses `pyproject.toml` and `uv.lock`)

- Set required environment variables:
  - `OPENAI_API_KEY` (required)
  - Optional DB settings (defaults match docker-compose):
    - `DB_HOST=postgres`, `DB_PORT=5432`, `DB_NAME=core_db`, `DB_USER=core_user`, `DB_PASSWORD=core_password`

- Start the API:
  - `uv run uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload`

Health check: `http://localhost:8001/health`
