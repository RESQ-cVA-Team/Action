# Action Service — Runbook

This README only covers:

- How to run the action service as a developer
- How to run it in production

---

## Related repositories

- Webapp: https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/Webapp
- Action (this repo): https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/Action
- Rasa: https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/Rasa
- SSOT: https://github.com/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/SSOT

---

## 1) Development setup

### Prerequisites (recommended path)

- Docker + Docker Compose
- VS Code + Dev Containers extension

### Dev Container workflow (recommended)

1. Open this repository in VS Code.
2. Choose **Reopen in Container**.
3. Wait for post-create setup to finish (`pip install -e . && mypy --strict . && ruff check .`).
4. Configure `.env` (repo root) as shown below.
5. Run the service command.

### Option B: Local machine

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment (`.env` in repo root)

```env
RASA_PROXY_URL=http://host.docker.internal:3000/api/rasa-proxy
ACTION_SERVER_TOKEN=<set-a-shared-token>
LONG_TASK_CALLBACK_TOKEN=<set-a-shared-token>

RASA_PROXY_GRAPHQL_TARGET=graphql
RASA_PROXY_ANALYTICS_TARGET=analytics

LLM_PROVIDER=vllm
LLM_BASE_URL=https://<your-vllm-host>/v1
LLM_MODEL=<model-id>
LLM_API_KEY=<optional-or-required-by-provider>

# Optional: heuristic planner shortcut (disabled by default)
ACTIONS_ENABLE_HEURISTIC_SHORTCUT=false
ACTIONS_HEURISTIC_MIN_CONFIDENCE=0.85

LOGLEVEL=DEBUG
```

If using OpenAI:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=<openai-api-key>
LLM_BASE_URL=
```

### Run locally

```bash
python -m rasa_sdk --actions src.actions
```

The action server listens on port `5055`.

### VS Code tasks (optional)

This repo includes `.vscode/tasks.json` with:

- `Start Rasa Actions`

Run from VS Code: **Terminal → Run Task**.

---

## 2) Production run

### Required dependencies

- Webapp (routes chat traffic and callbacks)
- One or more Rasa runtime containers (language-specific)
- Redis (tracker + lock stores)
- Duckling

### Required Action environment variables

- `ACTION_SERVER_TOKEN`
- `LONG_TASK_CALLBACK_TOKEN`
- `RASA_PROXY_URL`
- `RASA_PROXY_GRAPHQL_TARGET`
- `RASA_PROXY_ANALYTICS_TARGET`
- `GRAPHQL_API_URL`
- `LOGLEVEL`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY` (provider-dependent)

### Recommended image tags

- `ghcr.io/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/action:latest`
- `ghcr.io/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/webapp:latest`
- `ghcr.io/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/rasa:<locale>-latest`

### Minimal production compose snippet (action)

```yaml
services:
  action:
    image: ghcr.io/09c7b0ed-f907-45d2-bc7c-48b17f2d9940/action:latest
    environment:
      ACTION_SERVER_TOKEN: <shared-action-token>
      LONG_TASK_CALLBACK_TOKEN: <shared-action-token>
      RASA_PROXY_URL: http://webapp:3000/api/rasa-proxy
      GRAPHQL_API_URL: https://<your-domain>/api/graphql/aggregation
      RASA_PROXY_GRAPHQL_TARGET: graphql
      RASA_PROXY_ANALYTICS_TARGET: analytics
      LOGLEVEL: INFO
      LLM_PROVIDER: openai
      LLM_MODEL: gpt-4o-mini
      LLM_API_KEY: <llm-api-key>
```

Start stack:

```bash
docker compose up -d
```

---

## 3) Quick verification

- Action endpoint is reachable at `http://<action-host>:5055/webhook`
- Rasa can reach `http://action:5055/webhook`
- Action can reach Webapp proxy endpoint `/api/rasa-proxy`
- Shared tokens match across services:
  - `ACTION_SERVER_TOKEN`
  - `LONG_TASK_CALLBACK_TOKEN`

---

## 4) Common commands

Run action server:

```bash
python -m rasa_sdk --actions src.actions
```

