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
3. Wait for post-create setup to finish (`pip install -e . && pyright && ruff check .`).
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
LONG_TASK_CALLBACK_TOKEN=<set-a-callback-token>
CALLBACK_BASE_URL=http://host.docker.internal:3000

# Optional: explicit callback origin allowlist for long-task callbacks.
# Defaults to the origin from CALLBACK_BASE_URL when unset.
# Example: http://webapp:3000;https://chat.example.org
# LONG_TASK_CALLBACK_ALLOWED_ORIGINS=http://webapp:3000

# Optional: local-only escape hatch. By default, values already present in the
# process environment win over .env entries. Set this only when you explicitly
# want .env to override existing environment variables during local debugging.
# ACTION_DOTENV_OVERRIDE=1

RASA_PROXY_GRAPHQL_TARGET=graphql
RASA_PROXY_ANALYTICS_TARGET=analytics

LLM_PROVIDER=vllm
LLM_BASE_URL=https://<your-vllm-host>/v1
LLM_MODEL=<model-id>
LLM_API_KEY=<optional-or-required-by-provider>

LOGLEVEL=DEBUG
LOG_FORMAT=text

# Optional: file logging for debugging
LOG_TO_FILE=true
LOG_FILE_DIR=.tmp/logs
LOG_FILE_LEVEL=DEBUG
LOG_FILE_FORMAT=text
LOG_FILE_SESSION=false
LOG_FILE_ROTATE=true
LOG_FILE_MAX_BYTES=10485760
LOG_FILE_BACKUP_COUNT=3
LOG_FILE_RETENTION_DAYS=7
LOG_COLOR=false

# Optional: targeted debug logging for analytics-center proxy calls.
# Keep disabled in normal environments.
ANALYTICS_CENTER_LOG_QUERY=false
# When true, logs the full analytics-center query payload at debug level.
# Default is false, which logs only a safe summary of query keys.
ANALYTICS_CENTER_LOG_UPSTREAM_PREVIEW=false
# When true, logs a truncated preview of the upstream error body for analytics-center proxy failures.
# Default is false, which logs only metadata such as status code, body length, and proxy reason.

# Optional: reduce repeated low-level third-party HTTP logs
LOG_NOISY_LIB_LEVEL=WARNING
# Leave empty (default) to disable per-library suppression, or opt in:
LOG_NOISY_LIB_LOGGERS=
# LOG_NOISY_LIB_LOGGERS=openai,httpx,httpcore,urllib3

# Optional: override levels per logger/module
LOG_MODULE_LEVELS=
# LOG_MODULE_LEVELS=src.executors.graphql.client=DEBUG,src.executors.analytics_center.client=DEBUG,src.planners=DEBUG
```

`ACTION_SERVER_TOKEN` authenticates Action -> Webapp proxy requests.
`LONG_TASK_CALLBACK_TOKEN` authenticates Action -> Webapp long-task callback requests.
`LONG_TASK_CALLBACK_ALLOWED_ORIGINS` optionally overrides which callback URL origins Action will trust.
When it is unset, Action trusts the origin derived from `CALLBACK_BASE_URL`.
If neither `LONG_TASK_CALLBACK_ALLOWED_ORIGINS` nor `CALLBACK_BASE_URL` is set,
callback mode is disabled and long actions fall back to synchronous execution.
`.env` loading fills in missing values only; set `ACTION_DOTENV_OVERRIDE=1` only
when you explicitly need `.env` to override the existing process environment.

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

### Logging configuration

The action service supports two output formats:

- `LOG_FORMAT=text` keeps developer-friendly single-line logs.
- `LOG_FORMAT=json` emits structured logs with top-level fields such as `level`, `logger`, `message`, `trace_id`, `source`, and a `context` object.

File logging can use the same or a different formatter via `LOG_FILE_FORMAT`.

Level policy:

- `INFO` is for request lifecycle milestones, meaningful fallbacks, and user-impacting warnings or failures.
- `DEBUG` is for high-volume diagnostic detail such as outbound GraphQL or Analytics Center request chatter, planner bootstrap details, compiler diagnostics, and cache behavior.
- `WARNING` and `ERROR` remain reserved for degraded behavior, retries, partial results, upstream failures, and invalid responses.

Use `LOG_MODULE_LEVELS` for targeted overrides without changing the global level. The value is a comma-separated list of `logger=LEVEL` or `logger:LEVEL` entries, for example:

```env
LOGLEVEL=INFO
LOG_FORMAT=json
LOG_MODULE_LEVELS=src.executors.graphql.client=DEBUG,src.executors.analytics_center.client=DEBUG,src.planners=DEBUG
```

This is the recommended way to temporarily increase verbosity for one subsystem without turning on full-project `DEBUG` logging.

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
- `CALLBACK_BASE_URL` or `LONG_TASK_CALLBACK_ALLOWED_ORIGINS`
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
      LONG_TASK_CALLBACK_TOKEN: <shared-callback-token>
      CALLBACK_BASE_URL: http://webapp:3000
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
- `ACTION_SERVER_TOKEN` matches between Action and Webapp for proxy requests
- `LONG_TASK_CALLBACK_TOKEN` matches between Action and Webapp for callback requests
- `CALLBACK_BASE_URL` or `LONG_TASK_CALLBACK_ALLOWED_ORIGINS` allows the Webapp callback origin

---

## 4) Common commands

Run action server:

```bash
python -m rasa_sdk --actions src.actions
```

