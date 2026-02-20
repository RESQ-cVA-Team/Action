## What this repo is

This is a **Rasa SDK action server** that turns natural-language clinical analytics questions into structured chart payloads.

At a high level:

- A user asks a question (e.g. “Show DTN by sex over the last 6 months”).
- The server creates an **AnalysisPlan** (either via a heuristic planner or an LLM-based planner).
- It executes the plan by querying a **GraphQL backend via a proxy**.
- It returns a typed **VisualizationResponse** (charts + optional stats) as a Rasa `json_message` payload.

The action server entrypoint is the standard Rasa SDK module invocation:

```bash
python -m rasa_sdk --actions src.actions
```

## Key actions

- `action_generate_visualization`: plans + executes a visualization request.
- `action_explain_metric`: explains a metric/KPI using SSOT YAML metadata (multi-language descriptions).

## SSOT (Source of Truth) YAML

This repo depends on YAML files under src/shared/SSOT (metrics, enums, chart/test types). It is configured as a git submodule.

In most normal workflows this directory is already present after cloning. If you ever see runtime/build errors about missing SSOT YAML (for example `ChartType.yml`), initialize the submodule:

```bash
git submodule update --init --recursive
```

## Running

### Option A: VS Code Dev Container

The devcontainer is defined in [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) and uses Docker Compose ([.devcontainer/docker-compose.yml](.devcontainer/docker-compose.yml)).

1) Open this repo in the Dev Container.
2) Start the action server:

- VS Code task: “Start Rasa Actions” ([.vscode/tasks.json](.vscode/tasks.json))
- Or manually:

```bash
python -m rasa_sdk --actions src.actions
```

The Compose file maps port `5055:5055`.

If the server fails to start due to missing packages, install deps inside the container:

```bash
pip install -r requirements.txt
```

### Option B: Run a prebuilt Docker image

This repo is typically built/published by CI (GitHub Actions). Pull the published image for your environment, then run it with the required env vars.

Run (example):

```bash
docker run --rm -p 5055:5055 \
	-e GRAPHQL_PROXY_URL=... \
	-e GRAPHQL_API_URL=... \
	-e LLM_MODEL=... \
	-e OPENAI_API_KEY=... \
	<your-published-image>:<tag>
```

## Environment variables

This codebase loads `.env` automatically (via python-dotenv) when you use helpers in [src/util/env.py](src/util/env.py). The repo’s `.gitignore` is configured to **not commit** `.env`.

### Required

- **`GRAPHQL_PROXY_URL`**: proxy endpoint that forwards GraphQL requests.
- **`GRAPHQL_API_URL`**: upstream GraphQL aggregation endpoint.
- **`LLM_MODEL`**: model name passed to `langchain_openai.ChatOpenAI(model=...)`.
- **`OPENAI_API_KEY`**: OpenAI-compatible API key.

### Optional

- `LOGLEVEL` (default: `INFO`): logging level used by [src/__init__.py](src/__init__.py).

### Optional: callback streaming

If the incoming message metadata includes a `callback_url`, `action_generate_visualization` can stream progress + results to that endpoint.

- `LONG_TASK_CALLBACK_TOKEN` (default: unset)
	- If set, callback POSTs include header `x-action-server-token`.

### Optional: debug / verbosity flags

All flags are parsed as booleans (truthy: `1/true/yes/on`; falsy: `0/false/no/off`).

- `ACTIONS_LOG_USER_TEXT` (default: `false`)
- `ACTIONS_ECHO_INTERNAL_ERRORS` (default: `false`)

- `PLANNER_ENABLE_COT` (default: `true`)
- `PLANNER_LOG_PROMPTS` (default: `false`)
- `PLANNER_LOG_REASONING` (default: `false`)

- `EXECUTOR_LOG_GRAPHQL_QUERY` (default: `false`)
- `GRAPHQL_LOG_QUERY` (default: `false`)
- `GRAPHQL_LOG_BODY` (default: `false`)

- `LONG_ACTION_LOG_CALLBACK_STATUS` (default: `false`)
- `LONG_ACTION_LOG_CALLBACK_ERRORS` (default: `false`)

- `CLI_LOG_GRAPHQL_QUERY` (default: `false`)
