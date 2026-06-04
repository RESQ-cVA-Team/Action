# Action

Run instructions for:

- Development using the Dev Container
- Production using GitHub workflow-built images

## Service Wiring

- Rasa calls Action at `http://action:5055/webhook`
- Action calls Webapp proxy at `RASA_PROXY_URL` (for GraphQL and analytics REST)
- Action sends long-task callbacks to Webapp callback endpoint

## Required Environment Variables

- `RASA_PROXY_URL` (example: `http://webapp:3000/api/rasa-proxy`)
- `ACTION_SERVER_TOKEN` (must match Webapp)
- `LONG_TASK_CALLBACK_TOKEN` (must match Webapp)
- `RASA_PROXY_GRAPHQL_TARGET` (typically `graphql`)
- `RASA_PROXY_ANALYTICS_TARGET` (typically `analytics`)
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY` (for providers that require a key)

Callback URL validation (recommended):

- `CALLBACK_BASE_URL` (example: `http://webapp:3000`)
- Optional explicit allow-lists:
  - `LONG_TASK_CALLBACK_ALLOWED_ORIGINS`
  - `LONG_TASK_CALLBACK_ALLOWED_PATHS`

## Development (Dev Container)

1. Open this repository in VS Code.
2. Reopen in container.
3. Start Action:

```bash
python -m rasa_sdk --actions src.actions
```

The dev container definition is in `.devcontainer/Dockerfile`.

## Production (Workflow-built image)

GitHub workflows build and publish Action images to GHCR.

Typical tags:

- `ghcr.io/<org>/action:latest`
- `ghcr.io/<org>/action:<git-sha>`

Run example:

```bash
docker run --rm -p 5055:5055 \
  -e RASA_PROXY_URL=http://webapp:3000/api/rasa-proxy \
  -e ACTION_SERVER_TOKEN=<shared-action-token> \
  -e LONG_TASK_CALLBACK_TOKEN=<shared-callback-token> \
  -e RASA_PROXY_GRAPHQL_TARGET=graphql \
  -e RASA_PROXY_ANALYTICS_TARGET=analytics \
  -e LLM_PROVIDER=openai \
  -e LLM_MODEL=gpt-4o-mini \
  -e LLM_API_KEY=<llm-api-key> \
  ghcr.io/<org>/action:latest
```
