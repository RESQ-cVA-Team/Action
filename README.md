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
- `KEYCLOAK_ISSUER`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET` (Action's own Keycloak service-account client; its token authenticates Action to Webapp)
- `ACTION_AUTH_TOKEN` (the token Rasa presents on Action's `/webhook`; must match Rasa)
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
  -e KEYCLOAK_ISSUER=<issuer-url> \
  -e KEYCLOAK_CLIENT_ID=<service-account-client-id> \
  -e KEYCLOAK_CLIENT_SECRET=<service-account-client-secret> \
  -e ACTION_AUTH_TOKEN=<token-rasa-presents> \
  -e RASA_PROXY_GRAPHQL_TARGET=graphql \
  -e RASA_PROXY_ANALYTICS_TARGET=analytics \
  -e LLM_PROVIDER=openai \
  -e LLM_MODEL=gpt-4o-mini \
  -e LLM_API_KEY=<llm-api-key> \
  ghcr.io/<org>/action:latest
```

## Prompt Contract Test Runs

Run these from the workspace root with Docker Compose so dependencies and env match service runtime.

Deterministic prompt-contract suite (repo-native, no external APIs):

```bash
docker compose exec action python -m unittest tests.test_prompt_contract_suite
```

Optional live external API smoke (OpenAI + proxy GraphQL preflight):

```bash
docker compose exec -T action env RUN_EXTERNAL_API_E2E=1 python -m unittest tests.test_external_api_e2e_smoke
```

Notes:
- The deterministic suite is the default gate for prompt behavior regressions.
- The external smoke is opt-in and may skip GraphQL execution when proxy user token cache is not seeded for the test sender id.
