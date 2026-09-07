import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pluggy
from sanic import Sanic, response
from sanic.response import HTTPResponse

from src.util import env as env_util

hookimpl = pluggy.HookimplMarker("rasa_sdk")

# Captured once, at plugin-module import time (which happens once per action
# server process, at startup) -- lets CVaLab tell whether few-shot
# examples/prompts on disk have been edited since this process started, since
# both are only ever read once at import time and won't take effect until a
# restart either way.
_STARTED_AT = datetime.now(timezone.utc).isoformat()


def _read_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def init_hooks(manager: pluggy.PluginManager) -> None:
    manager.register(sys.modules[__name__], name="cva_action_version_endpoint")


# rasa_sdk 3.14.2's own /webhook route (registered in rasa_sdk.endpoint,
# outside this codebase) has no built-in authentication of any kind --
# older rasa_sdk versions had a --auth-token/ACTION_TOKEN CLI mechanism, but
# it does not exist in this version. Anything that can reach this port could
# otherwise trigger any action for any sender_id. This plugin hook is the
# only place this codebase can extend the Sanic app rasa_sdk builds, so the
# check is added here as request middleware rather than inside rasa_sdk.
_ACTION_TOKEN = os.getenv("ACTION_TOKEN") or None
_ACTION_REQUIRE_AUTH_TOKEN = env_util.env_flag("ACTION_REQUIRE_AUTH_TOKEN", default=True)
if _ACTION_REQUIRE_AUTH_TOKEN and not _ACTION_TOKEN:
    raise RuntimeError(
        "ACTION_TOKEN is required when ACTION_REQUIRE_AUTH_TOKEN is enabled. "
        "Set ACTION_TOKEN or set ACTION_REQUIRE_AUTH_TOKEN=false only for local debugging."
    )


@hookimpl
def attach_sanic_app_extensions(app: Sanic) -> None:
    @app.on_request
    async def _require_action_token(request) -> Optional[HTTPResponse]:
        # Only the action-invocation endpoint is gated -- /health must stay
        # open for container health checks, and /version, /actions,
        # /debug/fewshot-relevance are separate, lower-sensitivity surfaces
        # (the debug endpoint's own auth is tracked separately).
        if request.path != "/webhook" or not _ACTION_TOKEN:
            return None

        query_token = request.args.get("token")
        auth_header = request.headers.get("Authorization", "")
        header_token = auth_header[7:] if auth_header.startswith("Bearer ") else None
        if (query_token or header_token) != _ACTION_TOKEN:
            return response.json({"error": "Unauthorized"}, status=401)
        return None

    @app.get("/version")
    async def version(_) -> HTTPResponse:
        body = {
            "service": "action",
            "version": _read_env("ACTION_VERSION"),
            "commitSha": _read_env("ACTION_COMMIT_SHA"),
            "imageTag": _read_env("ACTION_IMAGE_TAG"),
            "buildDate": _read_env("ACTION_BUILD_DATE"),
            "modelName": _read_env("LLM_MODEL"),
            "llmProvider": _read_env("LLM_PROVIDER"),
            "promptVersion": _read_env("ACTION_PROMPT_VERSION"),
            "ssotVersion": _read_env("ACTION_SSOT_VERSION") or _read_env("SSOT_VERSION"),
            "startedAt": _STARTED_AT,
        }
        return response.json(body, status=200)

    @app.post("/debug/fewshot-relevance")
    async def fewshot_relevance(request) -> HTTPResponse:
        # Imported here rather than at module level: this plugin module is
        # discovered by rasa_sdk's pluggy machinery separately from (and
        # potentially before) the custom-actions package, so importing the
        # heavier langchain pipeline eagerly at plugin-load time would couple
        # two independent loading orders together for no benefit.
        from src.planners.langchain.pipeline import score_few_shot_examples

        body = request.json or {}
        question = body.get("question")
        entities = body.get("entities")
        if not isinstance(question, str) or not isinstance(entities, dict):
            return response.json({"error": "Body must include question (string) and entities (object)."}, status=400)

        scored = score_few_shot_examples(question, entities)
        return response.json({"scored": scored}, status=200)