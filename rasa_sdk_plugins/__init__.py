import os
import sys
from typing import Optional

import pluggy
from sanic import Sanic, response
from sanic.response import HTTPResponse

hookimpl = pluggy.HookimplMarker("rasa_sdk")


def _read_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def init_hooks(manager: pluggy.PluginManager) -> None:
    manager.register(sys.modules[__name__], name="cva_action_version_endpoint")


@hookimpl
def attach_sanic_app_extensions(app: Sanic) -> None:
    @app.get("/version")
    async def version(_) -> HTTPResponse:
        body = {
            "service": "action",
            "version": _read_env("ACTION_VERSION"),
            "commitSha": _read_env("ACTION_COMMIT_SHA"),
            "imageTag": _read_env("ACTION_IMAGE_TAG"),
            "modelName": _read_env("LLM_MODEL"),
            "llmProvider": _read_env("LLM_PROVIDER"),
        }
        return response.json(body, status=200)