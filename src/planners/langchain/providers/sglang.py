from typing import Any, Dict

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.planners.langchain.providers.base import LlmProviderPlugin
from src.util import env


class SglangProviderPlugin(LlmProviderPlugin):
    name = "sglang"

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any:
        base_url = env.require_all_env("LLM_BASE_URL")
        api_key = env.get_env("LLM_API_KEY") or env.get_env("OPENAI_API_KEY") or "dummy"

        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
            **extra_kwargs,
        }
        kwargs["api_key"] = SecretStr(api_key)

        return ChatOpenAI(**kwargs)
