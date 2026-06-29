from typing import Any, Dict

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.planners.langchain.providers.base import LlmProviderPlugin
from src.util import env


class OpenAiCompatibleProviderPlugin(LlmProviderPlugin):
    name = "openai-compatible"

    @staticmethod
    def _resolve_api_key() -> str:
        api_key = env.get_env("LLM_API_KEY") or env.get_env("OPENAI_API_KEY")
        if api_key:
            return api_key
        raise OSError("Missing required environment variable: LLM_API_KEY or OPENAI_API_KEY")

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any:
        base_url = env.require_all_env("LLM_BASE_URL")
        api_key = self._resolve_api_key()
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=SecretStr(api_key),
            **extra_kwargs,
        )
