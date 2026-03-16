from typing import Any, Dict

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.executors.langchain.providers.base import LlmProviderPlugin
from src.util import env


class OpenAiCompatibleProviderPlugin(LlmProviderPlugin):
    name = "openai-compatible"

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any:
        base_url = env.require_all_env("LLM_BASE_URL")
        api_key = env.require_all_env("LLM_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=SecretStr(api_key),
            **extra_kwargs,
        )
