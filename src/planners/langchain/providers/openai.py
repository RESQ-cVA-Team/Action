from typing import Any, Dict

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.planners.langchain.providers.base import LlmProviderPlugin
from src.util import env


class OpenAiProviderPlugin(LlmProviderPlugin):
    name = "openai"

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any:
        api_key = env.require_all_env("LLM_API_KEY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=SecretStr(api_key),
            **extra_kwargs,
        )
