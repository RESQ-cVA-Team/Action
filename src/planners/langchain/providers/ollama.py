from typing import Any, Dict

from langchain_ollama import ChatOllama

from src.planners.langchain.providers.base import LlmProviderPlugin
from src.util import env


class OllamaProviderPlugin(LlmProviderPlugin):
    name = "ollama"

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any:
        base_url = env.get_env("LLM_BASE_URL", "http://ollama:11434")
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **extra_kwargs,
        )
