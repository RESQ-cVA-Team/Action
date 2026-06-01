import logging
from typing import Any, Dict, Literal, cast

from src.planners.langchain.providers import (
    OllamaProviderPlugin,
    OpenAiCompatibleProviderPlugin,
    OpenAiProviderPlugin,
    SglangProviderPlugin,
    VllmProviderPlugin,
)
from src.planners.langchain.providers.base import LlmProviderPlugin
from src.util import env

logger = logging.getLogger(__name__)

LlmProvider = Literal["openai", "openai-compatible", "ollama", "vllm", "sglang"]


class ProviderRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, LlmProviderPlugin] = {}

    def register(self, plugin: LlmProviderPlugin) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> LlmProviderPlugin:
        plugin = self._plugins.get(name)
        if plugin is None:
            supported = ", ".join(sorted(self._plugins.keys()))
            raise OSError(f"Unsupported LLM_PROVIDER '{name}'. Supported providers: {supported}")
        return plugin


registry = ProviderRegistry()
registry.register(OpenAiProviderPlugin())
registry.register(OpenAiCompatibleProviderPlugin())
registry.register(OllamaProviderPlugin())
registry.register(VllmProviderPlugin())
registry.register(SglangProviderPlugin())


def register_provider(plugin: LlmProviderPlugin) -> None:
    """Register or override a provider plugin at runtime."""

    registry.register(plugin)


def _load_extra_kwargs() -> Dict[str, Any]:
    import json

    raw = env.get_env("LLM_KWARGS_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OSError("LLM_KWARGS_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise OSError("LLM_KWARGS_JSON must decode to a JSON object")
    return cast(Dict[str, Any], parsed)


def get_llm_provider() -> LlmProvider:
    provider_raw = env.require_all_env("LLM_PROVIDER")
    provider = provider_raw.strip().lower()
    registry.get(provider)
    return cast(LlmProvider, provider)


def create_chat_llm(temperature: float = 0) -> Any:
    provider = get_llm_provider()
    model = env.require_all_env("LLM_MODEL")
    extra_kwargs = _load_extra_kwargs()
    plugin = registry.get(provider)
    logger.debug("[LLM Factory] Creating chat model provider=%s model=%s", provider, model)
    return plugin.create_chat_model(
        model=model,
        temperature=temperature,
        extra_kwargs=extra_kwargs,
    )
