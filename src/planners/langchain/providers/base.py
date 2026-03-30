from typing import Any, Dict, Protocol


class LlmProviderPlugin(Protocol):
    name: str

    def create_chat_model(
        self,
        model: str,
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any: ...
