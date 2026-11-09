from typing import Optional

import httpx
import openai

LLM_UNAVAILABLE = "llm_unavailable"

# Failures of the LLM service itself, not of the request: unreachable, timed
# out, out of quota or rate limited, refused credentials, or a server error.
_SERVICE_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.InternalServerError,
    httpx.TransportError,
)


def is_llm_service_error(exc: Optional[BaseException]) -> bool:
    """Whether `exc`, or an exception it was raised from, is a failure of the LLM service."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, _SERVICE_ERRORS):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False
