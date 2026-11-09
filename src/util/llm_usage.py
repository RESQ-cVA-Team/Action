"""Counts of the LLM calls made for one request: how many, how long and how
many tokens, by step. Never their content. The counts ride along on the
payloads the actions already send, so evaluation tools can read them."""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

_LEDGER: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("llm_usage_ledger", default=None)


def start_llm_usage() -> None:
    """Counts the calls made from here on in the current request (and the threads bound to it)."""
    _LEDGER.set([])


def _tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    metadata = getattr(response, "response_metadata", None)
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if isinstance(token_usage, dict):
        return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")
    return None, None


def record_llm_call(step: str, started_at: float, response: Any = None) -> None:
    """Records one call that started at `started_at` (time.monotonic()); `response` is None when it failed."""
    ledger = _LEDGER.get()
    if ledger is None:
        return
    input_tokens, output_tokens = _tokens(response)
    ledger.append(
        {
            "step": step,
            "seconds": time.monotonic() - started_at,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )


def llm_usage_field() -> Dict[str, Any]:
    """{"llm_usage": {...}} for a payload, or {} when nothing is being counted."""
    ledger = _LEDGER.get()
    if ledger is None:
        return {}
    steps: Dict[str, int] = {}
    for call in ledger:
        steps[call["step"]] = steps.get(call["step"], 0) + 1
    return {
        "llm_usage": {
            "calls": len(ledger),
            "seconds": round(sum(call["seconds"] for call in ledger), 3),
            "input_tokens": sum(call["input_tokens"] or 0 for call in ledger),
            "output_tokens": sum(call["output_tokens"] or 0 for call in ledger),
            "steps": steps,
        }
    }
