import asyncio
import time
from typing import Any
from unittest.mock import patch

from src.actions.actions import visualization_action


def test_orchestrations_for_different_requests_run_at_the_same_time() -> None:
    def blocking_orchestrator(**_kwargs: Any) -> str:
        time.sleep(0.5)  # an LLM call blocks its thread
        return "outcome"

    async def two_requests() -> float:
        started = time.monotonic()
        results = await asyncio.gather(visualization_action._orchestrate(question="a"), visualization_action._orchestrate(question="b"))
        assert results == ["outcome", "outcome"]
        return time.monotonic() - started

    with patch.object(visualization_action, "orchestrate_visualization_request", side_effect=blocking_orchestrator):
        elapsed = asyncio.run(two_requests())

    assert elapsed < 0.9, f"orchestrations ran one after the other ({elapsed:.2f}s)"
