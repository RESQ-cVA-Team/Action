import time
from contextvars import copy_context
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain import pipeline, request_orchestrator
from src.util.llm_usage import llm_usage_field, record_llm_call, start_llm_usage


def _response(content: str, input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(content=content, usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens})


class _Chain:
    def __init__(self, response: Any) -> None:
        self.response = response

    def invoke(self, _inputs: Any) -> Any:
        return self.response


def _in_fresh_request(fn: Any) -> Any:
    return copy_context().run(fn)


def test_nothing_is_counted_outside_a_request() -> None:
    def body() -> Any:
        record_llm_call("plan", time.monotonic(), _response("{}", 10, 5))
        return llm_usage_field()

    assert _in_fresh_request(body) == {}


def test_calls_are_counted_by_step_with_their_tokens() -> None:
    def body() -> Any:
        start_llm_usage()
        record_llm_call("decision", time.monotonic(), _response("{}", 100, 20))
        record_llm_call("plan", time.monotonic(), _response("{}", 1000, 300))
        record_llm_call("critique", time.monotonic())  # failed: no tokens
        return llm_usage_field()["llm_usage"]

    usage = _in_fresh_request(body)
    assert usage["calls"] == 3
    assert usage["input_tokens"] == 1100
    assert usage["output_tokens"] == 320
    assert usage["steps"] == {"decision": 1, "plan": 1, "critique": 1}
    assert usage["seconds"] >= 0


def test_both_call_helpers_record_their_calls_from_the_worker_thread() -> None:
    def body() -> Any:
        start_llm_usage()
        request_orchestrator._invoke_chain(_Chain(_response('{"decision": "proceed"}', 50, 10)), {})
        pipeline._invoke_with_timeout(_Chain(_response("{}", 500, 100)), {}, label="plan_chain_attempt_1")
        pipeline._invoke_with_timeout(_Chain(_response('{"ok": true}', 400, 5)), {}, label="plan_critique")
        return llm_usage_field()["llm_usage"]

    usage = _in_fresh_request(body)
    assert usage["steps"] == {"decision": 1, "plan": 1, "critique": 1}
    assert usage["input_tokens"] == 950
    assert usage["output_tokens"] == 115


def test_plan_generation_in_its_own_thread_is_counted() -> None:
    def fake_generate(**_kwargs: Any) -> Any:
        record_llm_call("plan", time.monotonic(), _response("{}", 700, 90))
        return AnalysisPlan.model_validate({"charts": [{"chart_type": "LINE", "metrics": [{"metric": "DTN"}], "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}}}]})

    def body() -> Any:
        start_llm_usage()
        with patch.object(request_orchestrator, "generate_analysis_plan", side_effect=fake_generate):
            request_orchestrator._generate_plan_with_timeout(question="q", entities={})
        return llm_usage_field()["llm_usage"]

    usage = _in_fresh_request(body)
    assert usage["steps"] == {"plan": 1}
    assert usage["input_tokens"] == 700
