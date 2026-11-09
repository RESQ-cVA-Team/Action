from unittest.mock import patch

import httpx
import openai

from src.actions.actions.visualization_action import _decision_text
from src.planners.langchain.llm_errors import LLM_UNAVAILABLE, is_llm_service_error
from src.planners.langchain.request_orchestrator import VisualizationRequestOutcome, orchestrate_visualization_request

_REQUEST = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")


def _quota_error() -> openai.RateLimitError:
    return openai.RateLimitError("You have no credits remaining.", response=httpx.Response(429, request=_REQUEST), body=None)


def test_service_errors_are_recognised_through_a_wrapping_exception() -> None:
    try:
        try:
            raise _quota_error()
        except Exception as exc:
            raise ValueError("Planner failed to produce a valid AnalysisPlan") from exc
    except ValueError as wrapped:
        assert is_llm_service_error(wrapped)
    assert is_llm_service_error(openai.APIConnectionError(request=_REQUEST))


def test_a_request_problem_is_not_a_service_error() -> None:
    assert not is_llm_service_error(ValueError("Plan does not faithfully match the request"))


def test_an_unreachable_llm_rejects_with_its_own_reason() -> None:
    with patch("src.planners.langchain.request_orchestrator._get_llm", return_value=lambda _payload: None), patch(
        "src.planners.langchain.request_orchestrator._invoke_chain", side_effect=_quota_error()
    ):
        outcome = orchestrate_visualization_request("Show me DTN", {"metric": "DTN"}, language="en", include_plan=False)
    assert (outcome.decision, outcome.reason) == ("reject", LLM_UNAVAILABLE)


def test_the_user_is_told_the_llm_is_unreachable() -> None:
    outcome = VisualizationRequestOutcome(decision="reject", reason=LLM_UNAVAILABLE)
    assert "language model" in _decision_text(outcome, "en", "action.visualization.reject_default")
    other = VisualizationRequestOutcome(decision="reject", reason="out_of_scope", message="Not a chart request.")
    assert _decision_text(other, "en", "action.visualization.reject_default") == "Not a chart request."
