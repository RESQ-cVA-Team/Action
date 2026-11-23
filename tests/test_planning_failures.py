from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.actions.actions.visualization_action import _decision_text
from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain import pipeline, request_orchestrator
from src.planners.langchain.pipeline import PlannerTimeoutError, PlanUnresolvedError
from src.planners.langchain.request_orchestrator import PLAN_UNRESOLVED, PLANNING_TIMED_OUT, VisualizationRequestOutcome

_LINE_REQUEST = ("Show me a line graph of DTN", {"metric": "DTN", "chart_type": "LINE"})


def _bar_plan() -> str:
    chart = {"chart_type": "BAR", "metrics": [{"metric": "DTN"}], "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}, "time": {"grain": "QUARTER"}}}
    return AnalysisPlan.model_validate({"charts": [chart]}).model_dump_json()


def _plan_with(reply: Any, labels: List[str]) -> None:
    def invoke(_chain: Any, _inputs: Any, label: str = "") -> Any:
        labels.append(label)
        return reply(label)

    with patch.object(pipeline, "_invoke_with_timeout", side_effect=invoke):
        pipeline.generate_analysis_plan(*_LINE_REQUEST, "en", max_retries=2)


def test_a_plan_that_fails_a_check_on_every_attempt_is_not_returned() -> None:
    labels: List[str] = []
    with pytest.raises(PlanUnresolvedError, match="chart type"):
        _plan_with(lambda _label: _bar_plan(), labels)
    assert labels == ["plan_chain_attempt_1", "plan_chain_attempt_2", "plan_chain_attempt_3"]


def test_a_timed_out_call_is_not_retried() -> None:
    def timed_out(label: str) -> Any:
        raise PlannerTimeoutError(f"{label} timed out after 45.0s")

    labels: List[str] = []
    with pytest.raises(PlannerTimeoutError):
        _plan_with(timed_out, labels)
    assert labels == ["plan_chain_attempt_1"]


def _orchestrate(failure: Exception, question: str) -> tuple[VisualizationRequestOutcome, int]:
    calls: List[Dict[str, Any]] = []

    def failing(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise failure

    with patch.object(request_orchestrator, "generate_analysis_plan", side_effect=failing):
        outcome = request_orchestrator.orchestrate_visualization_request(question, {"metric": "DTN"}, decided=VisualizationRequestOutcome(decision="proceed", reason="all_required_fields_present"))
    return outcome, len(calls)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [(PlannerTimeoutError("plan_chain_attempt_1 timed out after 45.0s"), PLANNING_TIMED_OUT), (PlanUnresolvedError("no plan passed"), PLAN_UNRESOLVED)],
)
def test_a_planning_failure_is_its_own_reject(failure: Exception, reason: str) -> None:
    outcome, calls = _orchestrate(failure, "Run a Mann-Whitney U test for DTN between provider group 289 and provider group 290")
    assert (outcome.decision, outcome.reason, outcome.plan) == ("reject", reason, None)
    assert calls == 1


def test_each_planning_failure_tells_the_user_what_happened() -> None:
    timed_out = _decision_text(VisualizationRequestOutcome(decision="reject", reason=PLANNING_TIMED_OUT), "en", "action.visualization.reject_default")
    unresolved = _decision_text(VisualizationRequestOutcome(decision="reject", reason=PLAN_UNRESOLVED), "cs", "action.visualization.reject_default")
    assert "fewer charts" in timed_out
    assert unresolved.startswith("Nepodařilo se mi sestavit plán")
