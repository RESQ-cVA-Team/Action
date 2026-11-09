from unittest.mock import patch

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain import request_orchestrator


def _plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "charts": [
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}, "time": {"grain": "QUARTER"}},
                }
            ]
        }
    )


def test_a_decision_taken_earlier_for_the_message_skips_the_decision_stage() -> None:
    with (
        patch.object(request_orchestrator, "_decision_stage") as decision_stage,
        patch.object(request_orchestrator, "_generate_plan_with_timeout", return_value=_plan()) as generate,
    ):
        outcome = request_orchestrator.orchestrate_visualization_request(
            question="Show me a line graph of DTN",
            entities={"metric": "DTN", "chart_type": "LINE"},
            decided_proceed="all_required_fields_present",
        )

    decision_stage.assert_not_called()
    generate.assert_called_once()
    assert outcome.decision == "proceed"
    assert outcome.plan is not None


def test_without_an_earlier_decision_the_decision_stage_runs() -> None:
    clarify = request_orchestrator.VisualizationRequestOutcome(decision="clarify", reason="missing_required_fields", message="Which metric?")
    with (
        patch.object(request_orchestrator, "_decision_stage", return_value=clarify) as decision_stage,
        patch.object(request_orchestrator, "_generate_plan_with_timeout") as generate,
    ):
        outcome = request_orchestrator.orchestrate_visualization_request(question="Show me a line graph", entities={"chart_type": "LINE"})

    decision_stage.assert_called_once()
    generate.assert_not_called()
    assert outcome.decision == "clarify"
