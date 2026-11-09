from typing import Any, Dict, List
from unittest.mock import patch

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain import pipeline
from src.planners.langchain.pipeline import _is_plain_request, _plain_plan_mismatch


def _plan(charts: List[Dict[str, Any]], tests: List[Dict[str, Any]] | None = None) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "charts": [
                {
                    "chart_type": chart["chart_type"],
                    "metrics": [{"metric": m} for m in chart["metrics"]],
                    "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}, "time": {"grain": "QUARTER"}},
                }
                for chart in charts
            ],
            "statistical_tests": tests,
        }
    )


def test_plain_requests() -> None:
    assert _is_plain_request("Show me a line graph of door to needle time", {"metric": "DTN", "chart_type": "LINE"})
    assert _is_plain_request("DTN per quarter", {"metric": ["DTN"], "group_by": "QUARTER"})
    assert _is_plain_request("can u plott door to needle time in a bar chart??", {"metric": "DTN", "chart_type": "BAR", "limit": None})


def test_requests_the_critique_pass_is_for() -> None:
    entities = {"metric": "DTN", "chart_type": "LINE"}
    assert not _is_plain_request("DTN in 2025", entities)
    assert not _is_plain_request("DTN for the last 6 months", entities)
    assert not _is_plain_request("three line charts of DTN", entities)
    assert not _is_plain_request("Previous chart plan: {...}. Now as a bar chart", entities)
    assert not _is_plain_request("DTN for women", {**entities, "sex": "FEMALE"})
    assert not _is_plain_request("DTN as a bar chart, not a line chart", {"metric": "DTN", "chart_type": ["BAR", "LINE"]})
    assert not _is_plain_request("DTN by hospital", {**entities, "group_by": "HOSPITAL"})
    assert not _is_plain_request("a line graph", {"chart_type": "LINE"})


def test_plain_plan_check() -> None:
    entities = {"metric": "DTN", "chart_type": "LINE"}
    assert _plain_plan_mismatch(entities, _plan([{"chart_type": "LINE", "metrics": ["DTN"]}])) is None
    assert "leaves out" in (_plain_plan_mismatch(entities, _plan([{"chart_type": "LINE", "metrics": ["DIDO"]}])) or "")
    assert "adds metric" in (_plain_plan_mismatch(entities, _plan([{"chart_type": "LINE", "metrics": ["DTN", "DIDO"]}])) or "")
    assert "chart type" in (_plain_plan_mismatch(entities, _plan([{"chart_type": "BAR", "metrics": ["DTN"]}])) or "")
    with_test = _plan([{"chart_type": "LINE", "metrics": ["DTN"]}], [{"test_type": "MANN_WHITNEY_U_TEST", "metrics": [{"metric": "DTN"}]}])
    assert "statistical test" in (_plain_plan_mismatch(entities, with_test) or "")


def _plan_json(chart_type: str, metric: str) -> str:
    return _plan([{"chart_type": chart_type, "metrics": [metric]}]).model_dump_json()


def test_a_plain_request_skips_the_critique_call_and_still_retries_a_mismatch() -> None:
    responses = iter([_plan_json("BAR", "DTN"), _plan_json("LINE", "DTN")])
    labels: List[str] = []

    def fake_invoke(_chain: Any, _inputs: Any, label: str = "") -> str:
        labels.append(label)
        return next(responses)

    with patch.object(pipeline, "_invoke_with_timeout", side_effect=fake_invoke):
        plan = pipeline.generate_analysis_plan("Show me a line graph of DTN", {"metric": "DTN", "chart_type": "LINE"}, "en", max_retries=2)

    assert plan.charts is not None and plan.charts[0].chart_type == "LINE"
    assert labels == ["plan_chain_attempt_1", "plan_chain_attempt_2"]
