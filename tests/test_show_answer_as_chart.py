import asyncio
import unittest
from typing import Any, Dict, List

from src.actions.actions.visualization_action import ActionShowAnswerAsChart, _latest_answer_values
from src.actions.helpers.visualization import serialize_plan_for_frontend
from src.actions.long_action.long_action_context import LongActionContext
from src.domain.langchain.schema import AnalysisPlan
from src.executors.planning.value_plan import value_period, value_trend_chart

_VALUE = {
    "metric": "IMAGING_TYPE",
    "categories": ["ct", "ct_cta", "ct_cta_perfusion"],
    "measure": {"type": "RATE"},
    "period": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
    "filters": {"type": "SexFilter", "value": "FEMALE"},
}
_CHART = {"chart_type": "LINE", "metrics": [{"metric": "DTN"}], "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}, "time": {"grain": "QUARTER"}}}


def _plan_event(plan: Dict[str, Any]) -> Dict[str, Any]:
    preview = serialize_plan_for_frontend(AnalysisPlan.model_validate(plan))
    return {"event": "bot", "data": {"custom": {"type": "visualization_plan", "plan": preview}}}


class _Dispatcher:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def utter_message(self, **kwargs: Any) -> None:
        self.messages.append(kwargs)


def _prework(events: List[Dict[str, Any]]) -> tuple[Any, LongActionContext, _Dispatcher]:
    dispatcher = _Dispatcher()
    ctx = LongActionContext(sender_id="user:thread:1", tracker_snapshot={"events": events, "latest_message": {"text": "/show_answer_as_chart"}, "slots": {}}, dispatcher=dispatcher)
    return asyncio.run(ActionShowAnswerAsChart().prework(ctx)), ctx, dispatcher


class ValueTrendChartTests(unittest.TestCase):
    def test_a_value_per_quarter_keeps_what_it_counts(self) -> None:
        value = AnalysisPlan.model_validate({"values": [_VALUE]}).values[0]
        chart = value_trend_chart(value)
        self.assertEqual(chart.semantics.time.grain, "QUARTER")
        self.assertEqual((chart.semantics.time.window.start_date, chart.semantics.time.window.end_date), value_period(value))
        self.assertEqual(chart.metrics[0].categories, ["ct", "ct_cta", "ct_cta_perfusion"])
        quarter = AnalysisPlan.model_validate({"values": [{**_VALUE, "period": {"start_date": "2026-01-01", "end_date": "2026-03-31"}}]}).values[0]
        self.assertEqual(value_trend_chart(quarter).semantics.time.grain, "MONTH")
        self.assertEqual((chart.semantics.measure.type, chart.filters), (value.measure.type, value.filters))


class ShowAnswerAsChartTests(unittest.TestCase):
    def test_the_latest_answer_is_read_back_as_planned(self) -> None:
        values = _latest_answer_values([_plan_event({"values": [_VALUE]})])
        self.assertEqual(values, AnalysisPlan.model_validate({"values": [_VALUE]}).values)

    def test_the_button_charts_the_latest_answer_per_quarter(self) -> None:
        result, ctx, _ = _prework([_plan_event({"values": [_VALUE]})])
        self.assertTrue(result.proceed)
        prepared = ctx.tracker_snapshot["_visualization_prepared_plan"]
        self.assertEqual([chart.semantics.time.grain for chart in prepared.charts], ["QUARTER"])

    def test_a_later_chart_leaves_no_answer_to_chart(self) -> None:
        result, _, dispatcher = _prework([_plan_event({"values": [_VALUE]}), _plan_event({"charts": [_CHART]})])
        self.assertFalse(result.proceed)
        self.assertEqual([message.get("text") for message in dispatcher.messages], ["There is no number answer in this conversation to show as a chart."])


if __name__ == "__main__":
    unittest.main()
