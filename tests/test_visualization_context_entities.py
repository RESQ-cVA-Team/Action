from typing import Any, Dict, List, Optional

from src.actions.actions import visualization_action as va
from src.actions.helpers.visualization import canonicalize_ssot_entities


def _user_event(text: str, entities: List[Dict[str, Any]], intent: str = "generate_visualization") -> Dict[str, Any]:
    return {"event": "user", "text": text, "parse_data": {"text": text, "intent": {"name": intent}, "entities": entities}}


def _latest(text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"text": text, "intent": {"name": "generate_visualization"}, "entities": entities}


def _decision(decision: str, message: Optional[str] = None, edits: Optional[bool] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": "visualization_query_decision", "decision": decision, "reason": "x", "message": message}
    if edits is not None:
        payload["edits_previous_plan"] = edits
    return {"event": "bot", "custom": payload}


def _plan(metric: str, sex: Optional[str] = None) -> Dict[str, Any]:
    chart: Dict[str, Any] = {"chart_type": "LINE", "metrics": [{"metric": metric}]}
    if sex:
        chart["filters"] = {"type": "SexFilter", "value": sex}
    return {"event": "bot", "custom": {"type": "visualization_plan", "plan": {"charts": [chart]}}}


FEMALE_DTN = [
    _user_event("Show me DTN for female patients as a line chart", [{"entity": "metric", "value": "DTN"}, {"entity": "sex", "value": "FEMALE"}]),
    _decision("proceed"),
    _plan("DTN", "FEMALE"),
]


def test_merge_request_entities_combines_the_earlier_turns() -> None:
    earlier = {"chart_type": "LINE", "metric": "DTN", "date": "past year"}
    merged = va._merge_request_entities({"metric": "DTN"}, earlier)
    assert merged["metric"] == "DTN"
    assert merged["chart_type"] == "LINE"
    assert merged["date"] == "past year"


def test_merge_request_entities_lets_a_restated_scope_replace_the_earlier_one() -> None:
    earlier = {"group_id": "provider group 2825", "date": "2023-01-01"}
    merged = va._merge_request_entities({"group_id": ["provider group 279"], "date": ["2024-01-01", "2026-12-31"]}, earlier)
    assert merged["group_id"] == ["provider group 279"]
    assert merged["date"] == ["2024-01-01", "2026-12-31"]


def test_canonicalize_ssot_entities_does_not_override_extracted_metric_from_question() -> None:
    assert canonicalize_ssot_entities({"chart_type": "LINE", "metric": "ICH_TREATMENT_TYPE"})["metric"] == "ICH_TREATMENT_TYPE"


def test_a_new_request_stands_alone() -> None:
    events = FEMALE_DTN + [_user_event("Show me a line graph of discharge NIHSS", [{"entity": "metric", "value": "DISCHARGE_NIHSS"}])]
    request = va._request_context(events, _latest("Show me a line graph of discharge NIHSS", [{"entity": "metric", "value": "DISCHARGE_NIHSS"}]), edits_previous_plan=False)
    assert request["latest_plan_summary"] is None
    assert "sex" not in request["extracted_entities"]
    assert request["conversation_history"] == []
    assert request["answers_clarification"] is False


def test_an_edit_builds_on_the_previous_plan_and_the_request_it_edits() -> None:
    events = FEMALE_DTN + [
        _user_event("Show me a line graph of discharge NIHSS", [{"entity": "metric", "value": "DISCHARGE_NIHSS"}]),
        _decision("proceed", edits=False),
        _plan("DISCHARGE_NIHSS"),
        _user_event("as a bar chart", [{"entity": "chart_type", "value": "BAR"}]),
    ]
    request = va._request_context(events, _latest("as a bar chart", [{"entity": "chart_type", "value": "BAR"}]), edits_previous_plan=True)
    assert request["planner_question"].startswith("Previous chart plan")
    assert "DISCHARGE_NIHSS" in request["planner_question"]
    # The edited request's entities carry over; the older chart's filter does not.
    assert request["extracted_entities"]["metric"] == "DISCHARGE_NIHSS"
    assert "sex" not in request["extracted_entities"]
    assert request["extracted_entities"]["chart_type"] == "BAR"


def test_a_chain_of_edits_inherits_back_to_the_request_it_started_from() -> None:
    events = FEMALE_DTN + [
        _user_event("as a bar chart", [{"entity": "chart_type", "value": "BAR"}]),
        _decision("proceed", edits=True),
        _plan("DTN", "FEMALE"),
        _user_event("per month instead", [{"entity": "group_by", "value": "MONTH"}]),
    ]
    request = va._request_context(events, _latest("per month instead", [{"entity": "group_by", "value": "MONTH"}]), edits_previous_plan=True)
    entities = request["extracted_entities"]
    assert entities["metric"] == "DTN"
    assert entities["sex"] == "FEMALE"
    assert entities["chart_type"] == "BAR"
    assert entities["group_by"] == "MONTH"


def test_a_clarification_reply_continues_the_request_that_was_asked_about() -> None:
    events = FEMALE_DTN + [
        _user_event("Show me nihss over time", [{"entity": "group_by", "value": "NIHSS"}]),
        _decision("clarify", "Which NIHSS metric would you like to see over time?", edits=False),
        _user_event("admission NIHSS", [{"entity": "metric", "value": "ADMISSION_NIHSS"}]),
    ]
    request = va._request_context(events, _latest("admission NIHSS", [{"entity": "metric", "value": "ADMISSION_NIHSS"}]), edits_previous_plan=True)
    question = request["planner_question"]
    assert question.startswith("Show me nihss over time")
    assert "The assistant asked: Which NIHSS metric would you like to see over time?" in question
    assert "The user answered:" in question
    # Not an edit, so neither the earlier chart's plan nor its filter carries over.
    assert request["latest_plan_summary"] is None
    assert "sex" not in request["extracted_entities"]
    assert request["extracted_entities"]["metric"] == "ADMISSION_NIHSS"
    assert request["answers_clarification"] is True
    assert request["pending_clarification"]["message"] == "Which NIHSS metric would you like to see over time?"
    assert request["conversation_history"] == ["Show me nihss over time", "admission NIHSS"]


def test_a_reply_to_a_clarified_edit_keeps_the_previous_plan() -> None:
    events = FEMALE_DTN + [
        _user_event("group by hospital", [{"entity": "group_by", "value": "HOSPITAL"}]),
        _decision("clarify", "Which hospitals should be compared?", edits=True),
        _user_event("Hospital X", [{"entity": "hospital_name", "value": "Hospital X"}]),
    ]
    request = va._request_context(events, _latest("Hospital X", [{"entity": "hospital_name", "value": "Hospital X"}]), edits_previous_plan=False)
    assert request["latest_plan_summary"] is not None
    assert request["planner_question"].startswith("Previous chart plan")
    assert request["edits_previous_plan"] is True


def test_a_button_reply_reads_as_the_value_it_chose() -> None:
    payload = '/generate_visualization{"metric":"ADMISSION_NIHSS"}'
    events = [
        _user_event("Show me nihss over time", [{"entity": "group_by", "value": "NIHSS"}]),
        _decision("clarify", "Which NIHSS metric?"),
        _user_event(payload, [{"entity": "metric", "value": "ADMISSION_NIHSS"}]),
    ]
    request = va._request_context(events, _latest(payload, [{"entity": "metric", "value": "ADMISSION_NIHSS"}]), edits_previous_plan=False)
    assert request["planner_question"].endswith("The user answered: ADMISSION_NIHSS")


def test_a_two_round_exchange_keeps_every_question_and_answer() -> None:
    events = [
        _user_event("compare dtn", [{"entity": "metric", "value": "DTN"}]),
        _decision("clarify", "Which cohorts?"),
        _user_event("my hospital vs Hospital X", [{"entity": "hospital_name", "value": "Hospital X"}]),
        _decision("clarify", "Which period?"),
        _user_event("2025", [{"entity": "date", "value": "2025"}]),
    ]
    exchange = va._clarification_exchange(events)
    assert [event["text"] for event, _ in exchange] == ["compare dtn", "my hospital vs Hospital X"]
    request = va._request_context(events, _latest("2025", [{"entity": "date", "value": "2025"}]), edits_previous_plan=False)
    question = request["planner_question"]
    assert question.index("Which cohorts?") < question.index("Which period?")
    assert request["extracted_entities"]["metric"] == "DTN"
    assert request["extracted_entities"]["hospital_name"] == "Hospital X"


def test_no_exchange_after_a_completed_request() -> None:
    events = FEMALE_DTN + [_user_event("as a bar chart", [{"entity": "chart_type", "value": "BAR"}])]
    assert va._clarification_exchange(events) == []


def test_a_proceed_decision_sent_for_the_current_message_is_found() -> None:
    message = _user_event("show me a line graph of dtn", [])
    assert va._proceed_decided_for_latest_message([message, _decision("proceed")]) == "x"
    assert va._proceed_decided_for_latest_message([message, _decision("clarify")]) is None
    # A decision before the current message belongs to an earlier one.
    assert va._proceed_decided_for_latest_message([_decision("proceed"), message]) is None
    assert va._proceed_decided_for_latest_message([message]) is None
