import asyncio
from typing import Any, Dict, List, Optional

from src.actions.actions.question_action import ActionAnswerQuestion, question_topic


def _message(*entities: tuple[str, str]) -> Dict[str, Any]:
    return {"text": "q", "intent": {"name": "ask_question"}, "entities": [{"entity": e, "value": v} for e, v in entities]}


class _Dispatcher:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None:
        self.sent.append({"text": text, **kwargs})


class _Tracker:
    sender_id = "s"

    def __init__(self, message: Dict[str, Any]) -> None:
        self.latest_message = message

    def get_slot(self, name: str) -> Any:
        return None


def _run(message: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Any]]:
    dispatcher = _Dispatcher()
    events = asyncio.run(ActionAnswerQuestion().run(dispatcher, _Tracker(message), {}))
    return dispatcher.sent, events


def test_a_topic_picks_its_catalogue() -> None:
    assert question_topic(_message(("topic", "chart_types"))) == "chart_types"


def test_a_named_value_picks_its_catalogue() -> None:
    assert question_topic(_message(("chart_type", "PIE"))) == "chart_types"
    assert question_topic(_message(("sex", "MALE"))) == "sex_types"


def test_a_question_about_a_metric_has_no_catalogue() -> None:
    assert question_topic(_message(("metric", "DTN"), ("group_by", "SEX"))) is None


def test_a_question_about_a_catalogue_metric_picks_its_catalogue() -> None:
    assert question_topic(_message(("metric", "SEX"))) == "sex_types"
    assert question_topic(_message(("metric", "STROKE_TYPE"))) == "stroke_types"


def test_a_catalogue_question_gets_its_faq_answer() -> None:
    sent, events = _run(_message(("topic", "operator_types")))
    assert [m.get("response") for m in sent] == ["utter_faq_operator_types", "utter_faq_again"]
    assert events == []


def test_a_question_about_a_metric_gets_its_definition() -> None:
    sent, _ = _run(_message(("metric", "DTN")))
    assert len(sent) == 1 and sent[0]["text"].startswith("DTN")


def test_a_question_naming_nothing_gets_the_ssot_overview() -> None:
    sent, _ = _run(_message())
    assert [m.get("response") for m in sent] == ["utter_faq_ssot_help", "utter_faq_again"]


def test_a_question_naming_a_metric_and_a_chart_type_asks_for_the_chart() -> None:
    sent, events = _run(_message(("metric", "DTN"), ("chart_type", "LINE")))
    assert sent == []
    assert events == [{"event": "followup", "timestamp": None, "name": "action_clarify_visualization_request"}]
