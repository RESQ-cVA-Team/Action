import logging
from typing import Any, Dict, List, Optional, Text, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore
from rasa_sdk.events import FollowupAction  # type: ignore

from src.actions.actions.metric_action import DispatcherLike, TrackerLike, explain_metric
from src.actions.helpers.metric import resolve_language
from src.actions.helpers.visualization import names_metric_and_chart_type
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

DomainDict = Dict[str, Any]
RasaEventList = List[Dict[Text, Any]]

_TOPIC_RESPONSES = {
    "ssot": "utter_faq_ssot_help",
    "metric_types": "utter_faq_metric_types",
    "stroke_types": "utter_faq_stroke_types",
    "chart_types": "utter_faq_chart_types",
    "group_by_types": "utter_faq_group_by_types",
    "statistical_test_types": "utter_faq_statistical_test_types",
    "operator_types": "utter_faq_operator_types",
    "sex_types": "utter_faq_sex_types",
    "boolean_types": "utter_faq_boolean_types",
}

# A question naming a value of one of these, and no topic, asks about its catalogue.
_ENTITY_TOPICS = {
    "chart_type": "chart_types",
    "sex": "sex_types",
    "stroke_type": "stroke_types",
    "group_by": "group_by_types",
    "statistical_test_type": "statistical_test_types",
    "boolean_type": "boolean_types",
}

# Metrics whose values are a catalogue: a question about one asks for that list.
_METRIC_TOPICS = {
    "SEX": "sex_types",
    "STROKE_TYPE": "stroke_types",
}


def _entity_values(latest_message: Dict[str, Any]) -> Dict[str, str]:
    """The first value of each entity in the message."""
    values: Dict[str, str] = {}
    for entity_any in cast(List[Any], latest_message.get("entities") or []):
        if not isinstance(entity_any, dict):
            continue
        entity = cast(Dict[str, Any], entity_any)
        name, value = entity.get("entity"), entity.get("value")
        if isinstance(name, str) and isinstance(value, str) and value.strip():
            values.setdefault(name, value.strip())
    return values


def question_topic(latest_message: Dict[str, Any]) -> Optional[str]:
    """The catalogue a question asks about: its topic, else the catalogue of a
    value it names, or of a metric whose values are a catalogue. None for a
    question about any other metric or about nothing named."""
    values = _entity_values(latest_message)
    topic = values.get("topic")
    if topic in _TOPIC_RESPONSES:
        return topic
    if "metric" in values:
        return _METRIC_TOPICS.get(values["metric"].upper())
    return next((topic for entity, topic in _ENTITY_TOPICS.items() if entity in values), None)


class ActionAnswerQuestion(Action):  # pyright: ignore
    """Answers ask_question: a catalogue question with its FAQ response, a
    question about a metric with the metric's SSOT description, anything else
    with the SSOT overview. A question that names a metric and a chart type
    asks for that chart and goes to the visualization triage."""

    def name(self) -> str:
        return "action_answer_question"

    async def run(self, dispatcher: DispatcherLike, tracker: TrackerLike, domain: DomainDict) -> RasaEventList:
        latest = tracker.latest_message or {}
        with log_context(trace_id=uuid4().hex, sender_id=str(tracker.sender_id), action=self.name()):
            if names_metric_and_chart_type(latest):
                logger.info("Question names a metric and a chart type; handing over to the visualization triage")
                return [FollowupAction("action_clarify_visualization_request")]

            topic = question_topic(latest)
            if topic is not None:
                dispatcher.utter_message(response=_TOPIC_RESPONSES[topic])
                dispatcher.utter_message(response="utter_faq_again")
                return []

            metric = _entity_values(latest).get("metric")
            if metric:
                explain_metric(dispatcher, metric, resolve_language(tracker))
                return []

            dispatcher.utter_message(response=_TOPIC_RESPONSES["ssot"])
            dispatcher.utter_message(response="utter_faq_again")
            return []
