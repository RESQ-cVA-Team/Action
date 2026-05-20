from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore
from rasa_sdk.events import SlotSet  # type: ignore
from rasa_sdk.forms import FormValidationAction  # type: ignore

from src.actions.error_messages import visualization_error_payload
from src.actions.guided_visualization_validation import (
    SKIP_SENTINEL,
    DispatcherLike,
    TrackerLike,
    build_guided_plan,
    guided_slots_clear_events,
    is_skip_signal,
    validate_guided_hospital_scope,
    validate_optional_catalog_slot,
    validate_required_metric,
)
from src.actions.helpers.visualization import format_execution_summary, serialize_plan_for_frontend
from src.actions.i18n import resolve_language_from_tracker, translate
from src.executors import execute_plan_async
from src.util import env as env_util
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)
_SHOW_EXECUTION_SUMMARY = env_util.env_flag("ACTIONS_SHOW_EXECUTION_SUMMARY", default=True)
_SHOW_NORMALIZATION_SUMMARY = env_util.env_flag("ACTIONS_SHOW_NORMALIZATION_SUMMARY", default=True)

_executor_concurrency_raw = env_util.get_env("ACTIONS_EXECUTOR_MAX_CONCURRENCY", default="4") or "4"
try:
    _executor_max_concurrency = max(1, int(_executor_concurrency_raw))
except Exception:
    _executor_max_concurrency = 4
_EXECUTOR_MAX_CONCURRENCY = _executor_max_concurrency

DomainDict = Dict[str, Any]
RasaEventList = List[Any]


def _normalize_trace_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
    else:
        token = str(value).strip()
    return token or None


def _tracker_trace_id(tracker: TrackerLike) -> Optional[str]:
    latest = tracker.latest_message
    metadata_any = latest.get("metadata")
    metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}

    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        trace_id = _normalize_trace_id(metadata.get(key))
        if trace_id:
            return trace_id

    headers_any = metadata.get("headers")
    headers = cast(Dict[str, Any], headers_any) if isinstance(headers_any, dict) else {}
    for key in ("x-trace-id", "x_trace_id", "trace_id", "traceId"):
        trace_id = _normalize_trace_id(headers.get(key))
        if trace_id:
            return trace_id

    return None


class ActionGuidedGenerateVisualization(Action):  # pyright: ignore
    def name(self) -> Text:
        return "action_guided_generate_visualization"

    async def run(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> RasaEventList:
        trace_id = _tracker_trace_id(tracker) or uuid4().hex
        language = resolve_language_from_tracker(tracker)
        execution_summary: Any = None
        with log_context(trace_id=trace_id, sender_id=str(tracker.sender_id), action=self.name()):
            try:
                logger.info("Starting guided visualization generation")
                slots_any = tracker.current_state().get("slots", {})
                slots = cast(Dict[str, Any], slots_any) if isinstance(slots_any, dict) else {}
                user_sub = str(tracker.sender_id)

                def on_summary(summary: Any) -> None:
                    nonlocal execution_summary
                    execution_summary = summary

                plan_obj = build_guided_plan(slots=slots, user_sub=user_sub, trace_id=trace_id)
                dispatcher.utter_message(
                    json_message={
                        "type": "visualization_plan",
                        "trace_id": trace_id,
                        "plan": serialize_plan_for_frontend(plan_obj),
                    }
                )

                visualization = await execute_plan_async(
                    plan_obj,
                    user_sub=user_sub,
                    max_concurrency=_EXECUTOR_MAX_CONCURRENCY,
                    progress_cb=None,
                    summary_cb=on_summary,
                    trace_id=trace_id,
                )
                visualization_payload = visualization.model_dump(mode="json")
                dispatcher.utter_message(json_message=visualization_payload)

                warnings_any = visualization_payload.get("warnings")
                if isinstance(warnings_any, list):
                    for warning in cast(List[object], warnings_any):
                        if isinstance(warning, str) and warning.strip():
                            dispatcher.utter_message(text=f"Note: {warning.strip()}")

                if _SHOW_EXECUTION_SUMMARY and execution_summary is not None:
                    dispatcher.utter_message(
                        text=format_execution_summary(
                            execution_summary,
                            show_normalization=_SHOW_NORMALIZATION_SUMMARY,
                            planner_diagnostics=None,
                            language=language,
                        )
                    )
                else:
                    dispatcher.utter_message(text=translate("action.visualization.success_complete", language=language))

                return [SlotSet("awaiting_visualization_clarification", False), *guided_slots_clear_events()]
            except Exception as e:
                logger.exception("Error generating guided visualization")
                payload = visualization_error_payload(e, trace_id=trace_id, language=language)
                dispatcher.utter_message(
                    json_message={
                        "type": "visualization_error",
                        "trace_id": payload.get("trace_id"),
                        "error_code": payload.get("code"),
                        "reason": payload.get("reason"),
                        "message": payload.get("message"),
                    }
                )
                dispatcher.utter_message(
                    text=translate(
                        "action.common.error_with_context",
                        language=language,
                        params={
                            "message": payload.get("message") or "",
                            "code": payload.get("code") or "-",
                            "trace_id": payload.get("trace_id") or "-",
                        },
                    )
                )
                if _ECHO_INTERNAL_ERRORS:
                    dispatcher.utter_message(
                        text=translate(
                            "action.visualization.internal_error_guided",
                            language=language,
                            params={"error": str(e)},
                        )
                    )
                return [SlotSet("awaiting_visualization_clarification", False)]


class ValidateGuidedVisualizationForm(FormValidationAction):  # pyright: ignore
    def name(self) -> Text:
        return "validate_guided_visualization_form"

    def validate_metric(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return validate_required_metric(slot_value=slot_value, dispatcher=dispatcher, tracker=tracker)

    def validate_guided_hospital_scope(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return validate_guided_hospital_scope(slot_value=slot_value, dispatcher=dispatcher, tracker=tracker)

    def validate_stroke_type(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if is_skip_signal(slot_value=slot_value, tracker=tracker):
            return {"stroke_type": SKIP_SENTINEL}
        language = resolve_language_from_tracker(tracker)
        return validate_optional_catalog_slot(
            slot_name="stroke_type",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="StrokeType.yml",
            prompt=translate("action.guided.prompt_stroke_type", language=language),
        )

    def validate_sex(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if is_skip_signal(slot_value=slot_value, tracker=tracker):
            return {"sex": SKIP_SENTINEL}
        language = resolve_language_from_tracker(tracker)
        return validate_optional_catalog_slot(
            slot_name="sex",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="SexType.yml",
            prompt=translate("action.guided.prompt_sex", language=language),
        )

    def validate_group_by(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if is_skip_signal(slot_value=slot_value, tracker=tracker):
            return {"group_by": SKIP_SENTINEL}
        language = resolve_language_from_tracker(tracker)
        return validate_optional_catalog_slot(
            slot_name="group_by",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="GroupByType.yml",
            prompt=translate("action.guided.prompt_group_by", language=language),
        )

    def validate_chart_type(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if is_skip_signal(slot_value=slot_value, tracker=tracker):
            return {"chart_type": SKIP_SENTINEL}
        language = resolve_language_from_tracker(tracker)
        return validate_optional_catalog_slot(
            slot_name="chart_type",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="ChartType.yml",
            prompt=translate("action.guided.prompt_chart_type", language=language),
        )
