from __future__ import annotations

import json
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
from src.actions.utils.visualization import format_execution_summary
from src.executors import execute_plan_async
from src.shared.ssot_loader import get_canonical_values
from src.util import env as env_util

logger = logging.getLogger(__name__)

_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)
_SHOW_EXECUTION_SUMMARY = env_util.env_flag(
    "ACTIONS_SHOW_EXECUTION_SUMMARY", default=True
)
_SHOW_NORMALIZATION_SUMMARY = env_util.env_flag(
    "ACTIONS_SHOW_NORMALIZATION_SUMMARY", default=True
)

_executor_concurrency_raw = (
    env_util.get_env("ACTIONS_EXECUTOR_MAX_CONCURRENCY", default="4") or "4"
)
try:
    _executor_max_concurrency = max(1, int(_executor_concurrency_raw))
except Exception:
    _executor_max_concurrency = 4
_EXECUTOR_MAX_CONCURRENCY = _executor_max_concurrency

DomainDict = Dict[str, Any]
RasaEventList = List[Any]


def _build_button_payload(
    prompt: str,
    slot_name: str,
    button_options: List[str],
    allow_skip: bool = True,
) -> Dict[str, Any]:
    """Build a JSON payload with buttons for slot prompts."""
    buttons = [{"title": option, "payload": option} for option in button_options]
    if allow_skip:
        buttons.append({"title": "Skip", "payload": "/skip_guided_step"})

    return {
        "type": "slot_prompt_with_buttons",
        "slot_name": slot_name,
        "text": prompt,
        "buttons": buttons,
    }


def _normalize_trace_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
    else:
        token = str(value).strip()
    return token or None


def _tracker_trace_id(tracker: TrackerLike) -> Optional[str]:
    latest = tracker.latest_message if isinstance(tracker.latest_message, dict) else {}
    metadata_any = latest.get("metadata")
    metadata = (
        cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
    )

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
        execution_summary: Any = None
        try:
            logger.info(
                "Starting guided visualization generation (trace_id=%s)", trace_id
            )
            slots_any = tracker.current_state().get("slots", {})
            slots = (
                cast(Dict[str, Any], slots_any) if isinstance(slots_any, dict) else {}
            )
            user_sub = str(tracker.sender_id)

            def on_summary(summary: Any) -> None:
                nonlocal execution_summary
                execution_summary = summary

            plan_obj = build_guided_plan(
                slots=slots, user_sub=user_sub, trace_id=trace_id
            )
            dispatcher.utter_message(
                json_message={
                    "type": "visualization_plan",
                    "trace_id": trace_id,
                    "plan": cast(Any, plan_obj).model_dump(
                        mode="json", exclude_none=True
                    ),
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
            visualization_json = cast(Any, visualization).model_dump_json()
            dispatcher.utter_message(
                json_message=json.loads(cast(str, visualization_json))
            )
            if _SHOW_EXECUTION_SUMMARY and execution_summary is not None:
                dispatcher.utter_message(
                    text=format_execution_summary(
                        execution_summary,
                        show_normalization=_SHOW_NORMALIZATION_SUMMARY,
                        planner_diagnostics=None,
                    )
                )
            else:
                dispatcher.utter_message(text="✅ Visualization generation complete.")

            return [
                SlotSet("awaiting_visualization_clarification", False),
                *guided_slots_clear_events(),
            ]
        except Exception as e:
            logger.exception(
                "Error generating guided visualization (trace_id=%s)", trace_id
            )
            payload = visualization_error_payload(e, trace_id=trace_id)
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
                text=f"❌ {payload.get('message')} (Error code: {payload.get('code')}, Trace ID: {payload.get('trace_id')})"
            )
            if _ECHO_INTERNAL_ERRORS:
                dispatcher.utter_message(
                    text=f"Error generating guided visualization: {str(e)}"
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
        return validate_required_metric(
            slot_value=slot_value, dispatcher=dispatcher, tracker=tracker
        )

    def validate_guided_hospital_scope(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        return validate_guided_hospital_scope(
            slot_value=slot_value, dispatcher=dispatcher, tracker=tracker
        )

    def validate_stroke_type(
        self,
        slot_value: Any,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if is_skip_signal(slot_value=slot_value, tracker=tracker):
            return {"stroke_type": SKIP_SENTINEL}
        return validate_optional_catalog_slot(
            slot_name="stroke_type",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="StrokeType.yml",
            prompt="I couldn't match that stroke type. Please enter a valid stroke type or skip.",
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
        return validate_optional_catalog_slot(
            slot_name="sex",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="SexType.yml",
            prompt="I couldn't match that sex value. Please enter a valid value or skip.",
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
        return validate_optional_catalog_slot(
            slot_name="group_by",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="GroupByType.yml",
            prompt="I couldn't match that grouping. Please enter a valid grouping field or skip.",
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
        return validate_optional_catalog_slot(
            slot_name="chart_type",
            slot_value=slot_value,
            dispatcher=dispatcher,
            tracker=tracker,
            filename="ChartType.yml",
            prompt="I couldn't match that chart type. Please enter a valid chart type or skip.",
        )

    def ask_for_sex(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> None:
        """Ask for sex filter with button options."""
        try:
            sex_options = list(get_canonical_values("SexType.yml"))
            payload = _build_button_payload(
                prompt="Would you like to filter by patient sex?",
                slot_name="sex",
                button_options=sex_options,
            )
            dispatcher.utter_message(json_message=payload)
        except Exception as e:
            logger.warning(
                "Failed to build sex buttons, falling back to text: %s", str(e)
            )
            dispatcher.utter_message(
                text="Would you like to filter by patient sex? (or type 'skip')"
            )

    def ask_for_stroke_type(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> None:
        """Ask for stroke type filter with button options."""
        try:
            stroke_options = list(get_canonical_values("StrokeType.yml"))
            payload = _build_button_payload(
                prompt="Would you like to filter by stroke type?",
                slot_name="stroke_type",
                button_options=stroke_options,
            )
            dispatcher.utter_message(json_message=payload)
        except Exception as e:
            logger.warning(
                "Failed to build stroke type buttons, falling back to text: %s", str(e)
            )
            dispatcher.utter_message(
                text="Would you like to filter by stroke type? (or type 'skip')"
            )

    def ask_for_group_by(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> None:
        """Ask for grouping field with button options."""
        try:
            group_by_options = list(get_canonical_values("GroupByType.yml"))
            payload = _build_button_payload(
                prompt="Would you like to group the results by any field?",
                slot_name="group_by",
                button_options=group_by_options,
            )
            dispatcher.utter_message(json_message=payload)
        except Exception as e:
            logger.warning(
                "Failed to build group by buttons, falling back to text: %s", str(e)
            )
            dispatcher.utter_message(
                text="Would you like to group the results by any field? (or type 'skip')"
            )

    def ask_for_chart_type(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> None:
        """Ask for chart type with button options."""
        try:
            chart_options = list(get_canonical_values("ChartType.yml"))
            payload = _build_button_payload(
                prompt="What type of chart would you like to visualize?",
                slot_name="chart_type",
                button_options=chart_options,
            )
            dispatcher.utter_message(json_message=payload)
        except Exception as e:
            logger.warning(
                "Failed to build chart type buttons, falling back to text: %s", str(e)
            )
            dispatcher.utter_message(
                text="What type of chart would you like to visualize? (or type 'skip')"
            )
