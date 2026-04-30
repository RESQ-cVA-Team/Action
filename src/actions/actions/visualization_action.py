import json
import logging
from typing import Any, Dict, List, Optional, Protocol, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore
from rasa_sdk.events import FollowupAction, SlotSet  # type: ignore

from src.actions.error_messages import visualization_error_payload
from src.actions.helpers.visualization import (
    extract_entities_from_latest_message,
    format_execution_summary,
    resolve_override_language,
    serialize_plan_for_frontend,
)
from src.actions.i18n import resolve_language, translate
from src.actions.long_action.long_action import LongAction, PreworkResult
from src.actions.long_action.long_action_context import LongActionContext
from src.domain.langchain import schema as lang_schema
from src.executors import execute_plan_async
from src.executors.orchestration.plan_executor import VisualizationExecutionError
from src.planners.langchain import pipeline as lang_pipeline
from src.planners.langchain.request_orchestrator import orchestrate_visualization_request
from src.util import env as env_util

logger = logging.getLogger(__name__)

_LOG_USER_TEXT = env_util.env_flag("ACTIONS_LOG_USER_TEXT", default=False)
_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)
_SHOW_EXECUTION_SUMMARY = env_util.env_flag("ACTIONS_SHOW_EXECUTION_SUMMARY", default=True)
_DEFER_CALLBACK_HANDOFF = env_util.env_flag("LONG_ACTION_DEFER_CALLBACK_HANDOFF", default=False)
_SHOW_NORMALIZATION_SUMMARY = env_util.env_flag("ACTIONS_SHOW_NORMALIZATION_SUMMARY", default=True)
_VISUALIZATION_REQUEST_INTENTS = {"generate_visualization", "update_visualization"}
_VISUALIZATION_CONTINUATION_INTENTS = {"generate_visualization", "update_visualization", "clarify_visualization"}
_VISUALIZATION_THREAD_INTENTS = {
    "generate_visualization",
    "update_visualization",
    "clarify_visualization",
}
_VISUALIZATION_PLAN_TYPE = "visualization_plan"
_VISUALIZATION_RESPONSE_SCHEMA_VERSION = 1

_PLANNER_MAX_RETRIES = 2
_EXECUTOR_MAX_CONCURRENCY = 4

DomainDict = Dict[str, Any]
RasaEventList = List[Any]


class DispatcherLike(Protocol):
    def utter_message(
        self,
        text: Optional[str] = None,
        json_message: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]
    events: List[Dict[str, Any]]

    def current_state(self) -> Dict[str, Any]: ...


def _extract_intent_name_from_user_event(event: Dict[str, Any]) -> str:
    parse_data_any = event.get("parse_data")
    parse_data = cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else {}

    intent_any = parse_data.get("intent")
    intent = cast(Dict[str, Any], intent_any) if isinstance(intent_any, dict) else {}
    name_any = intent.get("name")
    if isinstance(name_any, str) and name_any.strip():
        return name_any.strip()

    fallback_intent_any = event.get("intent")
    fallback_intent = cast(Dict[str, Any], fallback_intent_any) if isinstance(fallback_intent_any, dict) else {}
    fallback_name_any = fallback_intent.get("name")
    if isinstance(fallback_name_any, str) and fallback_name_any.strip():
        return fallback_name_any.strip()

    metadata_any = event.get("metadata")
    metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
    metadata_intent_any = metadata.get("intentName")
    if isinstance(metadata_intent_any, str) and metadata_intent_any.strip():
        return metadata_intent_any.strip()

    return ""


def _collect_recent_user_messages(events: List[Dict[str, Any]], fallback_limit: int) -> List[str]:
    messages: List[str] = []
    for ev in events:
        if ev.get("event") != "user":
            continue
        text_any = ev.get("text")
        if isinstance(text_any, str) and text_any.strip():
            messages.append(text_any.strip())

    if len(messages) > fallback_limit:
        return messages[-fallback_limit:]
    return messages


def _collect_visualization_thread_messages(events: List[Dict[str, Any]], fallback_limit: int = 12) -> List[str]:
    """Return the current visualization conversation thread from tracker events.

    We anchor to the latest user turn with an explicit non-visualization intent,
    then keep subsequent user turns. This preserves context across
    generate_visualization -> clarify_visualization cycles and avoids repeatedly
    asking for fields that were already provided.
    """

    recent_messages = _collect_recent_user_messages(events, fallback_limit=fallback_limit)
    if not events:
        return recent_messages

    user_events: List[tuple[str, str]] = []
    for ev in events:
        if ev.get("event") != "user":
            continue
        text_any = ev.get("text")
        if not isinstance(text_any, str) or not text_any.strip():
            continue
        user_events.append((text_any.strip(), _extract_intent_name_from_user_event(ev)))

    if not user_events:
        return recent_messages

    anchor_user_idx = _find_latest_visualization_anchor_user_ordinal(events)
    if anchor_user_idx < 0:
        return recent_messages

    thread_slice = user_events[anchor_user_idx:]
    thread_messages = [text for text, _ in thread_slice]
    if len(thread_messages) > fallback_limit:
        thread_messages = thread_messages[-fallback_limit:]
    return thread_messages or recent_messages


def _extract_bot_custom_payload(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    custom_any = event.get("custom")
    if isinstance(custom_any, dict):
        return cast(Dict[str, Any], custom_any)

    data_any = event.get("data")
    data = cast(Dict[str, Any], data_any) if isinstance(data_any, dict) else {}
    nested_custom_any = data.get("custom")
    if isinstance(nested_custom_any, dict):
        return cast(Dict[str, Any], nested_custom_any)

    return None


def _is_visualization_payload(payload: Dict[str, Any]) -> bool:
    payload_type_any = payload.get("type")
    if isinstance(payload_type_any, str) and payload_type_any.strip() == _VISUALIZATION_PLAN_TYPE:
        return True

    schema_version_any = payload.get("schema_version")
    charts_any = payload.get("charts")
    return schema_version_any == _VISUALIZATION_RESPONSE_SCHEMA_VERSION and isinstance(charts_any, list)


def _event_has_visualization_signal(event: Dict[str, Any]) -> bool:
    event_name_any = event.get("event")
    event_name = event_name_any.strip() if isinstance(event_name_any, str) else ""
    if event_name == "user":
        return _extract_intent_name_from_user_event(event) in _VISUALIZATION_THREAD_INTENTS

    if event_name == "bot":
        payload = _extract_bot_custom_payload(event)
        if payload is not None and _is_visualization_payload(payload):
            return True

    return False


def _find_latest_visualization_anchor_user_ordinal(events: List[Dict[str, Any]]) -> int:
    signal_idx = -1
    for idx in range(len(events) - 1, -1, -1):
        if _event_has_visualization_signal(events[idx]):
            signal_idx = idx
            break

    if signal_idx < 0:
        return -1

    user_ordinal = -1
    for idx, event in enumerate(events):
        if event.get("event") == "user":
            user_ordinal += 1
        if idx == signal_idx:
            break

    for idx in range(signal_idx, -1, -1):
        ev = events[idx]
        if ev.get("event") != "user":
            continue

        text_any = ev.get("text")
        if isinstance(text_any, str) and text_any.strip():
            return user_ordinal

        user_ordinal -= 1

    return -1


def _collect_latest_visualization_plan_summary(events: List[Dict[str, Any]]) -> Optional[str]:
    for idx in range(len(events) - 1, -1, -1):
        event = events[idx]
        if event.get("event") != "bot":
            continue

        payload = _extract_bot_custom_payload(event)
        if not isinstance(payload, dict):
            continue

        payload_type_any = payload.get("type")
        payload_type = payload_type_any.strip() if isinstance(payload_type_any, str) else ""
        if payload_type != _VISUALIZATION_PLAN_TYPE:
            continue

        plan_any = payload.get("plan")
        plan = cast(Dict[str, Any], plan_any) if isinstance(plan_any, dict) else {}

        charts_any = plan.get("charts")
        charts_list = cast(List[Any], charts_any) if isinstance(charts_any, list) else []
        chart_count = len(charts_list)

        statistical_tests_any = plan.get("statistical_tests")
        statistical_tests_list = cast(List[Any], statistical_tests_any) if isinstance(statistical_tests_any, list) else []
        stats_count = len(statistical_tests_list)

        trace_id_any = payload.get("trace_id")
        trace_id = trace_id_any.strip() if isinstance(trace_id_any, str) and trace_id_any.strip() else "unknown"

        compact_plan_json: str
        try:
            compact_plan_json = json.dumps(plan, ensure_ascii=False)
        except Exception:
            compact_plan_json = "{}"

        return f"Latest visualization plan context (trace_id={trace_id}, charts={chart_count}, statistical_tests={stats_count}):\n{compact_plan_json}"

    return None


class ActionClarifyVisualizationRequest(Action):  # pyright: ignore
    def name(self) -> str:
        return "action_clarify_visualization_request"

    async def run(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> RasaEventList:
        trace_id = uuid4().hex
        metadata: Dict[str, Any] = {}
        slots: Dict[str, Any] = {}
        try:
            fallback_limit = 12
            latest_msg = tracker.latest_message

            user_message_any = latest_msg.get("text")
            user_message = user_message_any if isinstance(user_message_any, str) else ""
            extracted_entities = extract_entities_from_latest_message(latest_msg)

            metadata_any = latest_msg.get("metadata")
            metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
            trace_id = _trace_id_from_metadata(metadata) or trace_id
            logger.info("Starting visualization clarification routing (trace_id=%s)", trace_id)
            slots_any = tracker.current_state().get("slots", {})
            slots = cast(Dict[str, Any], slots_any) if isinstance(slots_any, dict) else {}
            override_language = resolve_override_language(metadata, slots)
            language = resolve_language(metadata=metadata, slots=slots, tracker=tracker)

            events = tracker.events
            conversation_history = _collect_visualization_thread_messages(events, fallback_limit=fallback_limit)

            planner_question = "\n".join([m for m in conversation_history if m.strip()]).strip() or user_message

            outcome = orchestrate_visualization_request(
                question=planner_question,
                entities=extracted_entities,
                language=override_language,
                trace_id=trace_id,
                max_retries=_PLANNER_MAX_RETRIES,
                include_plan=False,
                conversation_history=conversation_history,
                progress_cb=None,
            )

            dispatcher.utter_message(
                json_message={
                    "type": "visualization_query_decision",
                    "trace_id": trace_id,
                    "decision": outcome.decision,
                    "reason": outcome.reason,
                    "clarification_type": outcome.clarification_type,
                    "clarification_options": outcome.clarification_options,
                    "message": outcome.message,
                    "missing_fields": outcome.missing_fields,
                }
            )

            if outcome.decision == "clarify":
                dispatcher.utter_message(text=outcome.message or translate("action.visualization.clarify_default", language=language))
                return [SlotSet("awaiting_visualization_clarification", True)]

            if outcome.decision == "reject":
                dispatcher.utter_message(text=outcome.message or translate("action.visualization.reject_default", language=language))
                return [SlotSet("awaiting_visualization_clarification", False)]

            return [
                SlotSet("awaiting_visualization_clarification", False),
                FollowupAction("action_oneshot_generate_visualization"),
            ]
        except Exception as e:
            logger.exception("Error routing visualization request (trace_id=%s)", trace_id)
            language = resolve_language(metadata=metadata, slots=slots, tracker=tracker)
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
                        "action.visualization.internal_error_routing",
                        language=language,
                        params={"error": str(e)},
                    )
                )
            return []


def _extract_request_context(ctx: LongActionContext) -> Dict[str, Any]:
    latest_meta = ctx.metadata
    latest_any = ctx.tracker_snapshot.get("latest_message")
    latest_msg = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}
    extracted_entities = extract_entities_from_latest_message(latest_msg)
    override_language = resolve_override_language(latest_meta, ctx.slots)
    language = resolve_language(metadata=latest_meta, slots=ctx.slots)
    events = ctx.events
    conversation_history = _collect_visualization_thread_messages(events, fallback_limit=12)
    latest_plan_summary = _collect_latest_visualization_plan_summary(events)

    planner_question = "\n".join([m for m in conversation_history if m.strip()]).strip() or ctx.text
    if latest_plan_summary:
        planner_question = (f"{latest_plan_summary}\n\nConversation context (oldest to newest user turns):\n{planner_question}").strip()

    update_target_trace_id_any = latest_meta.get("update_target_trace_id")
    update_target_trace_id = update_target_trace_id_any.strip() if isinstance(update_target_trace_id_any, str) and update_target_trace_id_any.strip() else None

    return {
        "user_message": ctx.text,
        "planner_question": planner_question,
        "user_sub": ctx.sender_id,
        "latest_meta": latest_meta,
        "latest_msg": latest_msg,
        "extracted_entities": extracted_entities,
        "override_language": override_language,
        "language": language,
        "conversation_history": conversation_history,
        "latest_plan_summary": latest_plan_summary,
        "update_target_trace_id": update_target_trace_id,
    }


_INTERNAL_TRACE_ID_KEY = "_visualization_trace_id"
_INTERNAL_PREPARED_PLAN_KEY = "_visualization_prepared_plan"
_INTERNAL_PLANNER_DIAGNOSTICS_KEY = "_visualization_planner_diagnostics"


def _normalize_trace_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
    else:
        token = str(value).strip()
    return token or None


def _trace_id_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
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


def _ensure_context_trace_id(ctx: LongActionContext) -> str:
    existing = ctx.tracker_snapshot.get(_INTERNAL_TRACE_ID_KEY)
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    metadata_trace_id = _trace_id_from_metadata(ctx.metadata)
    if metadata_trace_id:
        ctx.tracker_snapshot[_INTERNAL_TRACE_ID_KEY] = metadata_trace_id
        return metadata_trace_id

    generated = uuid4().hex
    ctx.tracker_snapshot[_INTERNAL_TRACE_ID_KEY] = generated
    return generated


def _is_guided_visualization_request(slots: Dict[str, Any]) -> bool:
    return slots.get("guided_hospital_scope") is not None


class ActionOneShotGenerateVisualization(LongAction):
    """Freeform one-shot visualization action backed by the planner chain.

    In callback mode this streams messages via the long-action callback URL;
    otherwise it behaves like a normal synchronous action and uses the
    dispatcher directly.
    """

    def name(self) -> str:
        return "action_oneshot_generate_visualization"

    async def prework(self, ctx: LongActionContext) -> PreworkResult:
        trace_id = _ensure_context_trace_id(ctx)
        logger.info("Starting one-shot prework (trace_id=%s)", trace_id)
        try:
            if _is_guided_visualization_request(ctx.slots):
                return PreworkResult(
                    events=[
                        SlotSet("awaiting_visualization_clarification", False),
                        FollowupAction("action_guided_generate_visualization"),
                    ],
                    proceed=False,
                )

            latest_any = ctx.tracker_snapshot.get("latest_message")
            latest_msg = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}
            parse_data_any = latest_msg.get("parse_data")
            parse_data = cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else {}
            intent_any = parse_data.get("intent")
            intent_obj = cast(Dict[str, Any], intent_any) if isinstance(intent_any, dict) else {}
            intent_name_any = intent_obj.get("name")

            if not isinstance(intent_name_any, str) or not intent_name_any.strip():
                metadata_any = latest_msg.get("metadata")
                metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
                metadata_intent_any = metadata.get("intentName")
                if isinstance(metadata_intent_any, str) and metadata_intent_any.strip():
                    intent_name_any = metadata_intent_any

            intent_name = intent_name_any.strip() if isinstance(intent_name_any, str) else ""
            awaiting_clarification = bool(ctx.slots.get("awaiting_visualization_clarification"))

            if awaiting_clarification and intent_name not in _VISUALIZATION_CONTINUATION_INTENTS:
                return PreworkResult(events=[FollowupAction("action_clarify_visualization_request")], proceed=False)

            # Defensive fallback: if routing reaches this action for an unrelated
            # intent, always send a user-facing response instead of returning
            # nothing and leaving the conversation hanging.
            if intent_name and intent_name not in _VISUALIZATION_CONTINUATION_INTENTS and not _is_guided_visualization_request(ctx.slots):
                language = resolve_language(metadata=ctx.metadata, slots=ctx.slots)
                ctx.say(text=translate("action.visualization.non_visualization_intent", language=language))
                return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=False)

            request_ctx = _extract_request_context(ctx)
            outcome = orchestrate_visualization_request(
                question=str(request_ctx.get("planner_question") or request_ctx.get("user_message") or ""),
                entities=cast(Dict[str, Any], request_ctx.get("extracted_entities") or {}),
                language=cast(Optional[str], request_ctx.get("override_language")),
                trace_id=trace_id,
                max_retries=_PLANNER_MAX_RETRIES,
                include_plan=False,
                conversation_history=cast(List[str], request_ctx.get("conversation_history") or []),
                progress_cb=None,
            )
            decision_name = str(outcome.decision or "").strip().lower()
            language = cast(str, request_ctx.get("language") or "en")
            if decision_name == "clarify":
                ctx.say(
                    json_message={
                        "type": "visualization_query_decision",
                        "trace_id": trace_id,
                        "decision": outcome.decision,
                        "reason": outcome.reason,
                        "clarification_type": outcome.clarification_type,
                        "clarification_options": outcome.clarification_options,
                        "message": outcome.message,
                    }
                )
                ctx.say(text=outcome.message or translate("action.visualization.clarify_default", language=language))
                return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", True)], proceed=False)
            if decision_name == "reject":
                ctx.say(
                    json_message={
                        "type": "visualization_query_decision",
                        "trace_id": trace_id,
                        "decision": outcome.decision,
                        "reason": outcome.reason,
                        "clarification_type": outcome.clarification_type,
                        "clarification_options": outcome.clarification_options,
                        "message": outcome.message,
                    }
                )
                ctx.say(text=outcome.message or translate("action.visualization.reject_default", language=language))
                return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=False)

            planner_question = str(request_ctx.get("planner_question") or request_ctx.get("user_message") or "")
            extracted_entities = cast(Dict[str, Any], request_ctx.get("extracted_entities") or {})
            override_language = cast(Optional[str], request_ctx.get("override_language"))

            prepared_plan = lang_pipeline.generate_analysis_plan(
                question=planner_question,
                entities=extracted_entities,
                language=override_language,
                max_retries=_PLANNER_MAX_RETRIES,
                debug=False,
                trace_id=trace_id,
                progress_cb=None,
            )

            ctx.tracker_snapshot[_INTERNAL_PREPARED_PLAN_KEY] = prepared_plan
            ctx.tracker_snapshot[_INTERNAL_PLANNER_DIAGNOSTICS_KEY] = lang_pipeline.get_plan_cache_diagnostics()

            ctx.say(
                json_message={
                    "type": "visualization_query_decision",
                    "trace_id": trace_id,
                    "decision": outcome.decision,
                    "reason": outcome.reason,
                    "clarification_type": outcome.clarification_type,
                    "clarification_options": outcome.clarification_options,
                    "message": outcome.message,
                }
            )
            return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=True)
        except Exception as e:
            logger.exception("Error generating visualization (prework, trace_id=%s)", trace_id)
            language = resolve_language(metadata=ctx.metadata, slots=ctx.slots)
            payload = visualization_error_payload(e, trace_id=trace_id, language=language)
            ctx.say(
                json_message={
                    "type": "visualization_error",
                    "trace_id": payload.get("trace_id"),
                    "error_code": payload.get("code"),
                    "reason": payload.get("reason"),
                    "message": payload.get("message"),
                }
            )
            ctx.say(
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
                ctx.say(
                    text=translate(
                        "action.visualization.internal_error_generating",
                        language=language,
                        params={"error": str(e)},
                    )
                )
            return PreworkResult(events=[], proceed=False)

    async def work(self, ctx: LongActionContext) -> Any:
        completed_successfully = False
        execution_summary: Optional[Any] = None
        planner_diagnostics: Optional[Dict[str, Any]] = None
        trace_id = _ensure_context_trace_id(ctx)
        language = resolve_language(metadata=ctx.metadata, slots=ctx.slots)
        try:
            request_ctx = _extract_request_context(ctx)
            user_message = cast(str, request_ctx["user_message"])
            user_sub = cast(str, request_ctx["user_sub"])
            language = cast(str, request_ctx.get("language") or "en")

            if _LOG_USER_TEXT:
                logger.info("Processing visualization request (trace_id=%s): '%s'", trace_id, user_message)
            else:
                logger.info(
                    "Processing visualization request (trace_id=%s, text_len=%s)",
                    trace_id,
                    len(user_message or ""),
                )

            def progress(msg: str) -> None:
                ctx.say(progress=msg)

            def on_summary(summary: Any) -> None:
                nonlocal execution_summary
                execution_summary = summary

            # In deferred-handoff mode, initial routing/clarification can use
            # normal dispatcher delivery and heavy generation streams via
            # callback after this explicit handoff.
            if _DEFER_CALLBACK_HANDOFF and not ctx.callback_mode_enabled:
                ctx.enable_callback_mode()

            plan_obj: lang_schema.AnalysisPlan
            prepared_any = ctx.tracker_snapshot.pop(_INTERNAL_PREPARED_PLAN_KEY, None)
            diagnostics_any = ctx.tracker_snapshot.pop(_INTERNAL_PLANNER_DIAGNOSTICS_KEY, None)

            if isinstance(prepared_any, lang_schema.AnalysisPlan):
                plan_obj = prepared_any
                planner_diagnostics = cast(Optional[Dict[str, Any]], diagnostics_any) if isinstance(diagnostics_any, dict) else None
                progress("Using prepared plan from prework")
            else:
                logger.warning(
                    "Prepared plan missing in work fallback; regenerating plan in work (trace_id=%s)",
                    trace_id,
                )
                planner_question = cast(str, request_ctx.get("planner_question") or user_message)
                extracted_entities = cast(Dict[str, Any], request_ctx["extracted_entities"])
                override_language = cast(Optional[str], request_ctx["override_language"])

                progress("Calling planner LLM to build a plan")
                plan_obj = lang_pipeline.generate_analysis_plan(
                    question=planner_question,
                    entities=extracted_entities,
                    language=override_language,
                    max_retries=_PLANNER_MAX_RETRIES,
                    debug=False,
                    trace_id=trace_id,
                    progress_cb=progress,
                )
                planner_diagnostics = lang_pipeline.get_plan_cache_diagnostics()

            ctx.say(
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
                progress_cb=progress,
                summary_cb=on_summary,
                trace_id=trace_id,
            )
            visualization_json = cast(Any, visualization).model_dump_json()
            ctx.say(json_message=json.loads(cast(str, visualization_json)))
            completed_successfully = True
        except Exception as e:
            if isinstance(e, VisualizationExecutionError) and e.reason == "origin_scope_resolution":
                ctx.say(
                    json_message={
                        "type": "visualization_query_decision",
                        "trace_id": trace_id,
                        "decision": "clarify",
                        "reason": e.reason,
                        "clarification_type": e.clarification_type,
                        "clarification_options": e.clarification_options,
                        "message": e.user_message,
                    }
                )
                ctx.say(text=e.user_message)
                return None

            logger.exception("Error generating visualization (trace_id=%s)", trace_id)
            payload = visualization_error_payload(e, trace_id=trace_id, language=language)
            ctx.say(
                json_message={
                    "type": "visualization_error",
                    "trace_id": payload.get("trace_id"),
                    "error_code": payload.get("code"),
                    "reason": payload.get("reason"),
                    "message": payload.get("message"),
                }
            )
            ctx.say(
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
                ctx.say(
                    text=translate(
                        "action.visualization.internal_error_generating",
                        language=language,
                        params={"error": str(e)},
                    )
                )
        finally:
            if completed_successfully:
                if _SHOW_EXECUTION_SUMMARY and execution_summary is not None:
                    ctx.say(
                        text=format_execution_summary(
                            execution_summary,
                            show_normalization=_SHOW_NORMALIZATION_SUMMARY,
                            planner_diagnostics=planner_diagnostics,
                            language=language,
                        )
                    )
                else:
                    ctx.say(text=translate("action.visualization.success_complete", language=language))
            ctx.done()
        return None
