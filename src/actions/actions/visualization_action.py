import json
import logging
from typing import Any, Dict, List, Optional, Protocol, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore
from rasa_sdk.events import FollowupAction, SlotSet  # type: ignore

from src.actions.error_messages import visualization_error_payload
from src.actions.long_action.long_action import LongAction, PreworkResult
from src.actions.long_action.long_action_context import LongActionContext
from src.actions.utils.visualization import (
    extract_entities_from_latest_message,
    format_execution_summary,
    resolve_override_language,
    serialize_plan_for_frontend,
)
from src.domain.langchain import schema as lang_schema
from src.executors import execute_plan_async
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

_planner_retries_raw = env_util.get_env("ACTIONS_PLANNER_MAX_RETRIES", default="2") or "2"
try:
    _planner_max_retries = max(0, int(_planner_retries_raw))
except Exception:
    _planner_max_retries = 2
_PLANNER_MAX_RETRIES = _planner_max_retries

_executor_concurrency_raw = env_util.get_env("ACTIONS_EXECUTOR_MAX_CONCURRENCY", default="4") or "4"
try:
    _executor_max_concurrency = max(1, int(_executor_concurrency_raw))
except Exception:
    _executor_max_concurrency = 4
_EXECUTOR_MAX_CONCURRENCY = _executor_max_concurrency

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

            events = tracker.events

            recent_messages: List[str] = []
            for ev in events:
                if ev.get("event") != "user":
                    continue
                text_any = ev.get("text")
                if isinstance(text_any, str) and text_any.strip():
                    recent_messages.append(text_any.strip())
            if len(recent_messages) > fallback_limit:
                recent_messages = recent_messages[-fallback_limit:]

            anchor_idx = -1
            for idx in range(len(events) - 1, -1, -1):
                ev = events[idx]
                if ev.get("event") != "user":
                    continue
                parse_data_any = ev.get("parse_data")
                parse_data = cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else {}
                intent_any = parse_data.get("intent")
                intent = cast(Dict[str, Any], intent_any) if isinstance(intent_any, dict) else {}
                name_any = intent.get("name")

                if not isinstance(name_any, str) or not name_any.strip():
                    fallback_intent_any = ev.get("intent")
                    fallback_intent = cast(Dict[str, Any], fallback_intent_any) if isinstance(fallback_intent_any, dict) else {}
                    fallback_name_any = fallback_intent.get("name")
                    if isinstance(fallback_name_any, str) and fallback_name_any.strip():
                        name_any = fallback_name_any

                if isinstance(name_any, str) and name_any.strip() in _VISUALIZATION_REQUEST_INTENTS:
                    anchor_idx = idx
                    break

            conversation_history: List[str]
            if anchor_idx < 0:
                conversation_history = recent_messages
            else:
                since_anchor: List[str] = []
                for ev in events[anchor_idx:]:
                    if ev.get("event") != "user":
                        continue
                    text_any = ev.get("text")
                    if isinstance(text_any, str) and text_any.strip():
                        since_anchor.append(text_any.strip())
                conversation_history = since_anchor if since_anchor else recent_messages

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
                dispatcher.utter_message(text=outcome.message or "I need a bit more detail before I can generate a visualization.")
                return [SlotSet("awaiting_visualization_clarification", True)]

            if outcome.decision == "reject":
                dispatcher.utter_message(text=outcome.message or "This request is outside the visualization flow.")
                return [SlotSet("awaiting_visualization_clarification", False)]

            return [
                SlotSet("awaiting_visualization_clarification", False),
                FollowupAction("action_oneshot_generate_visualization"),
            ]
        except Exception as e:
            logger.exception("Error routing visualization request (trace_id=%s)", trace_id)
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
            dispatcher.utter_message(text=f"❌ {payload.get('message')} (Error code: {payload.get('code')}, Trace ID: {payload.get('trace_id')})")
            if _ECHO_INTERNAL_ERRORS:
                dispatcher.utter_message(text=f"Error routing visualization request: {str(e)}")
            return []


def _extract_request_context(ctx: LongActionContext) -> Dict[str, Any]:
    latest_meta = ctx.metadata
    latest_any = ctx.tracker_snapshot.get("latest_message")
    latest_msg = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}
    extracted_entities = extract_entities_from_latest_message(latest_msg)
    override_language = resolve_override_language(latest_meta, ctx.slots)
    conversation_history = ctx.user_messages_since_intent("generate_visualization", fallback_limit=12)
    planner_question = "\n".join([m for m in conversation_history if m.strip()]).strip() or ctx.text
    return {
        "user_message": ctx.text,
        "planner_question": planner_question,
        "user_sub": ctx.sender_id,
        "latest_meta": latest_meta,
        "latest_msg": latest_msg,
        "extracted_entities": extracted_entities,
        "override_language": override_language,
        "conversation_history": conversation_history,
    }


_INTERNAL_TRACE_ID_KEY = "_visualization_trace_id"


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

            if awaiting_clarification and intent_name not in _VISUALIZATION_REQUEST_INTENTS:
                return PreworkResult(events=[FollowupAction("action_clarify_visualization_request")], proceed=False)

            # Defensive fallback: if routing reaches this action for an unrelated
            # intent, always send a user-facing response instead of returning
            # nothing and leaving the conversation hanging.
            if intent_name and intent_name not in _VISUALIZATION_REQUEST_INTENTS and not _is_guided_visualization_request(ctx.slots):
                ctx.say(text=("I can help with visualization requests. Try asking to generate a chart, or ask to update the current chart."))
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
            decision: Dict[str, Any] = {
                "decision": outcome.decision,
                "reason": outcome.reason,
                "message": outcome.message,
                "clarification_type": outcome.clarification_type,
                "clarification_options": outcome.clarification_options,
                "missing_fields": outcome.missing_fields,
            }
            ctx.say(
                json_message={
                    "type": "visualization_query_decision",
                    "trace_id": trace_id,
                    "decision": decision.get("decision"),
                    "reason": decision.get("reason"),
                    "clarification_type": decision.get("clarification_type"),
                    "clarification_options": decision.get("clarification_options", []),
                    "message": decision.get("message"),
                }
            )

            decision_name = str(decision.get("decision") or "").strip().lower()
            if decision_name in {"clarify", "reject"}:
                message = decision.get("message")
                if isinstance(message, str) and message.strip():
                    ctx.say(text=message.strip())
                else:
                    ctx.say(text="I need a bit more detail before I can generate a visualization.")

            decision_name = str(decision.get("decision") or "").strip().lower()
            if decision_name == "clarify":
                return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", True)], proceed=False)
            if decision_name == "reject":
                return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=False)
            return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=True)
        except Exception as e:
            logger.exception("Error generating visualization (prework, trace_id=%s)", trace_id)
            payload = visualization_error_payload(e, trace_id=trace_id)
            ctx.say(
                json_message={
                    "type": "visualization_error",
                    "trace_id": payload.get("trace_id"),
                    "error_code": payload.get("code"),
                    "reason": payload.get("reason"),
                    "message": payload.get("message"),
                }
            )
            ctx.say(text=f"❌ {payload.get('message')} (Error code: {payload.get('code')}, Trace ID: {payload.get('trace_id')})")
            if _ECHO_INTERNAL_ERRORS:
                ctx.say(text=f"Error generating visualization: {str(e)}")
            return PreworkResult(events=[], proceed=False)

    async def work(self, ctx: LongActionContext) -> Any:
        completed_successfully = False
        execution_summary: Optional[Any] = None
        planner_diagnostics: Optional[Dict[str, Any]] = None
        trace_id = _ensure_context_trace_id(ctx)
        try:
            request_ctx = _extract_request_context(ctx)
            user_message = cast(str, request_ctx["user_message"])
            planner_question = cast(str, request_ctx.get("planner_question") or user_message)
            user_sub = cast(str, request_ctx["user_sub"])
            extracted_entities = cast(Dict[str, Any], request_ctx["extracted_entities"])
            override_language = cast(Optional[str], request_ctx["override_language"])

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
            logger.exception("Error generating visualization (trace_id=%s)", trace_id)
            payload = visualization_error_payload(e, trace_id=trace_id)
            ctx.say(
                json_message={
                    "type": "visualization_error",
                    "trace_id": payload.get("trace_id"),
                    "error_code": payload.get("code"),
                    "reason": payload.get("reason"),
                    "message": payload.get("message"),
                }
            )
            ctx.say(text=f"❌ {payload.get('message')} (Error code: {payload.get('code')}, Trace ID: {payload.get('trace_id')})")
            if _ECHO_INTERNAL_ERRORS:
                ctx.say(text=f"Error generating visualization: {str(e)}")
        finally:
            if completed_successfully:
                if _SHOW_EXECUTION_SUMMARY and execution_summary is not None:
                    ctx.say(
                        text=format_execution_summary(
                            execution_summary,
                            show_normalization=_SHOW_NORMALIZATION_SUMMARY,
                            planner_diagnostics=planner_diagnostics,
                        )
                    )
                else:
                    ctx.say(text="✅ Visualization generation complete.")
            ctx.done()
        return None
