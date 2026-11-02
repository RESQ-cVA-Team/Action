import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore
from rasa_sdk.events import FollowupAction, SlotSet

from src.actions.error_messages import visualization_error_payload
from src.actions.helpers.metric import resolve_next_metric_candidate
from src.actions.helpers.visualization import (
    canonicalize_ssot_entities,
    extract_entities_from_latest_message,
    question_with_resolved_metrics,
    format_execution_summary,
    is_plan_only_request,
    names_metric_and_chart_type,
    pretty_print_graphql_query,
    resolve_override_language,
    serialize_plan_for_frontend,
)
from src.actions.i18n import resolve_language, translate
from src.actions.long_action.long_action import LongAction, PreworkResult
from src.actions.long_action.long_action_context import LongActionContext
from src.domain.langchain import schema as lang_schema
from src.executors import execute_plan_async
from src.executors.orchestration.plan_executor import VisualizationExecutionError
from src.planners.langchain.llm_errors import LLM_UNAVAILABLE
from src.planners.langchain.request_orchestrator import (
    _PLANNER_MAX_RETRIES,
    VisualizationRequestOutcome,
    orchestrate_visualization_request,
)
from src.shared import ssot_loader
from src.util import env as env_util
from src.util.llm_usage import llm_usage_field, start_llm_usage
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

_LOG_USER_TEXT = env_util.env_flag("ACTIONS_LOG_USER_TEXT", default=False)
_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)
_SHOW_EXECUTION_SUMMARY = env_util.env_flag("ACTIONS_SHOW_EXECUTION_SUMMARY", default=True)
_DEFER_CALLBACK_HANDOFF = env_util.env_flag("LONG_ACTION_DEFER_CALLBACK_HANDOFF", default=False)
_EMIT_QUERY_DEBUG = env_util.env_flag("ACTIONS_EMIT_QUERY_DEBUG", default=False)
_VISUALIZATION_INTENT = "generate_visualization"
_VISUALIZATION_PLAN_TYPE = "visualization_plan"
_VISUALIZATION_GRAPHQL_QUERY_TYPE = "visualization_graphql_query"
_VISUALIZATION_RESPONSE_SCHEMA_VERSION = 1

_EXECUTOR_MAX_CONCURRENCY = 4


def _parse_positive_float_env(name: str, raw_value: str, minimum: float) -> float:
    token = (raw_value or "").strip()
    try:
        parsed = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got: {raw_value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {parsed}")
    return parsed


_execute_plan_timeout_raw = env_util.get_env("ACTIONS_EXECUTE_PLAN_TIMEOUT_SECONDS", default="90") or "90"
_EXECUTE_PLAN_TIMEOUT_SECONDS = _parse_positive_float_env("ACTIONS_EXECUTE_PLAN_TIMEOUT_SECONDS", _execute_plan_timeout_raw, 5.0)

DomainDict = Dict[str, Any]
RasaEventList = List[Any]


def _action_log_context(
    *,
    trace_id: str,
    action_name: str,
    event: str,
    outcome: str,
    **fields: Any,
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Any] = {
        "trace_id": trace_id,
        "action": action_name,
        "event": event,
        "outcome": outcome,
    }
    for key, value in fields.items():
        if value is None:
            continue
        context[key] = value
    return {"log_context": context}


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


def _emit_next_metric_followup(
    ctx: LongActionContext,
    plan_obj: lang_schema.AnalysisPlan,
    language: str,
) -> None:
    if not plan_obj.charts:
        return
    chart = plan_obj.charts[0]
    if not chart.metrics:
        return
    current_metric = (chart.metrics[0].metric or "").strip()
    if not current_metric:
        return
    next_metric = resolve_next_metric_candidate(current_metric)
    if not next_metric:
        return

    next_label = ssot_loader.get_metric_display_name(next_metric)

    payload = translate(
        "action.visualization.next_metric_payload",
        language=language,
        params={"metric": next_metric},
    )
    ctx.say(
        text=translate(
            "action.visualization.next_metric_suggestion",
            language=language,
            params={"metric": next_label},
        ),
        buttons=[
            {  # New KPI, Same Filters
                "title": translate(
                    "action.visualization.next_metric_button",
                    language=language,
                    params={"metric": next_label},
                ),
                "payload": payload,
            },
            {  # New KPI, Clear Filters
                "title": translate(
                    "action.visualization.next_metric_button",
                    language=language,
                    params={"metric": next_label},
                )
                + " (clear filters)",
                "payload": payload + " with no filters",
            },
        ],
    )


def _build_metric_clarification_prompt(
    clarification_type: Optional[str],
    clarification_options: Optional[List[str]],
    language: str,
    fallback_message: Optional[str],
) -> tuple[str, List[Dict[str, str]]]:
    if str(clarification_type or "").strip().lower() != "metric":
        if isinstance(fallback_message, str) and fallback_message.strip():
            return fallback_message.strip(), []
        return translate("action.visualization.clarify_default", language=language), []

    metric_meta = ssot_loader.get_metric_metadata()
    canonical_options: List[str] = []
    seen: set[str] = set()
    for raw_option in clarification_options or []:
        if not isinstance(raw_option, str):
            continue
        canonical = raw_option.strip().upper()
        if not canonical or canonical in seen or canonical not in metric_meta:
            continue
        seen.add(canonical)
        canonical_options.append(canonical)

    if len(canonical_options) < 2:
        if isinstance(fallback_message, str) and fallback_message.strip():
            return fallback_message.strip(), []
        return translate("action.visualization.clarify_default", language=language), []

    display_names = [ssot_loader.get_metric_display_name(metric_code) for metric_code in canonical_options]
    if any(not isinstance(display_name, str) or not display_name.strip() for display_name in display_names):
        if isinstance(fallback_message, str) and fallback_message.strip():
            return fallback_message.strip(), []
        return translate("action.visualization.clarify_default", language=language), []

    message = translate(
        "action.visualization.metric_choice_clarify",
        language=language,
        params={"options": ", ".join(display_names)},
        default=f"Which metric did you mean: {', '.join(display_names)}?",
    )
    buttons = [
        {
            "title": display_name,
            "payload": f'/generate_visualization{{"metric":"{metric_code}"}}',
        }
        for metric_code, display_name in zip(canonical_options, display_names)
    ]
    return message, buttons


def _merge_entities(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = value
            continue

        existing = merged[key]
        if isinstance(existing, list):
            existing_list = cast(List[Any], existing)
            if isinstance(value, list):
                existing_list.extend(cast(List[Any], value))
            else:
                existing_list.append(value)
            continue

        if isinstance(value, list):
            merged[key] = [existing, *value]
        else:
            merged[key] = value
    return merged


def _extract_entities_from_user_event(event: Dict[str, Any]) -> Dict[str, Any]:
    parse_data_any = event.get("parse_data")
    parse_data = cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else {}

    parse_entities_any = parse_data.get("entities")
    event_entities_any = event.get("entities")

    entities_list: List[Any]
    if isinstance(parse_entities_any, list):
        entities_list = cast(List[Any], parse_entities_any)
    elif isinstance(event_entities_any, list):
        entities_list = cast(List[Any], event_entities_any)
    else:
        return {}

    extracted: Dict[str, Any] = {}
    for ent_any in entities_list:
        if not isinstance(ent_any, dict):
            continue
        ent = cast(Dict[str, Any], ent_any)
        key_any = ent.get("entity")
        if not isinstance(key_any, str) or "value" not in ent:
            continue

        value = ent["value"]
        keys = [key_any]
        role_any = ent.get("role")
        if isinstance(role_any, str) and role_any.strip():
            keys.append(f"{key_any}_{role_any.strip().lower()}")
        for key in keys:
            if key not in extracted:
                extracted[key] = value
                continue
            existing = extracted[key]
            if isinstance(existing, list):
                cast(List[Any], existing).append(value)
            else:
                extracted[key] = [existing, value]

    return extracted


def _dedupe_list_values(values: List[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen: set[str] = set()
    for item in values:
        marker = json.dumps(item, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


_LATEST_ENTITY_PRECEDENCE_KEYS = {
    "hospital_name",
    "hospital_scope_reference",
    "country_code",
    "group_id",
    "date",
}


def _decision_for_latest_message(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The decision action_clarify_visualization_request already sent for the
    user's current message, before it handed over to the one-shot action."""
    last_user_idx = -1
    for idx in range(len(events) - 1, -1, -1):
        if events[idx].get("event") == "user":
            last_user_idx = idx
            break
    if last_user_idx < 0:
        return None

    for ev in reversed(events[last_user_idx + 1 :]):
        if ev.get("event") != "bot":
            continue
        payload = _extract_bot_custom_payload(ev)
        if payload and payload.get("type") == "visualization_query_decision":
            return payload
    return None


def _proceed_decided_for_latest_message(events: List[Dict[str, Any]]) -> Optional[str]:
    """The reason of a proceed decision already taken for the user's current
    message; None when there is none. The one-shot action then plans without
    asking the decision stage the same question again."""
    decision = _decision_for_latest_message(events)
    if decision is None or decision.get("decision") != "proceed":
        return None
    return str(decision.get("reason") or "sufficient_information")


def _user_turns(events: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Each user message with the decision sent for it, if any, oldest first."""
    turns: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for ev in events:
        if ev.get("event") == "user":
            turns.append((ev, None))
        elif ev.get("event") == "bot" and turns and turns[-1][1] is None:
            payload = _extract_bot_custom_payload(ev)
            if payload and payload.get("type") == "visualization_query_decision":
                turns[-1] = (turns[-1][0], payload)
    return turns


def _request_start(turns: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]], index: int) -> int:
    """The first turn of the request turn `index` belongs to: back over every
    earlier turn that was answered with a clarify."""
    start = index
    while start > 0 and (turns[start - 1][1] or {}).get("decision") == "clarify":
        start -= 1
    return start


def _edited_request_entities(turns: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]], start: int) -> Dict[str, Any]:
    """The entities of the request whose chart a request starting at `start`
    edits: the latest earlier request that proceeded."""
    for index in range(start - 1, -1, -1):
        if (turns[index][1] or {}).get("decision") == "proceed":
            return _request_entities(turns, index)
    return {}


def _request_entities(turns: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]], index: int) -> Dict[str, Any]:
    """The entities of the request turn `index` belongs to, up to that turn,
    including those of the request it edits; never those of older, unrelated
    requests."""
    start = _request_start(turns, index)
    edits = (turns[start][1] or {}).get("edits_previous_plan") is True
    entities = _edited_request_entities(turns, start) if edits else {}
    for event, _ in turns[start : index + 1]:
        entities = _merge_entities(entities, _extract_entities_from_user_event(event))
    return entities


def _clarification_exchange(events: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """The earlier turns of the request the current message answers: each user
    message that was answered with a clarify decision, oldest first, back to
    the message that started the request, with that decision. Empty when the
    current message answers no pending clarification."""
    turns = _user_turns(events)
    exchange: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for user_event, decision in reversed(turns[:-1]):
        if not decision or decision.get("decision") != "clarify":
            break
        exchange.append((user_event, decision))
    return list(reversed(exchange))


def _merge_request_entities(latest_entities: Dict[str, Any], earlier: Dict[str, Any]) -> Dict[str, Any]:
    """The current message's entities over those of the earlier turns of the
    same request; a restated identity or cohort key replaces the earlier one."""
    merged = _merge_entities(dict(earlier), latest_entities)
    for key, value in list(merged.items()):
        if isinstance(value, list):
            merged[key] = _dedupe_list_values(cast(List[Any], value))
    for key in _LATEST_ENTITY_PRECEDENCE_KEYS:
        if key in latest_entities:
            merged[key] = latest_entities[key]
    return merged


def _planner_text(message: Dict[str, Any]) -> str:
    """A message as the planner reads it: resolved metrics written as codes,
    and a button payload ("/intent{...}") as the values it chose."""
    text_any = message.get("text")
    text = text_any.strip() if isinstance(text_any, str) else ""
    if text.startswith("/"):
        values = [str(e.get("value")) for e in message.get("entities") or [] if isinstance(e, dict) and e.get("value")]
        return ", ".join(values) or text
    return question_with_resolved_metrics(message)


def _parse_data(event: Dict[str, Any]) -> Dict[str, Any]:
    parse_data_any = event.get("parse_data")
    return cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else event


def _request_context(events: List[Dict[str, Any]], latest_msg: Dict[str, Any], edits_previous_plan: bool) -> Dict[str, Any]:
    """Question, entities, history and pending clarification for the request
    the current message belongs to. A reply to a clarification continues the
    request that started the exchange: its question is that request with each
    question asked and answered, its entities those of the exchange, and the
    previous plan its base only if that request edited it. Otherwise the
    message stands alone, on top of the previous plan when edits_previous_plan.
    message_entities are the entities the message itself states, or those of
    the exchange it continues."""
    latest_entities = canonicalize_ssot_entities(extract_entities_from_latest_message(latest_msg))
    exchange = _clarification_exchange(events)
    if exchange:
        edits_previous_plan = exchange[0][1].get("edits_previous_plan") is True
        answers = [_parse_data(event) for event, _ in exchange[1:]] + [latest_msg]
        lines = [_planner_text(_parse_data(exchange[0][0]))]
        for (_, decision), answer in zip(exchange, answers):
            lines.append(f"The assistant asked: {decision.get('message') or ''}".rstrip())
            lines.append(f"The user answered: {_planner_text(answer)}")
        question = "\n".join(lines)
        turns = _user_turns(events)
        entities = canonicalize_ssot_entities(_merge_request_entities(latest_entities, _request_entities(turns, len(turns) - 2)))
        history = [str(_parse_data(event).get("text") or "") for event, _ in exchange] + [str(latest_msg.get("text") or "")]
        message_entities = entities
        pending_clarification: Optional[Dict[str, Any]] = exchange[-1][1]
    else:
        question = question_with_resolved_metrics(latest_msg)
        turns = _user_turns(events)
        earlier = _edited_request_entities(turns, len(turns) - 1) if edits_previous_plan and turns else {}
        entities = canonicalize_ssot_entities(_merge_request_entities(latest_entities, earlier)) if earlier else latest_entities
        message_entities = latest_entities
        history = []
        pending_clarification = None
    latest_plan_summary = _collect_latest_visualization_plan_summary(events) if edits_previous_plan else None
    if latest_plan_summary:
        question = f"{latest_plan_summary}\n\nUser's newest instruction:\n{question}".strip()
    return {
        "planner_question": question,
        "extracted_entities": entities,
        "message_entities": message_entities,
        "conversation_history": history,
        "latest_plan_summary": latest_plan_summary,
        "pending_clarification": pending_clarification,
        "answers_clarification": bool(exchange),
        "edits_previous_plan": edits_previous_plan,
    }


async def _orchestrate(**kwargs: Any) -> VisualizationRequestOutcome:
    """The orchestrator blocks on LLM calls for seconds at a time. Run in a worker
    thread, it leaves the action server free to answer other requests meanwhile."""
    return await asyncio.to_thread(orchestrate_visualization_request, **kwargs)


def _hand_over_plan(ctx: LongActionContext, trace_id: str, plan: lang_schema.AnalysisPlan) -> PreworkResult:
    """Hands the plan to the work phase, which fetches the data. A plan-only
    request gets the plan in this reply and ends here: nothing is fetched
    upstream and no callback follows."""
    if is_plan_only_request(ctx.metadata):
        ctx.say(json_message={"type": "visualization_plan", "trace_id": trace_id, "plan": serialize_plan_for_frontend(plan)})
        ctx.say(json_message={"type": "visualization_plan_only", "trace_id": trace_id})
        return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=False)
    ctx.tracker_snapshot[_INTERNAL_PREPARED_PLAN_KEY] = plan
    return PreworkResult(events=[SlotSet("awaiting_visualization_clarification", False)], proceed=True)


def _decision_text(outcome: VisualizationRequestOutcome, language: str, default_key: str, message: Optional[str] = None) -> str:
    """What the user reads for a clarify or reject. An unreachable LLM says so
    instead of asking for detail the request does not lack."""
    if outcome.reason == LLM_UNAVAILABLE:
        return translate("action.visualization.llm_unavailable", language=language)
    return message or outcome.message or translate(default_key, language=language)


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


def _collect_latest_visualization_plan_summary(
    events: List[Dict[str, Any]],
) -> Optional[str]:
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

        plan_json = json.dumps(plan, indent=2)
        summary = f"Previous chart plan (carry over everything except what the user explicitly changes):\n{plan_json}"
        return summary

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
        start_llm_usage()
        trace_id = uuid4().hex
        metadata: Dict[str, Any] = {}
        slots: Dict[str, Any] = {}
        latest_msg = tracker.latest_message

        metadata_any = latest_msg.get("metadata")
        metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
        trace_id = _trace_id_from_metadata(metadata) or trace_id
        slots_any = tracker.current_state().get("slots", {})
        slots = cast(Dict[str, Any], slots_any) if isinstance(slots_any, dict) else {}

        with log_context(trace_id=trace_id, sender_id=str(tracker.sender_id), action=self.name()):
            try:
                logger.info("Starting visualization clarification routing")
                override_language = resolve_override_language(metadata, slots)
                language = resolve_language(metadata=metadata, slots=slots, tracker=tracker)

                events = tracker.events
                # Judged against the previous plan unless the message answers a
                # pending clarification; the decision stage says whether it edits
                # it. It sees what the message states, not what an edit would
                # inherit, so the earlier request does not make it look like one.
                request = _request_context(events, latest_msg, edits_previous_plan=True)
                planner_question = request["planner_question"]
                conversation_history = request["conversation_history"]
                latest_plan_summary = request["latest_plan_summary"]
                pending_clarification = request["pending_clarification"]

                outcome = await _orchestrate(
                    question=planner_question,
                    entities=request["message_entities"],
                    edit_entities=request["extracted_entities"],
                    language=override_language,
                    trace_id=trace_id,
                    max_retries=_PLANNER_MAX_RETRIES,
                    include_plan=False,
                    conversation_history=conversation_history,
                    progress_cb=None,
                    pending_clarification=pending_clarification,
                )

                # A new request is judged as if there were no previous chart. The
                # previous plan in view colours that judgement, so a message not
                # judged an edit is judged once more without it.
                if not request["answers_clarification"] and latest_plan_summary and outcome.edits_previous_plan is not True:
                    request = _request_context(events, latest_msg, edits_previous_plan=False)
                    planner_question = request["planner_question"]
                    outcome = await _orchestrate(
                        question=planner_question,
                        entities=request["extracted_entities"],
                        language=override_language,
                        trace_id=trace_id,
                        max_retries=_PLANNER_MAX_RETRIES,
                        include_plan=False,
                        conversation_history=request["conversation_history"],
                        progress_cb=None,
                        pending_clarification=request["pending_clarification"],
                    )
                    latest_plan_summary = None

                if request["answers_clarification"]:
                    edits_previous_plan = bool(request["edits_previous_plan"])
                else:
                    edits_previous_plan = bool(latest_plan_summary and outcome.edits_previous_plan)

                if outcome.decision == "clarify":
                    clarification_message, clarification_buttons = _build_metric_clarification_prompt(
                        outcome.clarification_type,
                        outcome.clarification_options,
                        language,
                        outcome.message,
                    )
                    dispatcher.utter_message(
                        json_message={
                            "type": "visualization_query_decision",
                            "trace_id": trace_id,
                            **llm_usage_field(),
                            "decision": outcome.decision,
                            "reason": outcome.reason,
                            "clarification_type": outcome.clarification_type,
                            "clarification_options": outcome.clarification_options,
                            "message": clarification_message,
                            "missing_fields": outcome.missing_fields,
                            "edits_previous_plan": edits_previous_plan,
                        }
                    )
                    dispatcher.utter_message(text=clarification_message, buttons=clarification_buttons)
                    return [
                        SlotSet("awaiting_visualization_clarification", True),
                        SlotSet("guided_offer_shown", True),
                    ]

                dispatcher.utter_message(
                    json_message={
                        "type": "visualization_query_decision",
                        "trace_id": trace_id,
                        **llm_usage_field(),
                        "decision": outcome.decision,
                        "reason": outcome.reason,
                        "clarification_type": outcome.clarification_type,
                        "clarification_options": outcome.clarification_options,
                        "message": outcome.message,
                        "missing_fields": outcome.missing_fields,
                        "edits_previous_plan": edits_previous_plan,
                    }
                )

                if outcome.decision == "reject":
                    dispatcher.utter_message(text=_decision_text(outcome, language, "action.visualization.reject_default"))
                    return [SlotSet("awaiting_visualization_clarification", False)]

                return [
                    SlotSet("awaiting_visualization_clarification", False),
                    SlotSet("guided_hospital_scope", None),
                    FollowupAction("action_oneshot_generate_visualization"),
                ]
            except asyncio.TimeoutError:
                timeout_message = "The visualization execution took too long and timed out. Please try again, narrow the scope, or shorten the date range."
                logger.warning(
                    "Visualization execution timed out",
                    extra=_action_log_context(
                        trace_id=trace_id,
                        action_name=self.name(),
                        event="actions.visualization.work.timeout",
                        outcome="failure",
                        timeout_seconds=_EXECUTE_PLAN_TIMEOUT_SECONDS,
                    ),
                )
                dispatcher.utter_message(
                    json_message={
                        "type": "visualization_error",
                        "trace_id": trace_id,
                        "error_code": "EXEC_TIMEOUT",
                        "reason": "timeout",
                        "message": timeout_message,
                        "retry": True,
                    }
                )
                dispatcher.utter_message(
                    text=translate(
                        "action.common.error_with_context",
                        language=language,
                        params={
                            "message": timeout_message,
                            "code": "EXEC_TIMEOUT",
                            "trace_id": trace_id,
                        },
                    )
                )
                return None
            except Exception as e:
                logger.exception(
                    "Error routing visualization request",
                    extra=_action_log_context(
                        trace_id=trace_id,
                        action_name=self.name(),
                        event="actions.visualization.routing.failed",
                        outcome="failure",
                    ),
                )
                language = resolve_language(metadata=metadata, slots=slots, tracker=tracker)
                payload = visualization_error_payload(e, trace_id=trace_id, language=language)
                dispatcher.utter_message(
                    json_message={
                        "type": "visualization_error",
                        "trace_id": payload.get("trace_id"),
                        "error_code": payload.get("code"),
                        "reason": payload.get("reason"),
                        "message": payload.get("message"),
                        "retry": True,
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
    override_language = resolve_override_language(latest_meta, ctx.slots)
    language = resolve_language(metadata=latest_meta, slots=ctx.slots)
    events = ctx.events

    decision = _decision_for_latest_message(events)
    request = _request_context(events, latest_msg, edits_previous_plan=bool(decision and decision.get("edits_previous_plan") is True))
    planner_question = request["planner_question"]
    extracted_entities = request["extracted_entities"]
    conversation_history = request["conversation_history"]
    latest_plan_summary = request["latest_plan_summary"]
    pending_clarification = request["pending_clarification"]

    return {
        "user_message": ctx.text,
        "planner_question": planner_question,
        "user_sub": ctx.sender_id,
        "webapp_job_id": ctx.webapp_job_id,
        "latest_meta": latest_meta,
        "latest_msg": latest_msg,
        "extracted_entities": extracted_entities,
        "override_language": override_language,
        "language": language,
        "conversation_history": conversation_history,
        "latest_plan_summary": latest_plan_summary,
        "pending_clarification": pending_clarification,
        "decided_proceed": _proceed_decided_for_latest_message(events),
    }


_INTERNAL_TRACE_ID_KEY = "_visualization_trace_id"
_INTERNAL_PREPARED_PLAN_KEY = "_visualization_prepared_plan"


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


def _build_confirmation_message(plan_obj: lang_schema.AnalysisPlan, is_update: bool) -> str:
    """Build a short confirmation message describing what was just visualized."""
    charts = list(plan_obj.charts or [])
    tests = list(plan_obj.statistical_tests or [])
    if not charts:
        if tests:
            verb = "Updated" if is_update else "Completed"
            if len(tests) == 1:
                return f"{verb} statistical analysis: 1 test result is ready."
            return f"{verb} statistical analysis: {len(tests)} test results are ready."
        return "Done."

    chart = charts[0]
    metrics = [m.metric for m in (chart.metrics or [])]
    metric_str = " & ".join(metrics) if metrics else "the metric"
    chart_type = (chart.chart_type or "chart").lower()

    # Time grouping
    semantics = chart.semantics
    semantic_grain = None if semantics is None or semantics.time is None else semantics.time.grain
    if semantic_grain is not None:
        grain_str = f" per {str(semantic_grain).lower()}"
    else:
        grain_str = ""

    # Filters
    filters = chart.filters
    if filters is None:
        filter_str = " with no filters" if is_update else ""
    else:
        filter_str = " with filters applied"

    verb = "Updated —" if is_update else "Here's your"
    return f"{verb} {chart_type} chart of {metric_str}{grain_str}{filter_str}."


def _temporal_bounds_clarification(language: str) -> str:
    return translate(
        "action.visualization.missing_time_bounds",
        language=language,
        default=("I can plot monthly or quarterly trends, but I need an explicit time range first. Please include bounds such as 'from 2023-01-01 to 2023-12-31' or 'last 12 months'."),
    )


def _is_missing_temporal_bounds_error(exc: Exception) -> bool:
    return "Semantic time grouping requires explicit time window/range" in str(exc or "")


def _build_empty_plan_clarification(plan_obj: lang_schema.AnalysisPlan, language: str) -> Optional[str]:
    charts = list(plan_obj.charts or [])
    tests = list(plan_obj.statistical_tests or [])
    if charts or tests:
        return None

    return translate(
        "action.visualization.empty_plan_clarify",
        language=language,
        default=("I could not derive an executable analysis plan from that request. Please ask for a chart with metric and chart type, or for statistics provide two explicit cohorts to compare."),
    )


class ActionOneShotGenerateVisualization(LongAction):
    """Freeform one-shot visualization action backed by the planner chain.

    In callback mode this streams messages via the long-action callback URL;
    otherwise it behaves like a normal synchronous action and uses the
    dispatcher directly.
    """

    def name(self) -> str:
        return "action_oneshot_generate_visualization"

    async def prework(self, ctx: LongActionContext) -> PreworkResult:
        start_llm_usage()
        trace_id = _ensure_context_trace_id(ctx)
        with log_context(trace_id=trace_id, action=self.name()):
            logger.info("Starting one-shot prework")
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

                if awaiting_clarification:
                    if intent_name == _VISUALIZATION_INTENT:
                        return PreworkResult(
                            events=[SlotSet("awaiting_visualization_clarification", False)],
                            proceed=True,
                        )

                    return PreworkResult(
                        events=[FollowupAction("action_clarify_visualization_request")],
                        proceed=False,
                    )

                # Defensive fallback: if routing reaches this action for an unrelated
                # intent, always send a user-facing response instead of returning
                # nothing and leaving the conversation hanging.
                if intent_name and intent_name != _VISUALIZATION_INTENT and not names_metric_and_chart_type(latest_msg) and not _is_guided_visualization_request(ctx.slots):
                    language = resolve_language(metadata=ctx.metadata, slots=ctx.slots)
                    ctx.say(
                        text=translate(
                            "action.visualization.non_visualization_intent",
                            language=language,
                        )
                    )
                    return PreworkResult(
                        events=[SlotSet("awaiting_visualization_clarification", False)],
                        proceed=False,
                    )

                request_ctx = _extract_request_context(ctx)
                outcome = await _orchestrate(
                    question=cast(str, request_ctx["planner_question"]),
                    entities=cast(Dict[str, Any], request_ctx.get("extracted_entities") or {}),
                    language=cast(Optional[str], request_ctx.get("override_language")),
                    trace_id=trace_id,
                    max_retries=_PLANNER_MAX_RETRIES,
                    include_plan=True,
                    conversation_history=cast(List[str], request_ctx.get("conversation_history") or []),
                    progress_cb=None,
                    pending_clarification=cast(Optional[Dict[str, Any]], request_ctx.get("pending_clarification")),
                    decided_proceed=cast(Optional[str], request_ctx.get("decided_proceed")),
                )
                decision_name = str(outcome.decision or "").strip().lower()
                language = cast(str, request_ctx.get("language") or "en")
                if decision_name == "clarify":
                    clarification_message, clarification_buttons = _build_metric_clarification_prompt(
                        outcome.clarification_type,
                        outcome.clarification_options,
                        language,
                        outcome.message,
                    )
                    ctx.say(
                        json_message={
                            "type": "visualization_query_decision",
                            "trace_id": trace_id,
                            **llm_usage_field(),
                            "decision": outcome.decision,
                            "reason": outcome.reason,
                            "clarification_type": outcome.clarification_type,
                            "clarification_options": outcome.clarification_options,
                            "message": clarification_message,
                        }
                    )
                    ctx.say(text=clarification_message, buttons=clarification_buttons)
                    return PreworkResult(
                        events=[SlotSet("awaiting_visualization_clarification", True)],
                        proceed=False,
                    )
                if decision_name == "reject":
                    ctx.say(
                        json_message={
                            "type": "visualization_query_decision",
                            "trace_id": trace_id,
                            **llm_usage_field(),
                            "decision": outcome.decision,
                            "reason": outcome.reason,
                            "clarification_type": outcome.clarification_type,
                            "clarification_options": outcome.clarification_options,
                            "message": outcome.message,
                        }
                    )
                    ctx.say(text=_decision_text(outcome, language, "action.visualization.reject_default"))
                    return PreworkResult(
                        events=[SlotSet("awaiting_visualization_clarification", False)],
                        proceed=False,
                    )

                prepared_plan = outcome.plan if isinstance(outcome.plan, lang_schema.AnalysisPlan) else None
                if prepared_plan is None:
                    raise RuntimeError("Orchestrator returned proceed without an analysis plan")

                empty_plan_message = _build_empty_plan_clarification(prepared_plan, language)
                if empty_plan_message:
                    ctx.say(
                        json_message={
                            "type": "visualization_query_decision",
                            "trace_id": trace_id,
                            **llm_usage_field(),
                            "decision": "clarify",
                            "reason": "empty_plan",
                            "clarification_type": "analysis_plan",
                            "clarification_options": [],
                            "message": empty_plan_message,
                        }
                    )
                    ctx.say(text=empty_plan_message)
                    return PreworkResult(
                        events=[SlotSet("awaiting_visualization_clarification", True)],
                        proceed=False,
                    )

                ctx.say(
                    json_message={
                        "type": "visualization_query_decision",
                        "trace_id": trace_id,
                        **llm_usage_field(),
                        "decision": outcome.decision,
                        "reason": outcome.reason,
                        "clarification_type": outcome.clarification_type,
                        "clarification_options": outcome.clarification_options,
                        "message": outcome.message,
                    }
                )
                return _hand_over_plan(ctx, trace_id, prepared_plan)
            except Exception as e:
                logger.exception(
                    "Error generating visualization during prework",
                    extra=_action_log_context(
                        trace_id=trace_id,
                        action_name=self.name(),
                        event="actions.visualization.prework.failed",
                        outcome="failure",
                    ),
                )
                language = resolve_language(metadata=ctx.metadata, slots=ctx.slots)
                payload = visualization_error_payload(e, trace_id=trace_id, language=language)
                ctx.say(
                    json_message={
                        "type": "visualization_error",
                        "trace_id": payload.get("trace_id"),
                        "error_code": payload.get("code"),
                        "reason": payload.get("reason"),
                        "message": payload.get("message"),
                        "retry": True,
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
        start_llm_usage()
        completed_successfully = False
        execution_summary: Optional[Any] = None
        plan_obj: Optional[lang_schema.AnalysisPlan] = None
        trace_id = _ensure_context_trace_id(ctx)
        language = resolve_language(metadata=ctx.metadata, slots=ctx.slots)
        with log_context(trace_id=trace_id, action=self.name()):
            try:
                request_ctx = _extract_request_context(ctx)
                user_message = cast(str, request_ctx["user_message"])
                user_sub = cast(str, request_ctx["user_sub"])
                webapp_job_id = cast(Optional[str], request_ctx.get("webapp_job_id"))
                language = cast(str, request_ctx.get("language") or "en")

                if _LOG_USER_TEXT:
                    logger.info("Processing visualization request: '%s'", user_message)
                else:
                    logger.info(
                        "Processing visualization request",
                        extra={"log_context": {"text_length": len(user_message or "")}},
                    )

                def progress(msg: str) -> None:
                    ctx.say(progress=msg)

                def on_summary(summary: Any) -> None:
                    nonlocal execution_summary
                    execution_summary = summary

                def on_graphql_query(payload: Dict[str, Any]) -> None:
                    query_text_any = payload.get("query")
                    if not isinstance(query_text_any, str) or not query_text_any.strip():
                        return
                    query_pretty = pretty_print_graphql_query(query_text_any)

                    if _EMIT_QUERY_DEBUG:
                        ctx.say(
                            json_message={
                                "type": _VISUALIZATION_GRAPHQL_QUERY_TYPE,
                                "trace_id": trace_id,
                                "request_label": payload.get("request_label"),
                                "group_by_field": payload.get("group_by_field"),
                                "query_hash": payload.get("query_hash"),
                                "query": query_pretty,
                            }
                        )
                    if _EMIT_QUERY_DEBUG:
                        ctx.say(text=(f"[dev] GraphQL query\nhash={payload.get('query_hash') or '-'}\n{query_pretty}"))

                # In deferred-handoff mode, initial routing/clarification can use
                # normal dispatcher delivery and heavy generation streams via
                # callback after this explicit handoff.
                if _DEFER_CALLBACK_HANDOFF and not ctx.callback_mode_enabled:
                    ctx.enable_callback_mode()

                prepared_any = ctx.tracker_snapshot.pop(_INTERNAL_PREPARED_PLAN_KEY, None)

                if isinstance(prepared_any, lang_schema.AnalysisPlan):
                    plan_obj = prepared_any
                    progress("Using prepared plan from prework")
                else:
                    logger.warning(
                        "Prepared plan missing in work fallback; regenerating plan in work",
                        extra=_action_log_context(
                            trace_id=trace_id,
                            action_name=self.name(),
                            event="actions.visualization.prepared_plan_missing",
                            outcome="degraded",
                        ),
                    )
                    planner_question = cast(str, request_ctx["planner_question"])
                    extracted_entities = cast(Dict[str, Any], request_ctx["extracted_entities"])
                    override_language = cast(Optional[str], request_ctx["override_language"])
                    conversation_history = cast(List[str], request_ctx.get("conversation_history") or [])
                    pending_clarification = cast(Optional[Dict[str, Any]], request_ctx.get("pending_clarification"))

                    progress("Calling orchestrator to build a plan")
                    outcome = await _orchestrate(
                        question=planner_question,
                        entities=extracted_entities,
                        language=override_language,
                        trace_id=trace_id,
                        max_retries=_PLANNER_MAX_RETRIES,
                        include_plan=True,
                        conversation_history=conversation_history,
                        progress_cb=progress,
                        pending_clarification=pending_clarification,
                        decided_proceed=cast(Optional[str], request_ctx.get("decided_proceed")),
                    )

                    decision_name = str(outcome.decision or "").strip().lower()
                    if decision_name in {"clarify", "reject"}:
                        clarification_message = outcome.message
                        clarification_buttons: List[Dict[str, str]] = []
                        if decision_name == "clarify":
                            clarification_message, clarification_buttons = _build_metric_clarification_prompt(
                                outcome.clarification_type,
                                outcome.clarification_options,
                                language,
                                outcome.message,
                            )
                        ctx.say(
                            json_message={
                                "type": "visualization_query_decision",
                                "trace_id": trace_id,
                                **llm_usage_field(),
                                "decision": outcome.decision,
                                "reason": outcome.reason,
                                "clarification_type": outcome.clarification_type,
                                "clarification_options": outcome.clarification_options,
                                "message": clarification_message,
                            }
                        )
                        default_key = "action.visualization.clarify_default" if decision_name == "clarify" else "action.visualization.reject_default"
                        ctx.say(text=_decision_text(outcome, language, default_key, clarification_message), buttons=clarification_buttons)
                        return None

                    plan_obj = outcome.plan if isinstance(outcome.plan, lang_schema.AnalysisPlan) else None
                    if plan_obj is None:
                        raise RuntimeError("Orchestrator returned proceed without an analysis plan")

                if plan_obj is not None:
                    empty_plan_message = _build_empty_plan_clarification(plan_obj, language)
                    if empty_plan_message:
                        ctx.say(
                            json_message={
                                "type": "visualization_query_decision",
                                "trace_id": trace_id,
                                **llm_usage_field(),
                                "decision": "clarify",
                                "reason": "empty_plan",
                                "clarification_type": "analysis_plan",
                                "clarification_options": [],
                                "message": empty_plan_message,
                            }
                        )
                        ctx.say(text=empty_plan_message)
                        return None

                ctx.say(
                    json_message={
                        "type": "visualization_plan",
                        "trace_id": trace_id,
                        "plan": serialize_plan_for_frontend(plan_obj),
                        **llm_usage_field(),
                    }
                )

                def _run_execute_plan() -> Any:
                    return asyncio.run(
                        execute_plan_async(
                            plan_obj,
                            user_sub=user_sub,
                            job_id=webapp_job_id,
                            max_concurrency=_EXECUTOR_MAX_CONCURRENCY,
                            progress_cb=progress,
                            summary_cb=on_summary,
                            trace_id=trace_id,
                            query_cb=on_graphql_query,
                        )
                    )

                visualization = await asyncio.wait_for(
                    asyncio.to_thread(_run_execute_plan),
                    timeout=_EXECUTE_PLAN_TIMEOUT_SECONDS,
                )
                visualization_payload = visualization.model_dump(mode="json")
                ctx.say(json_message=visualization_payload)

                warnings_any = visualization_payload.get("warnings")
                if isinstance(warnings_any, list):
                    for warning in cast(List[object], warnings_any):
                        if isinstance(warning, str) and warning.strip():
                            ctx.say(text=f"Note: {warning.strip()}")

                completed_successfully = True
                is_update_flow = bool(request_ctx["latest_plan_summary"])
                confirmation = _build_confirmation_message(plan_obj, is_update=is_update_flow)
                ctx.say(text=confirmation)
            except asyncio.TimeoutError:
                timeout_message = "The visualization execution took too long and timed out. Please try again, narrow the scope, or shorten the date range."
                logger.warning(
                    "Visualization execution timed out",
                    extra=_action_log_context(
                        trace_id=trace_id,
                        action_name=self.name(),
                        event="actions.visualization.work.timeout",
                        outcome="failure",
                        timeout_seconds=_EXECUTE_PLAN_TIMEOUT_SECONDS,
                    ),
                )
                ctx.say(
                    json_message={
                        "type": "visualization_error",
                        "trace_id": trace_id,
                        "error_code": "EXEC_TIMEOUT",
                        "reason": "timeout",
                        "message": timeout_message,
                        "retry": True,
                    }
                )
                ctx.say(
                    text=translate(
                        "action.common.error_with_context",
                        language=language,
                        params={
                            "message": timeout_message,
                            "code": "EXEC_TIMEOUT",
                            "trace_id": trace_id,
                        },
                    )
                )
                return None
            except Exception as e:
                if _is_missing_temporal_bounds_error(e):
                    clarification_message = _temporal_bounds_clarification(language)
                    ctx.say(
                        json_message={
                            "type": "visualization_query_decision",
                            "trace_id": trace_id,
                            **llm_usage_field(),
                            "decision": "clarify",
                            "reason": "missing_time_bounds",
                            "clarification_type": "date",
                            "clarification_options": [],
                            "message": clarification_message,
                        }
                    )
                    ctx.say(text=clarification_message)
                    return None

                if isinstance(e, VisualizationExecutionError) and (e.reason == "origin_scope_resolution" or e.clarification_type):
                    ctx.say(
                        json_message={
                            "type": "visualization_query_decision",
                            "trace_id": trace_id,
                            **llm_usage_field(),
                            "decision": "clarify",
                            "reason": e.reason,
                            "clarification_type": e.clarification_type,
                            "clarification_options": e.clarification_options,
                            "message": e.user_message,
                        }
                    )
                    ctx.say(text=e.user_message)
                    return None

                logger.exception(
                    "Error generating visualization",
                    extra=_action_log_context(
                        trace_id=trace_id,
                        action_name=self.name(),
                        event="actions.visualization.work.failed",
                        outcome="failure",
                    ),
                )
                payload = visualization_error_payload(e, trace_id=trace_id, language=language)
                ctx.say(
                    json_message={
                        "type": "visualization_error",
                        "trace_id": payload.get("trace_id"),
                        "error_code": payload.get("code"),
                        "reason": payload.get("reason"),
                        "message": payload.get("message"),
                        "retry": True,
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
                # _build_confirmation_message already sent the "here's your
                # X" confirmation above (unconditionally, right after
                # completed_successfully is set) -- this only adds anything
                # when there's a genuine caveat (skipped/failed stats) to
                # report, never a second redundant "here's your chart".
                if completed_successfully and _SHOW_EXECUTION_SUMMARY and execution_summary is not None:
                    caveat = format_execution_summary(execution_summary, language=language, include_opening_line=False)
                    if caveat:
                        ctx.say(text=caveat)
                    if plan_obj is not None:
                        _emit_next_metric_followup(ctx=ctx, plan_obj=plan_obj, language=language)
                ctx.done()
        return None
