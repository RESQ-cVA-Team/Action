from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Protocol, cast
from uuid import uuid4

from src.actions.guided_scope import (
    GuidedScopeIntent,
    parse_guided_scope_intent,
    resolve_mine_scope_json,
    resolve_provider_group_mine_scope_json,
    scope_json,
)
from src.actions.i18n import resolve_language_from_tracker, translate
from src.actions.ssot_lookup import resolve_catalog_candidates, resolve_metric_candidates
from src.domain.langchain import schema as S
from src.executors.analytics_center.client import AnalyticsCenterError, get_analytics_center_client
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

SKIP_SENTINEL = "__skip__"
SKIP_TOKENS = {
    SKIP_SENTINEL,
    "/skip_guided_step",
    "skip",
    "skip step",
    "skip this step",
    "skip that step",
    "lets skip that step",
    "let's skip that step",
}


def _guided_scope_log_context(
    *,
    trace_id: Optional[str],
    event: str,
    operation: str,
    **fields: Any,
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Any] = {
        "trace_id": trace_id or "-",
        "event": event,
        "operation": operation,
        "outcome": "degraded",
    }
    for key, value in fields.items():
        if value is None:
            continue
        context[key] = value
    return {"log_context": context}


class DispatcherLike(Protocol):
    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]

    def current_state(self) -> Dict[str, Any]: ...


def _latest_entities(tracker: TrackerLike) -> Dict[str, Any]:
    latest_any: Any = getattr(tracker, "latest_message", None) or {}
    latest = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}
    entities_any = latest.get("entities")
    out: Dict[str, Any] = {}
    if isinstance(entities_any, list):
        for item_any in cast(List[Any], entities_any):
            if not isinstance(item_any, dict):
                continue
            item = cast(Dict[str, Any], item_any)
            key = item.get("entity")
            value = item.get("value")
            if not isinstance(key, str) or not key.strip() or value is None:
                continue
            if key not in out:
                out[key] = value
    return out


def _trace_id_from_tracker(tracker: TrackerLike) -> Optional[str]:
    latest_any: Any = getattr(tracker, "latest_message", None) or {}
    latest = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}
    metadata_any = latest.get("metadata")
    metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}

    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        value = metadata.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token:
            return token

    headers_any = metadata.get("headers")
    headers = cast(Dict[str, Any], headers_any) if isinstance(headers_any, dict) else {}
    for key in ("x-trace-id", "x_trace_id", "trace_id", "traceId"):
        value = headers.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token:
            return token

    return None


def _utter_invalid(dispatcher: DispatcherLike, message: str) -> None:
    dispatcher.utter_message(text=message)


def is_skip_signal(slot_value: Any, tracker: Optional[TrackerLike] = None) -> bool:
    candidates: List[str] = []

    if isinstance(slot_value, str) and slot_value.strip():
        candidates.append(slot_value.strip())

    if tracker is not None:
        latest_any: Any = getattr(tracker, "latest_message", None) or {}
        latest = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}

        text_any = latest.get("text")
        if isinstance(text_any, str) and text_any.strip():
            candidates.append(text_any.strip())

        intent_name = ""
        parse_data_any = latest.get("parse_data")
        parse_data = cast(Dict[str, Any], parse_data_any) if isinstance(parse_data_any, dict) else {}
        intent_any = parse_data.get("intent")
        intent = cast(Dict[str, Any], intent_any) if isinstance(intent_any, dict) else {}
        name_any = intent.get("name")
        if isinstance(name_any, str) and name_any.strip():
            intent_name = name_any.strip()

        if not intent_name:
            metadata_any = latest.get("metadata")
            metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}
            metadata_intent_any = metadata.get("intentName")
            if isinstance(metadata_intent_any, str) and metadata_intent_any.strip():
                intent_name = metadata_intent_any.strip()

        if intent_name:
            candidates.append(intent_name)
            candidates.append(f"/{intent_name}")

    for candidate in candidates:
        lowered = candidate.strip().lower()
        if lowered in SKIP_TOKENS:
            return True
    return False


def validate_required_metric(slot_value: Any, dispatcher: DispatcherLike, tracker: TrackerLike) -> Dict[str, Any]:
    language = resolve_language_from_tracker(tracker)
    entities = _latest_entities(tracker)
    source = entities.get("metric") if entities.get("metric") is not None else slot_value
    raw = source if isinstance(source, str) else str(source or "")
    candidates = resolve_metric_candidates(raw)
    if len(candidates) == 1:
        return {"metric": candidates[0]}
    if len(candidates) > 1:
        _utter_invalid(dispatcher, translate("action.guided.validate_metric_ambiguous", language=language))
        return {"metric": None}
    _utter_invalid(dispatcher, translate("action.guided.validate_metric_invalid", language=language))
    return {"metric": None}


def validate_optional_catalog_slot(
    slot_name: str,
    slot_value: Any,
    dispatcher: DispatcherLike,
    tracker: Optional[TrackerLike],
    filename: str,
    prompt: str,
) -> Dict[str, Any]:
    if slot_value is None:
        return {slot_name: None}
    if is_skip_signal(slot_value=slot_value, tracker=tracker):
        return {slot_name: None}

    raw = slot_value if isinstance(slot_value, str) else str(slot_value)
    candidates = resolve_catalog_candidates(filename, raw)
    if len(candidates) == 1:
        return {slot_name: candidates[0]}
    if len(candidates) > 1:
        _utter_invalid(dispatcher, prompt)
        return {slot_name: None}
    _utter_invalid(dispatcher, prompt)
    return {slot_name: None}


def validate_guided_hospital_scope(slot_value: Any, dispatcher: DispatcherLike, tracker: TrackerLike) -> Dict[str, Any]:
    """Resolve hospital scope input to a canonical scope JSON blob.

    Priority order (first match wins, no cross-entity fallbacks):
            1. provider_id / provider_group_id entities
            2. scope_kind entity (`mine`, `provider_group`)
            3. country_code entity
            4. region entity -> unsupported clarify
            5. Bare numeric text with no typed entity -> provider/group clarify
            6. Missing structured scope -> clarify
    """
    language = resolve_language_from_tracker(tracker)
    user_sub = tracker.sender_id
    trace_id = _trace_id_from_tracker(tracker) or uuid4().hex
    entities = _latest_entities(tracker)
    intent: GuidedScopeIntent = parse_guided_scope_intent(slot_value=slot_value, entities=entities)

    with log_context(trace_id=trace_id, sender_id=tracker.sender_id, user_sub=user_sub, validator="guided_hospital_scope"):
        if intent.kind == "all_accessible":
            return {
                "guided_hospital_scope": scope_json(
                    "all_accessible",
                    intent.value if intent.value is not None else "all",
                    label=translate("action.guided.all_hospitals_label", language=language),
                )
            }

        if intent.kind == "mine":
            try:
                mine_scope = resolve_mine_scope_json(user_sub=user_sub, trace_id=trace_id)
            except AnalyticsCenterError as exc:
                details = exc.details if isinstance(exc.details, dict) else {}
                proxy_info = cast(Dict[str, Any], details.get("proxy") or {})
                reason = str(proxy_info.get("reason") or "").strip().lower()
                if exc.status_code == 401 or "cached user access token" in reason or "user token unavailable" in reason:
                    _utter_invalid(dispatcher, translate("action.guided.mine_scope_auth_unavailable", language=language))
                else:
                    _utter_invalid(dispatcher, translate("action.guided.mine_scope_unknown", language=language))
                return {"guided_hospital_scope": None}
            if mine_scope is not None:
                return {"guided_hospital_scope": mine_scope}
            _utter_invalid(dispatcher, translate("action.guided.mine_scope_not_found", language=language))
            return {"guided_hospital_scope": None}

        if intent.kind == "provider_group_id":
            if isinstance(intent.value, int):
                return {"guided_hospital_scope": scope_json("provider_group_id", intent.value, label=str(intent.value))}

        if intent.kind == "provider_id":
            if isinstance(intent.value, int):
                return {"guided_hospital_scope": scope_json("provider_id", intent.value, label=str(intent.value))}

        if intent.kind == "provider_group_mine":
            group_scope = resolve_provider_group_mine_scope_json(user_sub=user_sub, trace_id=trace_id)
            if group_scope is not None:
                return {"guided_hospital_scope": group_scope}
            _utter_invalid(dispatcher, translate("action.guided.mine_scope_not_found", language=language))
            return {"guided_hospital_scope": None}

        if intent.kind == "country_code" and isinstance(intent.value, str):
            client = get_analytics_center_client()
            resolved = client.resolve_country_code(user_sub=user_sub, country_input=intent.value, trace_id=trace_id, raise_on_error=False)
            if resolved:
                return {"guided_hospital_scope": scope_json("country_code", resolved, label=resolved)}

        if intent.kind == "numeric_ambiguous":
            _utter_invalid(dispatcher, translate("action.guided.hospital_scope_numeric_ambiguous", language=language))
            return {"guided_hospital_scope": None}

        if intent.kind == "region_unsupported":
            _utter_invalid(dispatcher, translate("action.guided.hospital_scope_region_unsupported", language=language))
            return {"guided_hospital_scope": None}

        if intent.kind == "missing_structured_scope":
            _utter_invalid(dispatcher, translate("action.guided.hospital_scope_missing_structured", language=language))
            return {"guided_hospital_scope": None}

        _utter_invalid(dispatcher, translate("action.guided.hospital_scope_invalid", language=language))
        return {"guided_hospital_scope": None}


def parse_guided_scope(scope_raw: Any, trace_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if isinstance(scope_raw, dict):
        return cast(Dict[str, Any], scope_raw)
    if isinstance(scope_raw, str) and scope_raw.strip():
        try:
            parsed = json.loads(scope_raw)
            if isinstance(parsed, dict):
                return cast(Dict[str, Any], parsed)
        except Exception:
            logger.debug(
                "Failed to parse guided scope JSON; falling back to None",
                exc_info=True,
                extra=_guided_scope_log_context(
                    trace_id=trace_id,
                    event="actions.guided.scope_parse_failed",
                    operation="parse_guided_scope",
                    raw_scope_type=type(scope_raw).__name__,
                    raw_scope_length=len(scope_raw),
                ),
            )
            return None
    return None


def _optional_slot_value(slots: Dict[str, Any], name: str) -> Optional[str]:
    value = slots.get(name)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered == SKIP_SENTINEL:
        return None

    return text


def build_guided_plan(slots: Dict[str, Any], user_sub: str, trace_id: Optional[str] = None) -> S.AnalysisPlan:
    metric = str(slots.get("metric") or "").strip().upper()
    stroke_type = _optional_slot_value(slots, "stroke_type")
    sex = _optional_slot_value(slots, "sex")
    guided_scope = parse_guided_scope(slots.get("guided_hospital_scope"), trace_id=trace_id)

    filter_clauses: List[S.PredicateFilter] = []
    if isinstance(stroke_type, str) and stroke_type.strip():
        filter_clauses.append(S.PredicateFilter(op="predicate", field="STROKE_TYPE", operator="EQ", value=stroke_type.strip().upper()))
    if isinstance(sex, str) and sex.strip():
        filter_clauses.append(S.PredicateFilter(op="predicate", field="SEX", operator="EQ", value=sex.strip().upper()))

    filter_node: Optional[S.FilterNode] = None
    if len(filter_clauses) == 1:
        filter_node = filter_clauses[0]
    elif len(filter_clauses) > 1:
        filter_node = S.AndFilter(op="and", clauses=filter_clauses)

    metric_origin_scope: Optional[S.OriginScopeSpec] = None
    if isinstance(guided_scope, dict) and guided_scope:
        try:
            metric_origin_scope = S.OriginScopeSpec.model_validate(guided_scope)
        except Exception:
            logger.debug(
                "Failed to validate guided origin scope; falling back to None",
                exc_info=True,
                extra=_guided_scope_log_context(
                    trace_id=trace_id,
                    event="actions.guided.scope_validation_failed",
                    operation="build_guided_plan",
                    raw_scope_keys=sorted(str(key) for key in guided_scope.keys()),
                    scope_type=guided_scope.get("scope_type"),
                    scope_value=guided_scope.get("value"),
                ),
            )
            metric_origin_scope = None

    return S.AnalysisPlan(
        schemaVersion=2,
        charts=[
            S.HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=S.NumericMetricXAxis(kind="numeric_metric", metric=metric),
                yAxis=S.CountAxis(kind="count"),
                filters=filter_node,
                originScope=metric_origin_scope,
            )
        ],
    )


def guided_slots_clear_events() -> List[Dict[str, Any]]:
    return [
        {"event": "slot", "name": "metric", "value": None},
        {"event": "slot", "name": "guided_hospital_scope", "value": None},
        {"event": "slot", "name": "stroke_type", "value": None},
        {"event": "slot", "name": "sex", "value": None},
        {"event": "slot", "name": "group_by", "value": None},
        {"event": "slot", "name": "chart_type", "value": None},
    ]
