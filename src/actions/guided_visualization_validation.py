from __future__ import annotations

import json
import logging
import os
import socket
from typing import Any, Dict, List, Optional, Protocol, cast
from uuid import uuid4

from src.actions.i18n import resolve_language_from_tracker, translate
from src.actions.ssot_lookup import normalize_text, resolve_catalog_candidates, resolve_metric_candidates
from src.domain.langchain import schema as S
from src.executors.analytics_center.client import AnalyticsCenterError, get_analytics_center_client
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

SKIP_SENTINEL = "__skip__"
ALL_SCOPE_TOKENS = {"all", "all hospitals", "all sites", "all providers"}
MINE_SCOPE_TOKENS = {
    "mine",
    "my",
    "my hospital",
    "my site",
    "my center",
    "my centre",
    "our hospital",
    "our site",
    "our center",
    "our centre",
}
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


def _runtime_instance_fields() -> tuple[int, str]:
    return os.getpid(), socket.gethostname()


def _extract_provider_id(provider: Dict[str, Any]) -> Optional[int]:
    for key in ("id", "providerId", "provider_id"):
        value = provider.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _provider_name(provider: Dict[str, Any]) -> str:
    for key in ("nameEnglish", "nameNative", "shortName", "name"):
        value = provider.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _resolve_mine_scope(user_sub: str, trace_id: str) -> Optional[str]:
    """Resolve personal scope ('mine') to a concrete provider scope when possible.

    Strategy:
    - Use analytics-center /myself settings first (currentProvider/currentProviderGroup).
    - If exactly one provider is visible for the user, use it.
    - Else, look for an explicitly flagged default/current provider.
    - Otherwise, return None and let the caller request clarification.
    """

    client = get_analytics_center_client()
    default_scope = client.resolve_my_default_scope(user_sub=user_sub, trace_id=trace_id, raise_on_error=False)
    if isinstance(default_scope, dict):
        provider_id_any = default_scope.get("provider_id")
        if isinstance(provider_id_any, int):
            return _json_scope("provider_id", provider_id_any)

        provider_group_id_any = default_scope.get("provider_group_id")
        if isinstance(provider_group_id_any, int):
            return _json_scope("group_id", provider_group_id_any)

    page = client.list_providers(user_sub=user_sub, user=user_sub, limit=200, offset=0, trace_id=trace_id, raise_on_error=False)
    if not page:
        page = client.list_providers(user_sub=user_sub, limit=200, offset=0, trace_id=trace_id, raise_on_error=False)
    providers_any: Any = page.get("results", []) if isinstance(page, dict) else []
    providers: List[Dict[str, Any]] = []
    if isinstance(providers_any, list):
        for provider_any in cast(List[Any], providers_any):
            if isinstance(provider_any, dict):
                providers.append(cast(Dict[str, Any], provider_any))

    if not providers:
        return None

    if len(providers) == 1:
        provider = providers[0]
        provider_id = _extract_provider_id(provider)
        label = _provider_name(provider)
        if provider_id is not None:
            return _json_scope("provider_id", provider_id, label=label)
        if label:
            return _json_scope("hospital_name", label, label=label)
        return None

    default_flag_keys = (
        "isDefault",
        "default",
        "isPrimary",
        "primary",
        "isCurrent",
        "current",
        "selected",
        "isUserProvider",
        "isMine",
    )
    flagged = [p for p in providers if any(_is_truthy_flag(p.get(key)) for key in default_flag_keys)]
    if len(flagged) == 1:
        provider = flagged[0]
        provider_id = _extract_provider_id(provider)
        label = _provider_name(provider)
        if provider_id is not None:
            return _json_scope("provider_id", provider_id, label=label)
        if label:
            return _json_scope("hospital_name", label, label=label)

    return None


def _json_scope(scope_type: str, value: Any, label: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"scope_type": scope_type, "value": value}
    if isinstance(label, str) and label.strip():
        payload["label"] = label.strip()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
    language = resolve_language_from_tracker(tracker)
    user_sub = tracker.sender_id
    trace_id = _trace_id_from_tracker(tracker) or uuid4().hex
    client = get_analytics_center_client()
    entities = _latest_entities(tracker)
    with log_context(trace_id=trace_id, sender_id=tracker.sender_id, user_sub=user_sub, validator="guided_hospital_scope"):
        scope_ref_any = entities.get("hospital_scope_reference")
        if isinstance(scope_ref_any, str) and scope_ref_any.strip():
            scope_ref_norm = normalize_text(scope_ref_any)
            if scope_ref_norm in ALL_SCOPE_TOKENS:
                return {
                    "guided_hospital_scope": _json_scope(
                        "all",
                        "all",
                        label=translate("action.guided.all_hospitals_label", language=language),
                    )
                }
            if scope_ref_norm in MINE_SCOPE_TOKENS:
                process_id, hostname = _runtime_instance_fields()
                logger.debug(
                    "[GuidedVisualizationValidation] Mine scope validation",
                    extra={
                        "log_context": {
                            "pid": process_id,
                            "host": hostname,
                            "scope_ref": scope_ref_any,
                        }
                    },
                )
                try:
                    mine_scope = _resolve_mine_scope(user_sub=user_sub, trace_id=trace_id)
                except AnalyticsCenterError as exc:
                    details = exc.details if isinstance(exc.details, dict) else {}
                    proxy_any = details.get("proxy")
                    proxy_info = cast(Dict[str, Any], proxy_any) if isinstance(proxy_any, dict) else {}
                    reason_any = proxy_info.get("reason")
                    reason = reason_any.strip() if isinstance(reason_any, str) and reason_any.strip() else ""

                    if exc.status_code == 401 or "cached user access token" in reason.lower() or "user token unavailable" in reason.lower():
                        _utter_invalid(
                            dispatcher,
                            translate("action.guided.mine_scope_auth_unavailable", language=language),
                        )
                        return {"guided_hospital_scope": None}

                    _utter_invalid(
                        dispatcher,
                        translate("action.guided.mine_scope_unknown", language=language),
                    )
                    return {"guided_hospital_scope": None}
                if mine_scope is not None:
                    return {"guided_hospital_scope": mine_scope}
                _utter_invalid(
                    dispatcher,
                    translate("action.guided.mine_scope_not_found", language=language),
                )
                return {"guided_hospital_scope": None}

        country_any = entities.get("country_code") or entities.get("countryCode") or entities.get("country")
        if isinstance(country_any, str) and country_any.strip():
            resolved = client.resolve_country_code(user_sub=user_sub, country_input=country_any.strip(), trace_id=trace_id, raise_on_error=False)
            if resolved:
                return {"guided_hospital_scope": _json_scope("country_code", resolved, label=resolved)}

        hospital_any = entities.get("hospital_name") or entities.get("hospital") or entities.get("provider") or slot_value
        if isinstance(hospital_any, str) and hospital_any.strip() and normalize_text(hospital_any) not in ALL_SCOPE_TOKENS:
            page = client.list_providers(user_sub=user_sub, limit=200, offset=0, trace_id=trace_id, raise_on_error=False)
            providers_any: Any = page.get("results", []) if isinstance(page, dict) else []
            providers_list: List[Dict[str, Any]] = []
            if isinstance(providers_any, list):
                for provider_any in cast(List[Any], providers_any):
                    if isinstance(provider_any, dict):
                        providers_list.append(cast(Dict[str, Any], provider_any))
            normalized = normalize_text(hospital_any)
            exact_matches: List[Dict[str, Any]] = []
            fuzzy_matches: List[Dict[str, Any]] = []
            for provider_any in providers_list:
                provider = provider_any
                name = _provider_name(provider)
                if not name:
                    continue
                provider_norm = normalize_text(name)
                if provider_norm == normalized:
                    exact_matches.append(provider)
                elif normalized in provider_norm or provider_norm in normalized:
                    fuzzy_matches.append(provider)
            matches = exact_matches or fuzzy_matches
            if len(matches) == 1:
                provider = matches[0]
                provider_id = _extract_provider_id(provider)
                label = _provider_name(provider)
                if provider_id is not None:
                    return {"guided_hospital_scope": _json_scope("provider_id", provider_id, label=label)}
                return {"guided_hospital_scope": _json_scope("hospital_name", label or hospital_any.strip(), label=label or hospital_any.strip())}
            if len(matches) > 1:
                _utter_invalid(dispatcher, translate("action.guided.hospital_name_ambiguous", language=language))
                return {"guided_hospital_scope": None}

        group_any = entities.get("group_id") or entities.get("group") or slot_value
        if isinstance(group_any, int):
            return {"guided_hospital_scope": _json_scope("group_id", group_any, label=str(group_any))}
        if isinstance(group_any, str) and group_any.strip().isdigit():
            return {"guided_hospital_scope": _json_scope("group_id", int(group_any.strip()), label=group_any.strip())}

        raw_source = scope_ref_any if scope_ref_any is not None else slot_value
        raw = raw_source if isinstance(raw_source, str) else str(raw_source or "")
        if normalize_text(raw) in ALL_SCOPE_TOKENS:
            return {
                "guided_hospital_scope": _json_scope(
                    "all",
                    "all",
                    label=translate("action.guided.all_hospitals_label", language=language),
                )
            }

        _utter_invalid(dispatcher, translate("action.guided.hospital_scope_invalid", language=language))
        return {"guided_hospital_scope": None}


def parse_guided_scope(scope_raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(scope_raw, dict):
        return cast(Dict[str, Any], scope_raw)
    if isinstance(scope_raw, str) and scope_raw.strip():
        try:
            parsed = json.loads(scope_raw)
            if isinstance(parsed, dict):
                return cast(Dict[str, Any], parsed)
        except Exception:
            return None
    return None


def resolve_scope_to_data_origin(scope: Dict[str, Any], user_sub: str, trace_id: str) -> Optional[Dict[str, Any]]:
    scope_type = str(scope.get("scope_type") or "").strip().lower()
    value = scope.get("value")
    client = get_analytics_center_client()

    if scope_type == "provider_id":
        if isinstance(value, int):
            return {"providerId": [value]}
        if isinstance(value, str) and value.strip().isdigit():
            return {"providerId": [int(value.strip())]}
        return None

    if scope_type == "group_id":
        if isinstance(value, int):
            return {"providerGroupId": [value]}
        if isinstance(value, str) and value.strip().isdigit():
            return {"providerGroupId": [int(value.strip())]}
        return None

    if scope_type == "country_code":
        if not isinstance(value, str) or not value.strip():
            return None
        country_code = value.strip().upper()
        provider_ids: List[int] = []
        offset = 0
        limit = 200
        while True:
            page = client.list_providers(
                user_sub=user_sub,
                limit=limit,
                offset=offset,
                country_code=country_code,
                trace_id=trace_id,
                raise_on_error=False,
            )
            if not page:
                break
            providers_any: Any = page.get("results", [])
            providers_list: List[Dict[str, Any]] = []
            if isinstance(providers_any, list):
                for provider_any in cast(List[Any], providers_any):
                    if isinstance(provider_any, dict):
                        providers_list.append(cast(Dict[str, Any], provider_any))
            for provider_any in providers_list:
                provider_id = _extract_provider_id(provider_any)
                if provider_id is not None:
                    provider_ids.append(provider_id)
            returned = len(providers_list)
            total = page.get("count")
            offset += returned
            if returned == 0 or offset >= total:
                break
        provider_ids = sorted(set(provider_ids))
        return {"providerId": provider_ids} if provider_ids else None

    if scope_type == "all":
        return None

    if scope_type == "hospital_name" and isinstance(value, str) and value.strip():
        page = client.list_providers(user_sub=user_sub, limit=200, offset=0, trace_id=trace_id, raise_on_error=False)
        providers_any: Any = page.get("results", []) if isinstance(page, dict) else []
        providers_list: List[Dict[str, Any]] = []
        if isinstance(providers_any, list):
            for provider_any in cast(List[Any], providers_any):
                if isinstance(provider_any, dict):
                    providers_list.append(cast(Dict[str, Any], provider_any))
        normalized = normalize_text(value)
        for provider_any in providers_list:
            provider = provider_any
            if normalize_text(_provider_name(provider)) != normalized:
                continue
            provider_id = _extract_provider_id(provider)
            if provider_id is not None:
                return {"providerId": [provider_id]}
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
    chart_type = (_optional_slot_value(slots, "chart_type") or "LINE").upper()
    group_by = _optional_slot_value(slots, "group_by")
    stroke_type = _optional_slot_value(slots, "stroke_type")
    sex = _optional_slot_value(slots, "sex")
    guided_scope = parse_guided_scope(slots.get("guided_hospital_scope"))

    filters: List[S.FilterNode] = []
    if isinstance(stroke_type, str) and stroke_type.strip():
        filters.append(S.StrokeFilter(value=stroke_type.strip().upper()))
    if isinstance(sex, str) and sex.strip():
        filters.append(S.SexFilter(value=sex.strip().upper()))

    filter_node: Optional[S.FilterNode] = None
    if len(filters) == 1:
        filter_node = filters[0]
    elif len(filters) > 1:
        filter_node = S.AndFilter(and_=filters)

    group_specs: List[S.GroupBySpec] = []
    if isinstance(group_by, str) and group_by.strip():
        group_specs.append(S.GroupByCanonicalField(field=group_by.strip().upper()))

    metric_origin_scope: Optional[S.OriginScopeSpec] = None
    if isinstance(guided_scope, dict) and guided_scope:
        try:
            metric_origin_scope = S.OriginScopeSpec.model_validate(guided_scope)
        except Exception:
            metric_origin_scope = None

    return S.AnalysisPlan(
        charts=[
            S.ChartSpec(
                chart_type=chart_type,
                filters=filter_node,
                group_by=group_specs or None,
                metrics=[S.MetricSpec(metric=metric, originScope=metric_origin_scope)],
            )
        ],
        statistical_tests=None,
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
