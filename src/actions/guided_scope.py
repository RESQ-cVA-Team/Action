from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, cast

from src.actions.ssot_lookup import normalize_text
from src.executors.analytics_center.client import get_analytics_center_client

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

IntentKind = Literal[
    "all_accessible",
    "mine",
    "provider_id",
    "provider_group_mine",
    "provider_group_id",
    "country_code",
    "region_unsupported",
    "numeric_ambiguous",
    "missing_structured_scope",
    "invalid",
]


@dataclass(frozen=True)
class GuidedScopeIntent:
    kind: IntentKind
    value: Optional[Any] = None


@dataclass(frozen=True)
class ProviderNameResolution:
    status: Literal["match", "ambiguous", "not_found"]
    scope_json: Optional[str] = None


def _json_scope(scope_type: str, value: Any, label: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"scope_type": scope_type, "value": value}
    if isinstance(label, str) and label.strip():
        payload["label"] = label.strip()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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


def parse_guided_scope_intent(slot_value: Any, entities: Dict[str, Any]) -> GuidedScopeIntent:
    provider_id_raw = entities.get("provider_id") or entities.get("providerId")
    provider_group_id_raw = (
        entities.get("provider_group_id")
        or entities.get("providerGroupId")
    )

    if provider_id_raw is not None and provider_group_id_raw is not None:
        return GuidedScopeIntent(kind="numeric_ambiguous")

    if provider_id_raw is not None:
        if isinstance(provider_id_raw, int):
            return GuidedScopeIntent(kind="provider_id", value=provider_id_raw)
        if isinstance(provider_id_raw, str) and provider_id_raw.strip().isdigit():
            return GuidedScopeIntent(kind="provider_id", value=int(provider_id_raw.strip()))

    if provider_group_id_raw is not None:
        if isinstance(provider_group_id_raw, int):
            return GuidedScopeIntent(kind="provider_group_id", value=provider_group_id_raw)
        if isinstance(provider_group_id_raw, str) and provider_group_id_raw.strip().isdigit():
            return GuidedScopeIntent(kind="provider_group_id", value=int(provider_group_id_raw.strip()))

    scope_kind_raw = entities.get("scope_kind")
    if isinstance(scope_kind_raw, str) and scope_kind_raw.strip():
        scope_kind = normalize_text(scope_kind_raw)
        if scope_kind == "mine":
            return GuidedScopeIntent(kind="mine")
        if scope_kind in {"provider_group", "provider group", "provider-group"}:
            return GuidedScopeIntent(kind="provider_group_mine")

    region_raw = entities.get("region")
    if isinstance(region_raw, str) and region_raw.strip():
        return GuidedScopeIntent(kind="region_unsupported", value=region_raw.strip().upper())

    scope_ref = entities.get("hospital_scope_reference")
    if isinstance(scope_ref, str) and scope_ref.strip():
        norm = normalize_text(scope_ref)
        if norm in ALL_SCOPE_TOKENS:
            return GuidedScopeIntent(kind="all_accessible", value="all")
        if norm in MINE_SCOPE_TOKENS:
            return GuidedScopeIntent(kind="mine")

    raw_text = str(slot_value or "").strip()
    raw_norm = normalize_text(raw_text)

    if raw_norm in ALL_SCOPE_TOKENS:
        return GuidedScopeIntent(kind="all_accessible", value="all")
    if raw_norm in MINE_SCOPE_TOKENS:
        return GuidedScopeIntent(kind="mine")

    country_raw = entities.get("country_code") or entities.get("countryCode") or entities.get("country")
    if isinstance(country_raw, str) and country_raw.strip():
        return GuidedScopeIntent(kind="country_code", value=country_raw.strip())

    # Strict post-Rasa behavior: no free-text hospital-name fallback in Action.
    if isinstance(raw_text, str) and raw_text.strip() and not raw_text.isdigit():
        return GuidedScopeIntent(kind="missing_structured_scope")

    if raw_text.isdigit():
        return GuidedScopeIntent(kind="numeric_ambiguous", value=raw_text)

    return GuidedScopeIntent(kind="invalid")


def resolve_provider_group_mine_scope_json(user_sub: str, trace_id: str) -> Optional[str]:
    client = get_analytics_center_client()
    default_scope = client.resolve_my_default_scope(user_sub=user_sub, trace_id=trace_id, raise_on_error=False)
    if isinstance(default_scope, dict):
        group_id_any = default_scope.get("provider_group_id")
        if isinstance(group_id_any, int):
            return _json_scope("provider_group_id", group_id_any)
    return None


def resolve_mine_scope_json(user_sub: str, trace_id: str) -> Optional[str]:
    client = get_analytics_center_client()
    default_scope = client.resolve_my_default_scope(user_sub=user_sub, trace_id=trace_id, raise_on_error=False)
    if isinstance(default_scope, dict):
        provider_id_any = default_scope.get("provider_id")
        if isinstance(provider_id_any, int):
            return _json_scope("provider_id", provider_id_any)

        provider_group_id_any = default_scope.get("provider_group_id")
        if isinstance(provider_group_id_any, int):
            return _json_scope("provider_group_id", provider_group_id_any)

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
            return _json_scope("provider_name", label, label=label)
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
            return _json_scope("provider_name", label, label=label)

    return None


def resolve_provider_name_scope_json(name: str, *, user_sub: str, trace_id: str) -> ProviderNameResolution:
    client = get_analytics_center_client()
    page = client.list_providers(user_sub=user_sub, limit=200, offset=0, trace_id=trace_id, raise_on_error=False)
    providers_any: Any = page.get("results", []) if isinstance(page, dict) else []
    providers: List[Dict[str, Any]] = [
        cast(Dict[str, Any], p) for p in (providers_any if isinstance(providers_any, list) else []) if isinstance(p, dict)
    ]

    normalized = normalize_text(name)
    exact: List[Dict[str, Any]] = []
    fuzzy: List[Dict[str, Any]] = []
    for provider in providers:
        pname = _provider_name(provider)
        if not pname:
            continue
        pnorm = normalize_text(pname)
        if pnorm == normalized:
            exact.append(provider)
        elif normalized in pnorm or pnorm in normalized:
            fuzzy.append(provider)

    matches = exact or fuzzy
    if len(matches) == 1:
        provider = matches[0]
        provider_id = _extract_provider_id(provider)
        label = _provider_name(provider)
        if provider_id is not None:
            return ProviderNameResolution(status="match", scope_json=_json_scope("provider_id", provider_id, label=label))
        return ProviderNameResolution(
            status="match",
            scope_json=_json_scope("provider_name", label or name.strip(), label=label or name.strip()),
        )
    if len(matches) > 1:
        return ProviderNameResolution(status="ambiguous")
    return ProviderNameResolution(status="not_found")


def scope_json(scope_type: str, value: Any, label: Optional[str] = None) -> str:
    return _json_scope(scope_type=scope_type, value=value, label=label)
