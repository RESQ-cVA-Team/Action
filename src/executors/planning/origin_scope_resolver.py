from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

from src.domain.langchain import schema as S
from src.executors.analytics_center.client import AnalyticsCenterError, get_analytics_center_client
from src.util import env as env_util

logger = logging.getLogger(__name__)

_DEFAULT_SCOPE_TYPE = (env_util.get_env("EXECUTOR_DEFAULT_ORIGIN_SCOPE", default="mine") or "mine").strip().lower()
_FAIL_OPEN = env_util.env_flag("EXECUTOR_ORIGIN_SCOPE_FAIL_OPEN", default=False)
_MAX_PROVIDER_IDS_RAW = env_util.get_env("EXECUTOR_ORIGIN_SCOPE_MAX_PROVIDER_IDS", default="500") or "500"


def _parse_max_provider_ids(raw: str) -> int:
    try:
        return max(1, int(raw))
    except Exception:
        return 500


_MAX_PROVIDER_IDS = _parse_max_provider_ids(_MAX_PROVIDER_IDS_RAW)

_PROVIDER_CACHE_TTL_SECONDS_RAW = (
    env_util.get_env("EXECUTOR_PROVIDER_CACHE_TTL_SECONDS", default="300") or "300"
)


def _parse_provider_cache_ttl_seconds(raw: str) -> int:
    try:
        return max(0, int(raw))
    except Exception:
        return 300


_PROVIDER_CACHE_TTL_SECONDS = _parse_provider_cache_ttl_seconds(
    _PROVIDER_CACHE_TTL_SECONDS_RAW
)
_MAX_PROVIDER_CACHE_ENTRIES = 64
_PROVIDER_LIST_CACHE: Dict[tuple[str, str, str], tuple[float, List[Dict[str, Any]]]] = {}

_AUTH_SESSION_MESSAGE = "I could not access your analytics session right now. Please refresh the app and try again."


def _provider_cache_key(
    *,
    cache_type: str,
    user_sub: str,
    country_code: Optional[str],
) -> tuple[str, str, str]:
    return (
        cache_type,
        (user_sub or "").strip(),
        (country_code or "").strip().upper(),
    )


def _provider_cache_get(
    *,
    cache_type: str,
    user_sub: str,
    country_code: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    if _PROVIDER_CACHE_TTL_SECONDS <= 0:
        return None

    key = _provider_cache_key(
        cache_type=cache_type,
        user_sub=user_sub,
        country_code=country_code,
    )
    cached = _PROVIDER_LIST_CACHE.get(key)
    if cached is None:
        return None

    cached_at, payload = cached
    if time.monotonic() - cached_at > float(_PROVIDER_CACHE_TTL_SECONDS):
        _PROVIDER_LIST_CACHE.pop(key, None)
        return None

    # Return a shallow copy so callers cannot mutate cached state.
    return list(payload)


def _provider_cache_set(
    *,
    cache_type: str,
    user_sub: str,
    country_code: Optional[str],
    providers: List[Dict[str, Any]],
) -> None:
    if _PROVIDER_CACHE_TTL_SECONDS <= 0:
        return

    key = _provider_cache_key(
        cache_type=cache_type,
        user_sub=user_sub,
        country_code=country_code,
    )
    _PROVIDER_LIST_CACHE[key] = (time.monotonic(), list(providers))

    if len(_PROVIDER_LIST_CACHE) <= _MAX_PROVIDER_CACHE_ENTRIES:
        return

    # Drop the oldest cache entry.
    oldest_key: Optional[tuple[str, str, str]] = None
    oldest_ts = float("inf")
    for candidate_key, (cached_at, _) in _PROVIDER_LIST_CACHE.items():
        if cached_at < oldest_ts:
            oldest_ts = cached_at
            oldest_key = candidate_key
    if oldest_key is not None:
        _PROVIDER_LIST_CACHE.pop(oldest_key, None)


def _origin_scope_log_context(
    *,
    trace_id: str,
    event: str,
    outcome: str,
    metric_code: str,
    scope: Optional[S.OriginScopeSpec],
    fail_open_for_metric: bool,
    fallback_expected: bool,
    inferred_country_code: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Any] = {
        "trace_id": trace_id,
        "event": event,
        "operation": "_resolve_metric_origin",
        "outcome": outcome,
        "metric_code": metric_code,
        "fail_open_for_metric": fail_open_for_metric,
        "fallback_expected": fallback_expected,
    }
    if scope is not None:
        context["scope_type"] = (scope.scope_type or "").strip().lower() or "-"
        if scope.value is not None:
            context["scope_value"] = scope.value
        if isinstance(scope.label, str) and scope.label.strip():
            context["scope_label"] = scope.label.strip()
        if isinstance(scope.country_code, str) and scope.country_code.strip():
            context["scope_country_code"] = scope.country_code.strip().upper()
    if inferred_country_code:
        context["inferred_country_code"] = inferred_country_code
    return {"log_context": context}


@dataclass
class OriginScopeResolutionError(RuntimeError):
    message: str
    reason: str = "origin_scope_resolution"
    clarification_type: Optional[str] = None
    clarification_options: List[str] = field(default_factory=lambda: cast(List[str], []))

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _provider_id(provider: Dict[str, Any]) -> Optional[int]:
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


def _provider_country_code(provider: Dict[str, Any]) -> Optional[str]:
    for key in ("countryCode", "country_code", "country"):
        value = provider.get(key)
        if isinstance(value, str) and value.strip():
            token = value.strip()
            if len(token) == 2 and token.isalpha():
                return token.upper()
            return token
    return None


def _provider_group_id(group: Dict[str, Any]) -> Optional[int]:
    for key in ("id", "groupId", "providerGroupId"):
        value = group.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _provider_group_name(group: Dict[str, Any]) -> str:
    for key in ("fullName", "name", "shortName", "title"):
        value = group.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _provider_group_country_code(group: Dict[str, Any]) -> Optional[str]:
    for key in ("countryCode", "country_code", "country"):
        value = group.get(key)
        if isinstance(value, str) and value.strip():
            token = value.strip()
            if len(token) == 2 and token.isalpha():
                return token.upper()
            return token
    return None


def _is_missing_cached_token_error(exc: AnalyticsCenterError) -> bool:
    if exc.status_code != 401:
        return False
    details = exc.details if isinstance(exc.details, dict) else {}
    proxy_any = details.get("proxy")
    proxy = cast(Dict[str, Any], proxy_any) if isinstance(proxy_any, dict) else {}
    reason_any = proxy.get("reason")
    reason = reason_any.strip().lower() if isinstance(reason_any, str) and reason_any.strip() else ""
    if "no cached user access token" in reason:
        return True
    return "no cached user access token" in (exc.message or "").lower()


def _raise_if_auth_session_error(exc: AnalyticsCenterError) -> None:
    if _is_missing_cached_token_error(exc):
        raise OriginScopeResolutionError(
            _AUTH_SESSION_MESSAGE,
            reason="authentication",
            clarification_type="auth_session",
        )


def _list_accessible_providers(user_sub: str, trace_id: str, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    cached = _provider_cache_get(
        cache_type="accessible",
        user_sub=user_sub,
        country_code=country_code,
    )
    if cached is not None:
        return cached[:_MAX_PROVIDER_IDS]

    client = get_analytics_center_client()
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 200

    while len(out) < _MAX_PROVIDER_IDS:
        try:
            page = client.list_providers(
                user_sub=user_sub,
                trace_id=trace_id,
                country_code=country_code,
                limit=limit,
                offset=offset,
                raise_on_error=True,
            )
        except AnalyticsCenterError as exc:
            _raise_if_auth_session_error(exc)
            break
        if page is None:
            try:
                page = client.list_providers(
                    user_sub=user_sub,
                    trace_id=trace_id,
                    country_code=country_code,
                    limit=limit,
                    offset=offset,
                    raise_on_error=True,
                )
            except AnalyticsCenterError as exc:
                _raise_if_auth_session_error(exc)
                break

        if not page:
            break

        results_any = page.get("results", [])
        providers: List[Dict[str, Any]] = list(results_any)

        if not providers:
            break

        out.extend(providers)
        total_any = page.get("count")
        total = total_any if isinstance(total_any, int) and total_any >= 0 else None
        offset += len(providers)

        if total is not None and offset >= total:
            break

    trimmed = out[:_MAX_PROVIDER_IDS]
    _provider_cache_set(
        cache_type="accessible",
        user_sub=user_sub,
        country_code=country_code,
        providers=trimmed,
    )
    return trimmed


def _list_all_providers_catalog(user_sub: str, trace_id: str, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    cached = _provider_cache_get(
        cache_type="catalog",
        user_sub=user_sub,
        country_code=country_code,
    )
    if cached is not None:
        return cached[:_MAX_PROVIDER_IDS]

    client = get_analytics_center_client()
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 200

    while len(out) < _MAX_PROVIDER_IDS:
        try:
            page = client.list_providers(
                user_sub=user_sub,
                trace_id=trace_id,
                country_code=country_code,
                limit=limit,
                offset=offset,
                raise_on_error=True,
            )
        except AnalyticsCenterError as exc:
            _raise_if_auth_session_error(exc)
            break

        if not page:
            break

        results_any = page.get("results", [])
        providers: List[Dict[str, Any]] = list(results_any)
        if not providers:
            break

        out.extend(providers)
        total_any = page.get("count")
        total = total_any if isinstance(total_any, int) and total_any >= 0 else None
        offset += len(providers)

        if total is not None and offset >= total:
            break

    trimmed = out[:_MAX_PROVIDER_IDS]
    _provider_cache_set(
        cache_type="catalog",
        user_sub=user_sub,
        country_code=country_code,
        providers=trimmed,
    )
    return trimmed


def _search_accessible_providers_by_name(
    requested_names: List[str],
    user_sub: str,
    trace_id: str,
    country_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    logger.info(
        "[origin_scope_resolver] searching accessible providers by name",
        extra={
            "trace_id": trace_id,
            "requested_names": requested_names,
            "country_code": country_code or "-",
        },
    )
    cached = _provider_cache_get(
        cache_type="accessible",
        user_sub=user_sub,
        country_code=country_code,
    )
    if cached is not None:
        matched: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()
        for requested in requested_names:
            normalized = _normalize_text(requested)
            for provider in _match_providers_by_name(cached, normalized):
                provider_id = _provider_id(provider)
                if provider_id is None or provider_id in seen_ids:
                    continue
                matched.append(provider)
                seen_ids.add(provider_id)
        return matched

    client = get_analytics_center_client()
    matched: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    exact_found: Dict[str, bool] = {
        requested: False for requested in requested_names
    }
    requested_norms = {
        requested: _normalize_text(requested) for requested in requested_names
    }
    offset = 0
    limit = 200

    while True:
        try:
            page = client.list_providers(
                user_sub=user_sub,
                trace_id=trace_id,
                country_code=country_code,
                limit=limit,
                offset=offset,
                raise_on_error=True,
            )
        except AnalyticsCenterError as exc:
            _raise_if_auth_session_error(exc)
            break

        if not page:
            break

        results_any = page.get("results", [])
        providers: List[Dict[str, Any]] = list(results_any)
        if not providers:
            break

        for requested, normalized in requested_norms.items():
            if exact_found.get(requested):
                continue
            page_matches = _match_providers_by_name(providers, normalized)
            for provider in page_matches:
                provider_id = _provider_id(provider)
                if provider_id is None or provider_id in seen_ids:
                    continue
                matched.append(provider)
                seen_ids.add(provider_id)
                if _normalize_text(_provider_name(provider)) == normalized:
                    exact_found[requested] = True

        if all(exact_found.values()):
            break

        total_any = page.get("count")
        total = total_any if isinstance(total_any, int) and total_any >= 0 else None
        offset += len(providers)
        if total is not None and offset >= total:
            break

    return matched


def _match_providers_by_name(providers: List[Dict[str, Any]], normalized: str) -> List[Dict[str, Any]]:
    exact: List[Dict[str, Any]] = []
    fuzzy: List[Dict[str, Any]] = []
    for provider in providers:
        name = _provider_name(provider)
        if not name:
            continue
        provider_norm = _normalize_text(name)
        if provider_norm == normalized:
            exact.append(provider)
        elif normalized in provider_norm or provider_norm in normalized:
            fuzzy.append(provider)
    return exact or fuzzy


def _requested_provider_names(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        out: List[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(token)
        return out
    return []


def _format_name_list(names: List[str], max_items: int = 5) -> str:
    selected = [name for name in names if isinstance(name, str) and name.strip()][:max_items]
    if not selected:
        return ""
    quoted = ['"' + name.replace('"', '\\"') + '"' for name in selected]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"


def _list_accessible_provider_groups(user_sub: str, trace_id: str, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    client = get_analytics_center_client()
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 200

    while len(out) < _MAX_PROVIDER_IDS:
        try:
            page = client.list_provider_groups(
                user_sub=user_sub,
                trace_id=trace_id,
                country=country_code,
                limit=limit,
                offset=offset,
                raise_on_error=True,
            )
        except AnalyticsCenterError as exc:
            _raise_if_auth_session_error(exc)
            break
        if not page:
            break

        results_any = page.get("results", [])
        groups: List[Dict[str, Any]] = list(results_any)

        if not groups:
            break

        out.extend(groups)
        total_any = page.get("count")
        total = total_any if isinstance(total_any, int) and total_any >= 0 else None
        offset += len(groups)

        if total is not None and offset >= total:
            break

    return out[:_MAX_PROVIDER_IDS]


def _resolve_mine_scope(user_sub: str, trace_id: str) -> Optional[S.DataOriginSpec]:
    client = get_analytics_center_client()
    try:
        scope = client.resolve_my_default_scope(user_sub=user_sub, trace_id=trace_id, raise_on_error=True)
    except AnalyticsCenterError as exc:
        _raise_if_auth_session_error(exc)
        return None
    if not isinstance(scope, dict):
        return None

    provider_id_any = scope.get("provider_id")
    if isinstance(provider_id_any, int):
        return S.DataOriginSpec(providerId=[provider_id_any])

    provider_group_id_any = scope.get("provider_group_id")
    if isinstance(provider_group_id_any, int):
        return S.DataOriginSpec(providerGroupId=[provider_group_id_any])

    return None


def _resolve_provider_name(value: Any, user_sub: str, trace_id: str) -> S.DataOriginSpec:
    requested_names = _requested_provider_names(value)
    if not requested_names:
        raise OriginScopeResolutionError(
            "I need a specific hospital name to resolve scope.",
            clarification_type="provider_name",
        )

    providers = _search_accessible_providers_by_name(
        requested_names=requested_names,
        user_sub=user_sub,
        trace_id=trace_id,
    )
    if not providers:
        # If we can still find the hospital in the global catalog, surface a
        # clear access-related message instead of a generic load failure.
        catalog = _list_all_providers_catalog(user_sub=user_sub, trace_id=trace_id)
        inaccessible_names: List[str] = []
        for requested in requested_names:
            normalized = _normalize_text(requested)
            if _match_providers_by_name(catalog, normalized):
                inaccessible_names.append(requested)
        if inaccessible_names:
            listed = _format_name_list(inaccessible_names)
            raise OriginScopeResolutionError(
                f"You do not currently have access to {listed}.",
                clarification_type="provider_name",
            )
        raise OriginScopeResolutionError(
            "I could not load accessible hospitals to resolve the requested scope.",
            clarification_type="provider_name",
        )

    provider_ids: List[int] = []
    inaccessible_names = []
    unknown_names: List[str] = []
    catalog: Optional[List[Dict[str, Any]]] = None

    for requested in requested_names:
        normalized = _normalize_text(requested)
        matches = _match_providers_by_name(providers, normalized)
        if not matches:
            if catalog is None:
                catalog = _list_all_providers_catalog(user_sub=user_sub, trace_id=trace_id)
            catalog_matches = _match_providers_by_name(catalog, normalized)
            if catalog_matches:
                inaccessible_names.append(requested)
            else:
                unknown_names.append(requested)
            continue

        if len(matches) > 1:
            options = [_provider_name(item) for item in matches[:5] if _provider_name(item)]
            raise OriginScopeResolutionError(
                f"I found multiple hospitals matching '{requested}'. Please be more specific.",
                clarification_type="provider_name",
                clarification_options=options,
            )

        provider_id = _provider_id(matches[0])
        if provider_id is None:
            raise OriginScopeResolutionError(
                f"I matched '{requested}', but its provider ID is unavailable.",
                clarification_type="provider_name",
            )
        if provider_id not in provider_ids:
            provider_ids.append(provider_id)

    if inaccessible_names:
        options = [_provider_name(item) for item in providers[:5] if _provider_name(item)]
        listed = _format_name_list(inaccessible_names)
        raise OriginScopeResolutionError(
            f"You do not currently have access to {listed}.",
            clarification_type="provider_name",
            clarification_options=options,
        )

    if unknown_names:
        listed = _format_name_list(unknown_names)
        raise OriginScopeResolutionError(
            f"I could not match {listed} to an accessible provider.",
            clarification_type="provider_name",
        )

    if not provider_ids:
        raise OriginScopeResolutionError(
            "I could not match the requested hospitals to accessible providers.",
            clarification_type="provider_name",
        )

    return S.DataOriginSpec(providerId=provider_ids)


def _resolve_provider_group_name(value: Any, user_sub: str, trace_id: str) -> S.DataOriginSpec:
    if not isinstance(value, str) or not value.strip():
        raise OriginScopeResolutionError(
            "I need a specific provider-group name to resolve scope.",
            clarification_type="provider_group_name",
        )

    normalized = _normalize_text(value)
    groups = _list_accessible_provider_groups(user_sub=user_sub, trace_id=trace_id)
    if not groups:
        raise OriginScopeResolutionError(
            "I could not load accessible provider groups to resolve the requested scope.",
            clarification_type="provider_group_name",
        )

    exact: List[Dict[str, Any]] = []
    fuzzy: List[Dict[str, Any]] = []
    for group in groups:
        name = _provider_group_name(group)
        if not name:
            continue
        group_norm = _normalize_text(name)
        if group_norm == normalized:
            exact.append(group)
        elif normalized in group_norm or group_norm in normalized:
            fuzzy.append(group)

    matches = exact or fuzzy
    if not matches:
        raise OriginScopeResolutionError(
            "I could not match that provider group.",
            clarification_type="provider_group_name",
        )

    if len(matches) > 1:
        options = [_provider_group_name(item) for item in matches[:5] if _provider_group_name(item)]
        raise OriginScopeResolutionError(
            "I found multiple provider groups matching that name. Please be more specific.",
            clarification_type="provider_group_name",
            clarification_options=options,
        )

    group_id = _provider_group_id(matches[0])
    if group_id is None:
        raise OriginScopeResolutionError(
            "I matched the provider group name, but its group ID is unavailable.",
            clarification_type="provider_group_name",
        )

    return S.DataOriginSpec(providerGroupId=[group_id])


def _resolve_country_scope(value: Any, country_code: Optional[str], user_sub: str, trace_id: str) -> S.DataOriginSpec:
    client = get_analytics_center_client()

    raw_country = None
    if isinstance(country_code, str) and country_code.strip():
        raw_country = country_code.strip()
    elif isinstance(value, str) and value.strip():
        raw_country = value.strip()

    if not raw_country:
        raise OriginScopeResolutionError(
            "I need a country code or country name to resolve national scope.",
            clarification_type="country_code",
        )

    resolved_country = client.resolve_country_code(
        user_sub=user_sub,
        country_input=raw_country,
        trace_id=trace_id,
        raise_on_error=False,
    )
    if not resolved_country:
        raise OriginScopeResolutionError(
            f"I could not recognize '{raw_country}' as a valid country name or code. Please use a country name or ISO 2-letter code (e.g. 'Denmark' or 'DK').",
            clarification_type="country_code",
        )

    providers = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id, country_code=resolved_country)
    provider_ids: List[int] = []
    for provider in providers:
        provider_id = _provider_id(provider)
        if provider_id is not None and provider_id not in provider_ids:
            provider_ids.append(provider_id)

    if not provider_ids:
        # Try to surface which countries the user does have access to
        accessible_countries: List[str] = []
        try:
            all_providers = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id)
            seen: set[str] = set()
            for provider in all_providers:
                country = _provider_country_code(provider)
                resolved_c = client.resolve_country_code(user_sub=user_sub, country_input=country or "", trace_id=trace_id, raise_on_error=False) if country else None
                if resolved_c and resolved_c not in seen:
                    seen.add(resolved_c)
                    accessible_countries.append(resolved_c)
        except Exception:
            pass

        if accessible_countries:
            countries_hint = ", ".join(accessible_countries)
            raise OriginScopeResolutionError(
                f"You do not have access to any hospitals in {resolved_country}. You can request national data for: {countries_hint}.",
                clarification_type="country_code",
            )
        raise OriginScopeResolutionError(
            f"You do not have access to any hospitals in {resolved_country}.",
            clarification_type="country_code",
        )

    return S.DataOriginSpec(providerId=provider_ids)


def _infer_country_code_from_data_origin(data_origin: S.DataOriginSpec, user_sub: str, trace_id: str) -> Optional[str]:
    client = get_analytics_center_client()

    provider_ids = list(cast(Optional[List[int]], getattr(data_origin, "provider_id", None)) or [])
    if provider_ids:
        providers = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id)
        by_id: Dict[int, Dict[str, Any]] = {}
        for provider in providers:
            pid = _provider_id(provider)
            if pid is not None:
                by_id[pid] = provider

        for pid in provider_ids:
            provider = by_id.get(pid)
            if provider is None:
                continue
            country = _provider_country_code(provider)
            if not country:
                continue
            resolved_country = client.resolve_country_code(
                user_sub=user_sub,
                country_input=country,
                trace_id=trace_id,
                raise_on_error=False,
            )
            if resolved_country:
                return resolved_country

    provider_group_ids = list(cast(Optional[List[int]], getattr(data_origin, "provider_group_id", None)) or [])
    if provider_group_ids:
        groups = _list_accessible_provider_groups(user_sub=user_sub, trace_id=trace_id)
        by_id_group: Dict[int, Dict[str, Any]] = {}
        for group in groups:
            gid = _provider_group_id(group)
            if gid is not None:
                by_id_group[gid] = group

        for gid in provider_group_ids:
            group = by_id_group.get(gid)
            if group is None:
                continue
            country = _provider_group_country_code(group)
            if not country:
                continue
            resolved_country = client.resolve_country_code(
                user_sub=user_sub,
                country_input=country,
                trace_id=trace_id,
                raise_on_error=False,
            )
            if resolved_country:
                return resolved_country

    return None


def _infer_user_country_code(user_sub: str, trace_id: str) -> Optional[str]:
    mine_scope = _resolve_mine_scope(user_sub=user_sub, trace_id=trace_id)
    if mine_scope is None:
        return None
    return _infer_country_code_from_data_origin(data_origin=mine_scope, user_sub=user_sub, trace_id=trace_id)


def _resolve_scope(
    scope: S.OriginScopeSpec,
    user_sub: str,
    trace_id: str,
    inferred_country_code: Optional[str] = None,
) -> Optional[S.DataOriginSpec]:
    scope_type = (scope.scope_type or "").strip().lower()
    value = scope.value

    if scope_type == "mine":
        resolved = _resolve_mine_scope(user_sub=user_sub, trace_id=trace_id)
        if resolved is None:
            raise OriginScopeResolutionError(
                "I could not resolve your default hospital scope right now.",
                clarification_type="mine",
            )
        return resolved

    if scope_type == "provider_id":
        if isinstance(value, int):
            provider_id = value
        elif isinstance(value, str) and value.strip().isdigit():
            provider_id = int(value.strip())
        else:
            raise OriginScopeResolutionError(
                "Provider scope requires a numeric provider ID.",
                clarification_type="provider_id",
            )
        # Unlike provider_name (which only ever matches within the caller's
        # own accessible-provider set), a numeric ID was previously accepted
        # as-is with no check that the caller can actually see that
        # provider -- any id the planner produced, from any source, granted
        # that provider's data outright.
        accessible = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id)
        accessible_ids = {pid for pid in (_provider_id(p) for p in accessible) if pid is not None}
        if provider_id not in accessible_ids:
            raise OriginScopeResolutionError(
                f"You do not currently have access to provider {provider_id}.",
                clarification_type="provider_id",
            )
        return S.DataOriginSpec(providerId=[provider_id])

    if scope_type == "provider_name":
        return _resolve_provider_name(value=value, user_sub=user_sub, trace_id=trace_id)

    if scope_type == "provider_group_id":
        if isinstance(value, int):
            group_id = value
        elif isinstance(value, str) and value.strip().isdigit():
            group_id = int(value.strip())
        else:
            raise OriginScopeResolutionError(
                "Provider-group scope requires a numeric group ID.",
                clarification_type="provider_group_id",
            )
        # Same gap as provider_id above -- a numeric group ID was accepted
        # with no check the caller can see that provider group.
        accessible_groups = _list_accessible_provider_groups(user_sub=user_sub, trace_id=trace_id)
        accessible_group_ids = {gid for gid in (_provider_group_id(g) for g in accessible_groups) if gid is not None}
        if group_id not in accessible_group_ids:
            raise OriginScopeResolutionError(
                f"You do not currently have access to provider group {group_id}.",
                clarification_type="provider_group_id",
            )
        return S.DataOriginSpec(providerGroupId=[group_id])

    if scope_type == "country_code":
        return _resolve_country_scope(
            value=value,
            country_code=scope.country_code or inferred_country_code,
            user_sub=user_sub,
            trace_id=trace_id,
        )

    if scope_type == "country_average":
        return _resolve_country_scope(
            value=value,
            country_code=scope.country_code or inferred_country_code,
            user_sub=user_sub,
            trace_id=trace_id,
        )

    if scope_type == "all_accessible":
        providers = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id)
        provider_ids: List[int] = []
        for provider in providers:
            provider_id = _provider_id(provider)
            if provider_id is not None and provider_id not in provider_ids:
                provider_ids.append(provider_id)
        if not provider_ids:
            return None
        return S.DataOriginSpec(providerId=provider_ids)

    if scope_type == "provider_group_name":
        return _resolve_provider_group_name(value=value, user_sub=user_sub, trace_id=trace_id)

    raise OriginScopeResolutionError(
        f"Unsupported origin scope type: {scope_type}",
        clarification_type="origin_scope",
    )


def _default_scope_ref() -> Optional[S.OriginScopeSpec]:
    token = (_DEFAULT_SCOPE_TYPE or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"", "none", "executor_default", "legacy_default"}:
        return None
    return S.OriginScopeSpec(scopeType=token)


def _resolve_metric_origin(
    metric: S.MetricSpec,
    *,
    default_scope: Optional[S.OriginScopeSpec],
    user_sub: str,
    trace_id: str,
    fail_open_for_default_scope: bool,
    inferred_country_code: Optional[str],
) -> S.MetricSpec:
    metric_data_origin = cast(Optional[S.DataOriginSpec], getattr(metric, "data_origin", None))
    metric_origin_scope = cast(Optional[S.OriginScopeSpec], getattr(metric, "origin_scope", None))

    if metric_data_origin is not None:
        return metric

    scope_ref = metric_origin_scope or default_scope
    resolved_data_origin: Optional[S.DataOriginSpec] = None
    fail_open_for_metric = fail_open_for_default_scope and metric_origin_scope is None

    if scope_ref is not None:
        try:
            resolved_data_origin = _resolve_scope(
                scope=scope_ref,
                user_sub=user_sub,
                trace_id=trace_id,
                inferred_country_code=inferred_country_code,
            )
        except OriginScopeResolutionError:
            if not fail_open_for_metric:
                raise
            logger.warning(
                "Origin scope resolution failed; falling back to executor default data origin",
                exc_info=True,
                extra=_origin_scope_log_context(
                    trace_id=trace_id,
                    event="origin_scope.metric_resolution.fail_open",
                    outcome="degraded",
                    metric_code=metric.metric,
                    scope=scope_ref,
                    fail_open_for_metric=fail_open_for_metric,
                    fallback_expected=True,
                    inferred_country_code=inferred_country_code,
                ),
            )
            resolved_data_origin = None
        except Exception:
            if not fail_open_for_metric:
                raise OriginScopeResolutionError("Origin scope resolution failed unexpectedly.")
            logger.warning(
                "Unexpected origin scope resolution failure; falling back to executor default data origin",
                exc_info=True,
                extra=_origin_scope_log_context(
                    trace_id=trace_id,
                    event="origin_scope.metric_resolution.fail_open_unexpected",
                    outcome="degraded",
                    metric_code=metric.metric,
                    scope=scope_ref,
                    fail_open_for_metric=fail_open_for_metric,
                    fallback_expected=False,
                    inferred_country_code=inferred_country_code,
                ),
            )
            resolved_data_origin = None

    return S.MetricSpec(
        metric=metric.metric,
        dataOrigin=resolved_data_origin,
        originScope=scope_ref,
    )


def _metric_needs_country_inference(metric: S.MetricSpec) -> bool:
    scope = cast(Optional[S.OriginScopeSpec], getattr(metric, "origin_scope", None))
    if scope is None:
        return False
    scope_type = (scope.scope_type or "").strip().lower()
    if scope_type not in {"country_code", "country_average"}:
        return False

    explicit_country = None
    if isinstance(scope.country_code, str) and scope.country_code.strip():
        explicit_country = scope.country_code.strip()
    elif isinstance(scope.value, str) and scope.value.strip():
        explicit_country = scope.value.strip()

    return not bool(explicit_country)


def _plan_needs_country_inference(plan: S.AnalysisPlan) -> bool:
    for chart in plan.charts or []:
        for metric in chart.metrics:
            if _metric_needs_country_inference(metric):
                return True

    for test in plan.statistical_tests or []:
        for metric in test.metrics:
            if _metric_needs_country_inference(metric):
                return True

    return False


def resolve_plan_metric_origins(plan: S.AnalysisPlan, user_sub: str, trace_id: str) -> S.AnalysisPlan:
    default_scope = _default_scope_ref()
    inferred_country_code = _infer_user_country_code(user_sub=user_sub, trace_id=trace_id) if _plan_needs_country_inference(plan) else None

    charts = plan.charts or []
    resolved_charts: List[S.ChartSpec] = []
    for chart in charts:
        resolved_metrics: List[S.MetricSpec] = []
        for metric in chart.metrics:
            resolved_metric = _resolve_metric_origin(
                metric,
                default_scope=default_scope,
                user_sub=user_sub,
                trace_id=trace_id,
                fail_open_for_default_scope=_FAIL_OPEN,
                inferred_country_code=inferred_country_code,
            )
            resolved_metrics.append(resolved_metric)

        chart_filters = cast(Optional[S.FilterNode], getattr(chart, "filters", None))
        chart_semantics = cast(Optional[S.AnalysisSemanticsSpec], getattr(chart, "semantics", None))
        chart_numeric_resolution = cast(Optional[S.NumericResolutionSpec], getattr(chart, "numeric_resolution", None))
        resolved_chart = S.ChartSpec(
            chart_type=chart.chart_type,
            semantics=chart_semantics,
            filters=chart_filters,
            metrics=resolved_metrics,
            numericResolution=chart_numeric_resolution,
        )
        resolved_charts.append(resolved_chart)

    statistical_tests = plan.statistical_tests or []
    resolved_tests: List[S.StatisticalTestSpec] = []
    for test in statistical_tests:
        resolved_test_metrics: List[S.MetricSpec] = []
        for metric in test.metrics:
            resolved_metric = _resolve_metric_origin(
                metric,
                default_scope=default_scope,
                user_sub=user_sub,
                trace_id=trace_id,
                fail_open_for_default_scope=False,
                inferred_country_code=inferred_country_code,
            )
            resolved_test_metrics.append(resolved_metric)

        resolved_test = S.StatisticalTestSpec(
            test_type=test.test_type,
            metrics=resolved_test_metrics,
            group_by=cast(Optional[List[S.GroupBySpec]], getattr(test, "group_by", None)),
            filters=cast(Optional[S.FilterNode], getattr(test, "filters", None)),
        )
        resolved_tests.append(resolved_test)

    return S.AnalysisPlan(charts=resolved_charts or None, statistical_tests=resolved_tests or None)
