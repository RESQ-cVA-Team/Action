from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

from src.domain.langchain import schema as S
from src.executors.analytics_center.client import get_analytics_center_client
from src.util import env as env_util

logger = logging.getLogger(__name__)

_DEFAULT_SCOPE_TYPE = (env_util.get_env("EXECUTOR_DEFAULT_ORIGIN_SCOPE", default="mine") or "mine").strip().lower()
_FAIL_OPEN = env_util.env_flag("EXECUTOR_ORIGIN_SCOPE_FAIL_OPEN", default=True)
_MAX_PROVIDER_IDS_RAW = env_util.get_env("EXECUTOR_ORIGIN_SCOPE_MAX_PROVIDER_IDS", default="500") or "500"


def _parse_max_provider_ids(raw: str) -> int:
    try:
        return max(1, int(raw))
    except Exception:
        return 500


_MAX_PROVIDER_IDS = _parse_max_provider_ids(_MAX_PROVIDER_IDS_RAW)


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


def _list_accessible_providers(user_sub: str, trace_id: str, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    client = get_analytics_center_client()
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 200

    while len(out) < _MAX_PROVIDER_IDS:
        page = client.list_providers(
            user_sub=user_sub,
            trace_id=trace_id,
            user=user_sub,
            country_code=country_code,
            limit=limit,
            offset=offset,
            raise_on_error=False,
        )
        if page is None:
            page = client.list_providers(
                user_sub=user_sub,
                trace_id=trace_id,
                country_code=country_code,
                limit=limit,
                offset=offset,
                raise_on_error=False,
            )

        if not page:
            break

        results_any = page.get("results", [])
        providers: List[Dict[str, Any]] = list(results_any)

        if not providers:
            break

        out.extend(providers)
        total_any = page.get("count")
        total = total_any if total_any >= 0 else None
        offset += len(providers)

        if total is not None and offset >= total:
            break

    return out[:_MAX_PROVIDER_IDS]


def _list_accessible_provider_groups(user_sub: str, trace_id: str, country_code: Optional[str] = None) -> List[Dict[str, Any]]:
    client = get_analytics_center_client()
    out: List[Dict[str, Any]] = []
    offset = 0
    limit = 200

    while len(out) < _MAX_PROVIDER_IDS:
        page = client.list_provider_groups(
            user_sub=user_sub,
            trace_id=trace_id,
            country=country_code,
            limit=limit,
            offset=offset,
            raise_on_error=False,
        )
        if not page:
            break

        results_any = page.get("results", [])
        groups: List[Dict[str, Any]] = list(results_any)

        if not groups:
            break

        out.extend(groups)
        total_any = page.get("count")
        total = total_any if total_any >= 0 else None
        offset += len(groups)

        if total is not None and offset >= total:
            break

    return out[:_MAX_PROVIDER_IDS]


def _resolve_mine_scope(user_sub: str, trace_id: str) -> Optional[S.DataOriginSpec]:
    client = get_analytics_center_client()
    scope = client.resolve_my_default_scope(user_sub=user_sub, trace_id=trace_id, raise_on_error=False)
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
    if not isinstance(value, str) or not value.strip():
        raise OriginScopeResolutionError(
            "I need a specific hospital name to resolve scope.",
            clarification_type="provider_name",
        )

    normalized = _normalize_text(value)
    providers = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id)
    if not providers:
        raise OriginScopeResolutionError(
            "I could not load accessible hospitals to resolve the requested scope.",
            clarification_type="provider_name",
        )

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

    matches = exact or fuzzy
    if not matches:
        raise OriginScopeResolutionError(
            "I could not match that hospital to an accessible provider.",
            clarification_type="provider_name",
        )

    if len(matches) > 1:
        options = [_provider_name(item) for item in matches[:5] if _provider_name(item)]
        raise OriginScopeResolutionError(
            "I found multiple hospitals matching that name. Please be more specific.",
            clarification_type="provider_name",
            clarification_options=options,
        )

    provider_id = _provider_id(matches[0])
    if provider_id is None:
        raise OriginScopeResolutionError(
            "I matched the hospital name, but its provider ID is unavailable.",
            clarification_type="provider_name",
        )

    return S.DataOriginSpec(providerId=[provider_id])


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
            f"I could not resolve country '{raw_country}'.",
            clarification_type="country_code",
        )

    providers = _list_accessible_providers(user_sub=user_sub, trace_id=trace_id, country_code=resolved_country)
    provider_ids: List[int] = []
    for provider in providers:
        provider_id = _provider_id(provider)
        if provider_id is not None and provider_id not in provider_ids:
            provider_ids.append(provider_id)

    if not provider_ids:
        raise OriginScopeResolutionError(
            f"No accessible providers were found for country '{resolved_country}'.",
            clarification_type="country_code",
        )

    return S.DataOriginSpec(providerId=provider_ids)


def _resolve_scope(scope: S.OriginScopeSpec, user_sub: str, trace_id: str) -> Optional[S.DataOriginSpec]:
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
            return S.DataOriginSpec(providerId=[value])
        if isinstance(value, str) and value.strip().isdigit():
            return S.DataOriginSpec(providerId=[int(value.strip())])
        raise OriginScopeResolutionError(
            "Provider scope requires a numeric provider ID.",
            clarification_type="provider_id",
        )

    if scope_type == "provider_name":
        return _resolve_provider_name(value=value, user_sub=user_sub, trace_id=trace_id)

    if scope_type == "provider_group_id":
        if isinstance(value, int):
            return S.DataOriginSpec(providerGroupId=[value])
        if isinstance(value, str) and value.strip().isdigit():
            return S.DataOriginSpec(providerGroupId=[int(value.strip())])
        raise OriginScopeResolutionError(
            "Provider-group scope requires a numeric group ID.",
            clarification_type="provider_group_id",
        )

    if scope_type == "country_code":
        return _resolve_country_scope(value=value, country_code=scope.country_code, user_sub=user_sub, trace_id=trace_id)

    if scope_type == "country_average":
        return _resolve_country_scope(value=value, country_code=scope.country_code, user_sub=user_sub, trace_id=trace_id)

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


def resolve_plan_metric_origins(plan: S.AnalysisPlan, user_sub: str, trace_id: str) -> S.AnalysisPlan:
    charts = plan.charts or []
    if not charts:
        return plan

    default_scope = _default_scope_ref()

    resolved_charts: List[S.ChartSpec] = []
    for chart in charts:
        resolved_metrics: List[S.MetricSpec] = []
        for metric in chart.metrics:
            metric_data_origin = cast(Optional[S.DataOriginSpec], getattr(metric, "data_origin", None))
            metric_origin_scope = cast(Optional[S.OriginScopeSpec], getattr(metric, "origin_scope", None))

            if metric_data_origin is not None:
                resolved_metrics.append(metric)
                continue

            scope_ref = metric_origin_scope or default_scope
            resolved_data_origin: Optional[S.DataOriginSpec] = None
            fail_open_for_metric = _FAIL_OPEN and metric_origin_scope is None
            metric_distribution = cast(Optional[S.DistributionSpec], getattr(metric, "distribution", None))

            if scope_ref is not None:
                try:
                    resolved_data_origin = _resolve_scope(scope=scope_ref, user_sub=user_sub, trace_id=trace_id)
                except OriginScopeResolutionError:
                    if not fail_open_for_metric:
                        raise
                    logger.warning("Origin scope resolution failed; falling back to executor default data origin", exc_info=True)
                    resolved_data_origin = None
                except Exception:
                    if not fail_open_for_metric:
                        raise OriginScopeResolutionError("Origin scope resolution failed unexpectedly.")
                    logger.warning("Unexpected origin scope resolution failure; falling back to executor default data origin", exc_info=True)
                    resolved_data_origin = None

            resolved_metric = S.MetricSpec(
                metric=metric.metric,
                distribution=metric_distribution,
                data_origin=resolved_data_origin,
                origin_scope=scope_ref,
            )
            resolved_metrics.append(resolved_metric)

        chart_filters = cast(Optional[S.FilterNode], getattr(chart, "filters", None))
        chart_group_by = cast(Optional[List[S.GroupBySpec]], getattr(chart, "group_by", None))
        resolved_chart = S.ChartSpec(
            chart_type=chart.chart_type,
            filters=chart_filters,
            group_by=chart_group_by,
            metrics=resolved_metrics,
        )
        resolved_charts.append(resolved_chart)

    return S.AnalysisPlan(charts=resolved_charts, statistical_tests=plan.statistical_tests)
