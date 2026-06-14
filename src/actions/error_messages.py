from typing import Any, Mapping, cast

from src.actions.i18n import translate
from src.executors.analytics_center.client import AnalyticsCenterError
from src.executors.orchestration.plan_executor import VisualizationExecutionError


def _mapping_to_dict(value: Any) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def visualization_error_payload(
    exc: Exception,
    trace_id: str | None = None,
    language: str | None = None,
) -> dict[str, str | None]:
    code = "ACTION_VIS_UNKNOWN"
    reason = "unknown"
    message = friendly_visualization_error(exc, language=language)

    if isinstance(exc, VisualizationExecutionError):
        code = exc.code
        reason = exc.reason

    return {
        "code": code,
        "reason": reason,
        "message": message,
        "trace_id": trace_id,
    }


def friendly_visualization_error(exc: Exception, language: str | None = None) -> str:
    if isinstance(exc, VisualizationExecutionError):
        return exc.user_message
    if isinstance(exc, TimeoutError):
        return translate("action.errors.visualization_timeout", language=language)
    if isinstance(exc, ValueError):
        text = str(exc).strip()
        lowered = text.lower()
        if "missing ssot distribution defaults for metric" in lowered:
            return (
                "Visualization failed because metric distribution defaults are missing in SSOT metadata. "
                "Please add default buckets and numeric range for the requested metric."
            )
        if "invalid ssot distribution" in lowered:
            return (
                "Visualization failed because SSOT distribution metadata is invalid for this metric. "
                "Please fix bucket/range values in SSOT."
            )
        if "distribution-first kpi policy" in lowered:
            return (
                "This request produced an invalid plan under the current distribution-first policy. "
                "Please try again with a KPI distribution request or an explicit categorical comparison."
            )
    return translate("action.errors.visualization_generic", language=language)


def friendly_hospital_error(exc: Exception, language: str | None = None) -> str:
    if isinstance(exc, AnalyticsCenterError):
        details = _mapping_to_dict(exc.details)
        proxy_info = _mapping_to_dict(details.get("proxy"))
        reason_any = proxy_info.get("reason")
        reason = reason_any.strip().lower() if isinstance(reason_any, str) and reason_any.strip() else ""

        if exc.status_code == 401 or "no cached user access token" in reason or "user token unavailable" in reason:
            return translate("action.errors.hospital_auth_unavailable", language=language)
        if exc.kind == "timeout":
            return translate("action.errors.hospital_timeout", language=language)
        if exc.kind == "http_error" and exc.status_code in {429, 500, 502, 503, 504}:
            return translate("action.errors.hospital_service_unavailable", language=language)
        return translate("action.errors.hospital_generic", language=language)
    return translate("action.errors.hospital_action_generic", language=language)


def friendly_metric_error(exc: Exception, language: str | None = None) -> str:
    if isinstance(exc, OSError):
        return translate("action.errors.metric_definitions_unavailable", language=language)
    if isinstance(exc, TimeoutError):
        return translate("action.errors.metric_timeout", language=language)
    return translate("action.errors.metric_generic", language=language)
