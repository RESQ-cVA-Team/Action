from src.actions.i18n import translate
from src.executors.analytics_center.client import AnalyticsCenterError
from src.executors.orchestration.plan_executor import VisualizationExecutionError


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
    return translate("action.errors.visualization_generic", language=language)


def friendly_hospital_error(exc: Exception, language: str | None = None) -> str:
    if isinstance(exc, AnalyticsCenterError):
        details = exc.details if isinstance(exc.details, dict) else {}
        proxy_any = details.get("proxy")
        proxy_info = proxy_any if isinstance(proxy_any, dict) else {}
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
