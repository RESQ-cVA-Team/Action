from src.executors.analytics_center.client import AnalyticsCenterError
from src.executors.orchestration.plan_executor import VisualizationExecutionError


def visualization_error_payload(exc: Exception, trace_id: str | None = None) -> dict[str, str | None]:
    code = "ACTION_VIS_UNKNOWN"
    reason = "unknown"
    message = friendly_visualization_error(exc)

    if isinstance(exc, VisualizationExecutionError):
        code = exc.code
        reason = exc.reason

    return {
        "code": code,
        "reason": reason,
        "message": message,
        "trace_id": trace_id,
    }


def friendly_visualization_error(exc: Exception) -> str:
    if isinstance(exc, VisualizationExecutionError):
        return exc.user_message
    if isinstance(exc, TimeoutError):
        return "Visualization generation timed out. Please try again."
    return "Error generating visualization."


def friendly_hospital_error(exc: Exception) -> str:
    if isinstance(exc, AnalyticsCenterError):
        details = exc.details if isinstance(exc.details, dict) else {}
        proxy_any = details.get("proxy")
        proxy_info = proxy_any if isinstance(proxy_any, dict) else {}
        reason_any = proxy_info.get("reason")
        reason = reason_any.strip().lower() if isinstance(reason_any, str) and reason_any.strip() else ""

        if exc.status_code == 401 or "no cached user access token" in reason or "user token unavailable" in reason:
            return "Your analytics session is not available to the action server right now. Please try again after signing in again, or provide a hospital manually."
        if exc.kind == "timeout":
            return "The provider directory request timed out. Please try again."
        if exc.kind == "http_error" and exc.status_code in {429, 500, 502, 503, 504}:
            return "The provider directory service is temporarily unavailable. Please try again in a moment."
        return "Could not retrieve hospital data right now. Please try again."
    return "Error listing hospitals."


def friendly_metric_error(exc: Exception) -> str:
    if isinstance(exc, OSError):
        return "Metric definitions are temporarily unavailable. Please try again."
    if isinstance(exc, TimeoutError):
        return "Metric lookup timed out. Please try again."
    return "Error explaining metric."
