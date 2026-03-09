from src.executors import plan_executor
from src.executors.analytics_center.client import AnalyticsCenterError


def friendly_visualization_error(exc: Exception) -> str:
    if isinstance(exc, plan_executor.VisualizationExecutionError):
        return exc.user_message
    if isinstance(exc, TimeoutError):
        return "Visualization generation timed out. Please try again."
    return "Error generating visualization."


def friendly_hospital_error(exc: Exception) -> str:
    if isinstance(exc, AnalyticsCenterError):
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
