import logging
from typing import Any, Dict, List, Optional, Protocol, Text, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore

from src.actions.error_messages import friendly_metric_error
from src.actions.helpers.metric import extract_kpi, pick_description, resolve_language, suggest_metrics
from src.actions.i18n import translate
from src.shared import ssot_loader
from src.util import env as env_util
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)
_METRIC_TEXT_LOOKUP = ssot_loader.get_metric_text_lookup()

DomainDict = Dict[str, Any]
RasaEventList = List[Dict[Text, Any]]


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
    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]


def _tracker_trace_id(tracker: TrackerLike) -> Optional[str]:
    metadata_any = tracker.latest_message.get("metadata")
    metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}

    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        raw_value = metadata.get(key)
        if raw_value is not None:
            trace_id = str(raw_value).strip()
            if trace_id:
                return trace_id

    headers_any = metadata.get("headers")
    headers = cast(Dict[str, Any], headers_any) if isinstance(headers_any, dict) else {}
    for key in ("x-trace-id", "x_trace_id", "trace_id", "traceId"):
        raw_value = headers.get(key)
        if raw_value is not None:
            trace_id = str(raw_value).strip()
            if trace_id:
                return trace_id

    return None


class ActionExplainMetric(Action):  # pyright: ignore
    """Explain a metric/KPI based on MetricType.yml and language.

    The metric is resolved from the latest user message `kpi` entity (or the
    `kpi` slot as a fallback), normalized using the same rules as the SSOT
    lookup. The response uses the localized description from the SSOT
    `descriptions` block, falling back to English or the first available
    language.
    """

    def name(self) -> str:
        return "action_explain_metric"

    async def run(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> RasaEventList:
        trace_id = _tracker_trace_id(tracker) or uuid4().hex
        with log_context(trace_id=trace_id, sender_id=str(tracker.sender_id), action=self.name()):
            try:
                language = resolve_language(tracker)

                raw_kpi = extract_kpi(tracker)
                if not raw_kpi:
                    dispatcher.utter_message(text=translate("action.metric.missing_metric", language=language))
                    return []

                norm_key = ssot_loader.normalize_metric_text_key(raw_kpi)
                if not norm_key:
                    dispatcher.utter_message(text=translate("action.metric.metric_not_understood", language=language))
                    return []

                record = _METRIC_TEXT_LOOKUP.get(norm_key)
                if not record:
                    suggestions = suggest_metrics(_METRIC_TEXT_LOOKUP, max_items=5)
                    if suggestions:
                        dispatcher.utter_message(
                            text=translate(
                                "action.metric.metric_unknown_with_suggestions",
                                language=language,
                                params={"suggestions": ", ".join(suggestions)},
                            )
                        )
                    else:
                        dispatcher.utter_message(text=translate("action.metric.metric_unknown", language=language))
                    return []

                canonical = cast(str, record.get("canonical") or "")
                descriptions = cast(Dict[str, str], record.get("descriptions") or {})
                data_type = cast(Optional[str], record.get("data_type"))
                unit = cast(Optional[str], record.get("unit"))

                description_text = pick_description(descriptions, language)
                if not description_text:
                    for txt in descriptions.values():
                        txt_val = txt.strip()
                        if txt_val:
                            description_text = txt_val
                            break

                if not canonical or not description_text:
                    dispatcher.utter_message(text=translate("action.metric.description_not_configured", language=language))
                    return []

                display_name = ssot_loader.get_metric_display_name(canonical)

                header: str
                if display_name and display_name != canonical:
                    header = f"{canonical} – {display_name}."
                else:
                    header = f"{canonical}."

                parts: List[str] = [header, description_text]
                if data_type:
                    if unit:
                        parts.append(
                            translate(
                                "action.metric.data_type_with_unit",
                                language=language,
                                params={"data_type": data_type, "unit": unit},
                            )
                        )
                    else:
                        parts.append(
                            translate(
                                "action.metric.data_type",
                                language=language,
                                params={"data_type": data_type},
                            )
                        )

                dispatcher.utter_message(text=" ".join(parts))
                return []
            except Exception as e:
                logger.exception(
                    "Error explaining metric",
                    extra=_action_log_context(
                        trace_id=trace_id,
                        action_name=self.name(),
                        event="actions.metric.explain.failed",
                        outcome="failure",
                    ),
                )
                language = resolve_language(tracker)
                dispatcher.utter_message(text=f"❌ {friendly_metric_error(e, language=language)}")
                if _ECHO_INTERNAL_ERRORS:
                    dispatcher.utter_message(
                        text=translate(
                            "action.metric.internal_error",
                            language=language,
                            params={"error": str(e)},
                        )
                    )
                return []