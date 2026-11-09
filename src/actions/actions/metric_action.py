from typing import Any, Dict, List, Optional, Protocol, cast

from src.actions.helpers.metric import pick_description, suggest_metrics
from src.actions.i18n import translate
from src.shared import ssot_loader

_METRIC_TEXT_LOOKUP = ssot_loader.get_metric_text_lookup()


class DispatcherLike(Protocol):
    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]


def explain_metric(dispatcher: DispatcherLike, raw_metric: str, language: str) -> None:
    """Answer with the SSOT description of the metric `raw_metric` names."""
    norm_key = ssot_loader.normalize_metric_text_key(raw_metric)
    if not norm_key:
        dispatcher.utter_message(text=translate("action.metric.metric_not_understood", language=language))
        return

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
        return

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
        return

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
