import logging
from typing import Any, Dict, List, Mapping, Optional, Protocol, Text, cast
from uuid import uuid4

from rasa_sdk import Action  # type: ignore

from src.actions.error_messages import friendly_hospital_error
from src.actions.helpers.hospital import extract_hospital_filters
from src.actions.i18n import resolve_language_from_tracker, translate
from src.executors.analytics_center.client import get_analytics_center_client
from src.util import env as env_util
from src.util.logging_utils import log_context

logger = logging.getLogger(__name__)

_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)

DomainDict = Dict[str, Any]
RasaEventList = List[Dict[Text, Any]]


class DispatcherLike(Protocol):
    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]


def _mapping_to_dict(value: object) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: Dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def _tracker_trace_id(tracker: TrackerLike) -> Optional[str]:
    latest = _mapping_to_dict(tracker.latest_message)
    metadata = _mapping_to_dict(latest.get("metadata"))

    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        raw = metadata.get(key)
        if raw is None:
            continue
        token = str(raw).strip()
        if token:
            return token

    headers = _mapping_to_dict(metadata.get("headers"))
    for key in ("x-trace-id", "x_trace_id", "trace_id", "traceId"):
        raw = headers.get(key)
        if raw is None:
            continue
        token = str(raw).strip()
        if token:
            return token
    return None


class ActionListHospitals(Action):  # pyright: ignore
    """List hospitals/providers available for comparison."""

    def name(self) -> str:
        return "action_list_hospitals"

    async def run(
        self,
        dispatcher: DispatcherLike,
        tracker: TrackerLike,
        domain: DomainDict,
    ) -> RasaEventList:
        user_sub = str(tracker.sender_id)
        language = resolve_language_from_tracker(tracker)
        trace_id = _tracker_trace_id(tracker) or uuid4().hex
        with log_context(trace_id=trace_id, sender_id=user_sub, action=self.name()):
            try:
                logger.info("Listing hospitals")
                filters = extract_hospital_filters(tracker)
                client = get_analytics_center_client()

                raw_country = filters.get("country_code")
                if isinstance(raw_country, str) and raw_country.strip():
                    resolved_country = client.resolve_country_code(
                        user_sub=user_sub,
                        country_input=raw_country,
                        trace_id=trace_id,
                        raise_on_error=True,
                    )
                    if resolved_country:
                        filters["country_code"] = resolved_country
                    else:
                        dispatcher.utter_message(
                            text=translate(
                                "action.hospitals.country_not_matched",
                                language=language,
                                params={"country": raw_country},
                            )
                        )
                        return []

                provider_page = client.list_providers(
                    user_sub=user_sub,
                    limit=filters["limit"],
                    offset=filters["offset"],
                    country_code=filters.get("country_code"),
                    sort=filters.get("sort"),
                    user=filters.get("user_id"),
                    group=filters.get("group_id"),
                    trace_id=trace_id,
                    raise_on_error=True,
                )
                if not provider_page:
                    dispatcher.utter_message(text=translate("action.hospitals.none_available", language=language))
                    return []

                providers = provider_page["results"]
                total_count = provider_page["count"]
                offset = provider_page["offset"]
                limit = provider_page["limit"]

                names: List[str] = []
                for provider in providers:
                    name = provider.get("nameEnglish") or provider.get("nameNative") or provider.get("shortName")
                    if isinstance(name, str) and name.strip():
                        names.append(name.strip())

                name_filter = filters.get("name_contains")
                if isinstance(name_filter, str) and name_filter.strip():
                    needle = name_filter.strip().lower()
                    names = [n for n in names if needle in n.lower()]

                if not names:
                    no_match_msg = translate("action.hospitals.no_matches_with_hint", language=language) if isinstance(name_filter, str) and name_filter.strip() else translate("action.hospitals.no_matches", language=language)
                    dispatcher.utter_message(text=no_match_msg)
                    return []

                preview = ", ".join(names[:10])
                more_count = max(len(names) - 10, 0)
                criteria: List[str] = []
                country_code = filters.get("country_code")
                sort = filters.get("sort")
                if isinstance(country_code, str) and country_code.strip():
                    criteria.append(
                        translate(
                            "action.hospitals.criteria_country",
                            language=language,
                            params={"country": country_code.strip().upper()},
                        )
                    )
                if isinstance(sort, str) and sort.strip():
                    criteria.append(
                        translate(
                            "action.hospitals.criteria_sort",
                            language=language,
                            params={"sort": sort.strip()},
                        )
                    )

                criteria_text = f" ({', '.join(criteria)})" if criteria else ""

                if isinstance(name_filter, str) and name_filter.strip():
                    prefix = translate(
                        "action.hospitals.summary_with_name_filter",
                        language=language,
                        params={
                            "matched_count": len(names),
                            "search": name_filter.strip(),
                            "offset": offset,
                            "limit": limit,
                            "total_count": total_count,
                        },
                    )
                else:
                    shown_start = offset + 1 if names else 0
                    shown_end = offset + len(names)
                    prefix = translate(
                        "action.hospitals.summary_general",
                        language=language,
                        params={
                            "total_count": total_count,
                            "criteria_text": criteria_text,
                            "shown_start": shown_start,
                            "shown_end": shown_end,
                        },
                    )

                more_suffix = (
                    translate(
                        "action.hospitals.more_in_page",
                        language=language,
                        params={"more_count": more_count},
                    )
                    if more_count
                    else ""
                )
                text_message = prefix + f" {preview}." + more_suffix
                dispatcher.utter_message(text=text_message)
                return []
            except Exception as exc:
                logger.exception("Error listing hospitals")
                language = resolve_language_from_tracker(tracker)
                dispatcher.utter_message(text=f"❌ {friendly_hospital_error(exc, language=language)}")
                if _ECHO_INTERNAL_ERRORS:
                    dispatcher.utter_message(
                        text=translate(
                            "action.hospitals.internal_error",
                            language=language,
                            params={"error": str(exc)},
                        )
                    )
                return []
