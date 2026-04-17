import logging
from typing import Any, Dict, List, Optional, Protocol, Text
from uuid import uuid4

from rasa_sdk import Action  # type: ignore

from src.actions.error_messages import friendly_hospital_error
from src.actions.utils.hospital import extract_hospital_filters
from src.executors.analytics_center.client import get_analytics_center_client
from src.util import env as env_util

logger = logging.getLogger(__name__)

_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)

DomainDict = Dict[str, Any]
RasaEventList = List[Dict[Text, Any]]


class DispatcherLike(Protocol):
    def utter_message(self, text: Optional[str] = None, **kwargs: Any) -> None: ...


class TrackerLike(Protocol):
    sender_id: str
    latest_message: Dict[str, Any]


def _tracker_trace_id(tracker: TrackerLike) -> Optional[str]:
    latest_any = getattr(tracker, "latest_message", None)
    latest = latest_any if isinstance(latest_any, dict) else {}
    metadata_any = latest.get("metadata") if isinstance(latest, dict) else None
    metadata = metadata_any if isinstance(metadata_any, dict) else {}

    for key in ("trace_id", "traceId", "x-trace-id", "x_trace_id"):
        raw = metadata.get(key)
        if raw is None:
            continue
        token = str(raw).strip()
        if token:
            return token

    headers_any = metadata.get("headers")
    headers = headers_any if isinstance(headers_any, dict) else {}
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
        try:
            user_sub = str(tracker.sender_id)
            trace_id = _tracker_trace_id(tracker) or uuid4().hex
            logger.info("Listing hospitals (trace_id=%s)", trace_id or "-")
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
                    dispatcher.utter_message(text=f"I couldn't match country '{raw_country}'. Please try a 2-letter code like ES, MX, DE, or FR.")
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
                dispatcher.utter_message(text="I couldn't find any hospitals you can compare against.")
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
                no_match_msg = "I couldn't find any hospitals matching your filters."
                if isinstance(name_filter, str) and name_filter.strip():
                    no_match_msg += " Try a shorter or less specific name."
                dispatcher.utter_message(text=no_match_msg)
                return []

            preview = ", ".join(names[:10])
            more_count = max(len(names) - 10, 0)
            criteria: List[str] = []
            country_code = filters.get("country_code")
            sort = filters.get("sort")
            if isinstance(country_code, str) and country_code.strip():
                criteria.append(f"country={country_code.strip().upper()}")
            if isinstance(sort, str) and sort.strip():
                criteria.append(f"sort={sort.strip()}")

            criteria_text = f" ({', '.join(criteria)})" if criteria else ""

            if isinstance(name_filter, str) and name_filter.strip():
                prefix = f"I found {len(names)} matching hospitals on this page (search='{name_filter.strip()}', offset={offset}, limit={limit}); total providers before name filter: {total_count}."
            else:
                shown_start = offset + 1 if names else 0
                shown_end = offset + len(names)
                prefix = f"I found {total_count} hospitals{criteria_text}; showing {shown_start}-{shown_end}."

            text_message = prefix + f" {preview}." + (f" (+{more_count} more in this page)" if more_count else "")
            dispatcher.utter_message(text=text_message)
            return []
        except Exception as exc:
            logger.exception("Error listing hospitals")
            dispatcher.utter_message(text=f"❌ {friendly_hospital_error(exc)}")
            if _ECHO_INTERNAL_ERRORS:
                dispatcher.utter_message(text=f"Error listing hospitals: {str(exc)}")
            return []
