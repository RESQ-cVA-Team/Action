from __future__ import annotations

from typing import Any, Dict, List, Optional

from rasa_sdk.events import EventType  # type: ignore

from src.executors.analytics_center.client import get_analytics_center_client

from .. import command, register


def _to_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if txt.isdigit():
            return int(txt)
    return default


def _to_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if txt.isdigit():
            return int(txt)
    return None


@command("test_hospitals")
def test_hospitals(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    user_sub = tracker.sender_id
    client = get_analytics_center_client()

    country_code = opts.get("country") or opts.get("country_code") or opts.get("countryCode")
    name_contains = opts.get("name") or opts.get("hospital") or opts.get("q")
    sort = opts.get("sort")
    user_id = opts.get("user") or opts.get("user_id")
    group_id = _to_optional_int(opts.get("group") or opts.get("group_id"))
    limit = max(1, min(_to_int(opts.get("limit"), 20), 200))
    offset = max(0, _to_int(opts.get("offset"), 0))

    resolved_country_code: Optional[str] = None
    if isinstance(country_code, str) and country_code.strip():
        resolved_country_code = client.resolve_country_code(user_sub=user_sub, country_input=country_code.strip())
        if not resolved_country_code:
            dispatcher.utter_message(text=f"❌ Unknown country filter '{country_code}'. Try a 2-letter code like ES, MX, DE.")
            return []

    page = client.list_providers(
        user_sub=user_sub,
        limit=limit,
        offset=offset,
        country_code=resolved_country_code,
        sort=sort.strip() if isinstance(sort, str) and sort.strip() else None,
        user=user_id.strip() if isinstance(user_id, str) and user_id.strip() else None,
        group=group_id,
    )

    if not page:
        dispatcher.utter_message(text="❌ providers request failed or returned no payload.")
        return []

    providers = page["results"]
    total_count = page["count"]
    used_offset = page["offset"]
    used_limit = page["limit"]

    rows: List[str] = []
    for provider in providers:
        name = provider.get("nameEnglish") or provider.get("nameNative") or provider.get("shortName") or "(unnamed)"
        city = provider.get("city") or "?"
        country = provider.get("country") or "?"
        rows.append(f"- {name} ({city}, {country})")

    if isinstance(name_contains, str) and name_contains.strip():
        needle = name_contains.strip().lower()
        rows = [r for r in rows if needle in r.lower()]

    header_parts: List[str] = [f"✅ providers OK: total={total_count}, offset={used_offset}, limit={used_limit}, returned={len(providers)}"]
    if isinstance(resolved_country_code, str) and resolved_country_code.strip():
        header_parts.append(f"country={resolved_country_code.strip().upper()}")
    if isinstance(sort, str) and sort.strip():
        header_parts.append(f"sort={sort.strip()}")
    if isinstance(name_contains, str) and name_contains.strip():
        header_parts.append(f"name~={name_contains.strip()}")

    max_lines = min(len(rows), 15)
    preview = "\n".join(rows[:max_lines]) if rows else "(no providers in this page after local name filter)"
    suffix = ""
    if len(rows) > max_lines:
        suffix = f"\n... (+{len(rows) - max_lines} more in this page)"

    dispatcher.utter_message(text=" | ".join(header_parts) + "\n" + preview + suffix)
    return []


register("hospitals", test_hospitals)
