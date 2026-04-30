from __future__ import annotations

from typing import Any, Dict, Optional, cast


def extract_hospital_filters(tracker: Any) -> Dict[str, Any]:
    tracker_any: Any = tracker
    latest_any: Any = tracker_any.latest_message or {}
    latest = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}

    metadata_any: Any = latest.get("metadata")
    metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}

    entities_by_name: Dict[str, Any] = {}
    entities_any: Any = latest.get("entities")
    if isinstance(entities_any, list):
        entities_list = cast(list[Any], entities_any)
        for ent_any in entities_list:
            if not isinstance(ent_any, dict):
                continue
            ent = cast(Dict[str, Any], ent_any)
            key = ent.get("entity")
            value = ent.get("value")
            if isinstance(key, str) and key.strip() and value is not None and key not in entities_by_name:
                entities_by_name[key] = value

    def first_str(*candidates: Any) -> Optional[str]:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def first_int(*candidates: Any) -> Optional[int]:
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
            if isinstance(candidate, str):
                txt = candidate.strip()
                if txt.isdigit():
                    return int(txt)
        return None

    country_code = first_str(
        metadata.get("countryCode"),
        metadata.get("country_code"),
        entities_by_name.get("countryCode"),
        entities_by_name.get("country_code"),
        entities_by_name.get("country"),
        tracker_any.get_slot("countryCode"),
        tracker_any.get_slot("country_code"),
        tracker_any.get_slot("country"),
    )

    name_contains = first_str(
        metadata.get("hospitalName"),
        metadata.get("hospital_name"),
        metadata.get("name"),
        entities_by_name.get("hospital_name"),
        entities_by_name.get("hospital"),
        entities_by_name.get("provider"),
        entities_by_name.get("name"),
        tracker_any.get_slot("hospital_name"),
        tracker_any.get_slot("hospital"),
        tracker_any.get_slot("provider"),
        tracker_any.get_slot("name"),
    )

    sort = first_str(
        metadata.get("sort"),
        tracker_any.get_slot("sort"),
    )

    user_id = first_str(
        metadata.get("user"),
        metadata.get("userId"),
        tracker_any.get_slot("user"),
        tracker_any.get_slot("user_id"),
    )

    group_id = first_int(
        metadata.get("group"),
        metadata.get("groupId"),
        tracker_any.get_slot("group"),
        tracker_any.get_slot("group_id"),
    )

    limit_val = first_int(
        metadata.get("limit"),
        tracker_any.get_slot("limit"),
    )
    offset_val = first_int(
        metadata.get("offset"),
        tracker_any.get_slot("offset"),
    )

    limit = 50 if limit_val is None else max(1, min(limit_val, 200))
    offset = 0 if offset_val is None else max(0, offset_val)

    return {
        "country_code": country_code.upper() if isinstance(country_code, str) and len(country_code) == 2 else country_code,
        "name_contains": name_contains,
        "sort": sort,
        "user_id": user_id,
        "group_id": group_id,
        "limit": limit,
        "offset": offset,
    }
