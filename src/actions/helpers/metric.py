from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from src.shared import ssot_loader


def resolve_language(tracker: Any) -> str:
    tracker_any: Any = tracker
    lang: Optional[str] = None

    meta_any = tracker_any.latest_message.get("metadata")
    if isinstance(meta_any, dict):
        meta = cast(Dict[str, Any], meta_any)
        lang_val = meta.get("language")
        if isinstance(lang_val, str) and lang_val.strip():
            lang = lang_val.strip()

    if not lang:
        slot_lang_any: Any = tracker_any.get_slot("language")
        if isinstance(slot_lang_any, str) and slot_lang_any.strip():
            lang = slot_lang_any.strip()

    if not lang:
        return "en"

    primary = lang.split("-")[0].strip().lower()
    return primary or "en"


def extract_kpi(tracker: Any) -> Optional[str]:
    tracker_any: Any = tracker
    latest: Dict[str, Any] = tracker_any.latest_message or {}
    entities_any: Any = latest.get("entities")
    if isinstance(entities_any, list):
        entities_list: List[Any] = cast(List[Any], entities_any)
        for ent_any in entities_list:
            if not isinstance(ent_any, dict):
                continue
            ent = cast(Dict[str, Any], ent_any)
            if ent.get("entity") == "kpi":
                val = ent.get("value")
                if isinstance(val, str) and val.strip():
                    return val.strip()

    slot_val_any: Any = tracker_any.get_slot("kpi")
    if isinstance(slot_val_any, str) and slot_val_any.strip():
        return slot_val_any.strip()

    return None


def pick_description(descriptions: Dict[str, str], language: str) -> str:
    if not descriptions:
        return ""

    txt = descriptions.get(language)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    txt = descriptions.get("en")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    return ""


def suggest_metrics(metric_lookup: Dict[str, Dict[str, Any]], max_items: int = 5) -> List[str]:
    seen: Dict[str, str] = {}
    for record in metric_lookup.values():
        canonical = record.get("canonical")
        if not isinstance(canonical, str) or not canonical:
            continue
        code = canonical.upper()
        if code in seen:
            continue
        label = ssot_loader.get_metric_display_name(code)
        seen[code] = label
        if len(seen) >= max_items:
            break

    return list(seen.values())
