from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, cast

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "el", "cs")
DEFAULT_LANGUAGE = "en"
_LOCALES_ROOT = Path(__file__).resolve().parents[1] / "locales"


class _SafeFormatMap(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _mapping_to_dict(value: object) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    normalized: Dict[str, Any] = {}
    for key, item in mapping.items():
        if isinstance(key, str):
            normalized[key] = item
    return normalized


def _mapping_get(value: object, key: str) -> object | None:
    if not isinstance(value, Mapping):
        return None

    mapping = cast(Mapping[object, object], value)
    for current_key, current_value in mapping.items():
        if current_key == key:
            return current_value
    return None


def normalize_language_code(raw_language: Any) -> Optional[str]:
    if raw_language is None:
        return None

    token = str(raw_language).strip()
    if not token:
        return None

    # Handle values such as "en-US,en;q=0.9" defensively.
    token = token.split(",", 1)[0].split(";", 1)[0].strip()
    if not token:
        return None

    primary = token.split("-", 1)[0].strip().lower()
    if primary in SUPPORTED_LANGUAGES:
        return primary
    return None


def _language_from_metadata(metadata: Mapping[str, Any]) -> Optional[str]:
    language = metadata.get("language")
    return normalize_language_code(language)


def _language_from_slots(slots: Mapping[str, Any]) -> Optional[str]:
    language = slots.get("language")
    return normalize_language_code(language)


def _tracker_slots(tracker: Any) -> Dict[str, Any]:
    current_state = getattr(tracker, "current_state", None)
    if callable(current_state):
        try:
            state = _mapping_to_dict(current_state())
            slots = _mapping_to_dict(state.get("slots"))
            if slots:
                return slots
        except Exception:
            return {}

    get_slot = getattr(tracker, "get_slot", None)
    if callable(get_slot):
        try:
            slot_language = get_slot("language")
        except Exception:
            slot_language = None
        if slot_language is not None:
            return {"language": slot_language}

    return {}


def resolve_language(
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    slots: Optional[Mapping[str, Any]] = None,
    tracker: Any = None,
) -> str:
    metadata_map = _mapping_to_dict(metadata)
    slots_map = _mapping_to_dict(slots)

    metadata_language = _language_from_metadata(metadata_map)
    if metadata_language is not None:
        return metadata_language

    slot_language = _language_from_slots(slots_map)
    if slot_language is not None:
        return slot_language

    if tracker is not None:
        tracker_slots = _tracker_slots(tracker)
        tracker_language = _language_from_slots(tracker_slots)
        if tracker_language is not None:
            return tracker_language

    return DEFAULT_LANGUAGE


def resolve_language_from_tracker(tracker: Any) -> str:
    latest = _mapping_to_dict(getattr(tracker, "latest_message", None))
    metadata = _mapping_to_dict(latest.get("metadata"))
    slots = _tracker_slots(tracker)
    return resolve_language(metadata=metadata, slots=slots, tracker=tracker)


@lru_cache(maxsize=len(SUPPORTED_LANGUAGES))
def _load_catalog(language: str) -> Dict[str, Any]:
    normalized = normalize_language_code(language) or DEFAULT_LANGUAGE
    path = _LOCALES_ROOT / normalized / "common.json"

    try:
        with path.open("r", encoding="utf-8") as handle:
            return _mapping_to_dict(json.load(handle))
    except Exception:
        logger.exception("Failed to load locale catalog for language='%s' from '%s'", normalized, path)

    return {}


def _lookup_catalog(catalog: Mapping[str, Any], key: str) -> Optional[str]:
    node: object = catalog
    for part in key.split("."):
        node = _mapping_get(node, part)
        if node is None:
            return None

    if isinstance(node, str):
        return node
    return None


def translate(
    key: str,
    *,
    language: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    default: Optional[str] = None,
) -> str:
    normalized_language = normalize_language_code(language) or DEFAULT_LANGUAGE

    template = _lookup_catalog(_load_catalog(normalized_language), key)
    if template is None and normalized_language != DEFAULT_LANGUAGE:
        template = _lookup_catalog(_load_catalog(DEFAULT_LANGUAGE), key)

    if template is None:
        if isinstance(default, str):
            template = default
        else:
            template = key
            logger.warning("Missing translation key='%s' for language='%s'", key, normalized_language)

    if not params:
        return template

    format_params = _SafeFormatMap({k: v for k, v in params.items()})
    try:
        return template.format_map(format_params)
    except Exception:
        logger.exception("Failed formatting translation key='%s' with params", key)
        return template
