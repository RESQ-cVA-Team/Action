from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "el", "cs")
DEFAULT_LANGUAGE = "en"
_LOCALES_ROOT = Path(__file__).resolve().parents[1] / "locales"


class _SafeFormatMap(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


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
    if hasattr(tracker, "current_state") and callable(getattr(tracker, "current_state")):
        try:
            state_any = tracker.current_state()
            if isinstance(state_any, dict):
                slots_any = state_any.get("slots")
                if isinstance(slots_any, dict):
                    return dict(slots_any)
        except Exception:
            return {}

    if hasattr(tracker, "get_slot") and callable(getattr(tracker, "get_slot")):
        try:
            slot_language = tracker.get_slot("language")
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
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    slots_map = slots if isinstance(slots, Mapping) else {}

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
    latest_any = getattr(tracker, "latest_message", None)
    latest = latest_any if isinstance(latest_any, dict) else {}
    metadata_any = latest.get("metadata")
    metadata = metadata_any if isinstance(metadata_any, dict) else {}
    slots = _tracker_slots(tracker)
    return resolve_language(metadata=metadata, slots=slots, tracker=tracker)


@lru_cache(maxsize=len(SUPPORTED_LANGUAGES))
def _load_catalog(language: str) -> Dict[str, Any]:
    normalized = normalize_language_code(language) or DEFAULT_LANGUAGE
    path = _LOCALES_ROOT / normalized / "common.json"

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        logger.exception("Failed to load locale catalog for language='%s' from '%s'", normalized, path)

    return {}


def _lookup_catalog(catalog: Mapping[str, Any], key: str) -> Optional[str]:
    node: Any = catalog
    for part in key.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)

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
