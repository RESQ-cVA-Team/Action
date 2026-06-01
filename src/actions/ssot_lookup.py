from __future__ import annotations

import logging
from typing import Any, Dict, List, cast

from src.shared.ssot_loader import get_metric_text_lookup, get_ssot_items, normalize_metric_text_key

logger = logging.getLogger(__name__)


def normalize_text(value: str) -> str:
    text = (value or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _load_ssot_items(filename: str) -> List[Dict[str, Any]]:
    try:
        raw = get_ssot_items(filename)
    except Exception:
        logger.debug(
            "Failed to load SSOT items; returning empty candidate set",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "actions.ssot_lookup.items_load_failed",
                    "operation": "_load_ssot_items",
                    "outcome": "degraded",
                    "filename": filename,
                }
            },
        )
        return []
    return raw


def resolve_metric_candidates(raw_value: str) -> List[str]:
    normalized = normalize_metric_text_key(raw_value or "")
    if not normalized:
        return []

    lookup = get_metric_text_lookup()
    exact = lookup.get(normalized)
    if isinstance(exact, dict):
        canonical = exact.get("canonical")
        if isinstance(canonical, str) and canonical.strip():
            return [canonical.strip().upper()]

    matches: List[str] = []
    for alias, record in lookup.items():
        if normalized not in alias and alias not in normalized:
            continue
        canonical = record.get("canonical")
        if isinstance(canonical, str) and canonical.strip():
            candidate = canonical.strip().upper()
            if candidate not in matches:
                matches.append(candidate)
    return matches


def resolve_catalog_candidates(filename: str, raw_value: str) -> List[str]:
    normalized = normalize_text(raw_value)
    if not normalized:
        return []

    exact_matches: List[str] = []
    fuzzy_matches: List[str] = []
    for item in _load_ssot_items(filename):
        canonical_any = item.get("canonical")
        if not isinstance(canonical_any, str) or not canonical_any.strip():
            continue
        canonical = canonical_any.strip().upper()
        aliases = [canonical_any]
        synonyms_any = item.get("synonyms")
        if isinstance(synonyms_any, list):
            for value in cast(List[Any], synonyms_any):
                if isinstance(value, str):
                    aliases.append(value)

        normalized_aliases = [normalize_text(alias) for alias in aliases if normalize_text(alias)]
        if normalized in normalized_aliases:
            if canonical not in exact_matches:
                exact_matches.append(canonical)
            continue

        if any(normalized in alias or alias in normalized for alias in normalized_aliases):
            if canonical not in fuzzy_matches:
                fuzzy_matches.append(canonical)

    return exact_matches or fuzzy_matches
