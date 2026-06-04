"""Unified SSOT loader and metadata access.

Provides:
- Cached YAML loading for SSOT files.
- Dynamic enum factory (string enums) based on canonical values.
- Metric metadata accessor for richer properties (unit, labels, etc.).

All existing code should prefer this module instead of ad-hoc YAML parsing.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml

BASE_SSOT = Path(__file__).resolve().parent / "SSOT"
_LOGGER = logging.getLogger(__name__)


class SSOTLoadError(FileNotFoundError):
    pass


@lru_cache(maxsize=64)
def _load_yaml(filename: str) -> List[Dict[str, Any]]:
    path = BASE_SSOT / filename
    if not path.exists():
        raise SSOTLoadError(
            f"Missing SSOT file: {path}. Base directory contents: {[p.name for p in BASE_SSOT.glob('*.yml')] if BASE_SSOT.exists() else 'N/A'}"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected YAML structure in {path}; expected list")
    # Filter only dict items
    data_list = cast(List[Any], raw)
    items: List[Dict[str, Any]] = [d for d in data_list if isinstance(d, dict)]
    return items


def get_ssot_items(filename: str) -> List[Dict[str, Any]]:
    """Return raw SSOT records for a given YAML file.

    This is a public wrapper around the cached YAML loader for modules that
    need read-only access to item-level data (canonical + synonyms + metadata).
    """

    return _load_yaml(filename)


@lru_cache(maxsize=64)
def get_canonical_values(filename: str) -> Tuple[str, ...]:
    items = _load_yaml(filename)
    values: List[str] = []
    for item in items:
        val = item.get("canonical")
        if isinstance(val, str) and val.strip():
            values.append(val.strip())
    if not values:
        raise ValueError(f"No canonical values found in {filename}")
    return tuple(values)


@lru_cache(maxsize=16)
def _canonical_lookup(filename: str) -> Dict[str, str]:
    """Return normalized text -> canonical mapping for an SSOT type file.

    Includes both canonical values and all synonyms.
    """
    items = _load_yaml(filename)
    lookup: Dict[str, str] = {}

    for item in items:
        canonical_any = item.get("canonical")
        if not isinstance(canonical_any, str) or not canonical_any.strip():
            continue
        canonical = canonical_any.strip().upper()

        keys: List[str] = [canonical]
        syn_any = item.get("synonyms")
        if isinstance(syn_any, list):
            for s_any in cast(List[Any], syn_any):
                if isinstance(s_any, str) and s_any.strip():
                    keys.append(s_any.strip())

        for key in keys:
            norm = normalize_metric_text_key(key)
            if not norm:
                continue
            lookup.setdefault(norm, canonical)

    return lookup


def _resolve_canonical_value(filename: str, value: str) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None

    lookup = _canonical_lookup(filename)
    norm = normalize_metric_text_key(text)
    if not norm:
        return None
    return lookup.get(norm)


def resolve_chart_type(value: str) -> Optional[str]:
    """Resolve a chart type (canonical or synonym) to ChartType canonical value."""
    return _resolve_canonical_value("ChartType.yml", value)


def resolve_groupby_canonical(value: str) -> Optional[str]:
    """Resolve a group-by field (canonical or synonym) to GroupByType canonical value."""
    return _resolve_canonical_value("GroupByType.yml", value)


def create_enum(name: str, filename: str) -> Any:
    values = get_canonical_values(filename)
    # Use functional API; maintain str subclassing for compatibility.
    return Enum(name, [(v, v) for v in values], type=str)


@lru_cache(maxsize=1)
def get_metric_metadata() -> Dict[str, Dict[str, Any]]:
    """Return metadata for metrics keyed by canonical value.

    Structure and compatibility:
    - Always returns common top-level keys when present: synonyms, data_type, properties.
    - Supports type-specific nested sections for cleanliness in SSOT YAML:
        • For Numeric metrics: a nested "numeric" block may contain unit, range_min, range_max, distribution_default_buckets.
        • For Enum metrics: a nested "enum" block may contain labels.
    - Backwards compatibility: numeric/enum fields are also promoted to top-level keys
      (e.g., unit, range_min, labels) for existing consumers.
    """
    items = _load_yaml("MetricType.yml")
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        canonical = item.get("canonical")
        if not isinstance(canonical, str):
            continue
        meta: Dict[str, Any] = {}

        # Common fields
        for key in ("synonyms", "data_type", "properties"):
            val = item.get(key)
            if val is not None:
                meta[key] = val

        # Display name: prefer first synonym if available
        syn = meta.get("synonyms")
        if isinstance(syn, list) and syn and isinstance(syn[0], str):
            meta["display_name"] = syn[0]

        # Numeric-specific (nested or flat)
        numeric = item.get("numeric")
        if isinstance(numeric, dict):
            numeric = cast(Dict[str, Any], numeric)
            # Preserve nested block
            meta["numeric"] = numeric
            # Promote known keys for compatibility
            for k in ("unit", "range_min", "range_max", "distribution_default_buckets"):
                if k in numeric and k not in meta:
                    meta[k] = numeric[k]
            # If a specific field is provided, synthesize properties if absent
            field = numeric.get("field")
            if isinstance(field, str) and field and "properties" not in meta:
                meta["properties"] = [field]
        # Also support legacy flat keys if present (pre-nested YAML)
        for k in ("unit", "range_min", "range_max", "distribution_default_buckets"):
            if k in item and k not in meta:
                meta[k] = item[k]

        # Enum-specific (nested or flat)
        enum_block = item.get("enum")
        if isinstance(enum_block, dict):
            enum_block = cast(Dict[str, Any], enum_block)
            meta["enum"] = enum_block
            # Promote labels when provided
            if "labels" in enum_block and "labels" not in meta:
                meta["labels"] = enum_block["labels"]
            # Promote id_field to properties for single-ID categoricals
            id_field = enum_block.get("id_field")
            if isinstance(id_field, str) and id_field and "properties" not in meta:
                meta["properties"] = [id_field]
            # Promote flags (multi-flag categoricals) to properties and labels
            flags_obj = enum_block.get("flags")
            if isinstance(flags_obj, list) and flags_obj and "properties" not in meta:
                flags = cast(List[Any], flags_obj)
                flag_keys: List[str] = []
                flag_labels: List[str] = []
                for f in flags:
                    if isinstance(f, dict):
                        f = cast(Dict[str, Any], f)
                        key = f.get("key")
                        label = f.get("label")
                        if isinstance(key, str) and key:
                            flag_keys.append(key)
                            flag_labels.append(
                                label if isinstance(label, str) and label else key
                            )
                    elif isinstance(f, str):
                        flag_keys.append(f)
                        flag_labels.append(f)
                if flag_keys:
                    meta["properties"] = flag_keys
                    # Only set labels from flags if not already provided
                    if "labels" not in meta:
                        meta["labels"] = flag_labels
            # New unified options format: enum.options with per-option synonyms.
            # Supports two shapes:
            #  - Single-choice (id_field present): properties = [id_field], labels derived from option.synonyms[0]
            #  - Multi-flag (no id_field): properties = [option.key ...], labels derived from option.synonyms[0]
            options_obj = enum_block.get("options")
            if isinstance(options_obj, list) and options_obj:
                options = cast(List[Any], options_obj)
                option_map: Dict[str, Any] = {}
                derived_labels: List[str] = []
                option_keys: List[str] = []
                for opt in options:
                    if not isinstance(opt, dict):
                        continue
                    opt = cast(Dict[str, Any], opt)
                    key = opt.get("key")
                    syns = opt.get("synonyms")
                    if not (isinstance(key, str) and key):
                        continue
                    if not (
                        isinstance(syns, list) and syns and isinstance(syns[0], str)
                    ):
                        # Fallback: fabricate a human label from key
                        human_label = key.replace("_", " ").title()
                        syns = [human_label]
                        opt["synonyms"] = syns
                    option_map[key] = {
                        k: v for k, v in opt.items() if k in ("synonyms", "value")
                    }
                    option_keys.append(key)
                    derived_labels.append(cast(str, syns[0]))
                # Attach raw options map for downstream richer usage
                enum_block["options"] = options  # preserve original list
                meta["enum_options"] = option_map
                # Single-choice vs multi-flag determination
                if enum_block.get("id_field"):
                    # Single-choice: keep properties as id_field (do not overwrite if already set)
                    if "labels" not in meta:
                        meta["labels"] = derived_labels
                else:
                    # Multi-flag: use option keys as properties if not already set
                    if "properties" not in meta:
                        meta["properties"] = option_keys
                    if "labels" not in meta:
                        meta["labels"] = derived_labels
        # Simplified Enum list form (uppercase key) for multi-flag-like enums
        enum_list = item.get("Enum")
        if isinstance(enum_list, list) and enum_list:
            entries = cast(List[Any], enum_list)
            option_map: Dict[str, Any] = {}
            option_keys: List[str] = []
            labels: List[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry = cast(Dict[str, Any], entry)
                key = entry.get("key")
                syns = entry.get("synonyms")
                if not (isinstance(key, str) and key):
                    continue
                if not (isinstance(syns, list) and syns and isinstance(syns[0], str)):
                    # fabricate human label from key
                    fabricated = key.replace("_", " ").title()
                    syns = [fabricated]
                    entry["synonyms"] = syns
                option_map[key] = {
                    k: v for k, v in entry.items() if k in ("synonyms", "value")
                }
                option_keys.append(key)
                labels.append(cast(str, syns[0]))
            if option_keys:
                # Do not overwrite properties/labels if already derived from other enum forms
                if "properties" not in meta:
                    meta["properties"] = option_keys
                if "labels" not in meta:
                    meta["labels"] = labels
                meta["enum_options"] = option_map
        # Legacy flat labels
        if "labels" in item and "labels" not in meta:
            meta["labels"] = item["labels"]

        # If properties still missing, leave absent to avoid guessing wrong field names.

        out[canonical] = meta
    return out


@lru_cache(maxsize=1)
def get_statistics_metric_enum_map() -> Dict[str, str]:
    """Return a mapping from SSOT metric canonical → StatisticsMetricEnum GQL value.

    Only metrics with a ``statistics_enum`` field in MetricType.yml are included;
    these are the metrics valid for ``getMannWhitneyUTest`` and similar endpoints.
    """
    items = _load_yaml("MetricType.yml")
    out: Dict[str, str] = {}
    for item in items:
        canonical = item.get("canonical")
        stats_enum = item.get("statistics_enum")
        if isinstance(canonical, str) and isinstance(stats_enum, str):
            out[canonical.strip()] = stats_enum.strip()
    return out


def _ci_get(d: Dict[str, Any], key: str) -> Any:
    """Case-insensitive dict.get for first-level keys."""
    if key in d:
        return d.get(key)
    for k, v in d.items():
        # keys are typed as str in this mapping, no isinstance needed
        if k.lower() == key.lower():
            return v
    return None


def normalize_metric_text_key(value: str) -> str:
    """Normalize free-text metric keys for lookup.

    - Lowercases
    - Strips leading/trailing whitespace
    - Replaces punctuation and symbols with single spaces
    - Collapses repeated whitespace
    """

    # Value is declared as str in the signature, so avoid redundant
    # isinstance checks and operate on it directly for type checkers.
    text = value.strip().lower()
    if not text:
        return ""
    # Replace any non-alphanumeric character with a space to be tolerant to
    # punctuation variants (hyphens, slashes, etc.).
    cleaned = re.sub(r"[^0-9a-z]+", " ", text)
    # Collapse multiple spaces
    return " ".join(cleaned.split())


@lru_cache(maxsize=1)
def get_metric_text_lookup() -> Dict[str, Dict[str, Any]]:
    """Return a normalized text lookup for metrics from MetricType.yml.

    The returned mapping uses normalized text keys (see ``normalize_metric_text_key``)
    built from both canonical codes and all synonyms. Each entry is a record of
    the form::

        {
            "canonical": str,
            "synonyms": List[str],
            "descriptions": Dict[str, str],  # language -> description
            "data_type": Optional[str],
            "unit": Optional[str],
        }

    This is intended for conversational lookup where the user provides a KPI
    or metric in natural language and we want to resolve it to the SSOT
    definition and multilingual description.
    """

    items = _load_yaml("MetricType.yml")
    lookup: Dict[str, Dict[str, Any]] = {}

    for item in items:
        canonical = item.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        code = canonical.strip().upper()

        syn_any: Any = item.get("synonyms") or []
        synonyms: List[str] = []
        if isinstance(syn_any, list):
            syn_list: List[Any] = cast(List[Any], syn_any)
            for s_any in syn_list:
                if isinstance(s_any, str):
                    s_val = s_any.strip()
                    if s_val:
                        synonyms.append(s_val)

        desc_any: Any = item.get("descriptions")
        descriptions: Dict[str, str] = {}
        if isinstance(desc_any, dict):
            desc_dict: Dict[Any, Any] = cast(Dict[Any, Any], desc_any)
            for lang_any, text_any in desc_dict.items():
                if not isinstance(lang_any, str) or not isinstance(text_any, str):
                    continue
                text_val = text_any.strip()
                if not text_val:
                    continue
                lang_key = lang_any.strip()
                if not lang_key:
                    continue
                descriptions[lang_key] = text_val

        data_type_any = _ci_get(item, "data_type")
        data_type: Optional[str]
        if isinstance(data_type_any, (str, bytes)):
            data_type = str(data_type_any).strip()
        else:
            data_type = None

        unit: Optional[str] = None
        # Support both "Numeric" and "numeric" blocks with a case-insensitive lookup.
        numeric_any = _ci_get(item, "numeric")
        if isinstance(numeric_any, dict):
            numeric = cast(Dict[str, Any], numeric_any)
            unit_any = _ci_get(numeric, "unit")
            if isinstance(unit_any, (str, bytes)) and str(unit_any).strip():
                unit = str(unit_any).strip()

        record: Dict[str, Any] = {
            "canonical": code,
            "synonyms": synonyms,
            "descriptions": descriptions,
            "data_type": data_type,
            "unit": unit,
        }

        # Register canonical and all synonyms under normalized text keys.
        keys: List[str] = [canonical] + synonyms
        for raw_key in keys:
            norm = normalize_metric_text_key(raw_key)
            if not norm:
                continue
            # Do not overwrite existing entries for the same normalized key;
            # first definition wins to keep behavior deterministic.
            if norm not in lookup:
                lookup[norm] = record

    return lookup


def validate_metric_metadata_complete(logger: Optional[Any] = None) -> List[str]:
    """Validate MetricType SSOT for basic completeness and log warnings.

    Checks performed:
    - Numeric metrics: require unit, range_min, range_max, default_buckets; also validate numeric types and ordering.
    - Enum metrics: require at least one option; each option needs key (str) and synonyms (non-empty list of str); duplicates flagged.

    Returns list of warning strings.
    """
    active_logger = logger or _LOGGER
    warnings: List[str] = []

    try:
        items = _load_yaml("MetricType.yml")
    except Exception as e:
        active_logger.warning("SSOT validation skipped: %s", e)
        return warnings

    for item in items:
        canonical = item.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            # Skip entries without canonical key
            continue
        code = canonical.strip()
        data_type = _ci_get(item, "data_type")
        data_type_str = (
            str(data_type).strip().lower()
            if isinstance(data_type, (str, bytes))
            else ""
        )

        if data_type_str == "numeric":
            numeric_any = _ci_get(item, "numeric")
            if not isinstance(numeric_any, dict):
                msg = f"SSOT incomplete [NUMERIC]: {code} missing Numeric block"
                warnings.append(msg)
                active_logger.warning(msg)
                continue
            numeric: Dict[str, Any] = cast(Dict[str, Any], numeric_any)
            unit = _ci_get(numeric, "unit")
            rmin = _ci_get(numeric, "range_min")
            rmax = _ci_get(numeric, "range_max")
            buckets = _ci_get(numeric, "default_buckets")

            missing: List[str] = []
            if unit is None or (isinstance(unit, str) and not unit.strip()):
                missing.append("unit")
            if rmin is None:
                missing.append("range_min")
            if rmax is None:
                missing.append("range_max")
            if buckets is None:
                missing.append("default_buckets")
            if missing:
                msg = f"SSOT incomplete [NUMERIC]: {code} missing {', '.join(missing)}"
                warnings.append(msg)
                active_logger.warning(msg)
            # Type and logical validation
            try:
                rmin_v = float(rmin) if rmin is not None else None
                rmax_v = float(rmax) if rmax is not None else None
                buckets_v = int(buckets) if buckets is not None else None
            except Exception:
                active_logger.debug(
                    "SSOT numeric validation fallback: failed to parse range or buckets",
                    exc_info=True,
                    extra={
                        "log_context": {
                            "event": "ssot_loader.metric_metadata.numeric_parse_fallback",
                            "operation": "validate_metric_metadata_complete",
                            "outcome": "degraded",
                            "metric_code": code,
                            "raw_range_min": rmin,
                            "raw_range_max": rmax,
                            "raw_buckets": buckets,
                        }
                    },
                )
                rmin_v = rmax_v = None
                buckets_v = None
            if rmin_v is None or rmax_v is None or buckets_v is None:
                msg = f"SSOT invalid [NUMERIC]: {code} non-numeric range/buckets"
                warnings.append(msg)
                active_logger.warning(msg)
            else:
                if rmin_v >= rmax_v:
                    msg = f"SSOT invalid [NUMERIC]: {code} range_min ({rmin_v}) >= range_max ({rmax_v})"
                    warnings.append(msg)
                    active_logger.warning(msg)
                if buckets_v <= 0:
                    msg = f"SSOT invalid [NUMERIC]: {code} default_buckets ({buckets_v}) must be > 0"
                    warnings.append(msg)
                    active_logger.warning(msg)

        elif data_type_str == "enum":
            # Support both simplified 'Enum' list and nested 'enum.options'
            enum_list_any = _ci_get(item, "Enum")
            options_raw: List[Any] = []
            if isinstance(enum_list_any, list):
                options_raw = cast(List[Any], enum_list_any)
            else:
                enum_block_any = _ci_get(item, "enum")
                if isinstance(enum_block_any, dict):
                    enum_block: Dict[str, Any] = cast(Dict[str, Any], enum_block_any)
                    opts_any = _ci_get(enum_block, "options")
                    if isinstance(opts_any, list):
                        options_raw = cast(List[Any], opts_any)
            if not options_raw:
                msg = f"SSOT incomplete [ENUM]: {code} has no options"
                warnings.append(msg)
                active_logger.warning(msg)
                continue
            seen_keys: set[str] = set()
            for idx, opt_any in enumerate(options_raw):
                if not isinstance(opt_any, dict):
                    msg = f"SSOT invalid [ENUM]: {code} option #{idx + 1} is not a dict"
                    warnings.append(msg)
                    active_logger.warning(msg)
                    continue
                opt: Dict[str, Any] = cast(Dict[str, Any], opt_any)
                key = _ci_get(opt, "key")
                syns = _ci_get(opt, "synonyms")
                if not isinstance(key, str) or not key.strip():
                    msg = (
                        f"SSOT incomplete [ENUM]: {code} option #{idx + 1} missing key"
                    )
                    warnings.append(msg)
                    active_logger.warning(msg)
                else:
                    k = key.strip()
                    if k in seen_keys:
                        msg = f"SSOT invalid [ENUM]: {code} duplicate option key '{k}'"
                        warnings.append(msg)
                        active_logger.warning(msg)
                    seen_keys.add(k)
                syns_list: List[Any] = (
                    cast(List[Any], syns) if isinstance(syns, list) else []
                )
                if not (
                    syns_list
                    and all(isinstance(s, str) and s.strip() for s in syns_list)
                ):
                    msg = f"SSOT incomplete [ENUM]: {code} option '{key}' missing synonyms"
                    warnings.append(msg)
                    active_logger.warning(msg)
        else:
            # Unknown or missing data_type
            msg = f"SSOT incomplete: {code} has unknown or missing data_type"
            warnings.append(msg)
            active_logger.warning(msg)

    return warnings


def _first_synonym(item: Dict[str, Any]) -> Optional[str]:
    syn = item.get("synonyms")
    if isinstance(syn, list) and syn and isinstance(syn[0], str):
        return syn[0]
    return None


@lru_cache(maxsize=1)
def _metric_meta_cached() -> Dict[str, Dict[str, Any]]:
    return get_metric_metadata()


def get_metric_display_name(metric_code: str) -> str:
    """Return SSOT-preferred display name for a metric (first synonym), fallback to canonical code.

    Avoids local formatting in executors; centralizes presentation in the loader.
    """
    code = (metric_code or "").upper()
    meta = _metric_meta_cached().get(code) or {}
    disp = meta.get("display_name")
    if isinstance(disp, str) and disp.strip():
        return disp
    return code


def get_enum_option_label(metric_code: str, key: str) -> Optional[str]:
    """Return preferred label for an enum option of a given metric from SSOT.

    Looks up MetricType.yml entry, then 'enum_options' mapping derived by get_metric_metadata.
    Fallbacks: None if not found.
    """
    code = (metric_code or "").upper()
    k = (key or "").strip()
    if not code or not k:
        return None
    meta = _metric_meta_cached().get(code) or {}
    options = cast(Dict[str, Any], meta.get("enum_options") or {})
    # Try exact key, then case-insensitive match
    entry = options.get(k)
    if entry is None:
        for ok, ov in options.items():
            if ok.lower() == k.lower():
                entry = ov
                break
    if isinstance(entry, dict):
        entry_dict: Dict[str, Any] = cast(Dict[str, Any], entry)
        syns = entry_dict.get("synonyms")
        if isinstance(syns, list) and syns and isinstance(syns[0], str):
            return syns[0]
    return None


def get_canonical_display_name(canonical: str) -> str:
    """Return display name for a canonical field/metric from SSOT synonyms.

    Prefers MetricType.yml first; if not found, falls back to canonical value.
    """
    code = (canonical or "").upper()
    meta = _metric_meta_cached().get(code)
    if meta:
        disp = meta.get("display_name")
        if isinstance(disp, str) and disp.strip():
            return disp
    # Optional: check GroupByType.yml for synonyms
    try:
        items = _load_yaml("GroupByType.yml")
        for it in items:
            can = it.get("canonical")
            if isinstance(can, str) and can.upper() == code:
                lbl = _first_synonym(it)
                if lbl:
                    return lbl
    except Exception:
        _LOGGER.debug(
            "Failed to resolve GroupByType display label; using canonical fallback",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "ssot_loader.groupby_display_name.fallback",
                    "operation": "get_canonical_display_name",
                    "outcome": "degraded",
                    "canonical": code,
                    "filename": "GroupByType.yml",
                }
            },
        )
    return code


def _label_from_simple_type_file(filename: str, value: str) -> Optional[str]:
    """Return label from simple SSOT type files (e.g., SexType.yml, StrokeType.yml) using first synonym.

    Matches by canonical (case-insensitive). Returns None if not found.
    """
    try:
        items = _load_yaml(filename)
        val_up = (value or "").upper()
        for it in items:
            can = it.get("canonical")
            if isinstance(can, str) and can.upper() == val_up:
                return _first_synonym(it) or can
    except Exception:
        _LOGGER.debug(
            "Failed to resolve SSOT simple-type label; returning None",
            exc_info=True,
            extra={
                "log_context": {
                    "event": "ssot_loader.simple_type_label.fallback",
                    "operation": "_label_from_simple_type_file",
                    "outcome": "degraded",
                    "filename": filename,
                    "value": value,
                }
            },
        )
        return None
    return None


def get_sex_label(value: str) -> str:
    return _label_from_simple_type_file("SexType.yml", value) or value


def get_stroke_label(value: str) -> str:
    return _label_from_simple_type_file("StrokeType.yml", value) or value


def resolve_sex(value: str) -> Optional[str]:
    """Resolve a sex value (canonical or synonym) to SexType canonical value."""
    return _resolve_canonical_value("SexType.yml", value)


def resolve_stroke_type(value: str) -> Optional[str]:
    """Resolve a stroke type (canonical or synonym) to StrokeType canonical value."""
    return _resolve_canonical_value("StrokeType.yml", value)


__all__ = [
    "BASE_SSOT",
    "create_enum",
    "get_ssot_items",
    "get_canonical_values",
    "get_metric_metadata",
    "get_metric_text_lookup",
    "validate_metric_metadata_complete",
    "get_metric_display_name",
    "get_enum_option_label",
    "get_canonical_display_name",
    "resolve_chart_type",
    "resolve_groupby_canonical",
    "get_sex_label",
    "get_stroke_label",
    "normalize_metric_text_key",
]
