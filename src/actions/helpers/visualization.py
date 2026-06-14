from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from src.actions.i18n import translate
from src.shared.ssot_loader import resolve_chart_type, resolve_sex, resolve_stroke_type

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(self, *args: Any, **kwargs: Any) -> object: ...


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: Dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def _maybe_model_dump_dict(value: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, SupportsModelDump):
        return None
    return _mapping_to_dict(value.model_dump(**kwargs))


_ENTITY_SSOT_RESOLVERS = {
    "sex": resolve_sex,
    "stroke_type": resolve_stroke_type,
    "chart_type": resolve_chart_type,
}


def normalize_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in entities.items():
        resolver = _ENTITY_SSOT_RESOLVERS.get(key)
        if resolver:
            if isinstance(value, list):
                normalized[key] = [resolver(v) or v for v in value]
            elif isinstance(value, str):
                normalized[key] = resolver(value) or value
        else:
            normalized[key] = value
    return normalized


def extract_entities_from_latest_message(
    latest_message: Dict[str, Any],
) -> Dict[str, Any]:
    entities_any = latest_message.get("entities", [])
    if not isinstance(entities_any, list):
        return {}

    entities_list = cast(List[Any], entities_any)
    extracted: Dict[str, Any] = {}
    for ent_any in entities_list:
        if not isinstance(ent_any, dict):
            continue
        ent = cast(Dict[str, Any], ent_any)
        key_any = ent.get("entity")
        if not isinstance(key_any, str) or "value" not in ent:
            continue

        value = ent["value"]
        if key_any not in extracted:
            extracted[key_any] = value
            continue

        existing = extracted[key_any]
        if isinstance(existing, list):
            existing_list = cast(List[Any], existing)
            existing_list.append(value)
        else:
            extracted[key_any] = [existing, value]

    return extracted


def resolve_override_language(
    metadata: Dict[str, Any], slots: Dict[str, Any]
) -> Optional[str]:
    override_language: Any = None
    lang_meta = metadata.get("language")
    if isinstance(lang_meta, str) and lang_meta.strip():
        override_language = lang_meta
    if override_language is None:
        slot_lang = slots.get("language")
        if isinstance(slot_lang, str) and slot_lang.strip():
            override_language = slot_lang

    if isinstance(override_language, str):
        normalized = override_language.split("-")[0].lower()
        return normalized or None
    return None


def _strip_text_fields(value: Any) -> Any:
    """Drop user-facing free-text fields to prevent LLM prose from reaching clients."""

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in _mapping_to_dict(value).items():
            if key in {"title", "description"}:
                continue
            out[key] = _strip_text_fields(child)
        return out
    if isinstance(value, list):
        items = cast(List[object], value)
        return [_strip_text_fields(item) for item in items]
    return value


def serialize_plan_for_frontend(plan: Any) -> Dict[str, Any]:
    """Serialize planner output for frontend consumption without mutating it."""

    payload = _maybe_model_dump_dict(
        plan, mode="json", by_alias=True, exclude_none=True
    )
    if payload is not None:
        payload_any: object = payload
    elif isinstance(plan, dict):
        payload_any = _mapping_to_dict(plan)
    else:
        return {}

    payload_dict = _mapping_to_dict(payload_any)
    if not payload_dict:
        return {}

    sanitized_any = _strip_text_fields(payload_dict)
    return (
        cast(Dict[str, Any], sanitized_any) if isinstance(sanitized_any, dict) else {}
    )


def format_execution_summary(
    summary: Dict[str, Any] | Any,
    show_normalization: bool = True,
    planner_diagnostics: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
) -> str:
    def t(key: str, default: str, params: Optional[Dict[str, Any]] = None) -> str:
        return translate(key, language=language, params=params, default=default)

    summary_dict = _maybe_model_dump_dict(summary)
    if summary_dict is None and isinstance(summary, dict):
        summary_dict = _mapping_to_dict(summary)
    if summary_dict is None:
        return t("action.summary.complete", "✅ Visualization generation complete.")

    estimated = summary_dict.get("estimated_queries")
    actual = summary_dict.get("actual_queries")
    chart_count = summary_dict.get("chart_count")
    trace_id = summary_dict.get("trace_id")
    normalization = _mapping_to_dict(summary_dict.get("normalization")) or None
    batches_any = summary_dict.get("batches")
    batches: List[Any] = (
        cast(List[Any], batches_any) if isinstance(batches_any, list) else []
    )

    lines: List[str] = [
        t("action.summary.complete", "✅ Visualization generation complete.")
    ]

    if isinstance(trace_id, str) and trace_id.strip():
        lines.append(
            t(
                "action.summary.trace_id",
                "Trace ID: {trace_id}",
                {"trace_id": trace_id.strip()},
            )
        )

    if isinstance(chart_count, int):
        if chart_count == 1:
            lines.append(
                t("action.summary.plan_produced_one_chart", "Plan produced 1 chart.")
            )
        else:
            lines.append(
                t(
                    "action.summary.plan_produced_many_charts",
                    "Plan produced {chart_count} charts.",
                    {"chart_count": chart_count},
                )
            )

    if isinstance(planner_diagnostics, dict):
        cache_hit = planner_diagnostics.get("last_call_cache_hit")
        total_hits = planner_diagnostics.get("total_hits")
        total_misses = planner_diagnostics.get("total_misses")
        total_expired = planner_diagnostics.get("total_expired")
        entries = planner_diagnostics.get("entries")
        capacity = planner_diagnostics.get("capacity")
        ttl_seconds = planner_diagnostics.get("ttl_seconds")
        key_version = planner_diagnostics.get("key_version")

        if cache_hit is True:
            lines.append(
                t(
                    "action.summary.planner_cache_hit",
                    "Planner cache: hit (reused a previously generated plan).",
                )
            )
        elif cache_hit is False:
            lines.append(
                t(
                    "action.summary.planner_cache_miss",
                    "Planner cache: miss (generated a fresh plan).",
                )
            )

        stats: List[str] = []
        if isinstance(total_hits, int) and isinstance(total_misses, int):
            stats.append(f"hits={total_hits}")
            stats.append(f"misses={total_misses}")
        if isinstance(total_expired, int):
            stats.append(f"expired={total_expired}")
        if isinstance(entries, int) and isinstance(capacity, int):
            stats.append(f"entries={entries}/{capacity}")
        if isinstance(ttl_seconds, (int, float)):
            stats.append(f"ttl={int(ttl_seconds)}s")
        if isinstance(key_version, str) and key_version:
            stats.append(f"cache_key={key_version}")
        if stats:
            lines.append(" - " + "; ".join(stats))

    if isinstance(actual, int):
        if actual == 1:
            lines.append(
                t(
                    "action.summary.queried_once",
                    "I queried the analytics service once.",
                )
            )
        else:
            lines.append(
                t(
                    "action.summary.queried_many",
                    "I queried the analytics service {actual} times.",
                    {"actual": actual},
                )
            )

        if isinstance(estimated, int) and estimated != actual:
            lines.append(
                t(
                    "action.summary.planner_estimate",
                    "Planner estimate was {estimated} request(s).",
                    {"estimated": estimated},
                )
            )

    if show_normalization and normalization is not None:
        charts_in = normalization.get("charts_in")
        charts_out = normalization.get("charts_out")
        dropped_charts = normalization.get("dropped_empty_charts")
        metrics_in = normalization.get("metrics_in")
        metrics_out = normalization.get("metrics_out")
        dropped_metrics = normalization.get("dropped_empty_metrics")
        metric_code_norm = normalization.get("normalized_metric_codes")
        deduped_groupby = normalization.get("deduped_groupby_entries")
        normalized_groupby_fields = normalization.get(
            "normalized_canonical_groupby_fields"
        )
        dropped_groupby_fields = normalization.get("dropped_invalid_groupby_fields")
        normalized_text = normalization.get("normalized_text_fields")

        if isinstance(charts_in, int) and isinstance(charts_out, int):
            lines.append(t("action.summary.plan_normalization", "Plan normalization:"))
            lines.append(
                " - "
                + t(
                    "action.summary.charts_transition",
                    "Charts: {charts_in} -> {charts_out}",
                    {"charts_in": charts_in, "charts_out": charts_out},
                )
            )

            details: List[str] = []
            if isinstance(dropped_charts, int) and dropped_charts > 0:
                details.append(
                    t(
                        "action.summary.detail_dropped_charts",
                        "dropped {dropped_charts} empty chart(s)",
                        {"dropped_charts": dropped_charts},
                    )
                )
            if (
                isinstance(metrics_in, int)
                and isinstance(metrics_out, int)
                and metrics_in != metrics_out
            ):
                details.append(
                    t(
                        "action.summary.detail_metrics_transition",
                        "metrics {metrics_in} -> {metrics_out}",
                        {"metrics_in": metrics_in, "metrics_out": metrics_out},
                    )
                )
            if isinstance(dropped_metrics, int) and dropped_metrics > 0:
                details.append(
                    t(
                        "action.summary.detail_dropped_metrics",
                        "dropped {dropped_metrics} empty metric(s)",
                        {"dropped_metrics": dropped_metrics},
                    )
                )
            if isinstance(metric_code_norm, int) and metric_code_norm > 0:
                details.append(
                    t(
                        "action.summary.detail_normalized_metric_codes",
                        "normalized {metric_code_norm} metric code(s)",
                        {"metric_code_norm": metric_code_norm},
                    )
                )
            if isinstance(deduped_groupby, int) and deduped_groupby > 0:
                details.append(
                    t(
                        "action.summary.detail_removed_groupby_duplicates",
                        "removed {deduped_groupby} duplicate group-by entries",
                        {"deduped_groupby": deduped_groupby},
                    )
                )
            if (
                isinstance(normalized_groupby_fields, int)
                and normalized_groupby_fields > 0
            ):
                details.append(
                    t(
                        "action.summary.detail_normalized_groupby_fields",
                        "normalized {normalized_groupby_fields} canonical group-by field(s)",
                        {"normalized_groupby_fields": normalized_groupby_fields},
                    )
                )
            if isinstance(dropped_groupby_fields, int) and dropped_groupby_fields > 0:
                details.append(
                    t(
                        "action.summary.detail_dropped_groupby_fields",
                        "dropped {dropped_groupby_fields} invalid group-by field(s)",
                        {"dropped_groupby_fields": dropped_groupby_fields},
                    )
                )
            if isinstance(normalized_text, int) and normalized_text > 0:
                details.append(
                    t(
                        "action.summary.detail_cleaned_text_fields",
                        "cleaned {normalized_text} text field(s)",
                        {"normalized_text": normalized_text},
                    )
                )

            if details:
                lines.append(" - " + "; ".join(details))
            else:
                lines.append(
                    " - "
                    + t(
                        "action.summary.no_structural_changes",
                        "No structural changes were needed.",
                    )
                )

    if batches:
        lines.append(t("action.summary.what_i_queried", "What I queried:"))
        max_batches = 4
        for idx, batch_any in enumerate(batches[:max_batches], start=1):
            if not isinstance(batch_any, dict):
                continue
            batch = cast(Dict[str, Any], batch_any)

            query_count = batch.get("query_count")
            groupby = batch.get("server_groupby")
            periods = batch.get("batched_time_period_count")
            filters_any = batch.get("filter_dimensions")
            filters_list: List[Any] = (
                cast(List[Any], filters_any) if isinstance(filters_any, list) else []
            )
            filter_names = [str(x).replace("GroupBy", "").strip() for x in filters_list]

            parts: List[str] = [f"{idx})"]
            if isinstance(query_count, int):
                parts.append(
                    t(
                        "action.summary.request_count",
                        "{query_count} request(s)",
                        {"query_count": query_count},
                    )
                )
            if isinstance(groupby, str) and groupby:
                parts.append(
                    t(
                        "action.summary.grouped_by",
                        "grouped by {groupby}",
                        {"groupby": groupby},
                    )
                )
            if isinstance(periods, int) and periods > 0:
                parts.append(
                    t(
                        "action.summary.across_periods",
                        "across {periods} time period(s)",
                        {"periods": periods},
                    )
                )
            if filter_names:
                parts.append(
                    t(
                        "action.summary.split_by",
                        "split by {filters}",
                        {"filters": ", ".join(filter_names)},
                    )
                )

            lines.append(" - " + " | ".join(parts))

        remaining = len(batches) - max_batches
        if remaining > 0:
            lines.append(
                " - "
                + t(
                    "action.summary.remaining_batches",
                    "... and {remaining} more query batch(es)",
                    {"remaining": remaining},
                )
            )

    return "\n".join(lines)
