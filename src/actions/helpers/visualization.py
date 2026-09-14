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
from src.util import env as env_util

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


def canonicalize_ssot_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic SSOT canonicalization only (no fallback inference).
    normalized: Dict[str, Any] = {}
    for key, value in entities.items():
        resolver = _ENTITY_SSOT_RESOLVERS.get(key)
        if resolver:
            if isinstance(value, list):
                normalized[key] = [resolver(v) or v for v in cast(List[Any], value)]
            elif isinstance(value, str):
                normalized[key] = resolver(value) or value
        else:
            normalized[key] = value
    return normalized


def pretty_print_graphql_query(query: str) -> str:
    """Format a compact GraphQL query into a readable, indented multiline string."""
    compact = " ".join((query or "").split())
    if not compact:
        return ""

    lines: List[str] = []
    token: List[str] = []
    indent = 0
    in_string = False
    escaped = False

    def _flush_current() -> None:
        text = "".join(token).strip()
        token.clear()
        if text:
            lines.append(f"{'  ' * indent}{text}")

    for ch in compact:
        if ch == '"' and not escaped:
            in_string = not in_string

        if in_string:
            token.append(ch)
            escaped = ch == "\\" and not escaped
            continue

        escaped = False

        if ch == "{":
            head = "".join(token).strip()
            token.clear()
            if head:
                lines.append(f"{'  ' * indent}{head} {{")
            else:
                lines.append(f"{'  ' * indent}{{")
            indent += 1
            continue

        if ch == "}":
            _flush_current()
            indent = max(0, indent - 1)
            lines.append(f"{'  ' * indent}}}")
            continue

        if ch.isspace():
            if token and token[-1] != " ":
                token.append(" ")
            continue

        token.append(ch)

    _flush_current()
    return "\n".join(lines)


def extract_entities_from_latest_message(
    latest_message: Dict[str, Any],
) -> Dict[str, Any]:
    entities_any = latest_message.get("entities", [])
    if not isinstance(entities_any, list):
        return {}

    entities_list = cast(List[Any], entities_any)

    # RegexEntityExtractor's SSOT-derived lookup tables match a literal word
    # anywhere it appears (e.g. "age" is a valid metric name AND a valid
    # group_by value), with zero regard for context. DIETClassifier, by
    # contrast, assigns at most one contextual label per token span. When the
    # two disagree about what the *same span* of text means (e.g. "age" in
    # "...grouped by age in 10-year buckets" is unambiguously the group_by,
    # but the metric lookup table also matches it), the regex-only reading is
    # a false positive, not a genuine second candidate -- it previously read
    # to the decision-stage LLM as a real ambiguity ("DTN or AGE?") for an
    # unambiguous request. Build the set of DIET-confirmed spans first so the
    # second pass can drop any entity type whose only support is a lookup-table
    # match contradicted by DIET's own reading of those exact tokens.
    diet_span_labels: Dict[tuple[Any, Any], str] = {}
    for ent_any in entities_list:
        if not isinstance(ent_any, dict):
            continue
        ent = cast(Dict[str, Any], ent_any)
        extractors = ent.get("extractors")
        extractor_names = (
            {e.get("extractor") for e in extractors if isinstance(e, dict)}
            if isinstance(extractors, list)
            else set()
        )
        if "DIETClassifier" in extractor_names:
            span = (ent.get("start"), ent.get("end"))
            entity_type = ent.get("entity")
            if isinstance(entity_type, str):
                diet_span_labels[span] = entity_type

    extracted: Dict[str, Any] = {}
    for ent_any in entities_list:
        if not isinstance(ent_any, dict):
            continue
        ent = cast(Dict[str, Any], ent_any)
        key_any = ent.get("entity")
        if not isinstance(key_any, str) or "value" not in ent:
            continue

        extractors = ent.get("extractors")
        extractor_names = (
            {e.get("extractor") for e in extractors if isinstance(e, dict)}
            if isinstance(extractors, list)
            else set()
        )
        span = (ent.get("start"), ent.get("end"))
        diet_label_for_span = diet_span_labels.get(span)
        if (
            "DIETClassifier" not in extractor_names
            and diet_label_for_span is not None
            and diet_label_for_span != key_any
        ):
            continue

        value = ent["value"]
        if key_any not in extracted:
            extracted[key_any] = value
            continue

        existing = extracted[key_any]
        if isinstance(existing, list):
            existing_list = cast(List[Any], existing)
            if value not in existing_list:
                existing_list.append(value)
        elif value != existing:
            extracted[key_any] = [existing, value]
        # else: identical repeat of an already-captured scalar value (e.g.
        # "male patients DTN" and "female patients DTN" both mention DTN) --
        # not a second distinct answer, so it must not turn a clean scalar
        # into a redundant [DTN, DTN] list. That shape previously read to the
        # decision-stage LLM as two different metric candidates to choose
        # between, producing a spurious "which metric?" clarification for an
        # unambiguous request.

    return extracted


def resolve_override_language(metadata: Dict[str, Any], slots: Dict[str, Any]) -> Optional[str]:
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


def serialize_plan_for_frontend(plan: Any) -> Dict[str, Any]:
    """Serialize planner output into a minimal preview contract for the frontend."""

    payload = _maybe_model_dump_dict(plan, mode="json", by_alias=True, exclude_none=True)
    if payload is not None:
        payload_any: object = payload
    elif isinstance(plan, dict):
        payload_any = _mapping_to_dict(plan)
    else:
        return {}

    payload_dict = _mapping_to_dict(payload_any)
    if not payload_dict:
        return {}

    def _metric_preview(metric_any: Any) -> Dict[str, Any]:
        metric = _mapping_to_dict(metric_any)
        metric_code = metric.get("metric")
        if not (isinstance(metric_code, str) and metric_code.strip()):
            return {}

        out: Dict[str, Any] = {"metric": metric_code.strip().upper()}

        data_origin = _mapping_to_dict(metric.get("data_origin"))
        if not data_origin:
            data_origin = _mapping_to_dict(metric.get("dataOrigin"))
        if data_origin:
            data_origin_out: Dict[str, Any] = {}
            provider_ids_any = data_origin.get("provider_id")
            if isinstance(provider_ids_any, list):
                provider_ids = []
                for raw in cast(List[Any], provider_ids_any):
                    if isinstance(raw, int) and raw > 0:
                        provider_ids.append(raw)
                if provider_ids:
                    data_origin_out["provider_id"] = provider_ids

            provider_group_ids_any = data_origin.get("provider_group_id")
            if isinstance(provider_group_ids_any, list):
                provider_group_ids = []
                for raw in cast(List[Any], provider_group_ids_any):
                    if isinstance(raw, int) and raw > 0:
                        provider_group_ids.append(raw)
                if provider_group_ids:
                    data_origin_out["provider_group_id"] = provider_group_ids

            if data_origin_out:
                out["data_origin"] = data_origin_out

        origin_scope = _mapping_to_dict(metric.get("origin_scope"))
        if not origin_scope:
            origin_scope = _mapping_to_dict(metric.get("originScope"))
        if origin_scope:
            origin_scope_out: Dict[str, Any] = {}
            scope_type = origin_scope.get("scope_type")
            if not isinstance(scope_type, str) or not scope_type.strip():
                scope_type = origin_scope.get("scopeType")
            if isinstance(scope_type, str) and scope_type.strip():
                origin_scope_out["scope_type"] = scope_type.strip().upper()

            label = origin_scope.get("label")
            if isinstance(label, str) and label.strip():
                origin_scope_out["label"] = label.strip()

            value = origin_scope.get("value")
            if isinstance(value, (str, int, float)) and value != "":
                origin_scope_out["value"] = value

            country_code = origin_scope.get("country_code")
            if not isinstance(country_code, str) or not country_code.strip():
                country_code = origin_scope.get("countryCode")
            if isinstance(country_code, str) and country_code.strip():
                origin_scope_out["country_code"] = country_code.strip().upper()

            if origin_scope_out:
                out["origin_scope"] = origin_scope_out

        return out

    def _split_preview(split_any: Any) -> Dict[str, Any]:
        split = _mapping_to_dict(split_any)
        kind = split.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            return {}

        out: Dict[str, Any] = {"kind": kind.strip().upper()}
        categories_any = split.get("categories")
        if isinstance(categories_any, list):
            categories: List[str] = []
            for raw in cast(List[Any], categories_any):
                if isinstance(raw, str) and raw.strip():
                    categories.append(raw.strip().upper())
            if categories:
                out["categories"] = categories
        return out

    def _filter_preview(filter_any: Any) -> Dict[str, Any]:
        filter_node = _mapping_to_dict(filter_any)
        if not filter_node:
            return {}

        filter_type = filter_node.get("type")
        if not isinstance(filter_type, str) or not filter_type.strip():
            return {}

        filter_type_norm = filter_type.strip().upper()
        out: Dict[str, Any] = {"type": filter_type_norm}

        operator = filter_node.get("operator")
        if isinstance(operator, str) and operator.strip():
            out["operator"] = operator.strip().upper()

        value = filter_node.get("value")
        if isinstance(value, (str, int, float, bool)):
            out["value"] = value

        if filter_type_norm == "ANDFILTER":
            clauses_any = filter_node.get("and_")
            if not isinstance(clauses_any, list):
                clauses_any = filter_node.get("and")
            if isinstance(clauses_any, list):
                clauses: List[Dict[str, Any]] = []
                for item in cast(List[Any], clauses_any):
                    preview = _filter_preview(item)
                    if preview:
                        clauses.append(preview)
                if clauses:
                    out["and"] = clauses

        elif filter_type_norm == "ORFILTER":
            clauses_any = filter_node.get("or_")
            if not isinstance(clauses_any, list):
                clauses_any = filter_node.get("or")
            if isinstance(clauses_any, list):
                clauses: List[Dict[str, Any]] = []
                for item in cast(List[Any], clauses_any):
                    preview = _filter_preview(item)
                    if preview:
                        clauses.append(preview)
                if clauses:
                    out["or"] = clauses

        elif filter_type_norm == "NOTFILTER":
            clause_any = filter_node.get("not_")
            if clause_any is None:
                clause_any = filter_node.get("not")
            preview = _filter_preview(clause_any)
            if preview:
                out["not"] = preview

        return out

    def _chart_preview(chart_any: Any) -> Dict[str, Any]:
        chart = _mapping_to_dict(chart_any)
        out: Dict[str, Any] = {}

        chart_type = chart.get("chart_type")
        if isinstance(chart_type, str) and chart_type.strip():
            out["chart_type"] = chart_type.strip().upper()

        metrics_any = chart.get("metrics")
        if isinstance(metrics_any, list):
            metrics: List[Dict[str, Any]] = []
            for item in cast(List[Any], metrics_any):
                preview = _metric_preview(item)
                if preview:
                    metrics.append(preview)
            if metrics:
                out["metrics"] = metrics

        semantics = _mapping_to_dict(chart.get("semantics"))
        if semantics:
            semantics_out: Dict[str, Any] = {}
            intent = semantics.get("intent")
            if isinstance(intent, str) and intent.strip():
                semantics_out["intent"] = intent.strip().upper()

            measure = _mapping_to_dict(semantics.get("measure"))
            measure_type = measure.get("type")
            if isinstance(measure_type, str) and measure_type.strip():
                semantics_out["measure"] = {"type": measure_type.strip().upper()}

            time_spec = _mapping_to_dict(semantics.get("time"))
            grain = time_spec.get("grain")
            if isinstance(grain, str) and grain.strip():
                semantics_out["time"] = {"grain": grain.strip().upper()}

            splits_any = semantics.get("splits")
            if isinstance(splits_any, list):
                splits: List[Dict[str, Any]] = []
                for item in cast(List[Any], splits_any):
                    preview = _split_preview(item)
                    if preview:
                        splits.append(preview)
                if splits:
                    semantics_out["splits"] = splits

            if semantics_out:
                out["semantics"] = semantics_out

        return out

    def _statistical_test_preview(test_any: Any) -> Dict[str, Any]:
        test = _mapping_to_dict(test_any)
        out: Dict[str, Any] = {}

        test_type = test.get("test_type")
        if isinstance(test_type, str) and test_type.strip():
            out["test_type"] = test_type.strip().upper()

        metrics_any = test.get("metrics")
        if isinstance(metrics_any, list):
            metrics: List[Dict[str, Any]] = []
            for item in cast(List[Any], metrics_any):
                preview = _metric_preview(item)
                if preview:
                    metrics.append(preview)
            if metrics:
                out["metrics"] = metrics

        group_by_any = test.get("group_by")
        if isinstance(group_by_any, list):
            group_by: List[Dict[str, Any]] = []
            for item in cast(List[Any], group_by_any):
                preview = _split_preview(item)
                if preview:
                    group_by.append(preview)
            if group_by:
                out["group_by"] = group_by

        filters_any = test.get("filters")
        filter_preview = _filter_preview(filters_any)
        if filter_preview:
            out["filters"] = filter_preview

        return out

    charts_any = payload_dict.get("charts")
    charts: List[Dict[str, Any]] = []
    if isinstance(charts_any, list):
        for item in cast(List[Any], charts_any):
            preview = _chart_preview(item)
            if preview:
                charts.append(preview)

    tests_any = payload_dict.get("statistical_tests")
    tests: List[Dict[str, Any]] = []
    if isinstance(tests_any, list):
        for item in cast(List[Any], tests_any):
            preview = _statistical_test_preview(item)
            if preview:
                tests.append(preview)

    out_payload: Dict[str, Any] = {}
    if charts:
        out_payload["charts"] = charts
    if tests:
        out_payload["statistical_tests"] = tests

    return out_payload


def format_execution_summary(
    summary: Dict[str, Any] | Any,
    language: Optional[str] = None,
    include_opening_line: bool = True,
) -> str:
    """A short, natural closing line for the chat -- the chart(s)/stat(s)
    themselves render in the UI, so this doesn't need to describe them in
    detail, just close the loop conversationally. Diagnostic detail (trace
    id, cache stats, query batching, plan normalization) used to be dumped
    here too; that's developer information, not something an end user
    needs, and CVaLab's own debug chat already surfaces it from the
    structured payload -- see visualization_plan/visualization_response.

    `include_opening_line` exists because the one-shot flow
    (visualization_action.py) already sends its own specific confirmation
    via `_build_confirmation_message` (e.g. "Here's your line chart of
    DTN.") before calling this -- without it, callers would get two
    redundant "here's your chart" sentences back to back. The guided flow
    has no such confirmation elsewhere, so it keeps the default. May return
    an empty string (e.g. opening line suppressed and nothing else to add);
    callers should skip sending a message in that case."""
    def t(key: str, default: str, params: Optional[Dict[str, Any]] = None) -> str:
        return translate(key, language=language, params=params, default=default)

    summary_dict = _maybe_model_dump_dict(summary)
    if summary_dict is None and isinstance(summary, dict):
        summary_dict = _mapping_to_dict(summary)
    if summary_dict is None:
        return t("action.summary.complete_fallback", "Done!") if include_opening_line else ""

    chart_count = summary_dict.get("chart_count")
    stats_count = summary_dict.get("stats_count")
    stats_skipped = summary_dict.get("stats_skipped")
    stats_errors = summary_dict.get("stats_errors")

    has_charts = isinstance(chart_count, int) and chart_count > 0
    has_stats = isinstance(stats_count, int) and stats_count > 0

    lines: List[str] = []
    if include_opening_line:
        if has_charts and has_stats:
            lines.append(t("action.summary.complete_mixed", "Here's your chart and statistical comparison."))
        elif has_charts:
            if chart_count == 1:
                lines.append(t("action.summary.complete_chart_one", "Here's your chart."))
            else:
                lines.append(t("action.summary.complete_chart_many", "Here are your {chart_count} charts.", {"chart_count": chart_count}))
        elif has_stats:
            if stats_count == 1:
                lines.append(t("action.summary.complete_stat_one", "Here's your statistical comparison."))
            else:
                lines.append(t("action.summary.complete_stat_many", "Here are your {stats_count} statistical comparisons.", {"stats_count": stats_count}))
        else:
            lines.append(t("action.summary.complete_fallback", "Done!"))

    if isinstance(stats_errors, int) and stats_errors > 0:
        if stats_errors == 1:
            lines.append(t("action.summary.stats_errors_one", "One statistical test couldn't be completed."))
        else:
            lines.append(
                t(
                    "action.summary.stats_errors_many",
                    "{stats_errors} statistical tests couldn't be completed.",
                    {"stats_errors": stats_errors},
                )
            )
    elif isinstance(stats_skipped, int) and stats_skipped > 0:
        if stats_skipped == 1:
            lines.append(t("action.summary.stats_skipped_one", "One statistical test was skipped."))
        else:
            lines.append(
                t(
                    "action.summary.stats_skipped_many",
                    "{stats_skipped} statistical tests were skipped.",
                    {"stats_skipped": stats_skipped},
                )
            )

    return "\n".join(lines)
