from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast

from langchain.prompts import ChatPromptTemplate

from src.actions.i18n import translate
from src.planners.langchain.llm_factory import create_chat_llm
from src.shared import ssot_loader
from src.util import env as env_util
from src.util.logging_utils import bind_current_context

logger = logging.getLogger(__name__)


class QueryDecisionBase(TypedDict):
    decision: Literal["proceed", "clarify", "reject"]
    reason: str
    message: Optional[str]


class QueryDecision(QueryDecisionBase, total=False):
    clarification_type: str
    clarification_options: List[str]


_ENABLE_QUERY_GUARD = env_util.env_flag("ACTIONS_ENABLE_QUERY_GUARD", default=True)
_QUERY_GUARD_TIMEOUT_SECONDS_RAW = env_util.get_env("ACTIONS_QUERY_GUARD_TIMEOUT_SECONDS", default="8") or "8"
try:
    _QUERY_GUARD_TIMEOUT_SECONDS = max(1.0, float(_QUERY_GUARD_TIMEOUT_SECONDS_RAW))
except Exception:
    _QUERY_GUARD_TIMEOUT_SECONDS = 8.0

_QUERY_GUARD_TEMPERATURE_RAW = env_util.get_env("ACTIONS_QUERY_GUARD_TEMPERATURE", default="0") or "0"
try:
    _QUERY_GUARD_TEMPERATURE = float(_QUERY_GUARD_TEMPERATURE_RAW)
except Exception:
    _QUERY_GUARD_TEMPERATURE = 0.0

_QUERY_GUARD_FAIL_OPEN = env_util.env_flag("ACTIONS_QUERY_GUARD_FAIL_OPEN", default=False)

_QUERY_GUARD_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            """
You are a triage assistant for a clinical analytics visualization system.
Decide whether the user's request can proceed, requires clarification, or should be rejected.

Return strict JSON only with this schema:
{{
  "decision": "proceed" | "clarify" | "reject",
  "reason": "short_snake_case_reason",
  "message": string | null,
  "clarification_type": string | null,
  "clarification_options": string[] | null
}}

Rules:
- If enough information exists for a reasonable visualization plan, return decision="proceed" and message=null.
- If unclear or ambiguous, return decision="clarify" and ask exactly one concise clarification question in message.
- If request is out of scope for visualization flow, return decision="reject" with short actionable message.
- clarification_type should name what is missing/ambiguous (e.g. metric, chart_type, time_scope, time_range, grouping_dimension).
- clarification_options should contain concrete options only when natural; otherwise null.
- Prefer using supplied entities to avoid unnecessary clarification.
- Do not include markdown, prose, or code fences.
            """.strip(),
        ),
        (
            "user",
            "USER_LANGUAGE: {language}\nUSER_QUESTION: {question}\nENTITIES_JSON: {entities_json}\nVALID_METRIC_CANDIDATES_JSON: {metric_candidates_json}",
        ),
    ]
)

_query_guard_llm: Optional[Any] = None
_query_guard_llm_lock = Lock()


def _get_query_guard_llm() -> Optional[Any]:
    global _query_guard_llm
    if _query_guard_llm is not None:
        return _query_guard_llm
    with _query_guard_llm_lock:
        if _query_guard_llm is not None:
            return _query_guard_llm
        try:
            _query_guard_llm = create_chat_llm(temperature=_QUERY_GUARD_TEMPERATURE)
        except Exception:
            logger.exception("Failed to initialize query-guard LLM")
            _query_guard_llm = None
    return _query_guard_llm


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        content_items = cast(List[Any], content)
        for item in content_items:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                dict_item = cast(Dict[str, Any], item)
                text_any = dict_item.get("text")
                if text_any is not None:
                    chunks.append(str(text_any))
        if chunks:
            return "\n".join(chunks)

    return str(response)


def _metric_candidates(question: str, limit: int = 8) -> List[str]:
    normalized = ssot_loader.normalize_metric_text_key(question)
    if not normalized:
        return []

    lookup = ssot_loader.get_metric_text_lookup()
    if normalized in lookup:
        match = str(lookup[normalized]).strip()
        return [match] if match else []

    matches: List[str] = []
    for key, value in lookup.items():
        if normalized in key or key in normalized:
            match = str(value).strip()
            if match and match not in matches:
                matches.append(match)
        if len(matches) >= limit:
            break
    return matches


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    if not (candidate.startswith("{") and candidate.endswith("}")):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output does not contain JSON object")
        candidate = candidate[start : end + 1]

    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be object")
    return cast(Dict[str, Any], parsed)


def _coerce_decision(raw: Dict[str, Any]) -> QueryDecision:
    decision_any = raw.get("decision")
    decision = str(decision_any).strip().lower() if decision_any is not None else ""
    if decision not in {"proceed", "clarify", "reject"}:
        raise ValueError("Invalid decision from model")

    reason_any = raw.get("reason")
    reason = str(reason_any).strip() if reason_any is not None else ""
    if not reason:
        reason = "llm_query_guard"

    message_any = raw.get("message")
    message: Optional[str]
    if isinstance(message_any, str):
        message = message_any.strip() or None
    else:
        message = None

    result: QueryDecision = {
        "decision": cast(Literal["proceed", "clarify", "reject"], decision),
        "reason": reason,
        "message": message,
    }

    clarification_type_any = raw.get("clarification_type")
    if isinstance(clarification_type_any, str) and clarification_type_any.strip():
        result["clarification_type"] = clarification_type_any.strip()

    options_any = raw.get("clarification_options")
    if isinstance(options_any, list):
        options: List[str] = []
        for item in cast(List[Any], options_any):
            if isinstance(item, str) and item.strip():
                options.append(item.strip())
        if options:
            result["clarification_options"] = options

    if result["decision"] in {"clarify", "reject"} and not result["message"]:
        result["message"] = "I need a bit more detail before I can continue."
    if result["decision"] == "proceed":
        result["message"] = None

    return result


def _invoke_query_guard(chain: Any, payload: Dict[str, Any]) -> QueryDecision:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(bind_current_context(chain.invoke), payload)
        try:
            response = future.result(timeout=_QUERY_GUARD_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Query guard timed out after {_QUERY_GUARD_TIMEOUT_SECONDS:.1f}s") from exc

    text = _extract_text(response)
    parsed = _extract_json_object(text)
    return _coerce_decision(parsed)


def evaluate_visualization_query(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str] = None,
) -> QueryDecision:
    if not _ENABLE_QUERY_GUARD:
        return {"decision": "proceed", "reason": "guard_disabled", "message": None}

    guard_llm = _get_query_guard_llm()
    if guard_llm is None:
        if _QUERY_GUARD_FAIL_OPEN:
            return {"decision": "proceed", "reason": "guard_llm_unavailable", "message": None}
        return {
            "decision": "clarify",
            "reason": "guard_llm_unavailable",
            "message": "I need a bit more detail before I can continue.",
        }

    chain = _QUERY_GUARD_PROMPT | guard_llm
    payload = {
        "language": (language or "en").strip() or "en",
        "question": question or "",
        "entities_json": json.dumps(entities or {}, ensure_ascii=False),
        "metric_candidates_json": json.dumps(_metric_candidates(question or ""), ensure_ascii=False),
    }

    try:
        return _invoke_query_guard(chain, payload)
    except Exception:
        logger.exception("LLM query guard failed")
        if _QUERY_GUARD_FAIL_OPEN:
            return {"decision": "proceed", "reason": "guard_llm_failed", "message": None}
        return {
            "decision": "clarify",
            "reason": "guard_llm_failed",
            "message": "I need a bit more detail before I can continue.",
        }


def extract_entities_from_latest_message(latest_message: Dict[str, Any]) -> Dict[str, Any]:
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


def _strip_text_fields(value: Any) -> Any:
    """Drop user-facing free-text fields to prevent LLM prose from reaching clients."""

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str) and key in {"title", "description"}:
                continue
            out[key] = _strip_text_fields(child)
        return out
    if isinstance(value, list):
        return [_strip_text_fields(item) for item in value]
    return value


def serialize_plan_for_frontend(plan: Any) -> Dict[str, Any]:
    """Serialize planner output for frontend consumption without mutating it."""

    if hasattr(plan, "model_dump") and callable(getattr(plan, "model_dump")):
        payload_any = plan.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif isinstance(plan, dict):
        payload_any = dict(plan)
    else:
        return {}

    if not isinstance(payload_any, dict):
        return {}

    sanitized_any = _strip_text_fields(payload_any)
    return cast(Dict[str, Any], sanitized_any) if isinstance(sanitized_any, dict) else {}


def format_execution_summary(
    summary: Dict[str, Any] | Any,
    show_normalization: bool = True,
    planner_diagnostics: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
) -> str:
    def t(key: str, default: str, params: Optional[Dict[str, Any]] = None) -> str:
        return translate(key, language=language, params=params, default=default)

    if hasattr(summary, "model_dump") and callable(getattr(summary, "model_dump")):
        summary = cast(Dict[str, Any], summary.model_dump())
    elif not isinstance(summary, dict):
        return t("action.summary.complete", "✅ Visualization generation complete.")

    estimated = summary.get("estimated_queries")
    actual = summary.get("actual_queries")
    chart_count = summary.get("chart_count")
    trace_id = summary.get("trace_id")
    normalization_any = summary.get("normalization")
    normalization = cast(Dict[str, Any], normalization_any) if isinstance(normalization_any, dict) else None
    batches_any = summary.get("batches")
    batches: List[Any] = cast(List[Any], batches_any) if isinstance(batches_any, list) else []

    lines: List[str] = [t("action.summary.complete", "✅ Visualization generation complete.")]

    if isinstance(trace_id, str) and trace_id.strip():
        lines.append(t("action.summary.trace_id", "Trace ID: {trace_id}", {"trace_id": trace_id.strip()}))

    if isinstance(chart_count, int):
        if chart_count == 1:
            lines.append(t("action.summary.plan_produced_one_chart", "Plan produced 1 chart."))
        else:
            lines.append(t("action.summary.plan_produced_many_charts", "Plan produced {chart_count} charts.", {"chart_count": chart_count}))

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
            lines.append(t("action.summary.planner_cache_hit", "Planner cache: hit (reused a previously generated plan)."))
        elif cache_hit is False:
            lines.append(t("action.summary.planner_cache_miss", "Planner cache: miss (generated a fresh plan)."))

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
            lines.append(t("action.summary.queried_once", "I queried the analytics service once."))
        else:
            lines.append(t("action.summary.queried_many", "I queried the analytics service {actual} times.", {"actual": actual}))

        if isinstance(estimated, int) and estimated != actual:
            lines.append(t("action.summary.planner_estimate", "Planner estimate was {estimated} request(s).", {"estimated": estimated}))

    if show_normalization and normalization is not None:
        charts_in = normalization.get("charts_in")
        charts_out = normalization.get("charts_out")
        dropped_charts = normalization.get("dropped_empty_charts")
        metrics_in = normalization.get("metrics_in")
        metrics_out = normalization.get("metrics_out")
        dropped_metrics = normalization.get("dropped_empty_metrics")
        metric_code_norm = normalization.get("normalized_metric_codes")
        chart_type_norm = normalization.get("normalized_chart_types")
        deduped_groupby = normalization.get("deduped_groupby_entries")
        normalized_groupby_fields = normalization.get("normalized_canonical_groupby_fields")
        dropped_groupby_fields = normalization.get("dropped_invalid_groupby_fields")
        chart_type_fallback = normalization.get("fallback_chart_type_count")
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
            if isinstance(metrics_in, int) and isinstance(metrics_out, int) and metrics_in != metrics_out:
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
            if isinstance(chart_type_norm, int) and chart_type_norm > 0:
                details.append(
                    t(
                        "action.summary.detail_normalized_chart_types",
                        "normalized {chart_type_norm} chart type(s)",
                        {"chart_type_norm": chart_type_norm},
                    )
                )
            if isinstance(chart_type_fallback, int) and chart_type_fallback > 0:
                details.append(
                    t(
                        "action.summary.detail_applied_chart_fallback",
                        "applied {chart_type_fallback} chart type fallback(s)",
                        {"chart_type_fallback": chart_type_fallback},
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
            if isinstance(normalized_groupby_fields, int) and normalized_groupby_fields > 0:
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
                lines.append(" - " + t("action.summary.no_structural_changes", "No structural changes were needed."))

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
            filters_list: List[Any] = cast(List[Any], filters_any) if isinstance(filters_any, list) else []
            filter_names = [str(x).replace("GroupBy", "").strip() for x in filters_list]

            parts: List[str] = [f"{idx})"]
            if isinstance(query_count, int):
                parts.append(t("action.summary.request_count", "{query_count} request(s)", {"query_count": query_count}))
            if isinstance(groupby, str) and groupby:
                parts.append(t("action.summary.grouped_by", "grouped by {groupby}", {"groupby": groupby}))
            if isinstance(periods, int) and periods > 0:
                parts.append(t("action.summary.across_periods", "across {periods} time period(s)", {"periods": periods}))
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
