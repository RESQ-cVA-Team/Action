from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Any, Callable, Dict, List, Literal, Optional, cast

from langchain_core.prompts import ChatPromptTemplate

from src.domain.langchain.schema import TIME_INTERVALS, AnalysisPlan, AndFilter, ChartType, DataOriginSpec, DateFilter, MetricSpec, OriginScopeSpec, SplitSpec, StatisticalTestSpec
from src.planners.langchain.llm_factory import create_chat_llm
from src.planners.langchain.pipeline import (
    _PLANNER_REQUEST_TIMEOUT_SECONDS,
    _STAT_TEST_KEYWORDS,
    generate_analysis_plan,
)
from src.planners.langchain.prompt_loader import load_prompt_text
from src.shared import ssot_loader
from src.util import env as env_util
from src.util.logging_utils import bind_current_context, log_context

logger = logging.getLogger(__name__)


OutcomeDecision = Literal["proceed", "clarify", "reject"]


@dataclass
class VisualizationRequestOutcome:
    decision: OutcomeDecision
    reason: str
    message: Optional[str] = None
    clarification_type: Optional[str] = None
    clarification_options: List[str] = field(default_factory=lambda: cast(List[str], []))
    missing_fields: List[str] = field(default_factory=lambda: cast(List[str], []))
    plan: Optional[AnalysisPlan] = None


_ORCHESTRATOR_ENABLED = env_util.env_flag("ACTIONS_LLM_REQUEST_ORCHESTRATOR_ENABLED", default=True)
# This call shares the same underlying model/latency profile as plan
# generation (see _PLANNER_REQUEST_TIMEOUT_SECONDS in pipeline.py), so a 10s
# cap was too tight for the same reason -- it just failed more quietly here,
# since a timeout falls through to the coarser fallback-to-plan path instead
# of a loud per-attempt error.
_ORCHESTRATOR_TIMEOUT_RAW = env_util.get_env("ACTIONS_LLM_REQUEST_ORCHESTRATOR_TIMEOUT_SECONDS", default="20") or "20"
_orchestrator_timeout_value = 20.0
try:
    _orchestrator_timeout_value = max(1.0, float(_ORCHESTRATOR_TIMEOUT_RAW))
except Exception:
    _orchestrator_timeout_value = 20.0
_ORCHESTRATOR_TIMEOUT_SECONDS = _orchestrator_timeout_value

# Single source of truth for plan-generation retry count. Previously duplicated
# as a separate constant in visualization_action.py, which is how the timeout
# below drifted out of sync with it: the retry count only lived where the
# retries actually got requested, not where the outer timeout that has to
# survive them lived.
_PLANNER_MAX_RETRIES = 2

# Plan generation involves a reasoning step and can legitimately take longer than the
# fast triage/decision call above. generate_analysis_plan (pipeline.py) retries up to
# _PLANNER_MAX_RETRIES times, each bounded by its own _PLANNER_REQUEST_TIMEOUT_SECONDS
# budget -- so this outer timeout must cover the full retry loop's worst case
# (attempts * per-attempt budget), not just one attempt. It used to only account for
# one, so on anything slow enough to need a retry, this outer wrapper killed the call
# before generate_analysis_plan's own retry ever got a chance to run.
#
# Since the plan-critique pass (pipeline.py's _critique_plan), each attempt can make
# TWO LLM calls -- the plan generation itself, then the critique review of it -- each
# independently bounded by _PLANNER_REQUEST_TIMEOUT_SECONDS. This formula previously
# still budgeted for one call per attempt, so a scenario whose plan legitimately needs
# every retry (critique keeps finding real, distinct issues -- observed live for the
# "10 line graphs" case) could exhaust this outer timeout entirely before the inner
# loop's own attempts ran out, turning a slow-but-eventually-correct plan into a hard
# failure instead.
_PLAN_GENERATION_TIMEOUT_RAW = env_util.get_env("ACTIONS_LLM_PLAN_GENERATION_TIMEOUT_SECONDS", default="") or ""
try:
    _plan_generation_timeout_value = max(1.0, float(_PLAN_GENERATION_TIMEOUT_RAW))
except Exception:
    _plan_generation_timeout_value = max(
        _ORCHESTRATOR_TIMEOUT_SECONDS,
        _PLANNER_REQUEST_TIMEOUT_SECONDS * (_PLANNER_MAX_RETRIES + 1) * 2 + 10.0,
    )
_PLAN_GENERATION_TIMEOUT_SECONDS = _plan_generation_timeout_value

_ORCHESTRATOR_TEMPERATURE_RAW = env_util.get_env("ACTIONS_LLM_REQUEST_ORCHESTRATOR_TEMPERATURE", default="0") or "0"
_orchestrator_temperature_value = 0.0
try:
    _orchestrator_temperature_value = float(_ORCHESTRATOR_TEMPERATURE_RAW)
except Exception:
    _orchestrator_temperature_value = 0.0
_ORCHESTRATOR_TEMPERATURE = _orchestrator_temperature_value

_ORCHESTRATOR_FAIL_OPEN = env_util.env_flag("ACTIONS_LLM_REQUEST_ORCHESTRATOR_FAIL_OPEN", default=False)

_TEMPORAL_MISSING_FIELDS = {
    "time",
    "time_scope",
    "time_range",
    "time_window",
    "time_period",
    "date",
    "date_range",
    "date_window",
    "date_period",
    "period",
    "timeframe",
    "time_frame",
    "window",
    "range",
}

_STAT_TEST_ENTITY_KEYS = {
    "statistical_test_type",
    "statistical_tests",
    "test_type",
    "test_types",
}

_STATISTICAL_COHORT_ENTITY_KEYS = {
    "provider_id",
    "group_id",
    "provider_name",
    "provider_group_name",
    "hospital_name",
    "hospital_scope_reference",
    "scope",
    "country_code",
    "country_average",
    "sex",
    "stroke_type",
    "age",
    "mine",
}

_PROVIDER_GROUP_HINTS = (
    "provider group",
    "provider-group",
    "cohort a is group",
    "cohort b is group",
)

_PROVIDER_HINTS = (
    "provider ",
    "cohort a is provider",
    "cohort b is provider",
)

_DECISION_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore[attr-defined]
    [
        ("system", load_prompt_text("decision_system")),
        (
            "user",
            "USER_LANGUAGE: {language}\nUSER_QUESTION: {question}\nCONVERSATION_HISTORY_JSON: {conversation_history_json}\nENTITIES_JSON: {entities_json}\nVALID_METRIC_CANDIDATES_JSON: {metric_candidates_json}\nVALID_CHART_TYPES_JSON: {chart_types_json}",
        ),
    ]
)

_llm: Optional[Any] = None
_llm_lock = Lock()


def _get_llm() -> Optional[Any]:
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is not None:
            return _llm
        try:
            _llm = create_chat_llm(temperature=_ORCHESTRATOR_TEMPERATURE)
        except Exception:
            logger.exception(
                "Failed to initialize LLM request orchestrator",
                extra={
                    "log_context": {
                        "event": "orchestrator.llm_init.failed",
                        "operation": "_get_llm",
                        "outcome": "failure",
                    }
                },
            )
            _llm = None
    return _llm


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in cast(List[Any], content):
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                maybe_text = cast(Dict[str, Any], item).get("text")
                if maybe_text is not None:
                    chunks.append(str(maybe_text))
        if chunks:
            return "\n".join(chunks)

    return str(response)


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
        raise ValueError("Model output JSON must be an object")
    return cast(Dict[str, Any], parsed)


def _invoke_chain(chain: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(bind_current_context(chain.invoke), payload)
        try:
            response = future.result(timeout=_ORCHESTRATOR_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Orchestrator timed out after {_ORCHESTRATOR_TIMEOUT_SECONDS:.1f}s") from exc

    return _extract_json_object(_extract_text(response))


def _generate_plan_with_timeout(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str] = None,
    max_retries: int = 2,
    trace_id: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> AnalysisPlan:
    """Wrap generate_analysis_plan with timeout protection to prevent indefinite hangs."""

    def _call_plan_gen() -> AnalysisPlan:
        result = generate_analysis_plan(
            question=question,
            entities=entities,
            language=language,
            max_retries=max_retries,
            debug=False,
            trace_id=trace_id,
            progress_cb=progress_cb,
        )
        if not isinstance(result, AnalysisPlan):
            raise RuntimeError(f"Expected AnalysisPlan but got {type(result).__name__}")
        return cast(AnalysisPlan, result)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_plan_gen)
        try:
            return future.result(timeout=_PLAN_GENERATION_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Plan generation timed out after {_PLAN_GENERATION_TIMEOUT_SECONDS:.1f}s for question: {question[:50]}") from exc


def _metric_candidates(question: str, limit: int = 8) -> List[str]:
    normalized = ssot_loader.normalize_metric_text_key(question)
    if not normalized:
        return []

    lookup = ssot_loader.get_metric_text_lookup()
    if normalized in lookup:
        entry = lookup[normalized]
        canonical = entry.get("canonical")
        if isinstance(canonical, str) and canonical.strip():
            return [canonical.strip()]
        return [str(entry)]

    out: List[str] = []
    for key, entry in lookup.items():
        if normalized not in key and key not in normalized:
            continue
        canonical = entry.get("canonical")
        if isinstance(canonical, str) and canonical.strip() and canonical not in out:
            out.append(canonical.strip())
        elif str(entry).strip() and str(entry).strip() not in out:
            out.append(str(entry).strip())
        if len(out) >= limit:
            break
    return out


def _chart_types() -> List[str]:
    try:
        return [str(member) for member in ChartType]
    except Exception:
        return ["LINE", "BAR", "AREA", "SCATTER", "HISTOGRAM", "BOX", "VIOLIN"]


def _coerce_missing_fields(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in cast(List[Any], raw):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _coerce_options(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in cast(List[Any], raw):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _is_missing_chart_type_only(missing_fields: List[str]) -> bool:
    normalized = [field.strip().lower() for field in missing_fields if field and field.strip()]
    return bool(normalized) and all(field == "chart_type" for field in normalized)


# Required fields per decision_system.txt's own rules ("required fields are
# metric and chart_type" / "...metric and statistical_test_type") where
# "present in ENTITIES_JSON" is an unambiguous, objective presence check --
# unlike e.g. "two distinct statistical cohorts", which needs judgment beyond
# "some value exists" and is deliberately excluded here.
_SELF_VERIFIABLE_REQUIRED_FIELDS = {"metric", "chart_type", "statistical_test_type"}

# Any of these being present in ENTITIES_JSON is objective proof a request is
# in-scope, not just a present "metric": Rasa only ever extracts these within
# this domain's own SSOT-defined entity types, and this action only runs for
# intents Rasa already classified as visualization-related in the first
# place. Deliberately excludes generic/structural entities (limit, offset,
# sort) that aren't themselves clinical-domain signals. Observed live: "show
# only patients older than 60" (a bare age filter, no metric yet) got
# rejected as out_of_scope with the message "This request is not related to
# clinical stroke-care analytics" -- objectively false, it's a well-formed
# partial clinical request missing a metric, which is a clarify.
_SCOPE_PROVING_ENTITY_KEYS = {
    "metric",
    "chart_type",
    "statistical_test_type",
    "group_by",
    "stroke_type",
    "country_code",
    "sex",
    "age",
    "nihss",
    "hospital_name",
    "provider_group_name",
    "boolean_type",
    "date",
    "operator_type",
    "hospital_scope_reference",
    "group_id",
}

# decision_system.txt's own closed reason taxonomy: each value unambiguously
# implies exactly one decision. Observed live (CVaLab): the decision stage
# occasionally puts a reason value straight into the "decision" field (e.g.
# {"decision": "ambiguous_request", ...} -- reproducible twice in a row for
# the same input, not one-off noise, so a same-prompt retry alone doesn't
# reliably recover it). Since each reason already tells us the decision the
# model meant, this recovers losslessly rather than failing outright.
_REASON_IMPLIES_DECISION: Dict[str, str] = {
    "all_required_fields_present": "proceed",
    "missing_required_fields": "clarify",
    "ambiguous_request": "clarify",
    "out_of_scope": "reject",
    "invalid_date_format": "reject",
    "insufficient_information": "reject",
}


def _entity_present(entities: Dict[str, Any], key: str) -> bool:
    value = entities.get(key)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(isinstance(item, str) and item.strip() for item in cast(List[Any], value))
    return value is not None and value is not False


def _drop_falsely_missing_fields(missing_fields: List[str], entities: Dict[str, Any]) -> List[str]:
    """Cross-check the decision stage's own missing_fields claim against the
    same ENTITIES_JSON it was given. Observed intermittently: the LLM claims
    "metric" (sometimes others) is missing on a terse follow-up turn even
    though it's plainly present in ENTITIES_JSON -- a self-contradiction
    against its own stated rule ("if required fields are present, return
    proceed"), not a real ambiguity. Scoped to _SELF_VERIFIABLE_REQUIRED_FIELDS
    so this only overrides objectively-false claims, never a field that
    genuinely needs semantic judgment to consider satisfied.
    """
    return [field for field in missing_fields if not (field.strip().lower() in _SELF_VERIFIABLE_REQUIRED_FIELDS and _entity_present(entities, field.strip().lower()))]


def _has_statistical_test_signal(question: str, entities: Dict[str, Any]) -> bool:
    for key in _STAT_TEST_ENTITY_KEYS:
        value = entities.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list):
            value_list = cast(List[Any], value)
            if any(isinstance(item, str) and item.strip() for item in value_list):
                return True

    question_norm = (question or "").strip().lower()
    if not question_norm:
        return False
    return any(keyword in question_norm for keyword in _STAT_TEST_KEYWORDS)


def _extract_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        out: List[str] = []
        for item in cast(List[Any], value):
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _extract_provider_group_ids(entities: Dict[str, Any]) -> List[int]:
    # Rasa (and the domain's own registered entity, filters.yml) extracts this
    # as "group_id", not "provider_group_id" -- the latter was never a real
    # entity key, so this always found zero IDs even when the request named
    # two provider groups explicitly.
    raw_values = _extract_string_list(entities.get("group_id"))
    out: List[int] = []
    for token in raw_values:
        for match in re.findall(r"\d+", token):
            try:
                group_id = int(match)
            except Exception:
                continue
            if group_id > 0 and group_id not in out:
                out.append(group_id)
    return out


def _extract_provider_ids(entities: Dict[str, Any]) -> List[int]:
    raw_values = _extract_string_list(entities.get("provider_id"))
    out: List[int] = []
    for token in raw_values:
        for match in re.findall(r"\d+", token):
            try:
                provider_id = int(match)
            except Exception:
                continue
            if provider_id > 0 and provider_id not in out:
                out.append(provider_id)
    return out


def _extract_metric_code(entities: Dict[str, Any]) -> Optional[str]:
    metrics = _extract_string_list(entities.get("metric"))
    if not metrics:
        return None
    return metrics[0].upper()


def _extract_date_bounds(entities: Dict[str, Any]) -> Optional[tuple[str, str]]:
    date_values = [d for d in _extract_string_list(entities.get("date")) if re.match(r"^\d{4}-\d{2}-\d{2}$", d)]
    if len(date_values) < 2:
        return None
    ordered = sorted(set(date_values))
    if len(ordered) < 2:
        return None
    return ordered[0], ordered[-1]


def _extract_statistical_cohort_tokens(entities: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []

    for key in _STATISTICAL_COHORT_ENTITY_KEYS:
        value = entities.get(key)
        if value is None:
            continue

        if key == "mine" and bool(value):
            tokens.append("mine")
            continue

        if isinstance(value, str):
            token = value.strip()
            if token:
                tokens.append(token)
            continue

        if isinstance(value, list):
            for item in cast(List[Any], value):
                if isinstance(item, str) and item.strip():
                    tokens.append(item.strip())

    normalized: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        marker = token.strip().lower()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        normalized.append(token.strip())
    return normalized


def _extract_quarter_tokens(entities: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for raw in _extract_string_list(entities.get("quarter")) + _extract_string_list(entities.get("date")):
        token = raw.strip().upper().replace("_", " ")
        token = re.sub(r"\s+", " ", token)
        if re.match(r"^Q[1-4](?:\s+\d{4})?$", token):
            tokens.append(token)

    unique: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(token)
    return unique


def _extract_iso_day_tokens(entities: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for raw in _extract_string_list(entities.get("date")):
        token = raw.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
            tokens.append(token)

    unique: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(token)
    return unique


def _extract_period_tokens(entities: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    month_names = (
        "JANUARY",
        "FEBRUARY",
        "MARCH",
        "APRIL",
        "MAY",
        "JUNE",
        "JULY",
        "AUGUST",
        "SEPTEMBER",
        "OCTOBER",
        "NOVEMBER",
        "DECEMBER",
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "SEPT",
        "OCT",
        "NOV",
        "DEC",
    )
    month_pattern = "(?:" + "|".join(month_names) + ")"

    for raw in _extract_string_list(entities.get("date")):
        token = raw.strip().upper().replace("_", " ")
        token = re.sub(r"\s+", " ", token)
        if not token:
            continue
        if re.match(r"^\d{4}$", token):
            tokens.append(token)
            continue
        if re.match(r"^\d{4}-\d{2}$", token):
            tokens.append(token)
            continue
        if re.match(rf"^{month_pattern}\s+\d{{4}}$", token):
            tokens.append(token)
            continue
        if token in {
            "THIS YEAR",
            "LAST YEAR",
            "PREVIOUS YEAR",
            "THIS QUARTER",
            "LAST QUARTER",
            "PREVIOUS QUARTER",
            "THIS MONTH",
            "LAST MONTH",
            "PREVIOUS MONTH",
        }:
            tokens.append(token)

    unique: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(token)
    return unique


def _has_explicit_temporal_comparison(entities: Dict[str, Any]) -> bool:
    if len(_extract_quarter_tokens(entities)) >= 2:
        return True

    if len(_extract_period_tokens(entities)) >= 2:
        return True

    # Two temporal cohorts may come as two explicit date ranges with four bounds.
    if len(_extract_iso_day_tokens(entities)) >= 4:
        return True

    return False


def _question_mentions_provider_group(question: str) -> bool:
    low = (question or "").strip().lower()
    if not low:
        return False
    return any(hint in low for hint in _PROVIDER_GROUP_HINTS)


def _question_mentions_provider(question: str) -> bool:
    low = (question or "").strip().lower()
    if not low:
        return False
    if _question_mentions_provider_group(low):
        return False
    return any(hint in low for hint in _PROVIDER_HINTS)


# Test names other than Mann-Whitney that SSOT/StatisticalTestType.yml has no entry
# for at all -- not a partial gap, these tests have zero execution support anywhere
# in the pipeline. Listed here purely so an explicit ask for one of them can be
# rejected immediately with an accurate name, instead of silently reaching the
# LLM planner, which has no valid schema representation for them either and
# only fails after several retries with a generic, unhelpful message.
_KNOWN_UNSUPPORTED_STAT_TEST_NAMES = (
    "wilcoxon",
    "student's t",
    "paired t-test",
    "paired t test",
    "t-test",
    "t test",
    "chi-square",
    "chi square",
    "anova",
    "kruskal-wallis",
    "kruskal wallis",
    "spearman",
    "pearson",
    "fisher's exact",
    "fisher exact",
    "kolmogorov",
)


def _detect_unsupported_statistical_test(question: str) -> Optional[str]:
    question_norm = (question or "").strip().lower()
    if not question_norm:
        return None
    if "mann-whitney" in question_norm or "mann whitney" in question_norm:
        return None
    for name in _KNOWN_UNSUPPORTED_STAT_TEST_NAMES:
        if name in question_norm:
            return name
    return None


def _validate_statistical_test_support(question: str) -> Optional[VisualizationRequestOutcome]:
    unsupported_name = _detect_unsupported_statistical_test(question)
    if unsupported_name is None:
        return None

    return VisualizationRequestOutcome(
        decision="reject",
        reason="unsupported_statistical_test",
        message=(f"I can't run a {unsupported_name} test yet -- Mann-Whitney U is the only statistical test currently supported. Ask me to compare two cohorts with a Mann-Whitney U test instead."),
        clarification_type=None,
        clarification_options=[],
        missing_fields=[],
    )


# Raw group_by entity values the NLU lookup table recognises that map directly onto a
# SplitSpec split kind rather than a time grain or a GroupByType.yml canonical field.
_DIRECT_SPLIT_KIND_GROUP_BY_TOKENS = {"STROKE_TYPE", "SEX_TYPE", "SEX", "AGE", "NIHSS"}
_DIRECT_SPLIT_KIND_BY_CANONICAL_FIELD = {
    "STROKE_TYPE": "STROKE_TYPE",
    "SEX_TYPE": "SEX",
    "SEX": "SEX",
    "AGE": "AGE",
    "NIHSS": "NIHSS",
}

_REDUNDANT_SELF_SPLIT_METRICS_BY_KIND = {
    "STROKE_TYPE": {"STROKE_TYPE"},
    "SEX": {"SEX_TYPE"},
}


def _normalize_plan_semantic_splits(plan: AnalysisPlan) -> AnalysisPlan:
    """Normalize planner split semantics into compiler-supported forms.

    The retrieval compiler expects direct split kinds (STROKE_TYPE/SEX/AGE/NIHSS)
    for category fan-out. Some generated plans encode these as CANONICAL splits,
    which are valid schema but can fail compilation at runtime.
    """
    charts = list(plan.charts or [])
    if not charts:
        return plan

    for chart in charts:
        semantics = chart.semantics
        if semantics is None or not semantics.splits:
            continue

        metric_codes = {(metric.metric or "").strip().upper() for metric in chart.metrics or [] if isinstance(metric.metric, str) and metric.metric.strip()}

        normalized_splits: List[SplitSpec] = []
        changed = False
        for split in semantics.splits:
            kind = (split.kind or "").strip().upper()

            # If the metric itself is already an enum distribution axis, splitting by
            # that exact same axis creates a self-split that distorts chart shape.
            if kind in _REDUNDANT_SELF_SPLIT_METRICS_BY_KIND:
                redundant_metrics = _REDUNDANT_SELF_SPLIT_METRICS_BY_KIND[kind]
                if any(metric in metric_codes for metric in redundant_metrics):
                    changed = True
                    continue

            if kind != "CANONICAL":
                normalized_splits.append(split)
                continue

            field = (split.field or "").strip().upper()
            mapped_kind = _DIRECT_SPLIT_KIND_BY_CANONICAL_FIELD.get(field)
            if mapped_kind is None and not field and "STROKE_TYPE" in metric_codes:
                mapped_kind = "STROKE_TYPE"

            if mapped_kind is None:
                normalized_splits.append(split)
                continue

            redundant_metrics = _REDUNDANT_SELF_SPLIT_METRICS_BY_KIND.get(mapped_kind)
            if redundant_metrics and any(metric in metric_codes for metric in redundant_metrics):
                changed = True
                continue

            replacement = SplitSpec(
                kind=mapped_kind,
                categories=list(split.categories) if split.categories is not None else None,
                buckets=list(split.buckets) if split.buckets is not None else None,
            )
            normalized_splits.append(replacement)
            changed = True

        if changed:
            semantics.splits = normalized_splits or None

    return plan


@lru_cache(maxsize=1)
def _risk_factor_filter_terms() -> Dict[str, str]:
    """Map lowercase risk-factor keywords -> a human label, sourced from the SSOT
    RISK_FACTORS_TYPE enum (MetricType.yml) rather than a hand-maintained list.

    RISK_FACTORS_TYPE is real and queryable as a metric, but confirmed (via the
    real GraphQL API's schema.gql: EnumCaseFilter only covers
    strokeType/sexType/arrivalMode) to have zero filter-input support -- a
    permanent third-party backend limitation, not an Action-side schema gap.
    This powers a fast, accurate rejection when a risk factor is mentioned as a
    filter/cohort qualifier on some other metric, instead of letting the LLM
    planner hallucinate an invalid filter and fail slowly with a generic message.
    """
    terms: Dict[str, str] = {}
    try:
        items = ssot_loader.get_ssot_items("MetricType.yml")
    except Exception:
        return terms

    risk_factors = next((item for item in items if item.get("canonical") == "RISK_FACTORS_TYPE"), None)
    if not risk_factors:
        return terms

    for entry in risk_factors.get("Enum") or []:
        key = str(entry.get("key") or "").strip()
        if not key:
            continue
        core = key
        for prefix in ("risk_", "before_onset_"):
            if core.startswith(prefix):
                core = core[len(prefix) :]
        label = core.replace("_", " ").strip()
        if label and len(label) >= 3:
            terms.setdefault(label.lower(), label)
        for synonym in _flatten_synonym_block(entry.get("synonyms")):
            cleaned = synonym.strip().lower()
            for prefix in ("risk ", "history of "):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :]
            if cleaned and len(cleaned) >= 3:
                terms.setdefault(cleaned, label or cleaned)

    return terms


def _flatten_synonym_block(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, dict):
        for localized in cast(Dict[str, Any], value).values():
            if isinstance(localized, list):
                out.extend(str(item) for item in cast(List[Any], localized) if isinstance(item, str) and item.strip())
    return out


def _detect_unsupported_risk_factor_filter(question: str, entities: Dict[str, Any]) -> Optional[str]:
    question_norm = (question or "").strip().lower()
    if not question_norm:
        return None

    # If the user is actually asking about risk factors as the metric itself
    # (e.g. "show risk factor distribution"), that's fully supported -- only
    # flag when a *different* metric is requested and a risk factor shows up
    # as an incidental qualifier ("...for patients with hypertension").
    metric_values = [m.upper() for m in _extract_string_list(entities.get("metric"))]
    if "RISK_FACTORS_TYPE" in metric_values:
        return None

    # Valid standalone metrics can legitimately contain the VTE abbreviation in
    # their canonical names (e.g. VTE_INTERVENTION_AIS). These must not be
    # mistaken for a risk-factor filter phrase just because the word "vte" is
    # present in the question.
    for metric in metric_values:
        if metric.startswith("VTE_"):
            return None

    for term, label in _risk_factor_filter_terms().items():
        # Leading boundary only (not trailing), so a plural like "smokers"
        # still matches the SSOT term "smoker". Best-effort, not exhaustive --
        # this is a fast path; anything it misses still falls through to the
        # slower LLM plan-generation path, which fails safely via
        # _build_empty_plan_clarification.
        if re.search(r"\b" + re.escape(term), question_norm):
            return label
    return None


def _validate_risk_factor_filter_support(question: str, entities: Dict[str, Any]) -> Optional[VisualizationRequestOutcome]:
    label = _detect_unsupported_risk_factor_filter(question, entities)
    if label is None:
        return None

    return VisualizationRequestOutcome(
        decision="reject",
        reason="unsupported_risk_factor_filter",
        message=(
            f"I can't filter by {label} yet -- risk factors like this are queryable as their own "
            "metric, but the underlying data API has no way to use them as a filter on a different "
            "metric. Ask me to chart risk factors directly instead, without combining it with another metric."
        ),
        clarification_type=None,
        clarification_options=[],
        missing_fields=[],
    )


def _validate_group_by_support(question: str, entities: Dict[str, Any]) -> Optional[VisualizationRequestOutcome]:
    # Statistical-test plans compare cohorts via OriginScope/DataOrigin on each
    # MetricSpec (see _build_deterministic_statistical_plan) -- they never use
    # SplitSpec/group_by at all. NLU can still attach a group_by=HOSPITAL entity
    # to "...against X hospital using a Mann-Whitney test" phrasing even though
    # it's irrelevant there, so this check only applies to chart requests.
    #
    # _has_statistical_test_signal's keyword scan matches broadly on words like
    # "compare", which also appears in plain chart requests ("compare my DTN
    # per quarter with X hospital using a line chart") -- too loose to gate on
    # here. A present chart_type entity is the precise signal: it means the
    # request is unambiguously routed toward a chart, so group_by must still be
    # validated even if the wording happens to also trip the stat-test scan.
    has_chart_type = bool(_extract_string_list(entities.get("chart_type")))
    if not has_chart_type and _has_statistical_test_signal(question, entities):
        return None

    raw_values = _extract_string_list(entities.get("group_by"))
    if not raw_values:
        return None

    unsupported: List[str] = []
    for raw in raw_values:
        token = raw.strip().upper()
        if not token:
            continue
        if token in TIME_INTERVALS:
            continue
        if token in _DIRECT_SPLIT_KIND_GROUP_BY_TOKENS:
            continue
        # HOSPITAL is intentionally not routed through SplitSpec -- there is no
        # server-side or client-side split kind for it. It's handled entirely
        # at the planning layer instead: the plan-generation prompt/examples
        # teach the LLM to express hospital comparison as separate MetricSpec
        # entries with their own originScope (see
        # example_dtn_my_hospital_vs_named_hospital_quarterly_line), which
        # resolve_plan_metric_origins already executes generically for both
        # charts and statistical tests. Nothing to validate here.
        if token == "HOSPITAL":
            continue
        if ssot_loader.resolve_groupby_canonical(token) is not None:
            continue
        unsupported.append(raw.strip())

    if not unsupported:
        return None

    names = ", ".join(sorted(set(unsupported)))
    return VisualizationRequestOutcome(
        decision="clarify",
        reason="unsupported_group_by_dimension",
        message=(
            f"I can't group or split a chart by {names} yet. I can group by stroke type, sex, time period (month/quarter/year), EMS prenotification, first contact place, IVT department, or INR mode."
        ),
        clarification_type="analysis_plan",
        clarification_options=[],
        missing_fields=[],
    )


def _validate_statistical_entity_readiness(question: str, entities: Dict[str, Any]) -> Optional[VisualizationRequestOutcome]:
    if not _has_statistical_test_signal(question, entities):
        return None

    provider_group_ids = _extract_provider_group_ids(entities)
    provider_ids = _extract_provider_ids(entities)
    mentions_provider_group = _question_mentions_provider_group(question)
    mentions_provider = _question_mentions_provider(question)

    if mentions_provider_group and len(provider_group_ids) < 2:
        return VisualizationRequestOutcome(
            decision="clarify",
            reason="missing_provider_group_cohorts",
            message=("Your request mentions provider groups, but I do not have two provider-group cohorts. Please provide provider group A and provider group B explicitly."),
            clarification_type="analysis_plan",
            clarification_options=[],
            missing_fields=["provider_group_id"],
        )

    if mentions_provider and len(provider_ids) < 2:
        return VisualizationRequestOutcome(
            decision="clarify",
            reason="missing_provider_cohorts",
            message=("Your request mentions providers, but I do not have two provider cohorts. Please provide provider A and provider B explicitly."),
            clarification_type="analysis_plan",
            clarification_options=[],
            missing_fields=["provider_id"],
        )

    cohort_tokens = _extract_statistical_cohort_tokens(entities)
    if _has_explicit_temporal_comparison(entities):
        return None

    if len(cohort_tokens) < 2:
        return VisualizationRequestOutcome(
            decision="clarify",
            reason="missing_statistical_cohorts",
            message=("I can run this statistical test only with two explicit cohorts. Please provide cohort A and cohort B (for example two providers or provider groups)."),
            clarification_type="analysis_plan",
            clarification_options=[],
            missing_fields=["statistical_cohorts"],
        )

    return None


def _coerce_origin_scope(scope_type: str, value: Any = None, label: Optional[str] = None, country_code: Optional[str] = None) -> OriginScopeSpec:
    payload: Dict[str, Any] = {"scopeType": scope_type}
    if value is not None:
        payload["value"] = value
    if label is not None:
        payload["label"] = label
    if country_code is not None:
        payload["countryCode"] = country_code
    return OriginScopeSpec.model_validate(payload)


def _extract_semantic_scopes(entities: Dict[str, Any]) -> List[OriginScopeSpec]:
    scopes: List[OriginScopeSpec] = []

    if entities.get("mine"):
        scopes.append(_coerce_origin_scope("mine", label="mine"))

    scope_refs = _extract_string_list(entities.get("hospital_scope_reference")) + _extract_string_list(entities.get("scope"))
    for scope_ref in scope_refs:
        normalized = scope_ref.strip().lower()
        if normalized in {"mine", "my hospital", "our hospital", "ours"}:
            scopes.append(_coerce_origin_scope("mine", label="mine"))
        elif normalized in {"all", "all hospitals", "all accessible"}:
            scopes.append(_coerce_origin_scope("all_accessible", label="all accessible"))

    hospital_names = _extract_string_list(entities.get("hospital_name"))
    for hospital_name in hospital_names:
        scopes.append(_coerce_origin_scope("provider_name", value=hospital_name, label=hospital_name))

    provider_names = _extract_string_list(entities.get("provider_name"))
    for provider_name in provider_names:
        scopes.append(_coerce_origin_scope("provider_name", value=provider_name, label=provider_name))

    provider_group_names = _extract_string_list(entities.get("provider_group_name"))
    for provider_group_name in provider_group_names:
        scopes.append(_coerce_origin_scope("provider_group_name", value=provider_group_name, label=provider_group_name))

    country_codes = _extract_string_list(entities.get("country_code"))
    for country_code in country_codes:
        scopes.append(_coerce_origin_scope("country_code", value=country_code, country_code=country_code, label=country_code))

    if entities.get("country_average"):
        country_average_values = _extract_string_list(entities.get("country_average"))
        if country_average_values:
            for country_code in country_average_values:
                scopes.append(_coerce_origin_scope("country_average", value=country_code, country_code=country_code, label=country_code))
        else:
            scopes.append(_coerce_origin_scope("country_average", label="country average"))

    return scopes


def _build_deterministic_statistical_plan(
    question: str,
    entities: Dict[str, Any],
) -> Optional[AnalysisPlan]:
    if not _has_statistical_test_signal(question, entities):
        return None

    metric = _extract_metric_code(entities)
    provider_group_ids = _extract_provider_group_ids(entities)
    provider_ids = _extract_provider_ids(entities)
    semantic_scopes = _extract_semantic_scopes(entities)
    mentions_provider_group = _question_mentions_provider_group(question)
    mentions_provider = _question_mentions_provider(question)
    bounds = _extract_date_bounds(entities)

    if metric is None or bounds is None:
        return None

    start_date, end_date = bounds
    metrics: List[MetricSpec]

    if len(semantic_scopes) >= 2:
        metrics = [
            MetricSpec(metric=metric, originScope=semantic_scopes[0]),
            MetricSpec(metric=metric, originScope=semantic_scopes[1]),
        ]
    elif mentions_provider_group and len(provider_group_ids) >= 2:
        cohort_a = provider_group_ids[0]
        cohort_b = provider_group_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_b])),
        ]
    elif mentions_provider and len(provider_ids) >= 2:
        cohort_a = provider_ids[0]
        cohort_b = provider_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_b])),
        ]
    elif len(provider_group_ids) >= 2:
        cohort_a = provider_group_ids[0]
        cohort_b = provider_group_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerGroupId=[cohort_b])),
        ]
    elif len(provider_ids) >= 2:
        cohort_a = provider_ids[0]
        cohort_b = provider_ids[1]
        metrics = [
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_a])),
            MetricSpec(metric=metric, dataOrigin=DataOriginSpec(providerId=[cohort_b])),
        ]
    else:
        return None

    return AnalysisPlan(
        charts=None,
        statistical_tests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=metrics,
                filters=AndFilter(
                    and_=[
                        DateFilter(operator="GE", value=start_date),
                        DateFilter(operator="LE", value=end_date),
                    ]
                ),
            )
        ],
    )


def _has_distinct_metric_cohorts(metric_a: Optional[Any], metric_b: Optional[Any]) -> bool:
    if metric_a is None or metric_b is None:
        return False

    metric_a_origin = getattr(metric_a, "data_origin", None)
    metric_b_origin = getattr(metric_b, "data_origin", None)
    metric_a_scope = getattr(metric_a, "origin_scope", None)
    metric_b_scope = getattr(metric_b, "origin_scope", None)

    if metric_a_origin is not None and metric_b_origin is not None:
        origin_a_payload = metric_a_origin.model_dump(by_alias=True, exclude_none=True)
        origin_b_payload = metric_b_origin.model_dump(by_alias=True, exclude_none=True)
        if origin_a_payload != origin_b_payload:
            return True

    if metric_a_scope is not None and metric_b_scope is not None:
        scope_a_payload = metric_a_scope.model_dump(by_alias=True, exclude_none=True)
        scope_b_payload = metric_b_scope.model_dump(by_alias=True, exclude_none=True)
        if scope_a_payload != scope_b_payload:
            return True

    if metric_a_origin is None or metric_b_origin is None:
        return False

    label_a = getattr(metric_a_scope, "label", None)
    label_b = getattr(metric_b_scope, "label", None)
    if isinstance(label_a, str) and isinstance(label_b, str):
        return bool(label_a.strip() and label_b.strip() and label_a.strip() != label_b.strip())

    return False


def _validate_statistical_plan_readiness(plan: AnalysisPlan) -> Optional[VisualizationRequestOutcome]:
    logger.info("Entering _validate_statistical_plan_readiness")
    tests = list(plan.statistical_tests or [])
    logger.info("Statistical readiness received %d test(s)", len(tests))
    if not tests:
        return None

    index = 0
    while index < len(tests):
        test = tests[index]
        test_type = (test.test_type or "").upper().strip()
        logger.info("Validating test index=%d type=%s", index, test_type)
        if test_type != "MANN_WHITNEY_U_TEST":
            index += 1
            continue

        # Temporal comparisons are represented as two adjacent Mann-Whitney tests
        # with matching metrics and distinct date filters.
        if index + 1 < len(tests):
            next_test = tests[index + 1]
            next_type = (next_test.test_type or "").upper().strip()
            if next_type == "MANN_WHITNEY_U_TEST":
                metrics = [m.metric.strip().upper() for m in (test.metrics or []) if m.metric and m.metric.strip()]
                next_metrics = [m.metric.strip().upper() for m in (next_test.metrics or []) if m.metric and m.metric.strip()]
                if test.filters is not None and next_test.filters is not None and metrics and metrics == next_metrics:
                    index += 2
                    continue

        metrics = list(test.metrics or [])
        logger.info("Mann-Whitney test index=%d has %d metric cohort(s)", index, len(metrics))
        if len(metrics) < 2:
            return VisualizationRequestOutcome(
                decision="clarify",
                reason="missing_statistical_cohorts",
                message=("I can run Mann-Whitney U only with two explicit cohorts. Please provide cohort A and cohort B to compare."),
                clarification_type="analysis_plan",
                clarification_options=[],
                missing_fields=["statistical_cohorts"],
            )

        logger.info("Checking cohort distinctness for Mann-Whitney test index=%d", index)
        if not _has_distinct_metric_cohorts(metrics[0], metrics[1]):
            return VisualizationRequestOutcome(
                decision="clarify",
                reason="missing_statistical_cohorts",
                message=("Mann-Whitney U requires two distinct cohorts. Please provide different cohort filters or scopes for each comparison group."),
                clarification_type="analysis_plan",
                clarification_options=[],
                missing_fields=["statistical_cohorts"],
            )

        index += 1

    return None


def _decision_stage(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str],
    conversation_history: Optional[List[str]] = None,
) -> VisualizationRequestOutcome:
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable")

    chain = _DECISION_PROMPT | llm
    payload = {
        "language": (language or "en").strip() or "en",
        "question": question or "",
        "entities_json": json.dumps(entities or {}, ensure_ascii=False),
        "metric_candidates_json": json.dumps(_metric_candidates(question or ""), ensure_ascii=False),
        "chart_types_json": json.dumps(_chart_types(), ensure_ascii=False),
        "conversation_history_json": json.dumps(conversation_history or [], ensure_ascii=False),
    }

    # One retry on a malformed decision value: unlike generate_analysis_plan
    # (pipeline.py), which already retries with validation feedback on a bad
    # response, this call previously had zero retry -- any single malformed
    # "decision" (observed live via CVaLab, outside {proceed, clarify,
    # reject}) fell straight through to orchestrate_visualization_request's
    # generic exception handler ("orchestrator_failed", a vague "I need more
    # detail" message even for a well-formed request). A same-input retry is
    # cheap and mirrors the plan-generation stage's own established pattern.
    parsed: Dict[str, Any] = {}
    decision_raw = ""
    recovered_reason: Optional[str] = None
    last_bad_decision: Optional[str] = None
    for attempt in range(2):
        parsed = _invoke_chain(chain, payload)
        logger.info("Decision stage raw response (attempt %d): %s", attempt + 1, parsed)
        decision_raw = str(parsed.get("decision") or "").strip().lower()
        if decision_raw in {"proceed", "clarify", "reject"}:
            break
        if decision_raw in _REASON_IMPLIES_DECISION:
            recovered_reason = decision_raw
            decision_raw = _REASON_IMPLIES_DECISION[decision_raw]
            break
        last_bad_decision = decision_raw
    else:
        raise ValueError(f"Invalid decision from decision stage after retry: {last_bad_decision!r}")

    reason = recovered_reason or str(parsed.get("reason") or "").strip() or "llm_orchestrator"
    missing_fields = _coerce_missing_fields(parsed.get("missing_fields"))
    clarification_type = parsed.get("clarification_type")
    clarification_options = _coerce_options(parsed.get("clarification_options"))
    llm_message_raw = parsed.get("message")
    llm_message = llm_message_raw.strip() if isinstance(llm_message_raw, str) and llm_message_raw.strip() else None

    outcome = VisualizationRequestOutcome(
        decision=cast(OutcomeDecision, decision_raw),
        reason=reason,
        message=llm_message,
        clarification_type=clarification_type.strip() if isinstance(clarification_type, str) and clarification_type.strip() else None,
        clarification_options=clarification_options,
        missing_fields=missing_fields,
    )

    # Deterministic safeguard: don't trust a missing_fields claim that
    # contradicts ENTITIES_JSON itself (see _drop_falsely_missing_fields).
    if outcome.decision != "proceed" and outcome.missing_fields:
        corrected_missing = _drop_falsely_missing_fields(outcome.missing_fields, entities)
        if not corrected_missing:
            return VisualizationRequestOutcome(
                decision="proceed",
                reason="all_required_fields_present",
                message=None,
                clarification_type=None,
                clarification_options=[],
                missing_fields=[],
            )
        if corrected_missing != outcome.missing_fields:
            outcome = VisualizationRequestOutcome(
                decision=outcome.decision,
                reason=outcome.reason,
                message=outcome.message,
                clarification_type=outcome.clarification_type,
                clarification_options=outcome.clarification_options,
                missing_fields=corrected_missing,
            )

    # Deterministic safeguard: a request with a real, present metric cannot be
    # genuinely out of scope -- Rasa's own intent routing already established
    # this is a visualization request before this stage ever runs (this
    # action only fires for generate_visualization/update_visualization/
    # clarify_visualization). Observed: the LLM sometimes rejects such a
    # request as out_of_scope when the actual gap is a missing chart_type --
    # that's a clarify, not a reject.
    #
    # Scoped tightly to reasons that are actually about scope: "reject" is
    # also legitimately used for unrelated data-validity problems (e.g.
    # invalid_date_format for a malformed date), which must NOT be silently
    # waved through just because a metric happens to be present too -- that
    # was a real regression caught via CVaLab's webapp_negative_invalid_time_period
    # scenario during this fix's own testing.
    if outcome.decision == "reject" and "out_of_scope" in outcome.reason.strip().lower().replace(" ", "_") and any(_entity_present(entities, key) for key in _SCOPE_PROVING_ENTITY_KEYS):
        still_missing = [field for field in ("metric", "chart_type") if not _entity_present(entities, field)]
        if still_missing:
            # The LLM's own message text ("This request is not related to...")
            # came from the wrong reject/out_of_scope judgment being overridden
            # here -- reusing it verbatim would still show the user a false
            # claim even though the decision itself is now correct.
            readable_fields = " and ".join(field.replace("_", " ") for field in still_missing)
            outcome = VisualizationRequestOutcome(
                decision="clarify",
                reason="missing_required_fields",
                message=f"Please specify the {readable_fields} you'd like to use.",
                clarification_type=still_missing[0],
                clarification_options=[],
                missing_fields=still_missing,
            )
        else:
            outcome = VisualizationRequestOutcome(
                decision="proceed",
                reason="all_required_fields_present",
                message=None,
                clarification_type=None,
                clarification_options=[],
                missing_fields=[],
            )

    # Deterministic safeguard: do not block statistical-test requests on chart_type.
    if outcome.decision == "clarify" and _is_missing_chart_type_only(outcome.missing_fields) and _has_statistical_test_signal(question, entities):
        return VisualizationRequestOutcome(
            decision="proceed",
            reason="statistical_test_without_chart_type",
            message=None,
            clarification_type=None,
            clarification_options=[],
            missing_fields=[],
        )

    return outcome


def orchestrate_visualization_request(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str] = None,
    trace_id: Optional[str] = None,
    max_retries: int = 2,
    include_plan: bool = True,
    conversation_history: Optional[List[str]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> VisualizationRequestOutcome:
    with log_context(trace_id=trace_id or "", orchestrator_include_plan=include_plan):
        if not _ORCHESTRATOR_ENABLED:
            if not include_plan:
                return VisualizationRequestOutcome(decision="proceed", reason="orchestrator_disabled")
            plan = _generate_plan_with_timeout(
                question=question,
                entities=entities,
                language=language,
                max_retries=max_retries,
                trace_id=trace_id,
                progress_cb=progress_cb,
            )
            return VisualizationRequestOutcome(decision="proceed", reason="orchestrator_disabled", plan=plan)

        def report(message: str) -> None:
            if progress_cb is not None:
                progress_cb(message)

        try:
            report("Analyzing request intent and feasibility")
            logger.info("Orchestrator input - question: %s, entities: %s", question, entities)

            stat_test_support_validation = _validate_statistical_test_support(question)
            if stat_test_support_validation is not None:
                logger.info(
                    "Orchestrator rejection: %s",
                    stat_test_support_validation.reason,
                )
                return stat_test_support_validation

            risk_factor_filter_validation = _validate_risk_factor_filter_support(question, entities)
            if risk_factor_filter_validation is not None:
                logger.info(
                    "Orchestrator rejection: %s",
                    risk_factor_filter_validation.reason,
                )
                return risk_factor_filter_validation

            group_by_validation = _validate_group_by_support(question, entities)
            if group_by_validation is not None:
                logger.info(
                    "Orchestrator clarification: %s",
                    group_by_validation.reason,
                )
                return group_by_validation

            stats_entity_validation = _validate_statistical_entity_readiness(question, entities)
            if stats_entity_validation is not None:
                logger.info(
                    "Orchestrator clarification: %s",
                    stats_entity_validation.reason,
                )
                return stats_entity_validation

            stage1 = _decision_stage(question, entities, language, conversation_history=conversation_history)
            logger.info(
                "Orchestrator decision: %s, message: %s, missing: %s",
                stage1.decision,
                stage1.message,
                stage1.missing_fields,
            )

            if stage1.decision == "reject":
                if stage1.message is None:
                    stage1.message = "This request is outside the visualization flow."
                return stage1

            if stage1.decision == "clarify":
                return stage1

            if not include_plan:
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason=stage1.reason or "sufficient_information",
                )

            report("Generating visualization plan")
            # `question` is already the correctly-scoped current-turn text (callers
            # prefix it with the prior plan JSON for update flows); no need to
            # re-join conversation_history here too — that previously duplicated
            # history the caller already folded in, and could feed stale keywords
            # from unrelated prior turns into the provider/provider-group checks.
            deterministic_plan = _build_deterministic_statistical_plan(question, entities)
            if deterministic_plan is not None:
                deterministic_plan = _normalize_plan_semantic_splits(deterministic_plan)
                stats_validation = _validate_statistical_plan_readiness(deterministic_plan)
                if stats_validation is not None:
                    return stats_validation
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason="deterministic_statistical_plan",
                    plan=deterministic_plan,
                )

            logger.info("Plan generation starting via timeout wrapper")
            plan = _generate_plan_with_timeout(
                question=question,
                entities=entities,
                language=language,
                max_retries=max_retries,
                trace_id=trace_id,
                progress_cb=progress_cb,
            )
            plan = _normalize_plan_semantic_splits(plan)
            logger.info("Plan generation completed successfully", extra={"plan_type": type(plan).__name__})

            logger.info("Starting validation of statistical plan readiness")
            stats_validation = _validate_statistical_plan_readiness(plan)
            logger.info("Statistical validation complete", extra={"validation_result": stats_validation is not None})

            if stats_validation is not None:
                logger.info("Returning statistical validation result")
                return stats_validation

            logger.info("Creating and returning VisualizationRequestOutcome")
            outcome = VisualizationRequestOutcome(
                decision="proceed",
                reason=stage1.reason or "sufficient_information",
                plan=plan,
            )
            logger.info("VisualizationRequestOutcome created", extra={"outcome": str(outcome)[:100]})
            return outcome
        except Exception:
            logger.exception(
                "Visualization request orchestration failed",
                extra={
                    "log_context": {
                        "event": "orchestrator.request.failed",
                        "operation": "orchestrate_visualization_request",
                        "outcome": "failure",
                        "include_plan": include_plan,
                        "fail_open_enabled": _ORCHESTRATOR_FAIL_OPEN,
                    }
                },
            )
            # Orchestration failed (LLM error, timeout, parsing failure, ...). Retry plan
            # generation once, skipping the normal triage/readiness checks, when either
            # the fail-open escape hatch is enabled (a broad opt-in for degraded-mode
            # operation) or the request clearly looks like a statistical test (a
            # narrower, always-on resilience case). Either way, a second failure falls
            # through to the same explicit clarify outcome rather than propagating.
            should_retry = include_plan and (_ORCHESTRATOR_FAIL_OPEN or _has_statistical_test_signal(question, entities))
            if should_retry:
                try:
                    report("Orchestration failed; retrying plan generation directly")
                    plan = _generate_plan_with_timeout(
                        question=question,
                        entities=entities,
                        language=language,
                        max_retries=max_retries,
                        trace_id=trace_id,
                        progress_cb=progress_cb,
                    )
                    return VisualizationRequestOutcome(
                        decision="proceed",
                        reason="orchestrator_fallback_to_plan",
                        plan=plan,
                    )
                except Exception:
                    logger.exception("Fallback plan generation failed", extra={"trace_id": trace_id})

            return VisualizationRequestOutcome(
                decision="clarify",
                reason="orchestrator_failed",
                message=(
                    "I could not produce a valid statistical plan from that request. Please provide a metric, two explicit cohorts (for example two provider groups), and explicit date bounds."
                    if _has_statistical_test_signal(question, entities)
                    else "I need a bit more detail before I can continue."
                ),
            )
