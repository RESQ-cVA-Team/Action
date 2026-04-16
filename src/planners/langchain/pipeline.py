import json
import logging
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypedDict, Union, cast, overload

from langchain.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.domain.graphql.ssot_enums import MetricType
from src.domain.langchain.schema import AnalysisPlan, ChartSpec, ChartType, MetricSpec, PlanMetadata
from src.planners.langchain.examples import get_few_shot_examples
from src.planners.langchain.llm_factory import create_chat_llm, get_llm_provider
from src.util import env

logger = logging.getLogger(__name__)
# Privacy/safety defaults:
# - Do not log prompts or chain-of-thought by default.
# - Allow opting in for debugging via env flags.
_LOG_PROMPTS = env.env_flag("PLANNER_LOG_PROMPTS", default=False)
_LOG_REASONING = env.env_flag("PLANNER_LOG_REASONING", default=False)
_ENABLE_COT = env.env_flag("PLANNER_ENABLE_COT", default=True)
_ENABLE_DYNAMIC_FEW_SHOTS = env.env_flag("PLANNER_DYNAMIC_FEW_SHOTS", default=True)
_ENABLE_PLAN_CACHE = env.env_flag("PLANNER_ENABLE_PLAN_CACHE", default=True)
_ENABLE_TIMEOUT_FALLBACK = env.env_flag("PLANNER_ENABLE_TIMEOUT_FALLBACK", default=True)
_ENABLE_WEIGHTED_FEW_SHOT_RANKING = env.env_flag("PLANNER_ENABLE_WEIGHTED_FEW_SHOT_RANKING", default=True)
_STRICT_MODE = env.env_flag("ANALYTICS_STRICT_MODE", default=False) or env.env_flag("PLANNER_STRICT_MODE", default=False)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = env.get_env(name, default=str(default)) or str(default)
    try:
        return max(minimum, float(raw))
    except Exception:
        return default


def _env_csv_tokens(name: str, default: str) -> List[str]:
    raw = env.get_env(name, default=default) or default
    parts = [part.strip().lower() for part in raw.split(",")]
    return [part for part in parts if part]


_FEWSHOT_TOKEN_WEIGHT = _env_float("PLANNER_FEWSHOT_TOKEN_WEIGHT", default=1.0)
_FEWSHOT_ENTITY_KEY_WEIGHT = _env_float("PLANNER_FEWSHOT_ENTITY_KEY_WEIGHT", default=6.0)
_FEWSHOT_ENTITY_VALUE_WEIGHT = _env_float("PLANNER_FEWSHOT_ENTITY_VALUE_WEIGHT", default=10.0)
_FEWSHOT_INTENT_WEIGHT = _env_float("PLANNER_FEWSHOT_INTENT_WEIGHT", default=8.0)

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "chart_line": _env_csv_tokens("PLANNER_INTENT_LINE_KEYWORDS", "line,trend"),
    "chart_bar": _env_csv_tokens("PLANNER_INTENT_BAR_KEYWORDS", "bar,column"),
    "chart_area": _env_csv_tokens("PLANNER_INTENT_AREA_KEYWORDS", "area"),
    "distribution": _env_csv_tokens("PLANNER_INTENT_DISTRIBUTION_KEYWORDS", "distribution,histogram,violin,box"),
    "time": _env_csv_tokens("PLANNER_INTENT_TIME_KEYWORDS", "over time,last ,monthly,weekly,yearly,time series"),
    "group_by": _env_csv_tokens("PLANNER_INTENT_GROUPBY_KEYWORDS", " by ,grouped by,split by"),
    "stat_test": _env_csv_tokens("PLANNER_INTENT_TEST_KEYWORDS", "test,anova,t-test,chi,wilcoxon,mann-whitney"),
}

_SUPPORTED_STAT_TESTS_RAW = env.get_env("EXECUTOR_SUPPORTED_STAT_TESTS", default="MANN_WHITNEY_U_TEST") or "MANN_WHITNEY_U_TEST"
_SUPPORTED_STAT_TESTS = [token.strip().upper() for token in _SUPPORTED_STAT_TESTS_RAW.split(",") if token.strip()]

_SINGLE_CHART_HINTS = [
    "one graph",
    "one chart",
    "single chart",
    "single graph",
    "single visual",
    "in one graph",
    "in one chart",
    "same graph",
    "same chart",
]
_MULTI_CHART_HINTS = [
    "separate charts",
    "separate chart",
    "multiple charts",
    "multiple visuals",
    "two charts",
    "three charts",
    "as separate visuals",
    "as separate charts",
]

_planner_timeout_raw = env.get_env("PLANNER_REQUEST_TIMEOUT_SECONDS", default="30") or "30"
_planner_timeout_seconds = 30.0
try:
    _planner_timeout_seconds = max(1.0, float(_planner_timeout_raw))
except Exception:
    _planner_timeout_seconds = 30.0
_PLANNER_REQUEST_TIMEOUT_SECONDS = _planner_timeout_seconds

_few_shot_limit_raw = env.get_env("PLANNER_MAX_FEW_SHOTS", default="2") or "2"
_max_few_shots_value = 2
try:
    _max_few_shots_value = max(1, int(_few_shot_limit_raw))
except Exception:
    _max_few_shots_value = 2
_MAX_FEW_SHOTS = _max_few_shots_value

_plan_cache_size_raw = env.get_env("PLANNER_CACHE_SIZE", default="256") or "256"
_plan_cache_size_value = 256
try:
    _plan_cache_size_value = max(1, int(_plan_cache_size_raw))
except Exception:
    _plan_cache_size_value = 256
_PLAN_CACHE_SIZE = _plan_cache_size_value

_plan_cache_ttl_raw = env.get_env("PLANNER_CACHE_TTL_SECONDS", default="900") or "900"
_plan_cache_ttl_value = 900.0
try:
    _plan_cache_ttl_value = max(1.0, float(_plan_cache_ttl_raw))
except Exception:
    _plan_cache_ttl_value = 900.0
_PLAN_CACHE_TTL_SECONDS = _plan_cache_ttl_value
_PLAN_CACHE_KEY_VERSION = (env.get_env("PLANNER_CACHE_KEY_VERSION", default="v2") or "v2").strip()
_TIMEOUT_FALLBACK_METRIC = (env.get_env("PLANNER_TIMEOUT_FALLBACK_METRIC", default="DTN") or "DTN").strip().upper()
_TIMEOUT_FALLBACK_CHART_TYPE = (env.get_env("PLANNER_TIMEOUT_FALLBACK_CHART_TYPE", default="LINE") or "LINE").strip().upper()

LLM_PROVIDER = get_llm_provider()
llm: Any = create_chat_llm(temperature=0)
logger.info("[Planner] Initialized LLM provider=%s", LLM_PROVIDER)
_LLM_MODEL = (env.get_env("LLM_MODEL", default="") or "").strip()


class PlannerTimeoutError(TimeoutError):
    pass


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


def _extract_json_block(text: str) -> str:
    candidate = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON object")
    return candidate[start : end + 1]


def _coerce_analysis_plan(response: Any) -> AnalysisPlan:
    import json

    if isinstance(response, AnalysisPlan):
        return response

    if isinstance(response, dict):
        return AnalysisPlan.model_validate(response)

    text = _extract_text(response)
    json_block = _extract_json_block(text)
    parsed = json.loads(json_block)
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return AnalysisPlan.model_validate(parsed)


def get_schema_description(model: Type[Any]) -> str:
    """
    Recursively extract field descriptions from a Pydantic model as a readable schema spec (Pydantic v2 compatible).
    """
    from typing import get_args, get_origin

    def describe(model: Type[Any], indent: int = 0) -> str:
        lines: List[str] = []
        if hasattr(model, "model_fields"):
            for name, field in model.model_fields.items():
                desc = field.description or field.title or ""
                typ = str(field.annotation)
                lines.append(" " * indent + f"- {name} ({typ}): {desc}")
                # Recurse for nested models
                outer = get_origin(field.annotation)
                inner = get_args(field.annotation)
                if outer in (list, List) and inner and hasattr(inner[0], "model_fields"):
                    lines.append(describe(inner[0], indent + 2))
                elif hasattr(field.annotation, "model_fields"):
                    lines.append(describe(field.annotation, indent + 2))
        return "\n".join(lines)

    return f"AnalysisPlan schema:\n{describe(model)}"


SCHEMA_DESCRIPTION: str = get_schema_description(AnalysisPlan)
cot_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            "You are a clinical analytics planner. User interface language: {language}. Think step by step about how to answer the user's query, considering all entities. Explain your reasoning in detail in English (for internal clarity) even if the user language is different. Do not output the plan yet.",
        ),
        ("user", "USER_UTTERANCE:\n{question}\n\nENTITIES_DETECTED(JSON):\n{entities}"),
    ]
)

few_shot_examples = get_few_shot_examples()
_PLAN_CACHE: "OrderedDict[str, tuple[AnalysisPlan, float]]" = OrderedDict()
_PLAN_CACHE_LOCK = Lock()
_plan_cache_hits = 0
_plan_cache_misses = 0
_plan_cache_expired = 0
_PLAN_CACHE_STATS_LOCK = Lock()
_LAST_CACHE_EVENT: ContextVar[Optional[bool]] = ContextVar("planner_last_cache_event", default=None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invoke_with_timeout(chain: Any, inputs: Dict[str, Any], label: str) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chain.invoke, inputs)
        try:
            return future.result(timeout=_PLANNER_REQUEST_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise PlannerTimeoutError(f"{label} timed out after {_PLANNER_REQUEST_TIMEOUT_SECONDS:.1f}s") from exc


def _enum_allowed_values(enum_cls: Any) -> set[str]:
    try:
        return {m.value for m in enum_cls}  # type: ignore[attr-defined]
    except Exception:
        try:
            return {str(m.value) if hasattr(m, "value") else str(m) for m in list(enum_cls)}  # type: ignore[arg-type]
        except Exception:
            return set()


def _iter_entity_values(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in cast(List[Any], value):
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _with_plan_metadata(
    plan: AnalysisPlan,
    trace_id: Optional[str],
    requested_visual_layout: Optional[str] = None,
    fallback_used: bool = False,
    fallback_reason: Optional[str] = None,
) -> AnalysisPlan:
    current = plan.metadata if isinstance(getattr(plan, "metadata", None), PlanMetadata) else PlanMetadata()
    metadata = PlanMetadata(
        trace_id=trace_id or current.trace_id,
        planner_provider=current.planner_provider or LLM_PROVIDER,
        planner_model=current.planner_model or (_LLM_MODEL or None),
        planner_version=current.planner_version or "pipeline-v2",
        requested_visual_layout=current.requested_visual_layout or requested_visual_layout,
        fallback_used=bool(current.fallback_used or fallback_used),
        fallback_reason=current.fallback_reason or fallback_reason,
        generated_at_utc=current.generated_at_utc or _utc_now_iso(),
    )
    return plan.model_copy(update={"metadata": metadata})


def _infer_requested_visual_layout(question: str) -> Optional[str]:
    lowered = (question or "").strip().lower()
    if not lowered:
        return None

    if any(token in lowered for token in _MULTI_CHART_HINTS):
        return "multi_chart"
    if any(token in lowered for token in _SINGLE_CHART_HINTS):
        return "single_chart"
    return None


def _plan_for_cache(plan: AnalysisPlan) -> AnalysisPlan:
    current = plan.metadata if isinstance(getattr(plan, "metadata", None), PlanMetadata) else PlanMetadata()
    metadata = current.model_copy(update={"trace_id": None})
    return plan.model_copy(update={"metadata": metadata})


def _build_timeout_fallback_plan(question: str, entities: Dict[str, Any], language: str, trace_id: Optional[str]) -> Optional[AnalysisPlan]:
    metric_allowed = {v.upper() for v in _enum_allowed_values(MetricType)}
    chart_allowed = {v.upper() for v in ChartType}

    metric_candidates: List[str] = []
    for key in ["metric", "metric_code", "metric_type", "kpi"]:
        metric_candidates.extend(_iter_entity_values(entities.get(key)))

    chosen_metric: Optional[str] = None
    for candidate in metric_candidates:
        up = candidate.upper()
        if up in metric_allowed:
            chosen_metric = up
            break

    if chosen_metric is None and _TIMEOUT_FALLBACK_METRIC in metric_allowed:
        chosen_metric = _TIMEOUT_FALLBACK_METRIC
    if chosen_metric is None and "DTN" in metric_allowed:
        chosen_metric = "DTN"
    if chosen_metric is None:
        return None

    chart_candidates: List[str] = []
    for key in ["chart_type", "chart"]:
        chart_candidates.extend(_iter_entity_values(entities.get(key)))

    chosen_chart = _TIMEOUT_FALLBACK_CHART_TYPE if _TIMEOUT_FALLBACK_CHART_TYPE in chart_allowed else ("LINE" if "LINE" in chart_allowed else (sorted(chart_allowed)[0] if chart_allowed else "LINE"))
    for candidate in chart_candidates:
        up = candidate.upper()
        if up in chart_allowed:
            chosen_chart = up
            break

    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                title=f"{chosen_metric} Overview",
                description=f"Fallback plan generated for '{(language or 'auto').lower()}' after planner timeout.",
                chart_type=chosen_chart,
                metrics=[MetricSpec(metric=chosen_metric)],
            )
        ],
        statistical_tests=None,
    )
    return _with_plan_metadata(
        plan,
        trace_id=trace_id,
        requested_visual_layout=_infer_requested_visual_layout(question),
        fallback_used=True,
        fallback_reason="planner_timeout",
    )


def _cache_key(question: str, entities: Dict[str, Any], language: str) -> str:
    payload: Dict[str, Any] = {
        "q": (question or "").strip(),
        "e": entities,
        "l": (language or "").strip().lower(),
        "dfs": _ENABLE_DYNAMIC_FEW_SHOTS,
        "mfs": _MAX_FEW_SHOTS,
        "provider": LLM_PROVIDER,
        "model": _LLM_MODEL,
        "version": _PLAN_CACHE_KEY_VERSION,
        "weighted_rank": _ENABLE_WEIGHTED_FEW_SHOT_RANKING,
        "token_w": _FEWSHOT_TOKEN_WEIGHT,
        "entity_key_w": _FEWSHOT_ENTITY_KEY_WEIGHT,
        "entity_value_w": _FEWSHOT_ENTITY_VALUE_WEIGHT,
        "intent_w": _FEWSHOT_INTENT_WEIGHT,
        "intent_keywords": _INTENT_KEYWORDS,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validation_error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return exc.json()
    return str(exc)


def _cache_get(key: str) -> Optional[AnalysisPlan]:
    global _plan_cache_expired
    now = datetime.now(timezone.utc).timestamp()
    with _PLAN_CACHE_LOCK:
        item = _PLAN_CACHE.get(key)
        if item is None:
            return None
        cached_plan, cached_at = item
        if now - cached_at > _PLAN_CACHE_TTL_SECONDS:
            del _PLAN_CACHE[key]
            with _PLAN_CACHE_STATS_LOCK:
                _plan_cache_expired += 1
            return None
        _PLAN_CACHE.move_to_end(key)
        return cached_plan.model_copy(deep=True)


def _cache_put(key: str, value: AnalysisPlan) -> None:
    now = datetime.now(timezone.utc).timestamp()
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE[key] = (_plan_for_cache(value).model_copy(deep=True), now)
        _PLAN_CACHE.move_to_end(key)
        while len(_PLAN_CACHE) > _PLAN_CACHE_SIZE:
            _PLAN_CACHE.popitem(last=False)


def _record_cache_event(hit: Optional[bool]) -> None:
    global _plan_cache_hits, _plan_cache_misses
    _LAST_CACHE_EVENT.set(hit)
    if hit is None:
        return
    with _PLAN_CACHE_STATS_LOCK:
        if hit:
            _plan_cache_hits += 1
        else:
            _plan_cache_misses += 1


def get_plan_cache_diagnostics() -> Dict[str, Any]:
    with _PLAN_CACHE_STATS_LOCK, _PLAN_CACHE_LOCK:
        return {
            "enabled": _ENABLE_PLAN_CACHE,
            "last_call_cache_hit": _LAST_CACHE_EVENT.get(),
            "total_hits": _plan_cache_hits,
            "total_misses": _plan_cache_misses,
            "total_expired": _plan_cache_expired,
            "entries": len(_PLAN_CACHE),
            "capacity": _PLAN_CACHE_SIZE,
            "ttl_seconds": _PLAN_CACHE_TTL_SECONDS,
            "key_version": _PLAN_CACHE_KEY_VERSION,
        }


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def _entity_values(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, str):
        val = value.strip()
        if val:
            out.add(val.upper())
        return out
    if isinstance(value, bool):
        out.add(str(value).upper())
        return out
    if isinstance(value, (int, float)):
        out.add(str(value))
        return out
    if isinstance(value, list):
        for item in cast(List[Any], value):
            out.update(_entity_values(item))
    return out


def _normalize_entities(raw_entities: Dict[str, Any]) -> Dict[str, set[str]]:
    normalized: Dict[str, set[str]] = {}
    for key, value in raw_entities.items():
        key_norm = (key or "").strip().lower()
        if not key_norm:
            continue
        values = _entity_values(value)
        if values:
            normalized[key_norm] = values
    return normalized


def _extract_example_entities(example: Dict[str, str]) -> Dict[str, Any]:
    user_block = example.get("user", "")
    marker = "ENTITIES_DETECTED(JSON):"
    idx = user_block.find(marker)
    if idx == -1:
        return {}

    payload = user_block[idx + len(marker) :].strip()
    if not payload:
        return {}

    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return cast(Dict[str, Any], parsed)
    except Exception:
        pass

    try:
        parsed_fallback = json.loads(_extract_json_block(payload))
        if isinstance(parsed_fallback, dict):
            return cast(Dict[str, Any], parsed_fallback)
    except Exception:
        return {}

    return {}


def _intent_features(text: str) -> set[str]:
    lowered = (text or "").lower()
    features: set[str] = set()

    for feature_name, keywords in _INTENT_KEYWORDS.items():
        if any(token in lowered for token in keywords):
            features.add(feature_name)

    return features


def _match_score(question: str, entities: Dict[str, Any], example: Dict[str, str]) -> float:
    query_text = f"{question}\n{json.dumps(entities, ensure_ascii=False, sort_keys=True)}"
    query_tokens = _tokenize(query_text)
    ex_text = f"{example.get('description', '')}\n{example.get('user', '')}"
    ex_tokens = _tokenize(ex_text)

    if not _ENABLE_WEIGHTED_FEW_SHOT_RANKING:
        return float(len(query_tokens & ex_tokens))

    token_overlap_score = float(len(query_tokens & ex_tokens)) * _FEWSHOT_TOKEN_WEIGHT

    query_entities = _normalize_entities(entities)
    example_entities = _normalize_entities(_extract_example_entities(example))
    query_keys = set(query_entities.keys())
    example_keys = set(example_entities.keys())
    key_overlap = query_keys & example_keys

    entity_key_score = float(len(key_overlap)) * _FEWSHOT_ENTITY_KEY_WEIGHT
    entity_value_score = 0.0
    for key in key_overlap:
        entity_value_score += float(len(query_entities[key] & example_entities[key])) * _FEWSHOT_ENTITY_VALUE_WEIGHT

    query_intent = _intent_features(query_text)
    example_intent = _intent_features(ex_text)
    intent_score = float(len(query_intent & example_intent)) * _FEWSHOT_INTENT_WEIGHT

    return token_overlap_score + entity_key_score + entity_value_score + intent_score


def _select_few_shot_examples(question: str, entities: Dict[str, Any], max_items: int) -> List[Dict[str, str]]:
    if not few_shot_examples:
        return []

    capped = max(1, min(max_items, len(few_shot_examples)))
    scored = [(_match_score(question, entities, ex), idx, ex) for idx, ex in enumerate(few_shot_examples)]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected = [item[2] for item in scored[:capped]]
    # If all scores are zero, preserve deterministic default ordering from examples.py.
    if scored and scored[0][0] <= 0:
        return few_shot_examples[:capped]
    return selected


def _build_few_shots_text(examples: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for idx, ex in enumerate(examples, start=1):
        parts.append(
            "\n".join(
                [
                    f"EXAMPLE {idx}:",
                    f"Description: {ex['description']}",
                    "User Message:",
                    ex["user"],
                    "Assistant Plan JSON:",
                    ex["assistant"],
                    "---",
                ]
            )
        )
    return "\n".join(parts)


FEW_SHOTS_TEXT = _build_few_shots_text(few_shot_examples)
plan_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            "You are a planner. Interface language: {language}. Produce ONLY a valid AnalysisPlan JSON according to the schema. "
            "All 'title' and 'description' fields MUST be written in the interface language ({language}). "
            "Keep enum-like codes (metric, chart_type, test_type, stroke categories, sex categories) in their canonical uppercase English forms. "
            "Use the reasoning and prior examples. Place detected entities into metrics (group_by / filters). "
            "Prefer LINE/BAR for trends or comparisons; BOX/VIOLIN/HISTOGRAM for distributions. "
            "Title guidance: Avoid phrases like 'Over Time' unless there is an explicit temporal grouping. If no time/canonical grouping is specified and a LINE chart is used for a distribution, prefer '<METRIC> Distribution' for the title. "
            "Chart intent guidance: If user asks for one graph/one chart/single visual with multiple splits, prefer one chart with multiple group_by dimensions. If user asks for separate charts/multiple visuals, produce multiple chart specs. "
            "Statistical test guidance: Only use test types listed in SUPPORTED_STAT_TESTS_JSON; otherwise omit statistical_tests and return charts.",
        ),
        ("system", "SCHEMA:\n" + SCHEMA_DESCRIPTION),
        ("system", "FEW_SHOT_EXAMPLES:\n{few_shots}"),
        ("system", "SUPPORTED_STAT_TESTS_JSON:\n{supported_stat_tests}"),
        ("system", "REASONING (English internal reasoning shown below can differ from output language):\n{reasoning}"),
        ("user", "USER_UTTERANCE:\n{question}\n\nENTITIES_DETECTED(JSON):\n{entities}"),
    ]
)

cot_chain: Any = cot_prompt | llm
plan_chain: Any = plan_prompt | llm


class GeneratePlanDebug(TypedDict):
    """Typed structure for debug output of generate_analysis_plan."""

    reasoning: Any
    steps: List[Any]
    attempts: List[Any]
    final_output: Optional[AnalysisPlan]


ProgressCallback = Callable[[str], None]


@overload
def generate_analysis_plan(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str],
    max_retries: int,
    debug: Literal[True],
    trace_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> GeneratePlanDebug: ...


@overload
def generate_analysis_plan(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str],
    max_retries: int,
    debug: Literal[False],
    trace_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> AnalysisPlan: ...


@overload
def generate_analysis_plan(
    question: str,
    entities: Dict[str, Any],
    language: Optional[str],
    max_retries: int,
    debug: bool,
    trace_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Union[AnalysisPlan, GeneratePlanDebug]: ...


def generate_analysis_plan(
    question: str,
    entities: Dict[str, Any],
    language: str | None = None,
    max_retries: int = 2,
    debug: bool = False,
    trace_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Union[AnalysisPlan, GeneratePlanDebug]:
    """
    Generate a validated AnalysisPlan from user input, with chain-of-thought reasoning and automatic
    correction/retry on validation failure.

    Returns:
    - AnalysisPlan when debug is False (default).
    - GeneratePlanDebug when debug is True, including reasoning, steps, attempts, and final_output
      (which will be an AnalysisPlan or None on failure).

    Always includes 'reasoning' in the debug output, even if an error occurs.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)
    if not language:
        language = "auto"

    requested_visual_layout = _infer_requested_visual_layout(question)

    _record_cache_event(None)

    cache_key: Optional[str] = None
    if _ENABLE_PLAN_CACHE and not debug:
        cache_key = _cache_key(question=question, entities=entities, language=language)
        cached_plan = _cache_get(cache_key)
        if cached_plan is not None:
            _record_cache_event(True)
            if progress_cb is not None:
                progress_cb("Using cached plan.")
            if _LOG_PROMPTS:
                logger.debug("[Planner] plan cache hit")
            return _with_plan_metadata(cached_plan, trace_id=trace_id, requested_visual_layout=requested_visual_layout)
        _record_cache_event(False)

    # High-level notification that planning has started.
    if progress_cb is not None:
        progress_cb("Thinking about a plan.")

    selected_examples = few_shot_examples
    if _ENABLE_DYNAMIC_FEW_SHOTS:
        selected_examples = _select_few_shot_examples(
            question=question,
            entities=entities,
            max_items=_MAX_FEW_SHOTS,
        )
    if _LOG_PROMPTS:
        selected_descriptions = [ex.get("description", "") for ex in selected_examples]
        logger.debug(
            "[Planner] Selected %s few-shot example(s): %s",
            len(selected_examples),
            selected_descriptions,
        )
    few_shots_text = _build_few_shots_text(selected_examples)

    input_dict: Dict[str, Any] = {
        "question": question,
        "entities": json.dumps(entities),
        "few_shots": few_shots_text,
        "language": language,
        "supported_stat_tests": json.dumps(_SUPPORTED_STAT_TESTS),
    }
    if _LOG_PROMPTS:
        logger.debug("[Planner] input_dict: %s", input_dict)
    logger.info("[Planner] generate_analysis_plan invoked (language=%s)", language)

    steps: List[Any] = []
    attempts: List[Any] = []

    # Optional chain-of-thought step (kept out of logs by default).
    # Optimization: defer COT until needed (e.g. after a validation failure),
    # instead of paying the extra LLM call on every successful first attempt.
    reasoning: Any = ""
    cot_inputs: Dict[str, Any] = {
        "question": question,
        "entities": json.dumps(entities),
        "language": language,
    }

    def _maybe_generate_reasoning() -> bool:
        nonlocal reasoning
        if not _ENABLE_COT or reasoning:
            return bool(reasoning)
        try:
            cot_prompt_rendered: str = cot_prompt.format_prompt(**cot_inputs).to_string()
            if _LOG_PROMPTS:
                logger.debug("[Planner] cot_prompt_rendered: %s", cot_prompt_rendered)
            cot_response: Any = _invoke_with_timeout(cot_chain, cot_inputs, label="cot_chain")
            reasoning_text: str = _extract_text(cot_response)
            if _LOG_REASONING:
                logger.debug("[Planner] cot_response(content): %s", reasoning_text)
            steps.append(
                {
                    "step": "chain_of_thought",
                    "prompt": cot_prompt_rendered if _LOG_PROMPTS else "(prompt logging disabled)",
                    "response": reasoning_text if _LOG_REASONING else "(reasoning logging disabled)",
                }
            )
            reasoning = reasoning_text
            return True
        except Exception as cot_exc:
            logger.warning("[Planner] COT step failed; continuing without it: %s", cot_exc)
            steps.append(
                {
                    "step": "chain_of_thought",
                    "prompt": "(prompt logging disabled)",
                    "response": f"ERROR: {cot_exc}",
                }
            )
            reasoning = ""
            return False

    plan_inputs: Dict[str, Any] = {
        "question": question,
        "entities": json.dumps(entities),
        "reasoning": reasoning,
        "few_shots": few_shots_text,
        "language": language,
        "supported_stat_tests": json.dumps(_SUPPORTED_STAT_TESTS),
    }

    plan_prompt_rendered: str = ""
    if _LOG_PROMPTS:
        try:
            plan_prompt_rendered = plan_prompt.format_prompt(**plan_inputs).to_string()
            logger.debug("[Planner] plan_prompt_rendered: %s", plan_prompt_rendered)
        except Exception:
            plan_prompt_rendered = "(failed to render prompt for logging)"
    for attempt in range(max_retries + 1):
        if progress_cb is not None:
            dots = "." * (attempt + 1)
            progress_cb(f"Thinking about a plan.{dots}")
        try:
            # Invoke only the plan chain here. (We already ran the optional COT step above.)
            if _LOG_PROMPTS:
                logger.debug("[Planner] Attempt %s: invoking plan_chain", attempt + 1)
            raw_result: Any = _invoke_with_timeout(plan_chain, plan_inputs, label="plan_chain")
            if _LOG_PROMPTS:
                logger.debug("[Planner] Attempt %s: result: %s", attempt + 1, raw_result)
            steps.append(
                {
                    "step": f"plan_attempt_{attempt + 1}",
                    "prompt": plan_prompt_rendered if _LOG_PROMPTS else "(prompt logging disabled)",
                    "response": raw_result,
                }
            )
            result: AnalysisPlan = _with_plan_metadata(
                _coerce_analysis_plan(raw_result),
                trace_id=trace_id,
                requested_visual_layout=requested_visual_layout,
            )

            if debug:
                debug_payload: GeneratePlanDebug = {
                    "reasoning": reasoning,
                    "steps": steps,
                    "attempts": attempts,
                    "final_output": result,
                }
                if progress_cb is not None:
                    progress_cb("Finished thinking about a plan.")
                return debug_payload
            if cache_key is not None:
                _cache_put(cache_key, result)
            if progress_cb is not None:
                progress_cb("Finished thinking about a plan.")
            return result
        except PlannerTimeoutError as te:
            logger.warning("[Planner] Timeout on attempt %s: %s", attempt + 1, te)
            attempts.append(
                {
                    "error": str(te),
                    "input": input_dict if _LOG_PROMPTS else "(input logging disabled)",
                    "output": "planner_timeout",
                }
            )

            if not debug and _ENABLE_TIMEOUT_FALLBACK and not _STRICT_MODE:
                fallback = _build_timeout_fallback_plan(
                    question=question,
                    entities=entities,
                    language=language,
                    trace_id=trace_id,
                )
                if fallback is not None:
                    if progress_cb is not None:
                        progress_cb("Planner timed out; using deterministic fallback plan.")
                    return fallback

            if attempt == max_retries:
                if debug:
                    return {
                        "reasoning": reasoning,
                        "steps": steps,
                        "attempts": attempts,
                        "final_output": None,
                    }
                raise

            continue
        except (ValidationError, ValueError, TypeError) as ve:
            logger.warning("[Planner] Plan parsing/validation error on attempt %s: %s", attempt + 1, ve)
            attempts.append(
                {
                    "error": str(ve),
                    "input": input_dict if _LOG_PROMPTS else "(input logging disabled)",
                    "output": _validation_error_text(ve),
                }
            )
            if attempt == max_retries:
                if debug:
                    return {
                        "reasoning": reasoning,
                        "steps": steps,
                        "attempts": attempts,
                        "final_output": None,
                    }
                raise

            if _ENABLE_COT and not reasoning:
                generated = _maybe_generate_reasoning()
                if generated:
                    plan_inputs["reasoning"] = reasoning
                    logger.info("[Planner] Retrying with deferred COT reasoning after initial validation failure")
                    continue

            invalid_output: str = _validation_error_text(ve)
            # Avoid inlining raw JSON with `{}` into template text.
            critique_prompt_obj: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore
                [
                    (
                        "system",
                        "The following output did not pass validation. Critique the output, explain what is wrong, and then return a corrected valid AnalysisPlan JSON. Only fix the error described.",
                    ),
                    (
                        "user",
                        "Original user input: {user_input}\n\nInvalid output: {invalid_output}\n\nValidation error: {validation_error}",
                    ),
                ]
            )

            critique_inputs: Dict[str, Any] = {
                "user_input": json.dumps(input_dict, ensure_ascii=False),
                "invalid_output": invalid_output,
                "validation_error": str(ve),
            }
            critique_prompt_rendered: str = ""
            if _LOG_PROMPTS:
                try:
                    critique_prompt_rendered = critique_prompt_obj.format_prompt(**critique_inputs).to_string()
                    logger.debug("[Planner] critique_prompt_rendered: %s", critique_prompt_rendered)
                except Exception:
                    critique_prompt_rendered = "(failed to render critique prompt for logging)"

            critique_chain: Any = critique_prompt_obj | llm
            try:
                critique_raw_response: Any = _invoke_with_timeout(critique_chain, critique_inputs, label="critique_chain")
                critique_response: AnalysisPlan = _with_plan_metadata(
                    _coerce_analysis_plan(critique_raw_response),
                    trace_id=trace_id,
                    requested_visual_layout=requested_visual_layout,
                )
            except Exception as critique_exc:
                logger.warning("[Planner] Correction step failed on attempt %s: %s", attempt + 1, critique_exc)
                steps.append(
                    {
                        "step": f"correction_attempt_{attempt + 1}",
                        "prompt": critique_prompt_rendered if _LOG_PROMPTS else "(prompt logging disabled)",
                        "response": f"ERROR: {critique_exc}",
                    }
                )
                continue

            # Treat the critique response as the corrected result immediately.
            steps.append(
                {
                    "step": f"correction_attempt_{attempt + 1}",
                    "prompt": critique_prompt_rendered if _LOG_PROMPTS else "(prompt logging disabled)",
                    "response": critique_raw_response,
                }
            )
            if debug:
                debug_payload: GeneratePlanDebug = {
                    "reasoning": reasoning,
                    "steps": steps,
                    "attempts": attempts,
                    "final_output": critique_response,
                }
                if progress_cb is not None:
                    progress_cb("Finished thinking about a plan.")
                return debug_payload
            if cache_key is not None:
                _cache_put(cache_key, critique_response)
            if progress_cb is not None:
                progress_cb("Finished thinking about a plan.")
            return critique_response
    # Should never reach here; all paths either return or raise inside the loop
    raise RuntimeError("generate_analysis_plan failed to produce a result")
