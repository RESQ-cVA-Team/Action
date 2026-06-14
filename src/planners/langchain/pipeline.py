import hashlib
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

from langchain_core.prompts import ChatPromptTemplate

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.examples import get_few_shot_examples
from src.planners.langchain.llm_factory import create_chat_llm, get_llm_provider
from src.planners.langchain.semantic_adapter import validate_analysis_plan_semantics
from src.util import env as env_util
from src.util.logging_utils import bind_current_context, log_context

logger = logging.getLogger(__name__)
_ENABLE_DYNAMIC_FEW_SHOTS = True
_ENABLE_PLAN_CACHE = True

_FEWSHOT_TOKEN_WEIGHT = 1.0
_FEWSHOT_ENTITY_KEY_WEIGHT = 6.0
_FEWSHOT_ENTITY_VALUE_WEIGHT = 10.0
_FEWSHOT_INTENT_WEIGHT = 8.0

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "distribution": ["distribution", "histogram", "violin", "box"],
    "group_by": [" by ", "grouped by", "split by"],
    "stat_test": ["compare", "mann-whitney", "statistical test", "significant", "difference between"],
}

_SUPPORTED_STAT_TESTS = ["MANN_WHITNEY_U_TEST"]

_PLANNER_REQUEST_TIMEOUT_SECONDS = 30.0
_MAX_FEW_SHOTS = 2
_PLAN_CACHE_SIZE = 256
_PLAN_CACHE_TTL_SECONDS = 900.0
_PLAN_CACHE_KEY_VERSION = "v3"

LLM_PROVIDER = get_llm_provider()
_LLM_MODEL = (env_util.get_env("LLM_MODEL", default="") or "").strip()
llm: Any = create_chat_llm(temperature=0)
logger.debug("[Planner] Initialized LLM provider=%s model=%s", LLM_PROVIDER, _LLM_MODEL or "-")


class PlannerTimeoutError(TimeoutError):
    pass


class PlannerIntentAmbiguityError(ValueError):
    """Raised when planner output remains semantically ambiguous after retries."""

    def __init__(self, message: str, clarification_options: Optional[List[str]] = None):
        super().__init__(message)
        self.clarification_options = clarification_options or ["DISTRIBUTION", "SUMMARY", "COMPARISON"]


def _is_semantic_ambiguity_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return (
        "unable to infer chart semantics" in text
        or "xaxis" in text
        or "xaxes" in text
        or "yaxes" in text
        or "seriesby" in text
        or "series" in text
    )


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


def _assert_no_empty_chart_semantics(payload: Dict[str, Any]) -> None:
    """Reject ambiguous planner output instead of auto-correcting it.

    Empty dict values in xAxes/yAxes/series can pass shallow JSON checks.
    We fail fast so retry feedback forces explicit and valid chart semantics.
    """
    charts_any = payload.get("charts")
    if not isinstance(charts_any, list):
        return
    chart_entries = cast(List[Any], charts_any)

    for chart_idx, chart_any in enumerate(chart_entries):
        if not isinstance(chart_any, dict):
            continue
        chart = cast(Dict[str, Any], chart_any)
        for key in ("xAxes", "yAxes"):
            value = chart.get(key)
            if isinstance(value, dict) and not value:
                raise ValueError(
                    f"Invalid AnalysisPlan: charts[{chart_idx}].{key} is an empty object. "
                    "Provide explicit chart semantics."
                )

        series_any = chart.get("series")
        if isinstance(series_any, list) and len(series_any) == 0:
            raise ValueError(
                f"Invalid AnalysisPlan: charts[{chart_idx}].series must contain at least one entry."
            )
        if isinstance(series_any, list):
            series_entries = cast(List[Any], series_any)
            for s_idx, item in enumerate(series_entries):
                if isinstance(item, dict) and not item:
                    raise ValueError(
                        f"Invalid AnalysisPlan: charts[{chart_idx}].series[{s_idx}] is an empty object. "
                        "Provide explicit metric/xAxis/yAxis references."
                    )


def _coerce_analysis_plan(response: Any) -> AnalysisPlan:
    import json

    if isinstance(response, AnalysisPlan):
        return response

    if isinstance(response, dict):
        _assert_no_empty_chart_semantics(cast(Dict[str, Any], response))
        return AnalysisPlan.model_validate(response)

    text = _extract_text(response)
    json_block = _extract_json_block(text)
    parsed = json.loads(json_block)
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    _assert_no_empty_chart_semantics(cast(Dict[str, Any], parsed))
    return AnalysisPlan.model_validate(parsed)


def get_schema_description(model: Type[Any]) -> str:
    """
    Recursively extract field names/types from a Pydantic model as a readable schema spec (Pydantic v2 compatible).
    """
    from typing import get_args, get_origin

    def describe(model: Type[Any], indent: int = 0) -> str:
        lines: List[str] = []
        if hasattr(model, "model_fields"):
            for name, field in model.model_fields.items():
                typ = str(field.annotation)
                lines.append(" " * indent + f"- {name} ({typ})")
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

few_shot_examples = get_few_shot_examples()
_PLAN_CACHE: "OrderedDict[str, tuple[AnalysisPlan, float]]" = OrderedDict()
_PLAN_CACHE_LOCK = Lock()
_plan_cache_hits = 0
_plan_cache_misses = 0
_plan_cache_expired = 0
_PLAN_CACHE_STATS_LOCK = Lock()
_LAST_CACHE_EVENT: ContextVar[Optional[bool]] = ContextVar("planner_last_cache_event", default=None)


def _invoke_with_timeout(chain: Any, inputs: Dict[str, Any], label: str) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(bind_current_context(chain.invoke), inputs)
        try:
            return future.result(timeout=_PLANNER_REQUEST_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise PlannerTimeoutError(f"{label} timed out after {_PLANNER_REQUEST_TIMEOUT_SECONDS:.1f}s") from exc


def _plan_for_cache(plan: AnalysisPlan) -> AnalysisPlan:
    return plan.model_copy(deep=True)


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
        "token_w": _FEWSHOT_TOKEN_WEIGHT,
        "entity_key_w": _FEWSHOT_ENTITY_KEY_WEIGHT,
        "entity_value_w": _FEWSHOT_ENTITY_VALUE_WEIGHT,
        "intent_w": _FEWSHOT_INTENT_WEIGHT,
        "intent_keywords": _INTENT_KEYWORDS,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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

    example_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return cast(Dict[str, Any], parsed)
    except json.JSONDecodeError:
        logger.debug(
            "[Planner] Few-shot example entity JSON parse failed; trying fallback parser",
            extra={
                "log_context": {
                    "event": "planner.example_entities.parse_fallback",
                    "operation": "extract_example_entities",
                    "outcome": "degraded",
                    "error_kind": "json_decode_error",
                    "example_hash": example_hash,
                }
            },
            exc_info=True,
        )

    try:
        parsed_fallback = json.loads(_extract_json_block(payload))
        if isinstance(parsed_fallback, dict):
            return cast(Dict[str, Any], parsed_fallback)
    except (json.JSONDecodeError, ValueError):
        logger.debug(
            "[Planner] Few-shot example entity fallback parse failed; skipping entity extraction",
            extra={
                "log_context": {
                    "event": "planner.example_entities.parse_skipped",
                    "operation": "extract_example_entities",
                    "outcome": "degraded",
                    "error_kind": "fallback_parse_failed",
                    "example_hash": example_hash,
                }
            },
            exc_info=True,
        )
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
    ex_text = example.get("user", "")
    ex_tokens = _tokenize(ex_text)

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
                    "User Message:",
                    ex["user"],
                    "Assistant Plan JSON:",
                    ex["assistant"],
                    "---",
                ]
            )
        )
    return "\n".join(parts)


plan_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            "You are a planner. Interface language: {language}. Produce ONLY a valid AnalysisPlan JSON according to the schema. "
            "Keep enum-like codes (metric, chart_type, test_type, stroke categories, sex categories) in their canonical uppercase English forms. "
            "Use the reasoning and prior examples. Place detected entities into metrics (group_by / filters). "
            "When scope is semantic (my hospital, hospital name, provider-group name, country average, all accessible), prefer metric-level originScope instead of dataOrigin. "
            "Use originScope.scopeType from: mine, provider_name, provider_group_name, country_code, country_average, all_accessible, provider_id, provider_group_id. "
            "Put human references in originScope.value and ISO country in originScope.countryCode when available. "
            "Only use dataOrigin when explicit numeric provider/group IDs are directly provided by the user. "
            "When comparing multiple scopes in one chart, use separate series entries with per-series metric-level originScope/dataOrigin. "
            "Use explicit chart semantics with the new contract: LINE charts require chartType=LINE, xAxes map, yAxes map, and series list with xAxis/yAxis keys; LINE may use a time/category x-axis with metric_value y-axis or a numeric_metric x-axis with count y-axis for a distribution line; HISTOGRAM requires chartType=HISTOGRAM, xAxis numeric metric, yAxis count. "
            "Always emit schemaVersion=2. Do not emit deprecated fields (xAxis object for LINE, yAxes list, seriesBy, summaries). "
            "Sex semantics guidance: phrases like 'males only' or 'females only' should usually be chart filters (SexFilter), while 'split/group by sex' should use GroupBySex. "
            "Chart intent precedence: if the user explicitly asks for a supported chart type (for example LINE or HISTOGRAM), honor it when semantically valid. "
            "Default behavior for KPI metrics: when chart type is not explicitly requested, use HISTOGRAM with numeric_metric xAxis. "
            "For explicit LINE requests for a numeric KPI without categorical grouping, prefer a distribution line: numeric_metric x-axis for the KPI and count y-axis. Use time x-axis with metric_value MEAN only when the user explicitly asks for a trend over time. "
            "Chart intent guidance: If user asks for one graph/one chart/single visual with multiple splits, prefer one LINE chart with multiple series. If user asks for separate charts/multiple visuals, produce multiple chart specs. "
            "Statistical test guidance: Only use test types listed in SUPPORTED_STAT_TESTS_JSON; otherwise omit statistical_tests and return charts.",
        ),
        ("system", "SCHEMA:\n" + SCHEMA_DESCRIPTION),
        ("system", "FEW_SHOT_EXAMPLES:\n{few_shots}"),
        ("system", "SUPPORTED_STAT_TESTS_JSON:\n{supported_stat_tests}"),
        ("system", "REASONING (English internal reasoning shown below can differ from output language):\n{reasoning}"),
        ("user", "USER_UTTERANCE:\n{question}\n\nENTITIES_DETECTED(JSON):\n{entities}"),
    ]
)

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
    Generate a validated AnalysisPlan from user input, retrying with validation feedback on failure.

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

    with log_context(trace_id=trace_id or "", planner_language=language):
        _record_cache_event(None)

        cache_key: Optional[str] = None
        if _ENABLE_PLAN_CACHE and not debug:
            cache_key = _cache_key(question=question, entities=entities, language=language)
            cached_plan = _cache_get(cache_key)
            if cached_plan is not None:
                _record_cache_event(True)
                logger.debug("[Planner] Returning cached analysis plan")
                if progress_cb is not None:
                    progress_cb("Using cached plan.")
                return cached_plan
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
        few_shots_text = _build_few_shots_text(selected_examples)

        logger.debug(
            "[Planner] generate_analysis_plan invoked",
            extra={"log_context": {"question_length": len(question or ""), "entity_count": len(entities or {})}},
        )

        steps: List[Any] = []
        attempts: List[Any] = []

        plan_inputs: Dict[str, Any] = {
            "question": question,
            "entities": json.dumps(entities),
            "reasoning": "",
            "few_shots": few_shots_text,
            "language": language,
            "supported_stat_tests": json.dumps(_SUPPORTED_STAT_TESTS),
        }

        retries = max(0, max_retries)
        total_attempts = retries + 1
        result: Optional[AnalysisPlan] = None
        last_error: Optional[Exception] = None

        for attempt in range(1, total_attempts + 1):
            if progress_cb is not None:
                progress_cb(f"Thinking about a plan (attempt {attempt}/{total_attempts}).")

            raw_result: Any = _invoke_with_timeout(plan_chain, plan_inputs, label=f"plan_chain_attempt_{attempt}")
            steps.append(
                {
                    "step": f"plan_attempt_{attempt}",
                    "prompt": "(prompt logging disabled)",
                    "response": raw_result,
                }
            )

            try:
                result = _coerce_analysis_plan(raw_result)
                result = validate_analysis_plan_semantics(result)
                break
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "ok": False,
                        "error": str(exc),
                    }
                )
                if attempt >= total_attempts:
                    break
                plan_inputs["reasoning"] = f"Previous output failed schema validation. Return ONLY a valid AnalysisPlan JSON object. Validation error: {exc}"

        if result is None:
            if last_error is not None and _is_semantic_ambiguity_error(last_error):
                message = "I need one clarification: which KPI metric should I visualize?"
                raise PlannerIntentAmbiguityError(
                    message
                ) from last_error
            raise ValueError(f"Planner failed to produce a valid AnalysisPlan after {total_attempts} attempts") from last_error

        if debug:
            debug_payload: GeneratePlanDebug = {
                "reasoning": "",
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
