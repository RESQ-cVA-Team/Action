from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Literal, Optional, cast

from langchain_core.prompts import ChatPromptTemplate

from src.domain.langchain.schema import AnalysisPlan, ChartType
from src.planners.langchain.llm_factory import create_chat_llm
from src.planners.langchain.pipeline import generate_analysis_plan
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
_ORCHESTRATOR_TIMEOUT_RAW = env_util.get_env("ACTIONS_LLM_REQUEST_ORCHESTRATOR_TIMEOUT_SECONDS", default="10") or "10"
_orchestrator_timeout_value = 10.0
try:
    _orchestrator_timeout_value = max(1.0, float(_ORCHESTRATOR_TIMEOUT_RAW))
except Exception:
    _orchestrator_timeout_value = 10.0
_ORCHESTRATOR_TIMEOUT_SECONDS = _orchestrator_timeout_value

_ORCHESTRATOR_TEMPERATURE_RAW = env_util.get_env("ACTIONS_LLM_REQUEST_ORCHESTRATOR_TEMPERATURE", default="0") or "0"
_orchestrator_temperature_value = 0.0
try:
    _orchestrator_temperature_value = float(_ORCHESTRATOR_TEMPERATURE_RAW)
except Exception:
    _orchestrator_temperature_value = 0.0
_ORCHESTRATOR_TEMPERATURE = _orchestrator_temperature_value

_ORCHESTRATOR_FAIL_OPEN = env_util.env_flag("ACTIONS_LLM_REQUEST_ORCHESTRATOR_FAIL_OPEN", default=False)
_ASSUME_DEFAULT_TIME_SCOPE = env_util.env_flag("ACTIONS_ASSUME_DEFAULT_TIME_SCOPE", default=True)

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

_DECISION_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            """
You are the request triage stage for a clinical analytics visualization assistant.
Your goal is to decide whether to proceed to plan generation, ask for clarification, or reject.

Return strict JSON only:
{{
  "decision": "proceed" | "clarify" | "reject",
  "reason": "short_snake_case_reason",
  "missing_fields": string[] | null,
  "clarification_type": string | null,
  "clarification_options": string[] | null,
  "reject_message": string | null
}}

Rules:
- Use "proceed" when enough information exists for a useful visualization plan.
- Use "clarify" when required details are missing/ambiguous.
- Use "reject" only when request is out of scope for visualization flow.
- Prefer resolving known metric names and aliases from VALID_METRIC_CANDIDATES_JSON.
- Keep missing_fields concrete (e.g. metric, chart_type, time_scope, time_range, grouping_dimension).
- Do not include markdown or prose outside JSON.
            """.strip(),
        ),
        (
            "user",
            "USER_LANGUAGE: {language}\nUSER_QUESTION: {question}\nCONVERSATION_HISTORY_JSON: {conversation_history_json}\nENTITIES_JSON: {entities_json}\nVALID_METRIC_CANDIDATES_JSON: {metric_candidates_json}\nVALID_CHART_TYPES_JSON: {chart_types_json}",
        ),
    ]
)

_CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore
    [
        (
            "system",
            """
You are the clarification message generator for a clinical analytics visualization assistant.
Given missing or ambiguous fields, ask exactly one concise clarification question.

Return strict JSON only:
{{
  "message": string,
  "clarification_type": string | null,
  "clarification_options": string[] | null
}}

Rules:
- Ask one direct question.
- If possible, provide a short list of options.
- Keep the question under 25 words.
- Do not include markdown or prose outside JSON.
            """.strip(),
        ),
        (
            "user",
            "USER_LANGUAGE: {language}\nUSER_QUESTION: {question}\nCONVERSATION_HISTORY_JSON: {conversation_history_json}\nMISSING_FIELDS_JSON: {missing_fields_json}\nENTITIES_JSON: {entities_json}\nVALID_METRIC_CANDIDATES_JSON: {metric_candidates_json}\nVALID_CHART_TYPES_JSON: {chart_types_json}",
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
            logger.exception("Failed to initialize LLM request orchestrator")
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


def _normalize_missing_field(value: str) -> str:
    token = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _is_temporal_missing_field(value: str) -> bool:
    token = _normalize_missing_field(value)
    if not token:
        return False
    if token in _TEMPORAL_MISSING_FIELDS:
        return True
    return token.startswith("time_") or token.startswith("date_")


def _should_assume_default_time_scope(stage1: VisualizationRequestOutcome) -> bool:
    if stage1.decision != "clarify":
        return False

    fields: List[str] = []
    fields.extend(stage1.missing_fields)
    if stage1.clarification_type:
        fields.append(stage1.clarification_type)

    normalized = [_normalize_missing_field(field) for field in fields if _normalize_missing_field(field)]
    if not normalized:
        return False

    return all(_is_temporal_missing_field(field) for field in normalized)


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
    parsed = _invoke_chain(chain, payload)

    decision_raw = str(parsed.get("decision") or "").strip().lower()
    if decision_raw not in {"proceed", "clarify", "reject"}:
        raise ValueError("Invalid decision from decision stage")

    reason = str(parsed.get("reason") or "").strip() or "llm_orchestrator"
    missing_fields = _coerce_missing_fields(parsed.get("missing_fields"))
    clarification_type = parsed.get("clarification_type")
    clarification_options = _coerce_options(parsed.get("clarification_options"))
    reject_message_raw = parsed.get("reject_message")
    reject_message = reject_message_raw.strip() if isinstance(reject_message_raw, str) and reject_message_raw.strip() else None

    return VisualizationRequestOutcome(
        decision=cast(OutcomeDecision, decision_raw),
        reason=reason,
        message=reject_message,
        clarification_type=clarification_type.strip() if isinstance(clarification_type, str) and clarification_type.strip() else None,
        clarification_options=clarification_options,
        missing_fields=missing_fields,
    )


def _clarification_stage(
    question: str,
    entities: Dict[str, Any],
    missing_fields: List[str],
    language: Optional[str],
    conversation_history: Optional[List[str]] = None,
) -> VisualizationRequestOutcome:
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable")

    chain = _CLARIFICATION_PROMPT | llm
    payload = {
        "language": (language or "en").strip() or "en",
        "question": question or "",
        "entities_json": json.dumps(entities or {}, ensure_ascii=False),
        "missing_fields_json": json.dumps(missing_fields, ensure_ascii=False),
        "metric_candidates_json": json.dumps(_metric_candidates(question or ""), ensure_ascii=False),
        "chart_types_json": json.dumps(_chart_types(), ensure_ascii=False),
        "conversation_history_json": json.dumps(conversation_history or [], ensure_ascii=False),
    }
    parsed = _invoke_chain(chain, payload)

    message_raw = parsed.get("message")
    message = message_raw.strip() if isinstance(message_raw, str) and message_raw.strip() else "I need a bit more detail before I can continue."

    clarification_type_raw = parsed.get("clarification_type")
    clarification_type = clarification_type_raw.strip() if isinstance(clarification_type_raw, str) and clarification_type_raw.strip() else (missing_fields[0] if missing_fields else None)

    clarification_options = _coerce_options(parsed.get("clarification_options"))

    return VisualizationRequestOutcome(
        decision="clarify",
        reason="llm_clarification_required",
        message=message,
        clarification_type=clarification_type,
        clarification_options=clarification_options,
        missing_fields=missing_fields,
    )


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
            plan = generate_analysis_plan(
                question=question,
                entities=entities,
                language=language,
                max_retries=max_retries,
                debug=False,
                trace_id=trace_id,
                progress_cb=progress_cb,
            )
            return VisualizationRequestOutcome(decision="proceed", reason="orchestrator_disabled", plan=plan)

        def report(message: str) -> None:
            if progress_cb is not None:
                progress_cb(message)

        try:
            report("Analyzing request intent and feasibility")
            stage1 = _decision_stage(question, entities, language, conversation_history=conversation_history)

            if stage1.decision == "reject":
                if stage1.message is None:
                    stage1.message = "This request is outside the visualization flow."
                return stage1

            if stage1.decision == "clarify":
                if _ASSUME_DEFAULT_TIME_SCOPE and _should_assume_default_time_scope(stage1):
                    logger.debug("Assuming default time scope; skipping clarification")
                    return VisualizationRequestOutcome(
                        decision="proceed",
                        reason="default_time_scope_assumed",
                    )

                report("Generating clarification question")
                return _clarification_stage(
                    question=question,
                    entities=entities,
                    missing_fields=stage1.missing_fields,
                    language=language,
                    conversation_history=conversation_history,
                )

            if not include_plan:
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason=stage1.reason or "sufficient_information",
                )

            report("Generating visualization plan")
            planner_question = question
            if conversation_history:
                cleaned_history = [item.strip() for item in conversation_history if item.strip()]
                if cleaned_history:
                    joined = "\n".join(f"- {item}" for item in cleaned_history)
                    planner_question = f"Conversation context (oldest to newest user turns):\n{joined}\n\nCurrent request to fulfill:\n{question}"

            plan = generate_analysis_plan(
                question=planner_question,
                entities=entities,
                language=language,
                max_retries=max_retries,
                debug=False,
                trace_id=trace_id,
                progress_cb=progress_cb,
            )
            return VisualizationRequestOutcome(
                decision="proceed",
                reason=stage1.reason or "sufficient_information",
                plan=plan,
            )
        except Exception:
            logger.exception("Visualization request orchestration failed")
            if _ORCHESTRATOR_FAIL_OPEN and include_plan:
                report("Orchestration fallback: generating plan directly")
                plan = generate_analysis_plan(
                    question=question,
                    entities=entities,
                    language=language,
                    max_retries=max_retries,
                    debug=False,
                    trace_id=trace_id,
                    progress_cb=progress_cb,
                )
                return VisualizationRequestOutcome(
                    decision="proceed",
                    reason="orchestrator_fallback_to_plan",
                    plan=plan,
                )

            return VisualizationRequestOutcome(
                decision="clarify",
                reason="orchestrator_failed",
                message="I need a bit more detail before I can continue.",
            )
