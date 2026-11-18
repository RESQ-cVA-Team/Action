"""One sentence per value, in the user's language. The model writes it from
the computed number; it is sent only if the value appears in it unchanged and
every other number in it is one of the facts', else a fixed sentence is."""

import asyncio
import json
import logging
import re
import time
from datetime import date
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate

from src.planners.langchain.llm_errors import is_llm_service_error
from src.planners.langchain.llm_factory import create_chat_llm
from src.planners.langchain.prompt_loader import load_prompt_text
from src.planners.langchain.user_terms import language_name, option_names, statistic_name
from src.shared.ssot_loader import get_metric_name, get_metric_unit
from src.util.llm_usage import record_llm_call

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 20.0
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# Units shown with a value; a score or a count is a plain number.
_UNITS = {
    "minutes": {"en": "minutes", "cs": "minut", "el": "λεπτά"},
    "years": {"en": "years", "cs": "let", "el": "έτη"},
    "days": {"en": "days", "cs": "dní", "el": "ημέρες"},
    "ml": {"en": "ml"},
    "mmHg": {"en": "mmHg"},
    "mg/dl": {"en": "mg/dl"},
    "mg": {"en": "mg"},
}

_FIXED = {
    "en": {"answer": "{statistic} – {subject}, {context}: {value} (patients: {patients}).", "no_cases": "{statistic} – {subject}, {context}: no cases."},
    "cs": {"answer": "{statistic} – {subject}, {context}: {value} (počet pacientů: {patients}).", "no_cases": "{statistic} – {subject}, {context}: žádné případy."},
    "el": {"answer": "{statistic} – {subject}, {context}: {value} (ασθενείς: {patients}).", "no_cases": "{statistic} – {subject}, {context}: κανένα περιστατικό."},
}

_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore[attr-defined]
    [
        ("system", load_prompt_text("value_system", expected_variables=frozenset({"language"}))),
        ("user", "VALUE_FACTS_JSON: {facts}"),
    ]
)

_llm: Optional[Any] = None
_llm_lock = Lock()


def _get_llm() -> Any:
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = create_chat_llm(temperature=0)
        return _llm


def format_number(number: float, statistic: str, language: str) -> str:
    """A count as a whole number; anything else to one decimal, without a
    trailing .0, with the language's decimal separator."""
    text = str(int(round(number))) if statistic == "COUNT" else f"{number:.1f}".removesuffix(".0")
    return text.replace(".", ",") if language in ("cs", "el") else text


def period_label(start: str, end: str) -> str:
    """A calendar year as 2025, a quarter as 2026-Q1, a month as 2026-03, any other span as its dates."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    next_day = date.fromordinal(last.toordinal() + 1)
    if first.day == 1 and next_day.day == 1:
        months = (next_day.year - first.year) * 12 + next_day.month - first.month
        if months == 12 and first.month == 1:
            return str(first.year)
        if months == 3 and first.month % 3 == 1:
            return f"{first.year}-Q{(first.month - 1) // 3 + 1}"
        if months == 1:
            return f"{first.year}-{first.month:02d}"
    return f"{start} – {end}"


def _unit(value: Any, language: str) -> Optional[str]:
    if value.statistic == "RATE":
        return "%"
    if value.statistic == "COUNT":
        return None
    names = _UNITS.get(get_metric_unit(value.metric.metric) or "") or {}
    return names.get(language) or names.get("en")


def _subject(value: Any, language: str) -> str:
    options = option_names(value.metric, language)
    name = get_metric_name(value.metric.metric, language)
    if len(options) == 1:
        return options[0]
    return f"{name} ({', '.join(options)})" if options else name


def value_facts(value: Any, language: str) -> Dict[str, Any]:
    unit = _unit(value, language)
    facts: Dict[str, Any] = {
        "metric": get_metric_name(value.metric.metric, language),
        "statistic": statistic_name(value.statistic, language),
        "value": format_number(value.value, value.statistic, language),
        "unit": unit,
        "patients": value.patients,
        "period": period_label(value.start_date, value.end_date),
    }
    options = option_names(value.metric, language)
    if options:
        facts["options_counted_together"] = options
    if value.scope:
        facts["scope"] = value.scope
    if value.filters:
        facts["filters"] = value.filters
    return {key: item for key, item in facts.items() if item is not None}


def fixed_sentence(value: Any, language: str) -> str:
    texts = _FIXED.get(language) or _FIXED["en"]
    context = ", ".join(part for part in (period_label(value.start_date, value.end_date), value.scope, value.filters) if part)
    statistic = str(statistic_name(value.statistic, language))
    params = {"statistic": statistic[:1].upper() + statistic[1:], "subject": _subject(value, language), "context": context}
    if value.value is None:
        return texts["no_cases"].format(**params)
    unit = _unit(value, language)
    number = format_number(value.value, value.statistic, language)
    return texts["answer"].format(**params, value=f"{number} {unit}" if unit else number, patients=value.patients)


def sentence_keeps_the_numbers(sentence: str, facts: Dict[str, Any]) -> bool:
    found = _NUMBER.findall(sentence)
    allowed = {facts["value"], str(facts.get("patients"))}
    for text in (facts.get("period"), facts.get("scope"), facts.get("filters")):
        for number in _NUMBER.findall(str(text or "")):
            allowed.update({number, number.lstrip("0") or "0"})
    return facts["value"] in found and all(number in allowed for number in found)


def _write_now(value: Any, language: str) -> Optional[str]:
    facts = value_facts(value, language)
    inputs = {"language": language_name(language), "facts": json.dumps(facts, ensure_ascii=False)}
    started_at = time.monotonic()
    try:
        response = (_PROMPT | _get_llm()).invoke(inputs)
    except Exception:
        record_llm_call("value_answer", started_at)
        raise
    record_llm_call("value_answer", started_at, response)
    sentence = str(getattr(response, "content", "") or "").strip()
    if sentence and sentence_keeps_the_numbers(sentence, facts):
        return sentence
    logger.warning("Value answer changed or added a number; the fixed sentence is sent instead")
    return None


async def _answer(value: Any, language: str) -> str:
    if value.value is None:
        return fixed_sentence(value, language)
    try:
        sentence = await asyncio.wait_for(asyncio.to_thread(_write_now, value, language), timeout=_TIMEOUT_SECONDS)
    except Exception as exc:
        reason = "llm_unavailable" if is_llm_service_error(exc) else type(exc).__name__
        logger.warning("Value answer failed (%s); the fixed sentence is sent instead", reason, exc_info=not is_llm_service_error(exc))
        sentence = None
    return sentence or fixed_sentence(value, language)


async def answer_values(values: Sequence[Any], language: str) -> List[str]:
    """A sentence per value, in value order."""
    return list(await asyncio.gather(*(_answer(value, language) for value in values)))
