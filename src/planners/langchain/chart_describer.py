"""One short sentence per finished chart, in the user's language, saying what
it shows. The model gets how the chart is built, never its data, so it has no
values to report on."""

import asyncio
import json
import logging
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate

from src.planners.langchain.llm_errors import is_llm_service_error
from src.planners.langchain.llm_factory import create_chat_llm
from src.planners.langchain.prompt_loader import load_prompt_text
from src.planners.langchain.user_terms import language_name, option_names, statistic_name
from src.shared.ssot_loader import get_metric_description, get_metric_name
from src.util.llm_usage import record_llm_call

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 20.0
_MAX_NAMED = 3

_BANDS_NOTE = {
    "en": "coloured bands mark the Angels Awards gold, platinum and diamond thresholds",
    "cs": "barevné pásy označují prahy Angels Awards pro zlato, platinu a diamant",
    "el": "οι χρωματιστές ζώνες δείχνουν τα όρια χρυσού, πλατίνας και διαμαντιού των Angels Awards",
}

_COUNTED_TOGETHER_NOTE = {
    "en": "{options} are counted together as one group",
    "cs": "{options} se počítají dohromady jako jedna skupina",
    "el": "{options} μετρώνται μαζί ως μία ομάδα",
}

_PROMPT = ChatPromptTemplate.from_messages(  # type: ignore[attr-defined]
    [
        ("system", load_prompt_text("describe_system", expected_variables=frozenset({"language"}))),
        ("user", "CHART_FACTS_JSON: {facts}"),
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


def _x_values(chart: Any) -> Dict[str, Any]:
    values: List[Any] = []
    for series in getattr(chart, "series", None) or []:
        for point in series.data:
            if point.x not in values:
                values.append(point.x)
    return {"count": len(values), "first": values[0] if values else None, "last": values[-1] if values else None}


def _names(names: List[Any]) -> Any:
    """A few names as they are; for more, only how many, so the sentence stays short."""
    return names if len(names) <= _MAX_NAMED else {"count": len(names)}


def _metric_facts(metric: Any, language: str, with_meaning: bool) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"name": get_metric_name(metric.metric, language)}
    if with_meaning:
        facts["meaning"] = get_metric_description(metric.metric, language)
    if len(metric.categories) == 1:
        facts["option_counted"] = option_names(metric, language)[0]
    return facts


def chart_facts(chart: Any, language: str) -> Dict[str, Any]:
    """How a finished chart is built, without any of its values."""
    metadata = chart.metadata
    measure = (metadata.statistic or "").upper()
    # A count or share is of cases; the metric's own unit would only mislead.
    with_meaning = measure not in ("COUNT", "RATE", "DISTRIBUTION")
    facts: Dict[str, Any] = {
        "chart_type": str(getattr(chart.type, "value", chart.type)),
        "title": metadata.title,
        "x_axis": metadata.x_axis.label if metadata.x_axis else None,
        "y_axis": metadata.y_axis.label if metadata.y_axis else None,
        "metrics": [_metric_facts(metric, language, with_meaning) for metric in metadata.metrics],
        "statistic": statistic_name(measure, language),
    }
    notes = [
        (_COUNTED_TOGETHER_NOTE.get(language) or _COUNTED_TOGETHER_NOTE["en"]).format(options=", ".join(option_names(metric, language)))
        for metric in metadata.metrics
        if len(metric.categories) > 1
    ]
    if metadata.angels_awards:
        notes.append(_BANDS_NOTE.get(language) or _BANDS_NOTE["en"])
    if notes:
        facts["notes"] = notes
    if getattr(chart, "series", None) is not None:
        facts["series"] = _names([series.name for series in chart.series])
        facts["x_values"] = _x_values(chart)
    else:
        facts["categories"] = _names([getattr(item, "label", None) or getattr(item, "name", None) for item in getattr(chart, "data", None) or []])
    return facts


def _describe_now(chart: Any, language: str) -> Optional[str]:
    inputs = {"language": language_name(language), "facts": json.dumps(chart_facts(chart, language), ensure_ascii=False)}
    started_at = time.monotonic()
    try:
        response = (_PROMPT | _get_llm()).invoke(inputs)
    except Exception:
        record_llm_call("describe", started_at)
        raise
    record_llm_call("describe", started_at, response)
    text = str(getattr(response, "content", "") or "").strip()
    return text or None


async def _describe(chart: Any, language: str) -> Optional[str]:
    try:
        return await asyncio.wait_for(asyncio.to_thread(_describe_now, chart, language), timeout=_TIMEOUT_SECONDS)
    except Exception as exc:
        reason = "llm_unavailable" if is_llm_service_error(exc) else type(exc).__name__
        logger.warning("Chart description failed (%s); the chart is sent without one", reason, exc_info=not is_llm_service_error(exc))
        return None


async def describe_charts(charts: Sequence[Any], language: str) -> List[str]:
    """A description per chart, in chart order; a chart whose description failed has none."""
    descriptions = await asyncio.gather(*(_describe(chart, language) for chart in charts))
    return [text for text in descriptions if text]
