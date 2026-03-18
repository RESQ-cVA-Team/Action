from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, cast

from src.domain.langchain.schema import AnalysisPlan, ChartSpec, GroupBySex, GroupByStrokeType, GroupByTime, MetricSpec, TimeWindow
from src.shared.ssot_loader import get_metric_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeuristicPlannerResult:
    plan: Optional[AnalysisPlan]
    confidence: float
    reason: str


# Build a lightweight synonym index for metrics from SSOT so we can
# recognise more than just DTN in simple queries.
_METRIC_META: Dict[str, Dict[str, Any]] = get_metric_metadata()


def _build_metric_synonym_index() -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = {}
    for code, meta in _METRIC_META.items():
        code_up = (code or "").upper()
        names: Set[str] = set()

        # Canonical code as token.
        if code_up:
            names.add(code_up.lower())

        # Display name and synonyms.
        display = meta.get("display_name")
        if isinstance(display, str) and display.strip():
            names.add(display.strip().lower())
        syn_any = meta.get("synonyms")
        if isinstance(syn_any, list):
            syn_list = cast(List[Any], syn_any)
            for s in syn_list:
                if isinstance(s, str) and s.strip():
                    names.add(s.strip().lower())

        if names:
            index[code_up] = names
    return index


_METRIC_SYNONYMS: Dict[str, Set[str]] = _build_metric_synonym_index()


def _tokenise(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _find_metric_from_text(q_lower: str) -> Optional[str]:
    tokens = _tokenise(q_lower)
    candidates: Set[str] = set()

    for code, names in _METRIC_SYNONYMS.items():
        for name in names:
            if not name:
                continue
            if " " in name:
                # Multi-word synonym: simple substring match.
                if name in q_lower:
                    candidates.add(code)
                    break
            else:
                # Single token: require token match to avoid spurious substring hits.
                if name in tokens:
                    candidates.add(code)
                    break

    if len(candidates) == 1:
        return next(iter(candidates))
    return None


class HeuristicVisualizationPlanner:
    """Lightweight, non-LLM planner for simple visualization requests.

    This is intentionally conservative: it only handles very simple patterns
    (e.g. "show a line chart of DTN"). When it cannot confidently interpret
    the request, it returns None so the caller can fall back to the LangChain
    planner.
    """

    @staticmethod
    def try_plan_with_diagnostics(question: str, entities: Dict[str, Any], language: Optional[str]) -> HeuristicPlannerResult:
        text = (question or "").strip()
        if not text:
            return HeuristicPlannerResult(plan=None, confidence=0.0, reason="empty_question")

        q_lower = text.lower()

        complexity_markers = [" vs ", " versus ", " compare ", " correlation", " impact ", " per "]
        if any(marker in q_lower for marker in complexity_markers):
            return HeuristicPlannerResult(plan=None, confidence=0.0, reason="complex_query_marker")

        chart_type = "LINE"
        chart_confidence = 0.55
        if "bar chart" in q_lower or "bar graph" in q_lower:
            chart_type = "BAR"
            chart_confidence = 0.9
        elif "area chart" in q_lower or "area graph" in q_lower:
            chart_type = "AREA"
            chart_confidence = 0.9
        elif "histogram" in q_lower or "distribution" in q_lower:
            chart_type = "HISTOGRAM"
            chart_confidence = 0.8
        elif "line chart" in q_lower or "line graph" in q_lower or "trend" in q_lower:
            chart_type = "LINE"
            chart_confidence = 0.9

        metric_code: Optional[str] = None
        metric_confidence = 0.0

        metric_entity_keys = ["metric", "metric_code", "metric_type"]
        for key in metric_entity_keys:
            value = entities.get(key)
            if isinstance(value, str) and value.strip():
                metric_code = value.strip().upper()
                metric_confidence = 1.0
                break

        if metric_code is None:
            metric_code = _find_metric_from_text(q_lower)
            if metric_code is not None:
                metric_confidence = 0.75

        if metric_code is None:
            return HeuristicPlannerResult(plan=None, confidence=0.0, reason="metric_not_confident")

        group_by: list[Any] = []
        groupby_score_parts: List[float] = []

        if "by sex" in q_lower or "by gender" in q_lower:
            try:
                group_by.append(GroupBySex())
                groupby_score_parts.append(1.0)
            except Exception:
                pass

        if "by stroke type" in q_lower or "by stroke" in q_lower:
            try:
                group_by.append(GroupByStrokeType())
                groupby_score_parts.append(1.0)
            except Exception:
                pass

        if "over time" in q_lower or "time series" in q_lower or "over the last" in q_lower:
            try:
                window = TimeWindow(last_n=6, unit="MONTH")
                group_by.append(GroupByTime(grain="MONTH", window=window, include_partial=True))
                groupby_score_parts.append(0.85)
            except Exception:
                pass

        if " by " in q_lower and not group_by:
            return HeuristicPlannerResult(plan=None, confidence=0.0, reason="unrecognized_groupby")

        groupby_confidence = min(groupby_score_parts) if groupby_score_parts else 1.0

        try:
            metric_spec = MetricSpec(metric=metric_code)
            chart_spec = ChartSpec(chart_type=chart_type, metrics=[metric_spec], group_by=group_by or None)
            plan = AnalysisPlan(charts=[chart_spec], statistical_tests=None)
        except Exception as exc:
            logger.debug("HeuristicVisualizationPlanner failed to build plan for question %r: %s", question, exc)
            return HeuristicPlannerResult(plan=None, confidence=0.0, reason="plan_build_failed")

        confidence = round(metric_confidence * 0.6 + chart_confidence * 0.2 + groupby_confidence * 0.2, 3)
        logger.info(
            "HeuristicVisualizationPlanner produced a simple plan for question %r (metric=%s, chart_type=%s, group_by=%s, confidence=%.3f)",
            question,
            metric_code,
            chart_type,
            group_by,
            confidence,
        )
        return HeuristicPlannerResult(plan=plan, confidence=confidence, reason="heuristic_plan_ready")

    @staticmethod
    def try_plan(question: str, entities: Dict[str, Any], language: Optional[str]) -> Optional[AnalysisPlan]:
        return HeuristicVisualizationPlanner.try_plan_with_diagnostics(
            question=question,
            entities=entities,
            language=language,
        ).plan
