"""SSOT-backed metric defaults for distribution bucketing and axis labelling.

This module is the single source of truth for deriving distribution parameters
and histogram axis metadata from SSOT metric records.  It is called directly by
the metric-request factory so that orchestration never needs to inject axis/bucket
knowledge as callbacks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, cast

from src.domain.dto.charts.types import ChartAxis
from src.shared.ssot_loader import get_metric_display_name, get_metric_metadata

logger = logging.getLogger(__name__)

_METRIC_METADATA: Dict[str, Any] = get_metric_metadata()

_AXIS_ACRONYMS = {"NIHSS", "DTN", "IVT", "EVT", "TIA", "LVO", "ICH", "SAH", "CT", "MRI"}
_NICE_BIN_BASES: tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0)
_IMPLICIT_DISTRIBUTION_TARGET_BINS = 20
_IMPLICIT_DISTRIBUTION_MIN_BINS = 14
_IMPLICIT_DISTRIBUTION_MAX_BINS = 26
_MINUTES_BIN_WIDTH = 5


@dataclass(frozen=True)
class PrettyDistributionBins:
    lower_bound: float
    upper_bound: float
    bin_width: float
    bin_count: int


def _format_decimal_label(value: float) -> str:
    rounded_value = round(float(value), 9)
    if math.isclose(rounded_value, round(rounded_value), abs_tol=1e-9):
        rounded_value = float(round(rounded_value))
    decimal_value = Decimal(str(rounded_value))
    normalized = decimal_value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def _round_down_to_step(value: int, step: int) -> int:
    if step <= 0:
        return value
    return int(math.floor(value / step) * step)


def _round_up_to_step(value: int, step: int) -> int:
    if step <= 0:
        return value
    return int(math.ceil(value / step) * step)


def _nice_bin_width_candidates(raw_width: float) -> list[float]:
    if raw_width <= 0:
        return [1.0]

    exponent = int(math.floor(math.log10(raw_width)))
    candidates: set[float] = set()
    for power in range(exponent - 1, exponent + 3):
        scale = 10.0**power
        for base in _NICE_BIN_BASES:
            candidates.add(base * scale)
    return sorted(width for width in candidates if width > 0)


def _nearest_nice_width_distance(width: float) -> float:
    if width <= 0:
        return float("inf")
    candidates = _nice_bin_width_candidates(width)
    return min(abs(math.log10(width / candidate)) for candidate in candidates)


def _score_pretty_bin_layout(
    lower_bound: float,
    upper_bound: float,
    span: float,
    raw_width: float,
    target_bins: int,
    minimum_bins: int,
    maximum_bins: int,
    candidate_width: float,
) -> tuple[float, float, float, float, float]:
    if candidate_width <= 0:
        return (float("inf"), float("inf"), float("inf"), float("inf"), float("inf"))

    exact_bins = span / candidate_width if span > 0 else float(target_bins)
    rounded_bins = max(1, int(round(exact_bins)))
    bucket_penalty = 0.0
    if rounded_bins < minimum_bins:
        bucket_penalty = float(minimum_bins - rounded_bins)
    elif rounded_bins > maximum_bins:
        bucket_penalty = float(rounded_bins - maximum_bins)

    target_error = abs(rounded_bins - target_bins)
    snapped_lower = math.floor(lower_bound / candidate_width) * candidate_width
    snapped_upper = math.ceil(upper_bound / candidate_width) * candidate_width
    expansion_ratio = ((snapped_upper - snapped_lower) - span) / span if span > 0 else 0.0
    divisibility_error = abs(exact_bins - rounded_bins)
    width_distance = abs(math.log10(candidate_width / raw_width)) if raw_width > 0 else 0.0
    return (bucket_penalty, target_error, expansion_ratio, divisibility_error, width_distance)


def resolve_pretty_distribution_bins(
    lower_bound: float,
    upper_bound: float,
    target_bins: int,
    minimum_bins: int = 6,
    maximum_bins: int = 16,
) -> PrettyDistributionBins:
    """Return a human-readable bin layout for an implicit numeric distribution.

    The layout aims for a steady target bucket count while preferring widths
    that divide the selected span cleanly and produce stable labels like
    "30-40" or "0-2.5" instead of floating-point artifacts.
    """
    resolved_lower = float(lower_bound)
    resolved_upper = float(upper_bound)
    if resolved_lower > resolved_upper:
        resolved_lower, resolved_upper = resolved_upper, resolved_lower

    span = resolved_upper - resolved_lower
    if span <= 0:
        return PrettyDistributionBins(
            lower_bound=resolved_lower,
            upper_bound=resolved_upper,
            bin_width=1.0,
            bin_count=1,
        )

    effective_minimum = max(1, minimum_bins)
    effective_maximum = max(effective_minimum, maximum_bins)
    effective_target = min(max(1, target_bins), effective_maximum)
    effective_target = max(effective_target, effective_minimum)

    raw_width = span / float(effective_target)
    candidates = _nice_bin_width_candidates(raw_width)
    best_width = min(
        candidates,
        key=lambda width: _score_pretty_bin_layout(
            lower_bound=resolved_lower,
            upper_bound=resolved_upper,
            span=span,
            raw_width=raw_width,
            target_bins=effective_target,
            minimum_bins=effective_minimum,
            maximum_bins=effective_maximum,
            candidate_width=width,
        ),
    )

    snapped_lower = math.floor(resolved_lower / best_width) * best_width
    snapped_upper = math.ceil(resolved_upper / best_width) * best_width
    snapped_span = max(best_width, snapped_upper - snapped_lower)
    bin_count = max(1, int(round(snapped_span / best_width)))
    snapped_upper = snapped_lower + (bin_count * best_width)

    return PrettyDistributionBins(
        lower_bound=snapped_lower,
        upper_bound=snapped_upper,
        bin_width=best_width,
        bin_count=bin_count,
    )


def format_distribution_bin_label(start: float, end: float) -> str:
    """Return a stable, human-readable display label for one numeric bin."""
    try:
        start_decimal = Decimal(str(start))
        end_decimal = Decimal(str(end))
    except InvalidOperation as exc:
        raise ValueError("Distribution bin labels require numeric bounds") from exc
    return f"{_format_decimal_label(float(start_decimal))}-{_format_decimal_label(float(end_decimal))}"


def resolve_pretty_distribution_bin_count(
    lower_bound: int,
    upper_bound: int,
    target_bins: int,
    minimum_bins: int = 6,
    maximum_bins: int = 16,
) -> int:
    """Return an implicit bin count that keeps the span on readable increments."""
    resolved_lower = int(lower_bound)
    resolved_upper = int(upper_bound)
    if resolved_lower > resolved_upper:
        resolved_lower, resolved_upper = resolved_upper, resolved_lower

    span = max(0, resolved_upper - resolved_lower)
    if span <= 0:
        return 1

    effective_minimum = max(1, minimum_bins)
    effective_maximum = max(effective_minimum, maximum_bins)
    effective_target = min(max(1, target_bins), effective_maximum)
    effective_target = max(effective_target, effective_minimum)

    return min(
        range(effective_minimum, effective_maximum + 1),
        key=lambda count: (
            _nearest_nice_width_distance(span / float(count)),
            abs(count - effective_target),
            abs((span / float(count)) - (span / float(effective_target))),
        ),
    )


def resolve_implicit_distribution_layout(
    lower_bound: int,
    upper_bound: int,
    target_bins: int = _IMPLICIT_DISTRIBUTION_TARGET_BINS,
) -> PrettyDistributionBins:
    """Return the snapped default layout used for implicit numeric distributions."""
    return resolve_pretty_distribution_bins(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        target_bins=target_bins,
        minimum_bins=_IMPLICIT_DISTRIBUTION_MIN_BINS,
        maximum_bins=_IMPLICIT_DISTRIBUTION_MAX_BINS,
    )


def resolve_score_distribution_layout(
    lower_bound: int,
    upper_bound: int,
) -> PrettyDistributionBins:
    """Return a one-point-per-score layout for discrete score metrics."""
    resolved_lower = int(lower_bound)
    resolved_upper = int(upper_bound)
    if resolved_lower > resolved_upper:
        resolved_lower, resolved_upper = resolved_upper, resolved_lower

    bin_count = max(1, (resolved_upper - resolved_lower) + 1)
    return PrettyDistributionBins(
        lower_bound=float(resolved_lower),
        upper_bound=float(resolved_lower + bin_count),
        bin_width=1.0,
        bin_count=bin_count,
    )


def build_distribution_bin_ranges(bin_starts: list[object]) -> list[tuple[float, float]]:
    """Return stable numeric [start, end) ranges from backend histogram edges."""
    if not bin_starts:
        return []

    starts: list[float] = []
    for value in bin_starts:
        try:
            starts.append(float(str(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected numeric chart value, got {value!r}") from exc

    if len(starts) == 1:
        return [(starts[0], starts[0])]

    positive_deltas = [current - previous for previous, current in zip(starts, starts[1:]) if current > previous]
    inferred_width = positive_deltas[-1] if positive_deltas else 0.0

    ranges: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1]
        else:
            end = start + inferred_width if inferred_width > 0 else start
        ranges.append((start, end))
    return ranges


def format_distribution_point_label(metric_code: str, start: float, end: float) -> str:
    """Return a metric-aware label for a numeric distribution point."""
    if is_score_metric(metric_code) and math.isclose(end - start, 1.0, abs_tol=1e-9):
        return _format_decimal_label(start)
    return format_distribution_bin_label(start, end)


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(Dict[str, Any], value)


def _normalize_axis_display_label(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    def _word_case(word: str) -> str:
        token = word.strip()
        if not token:
            return token
        if token.upper() in _AXIS_ACRONYMS:
            return token.upper()
        if token.isupper() and len(token) <= 4:
            return token
        if token[:1].isdigit():
            return token
        return token[:1].upper() + token[1:].lower()

    out_words: list[str] = []
    for word in text.split():
        if "-" in word:
            out_words.append("-".join(_word_case(p) for p in word.split("-")))
        else:
            out_words.append(_word_case(word))
    return " ".join(out_words)


def get_enum_labels(metric_code: str) -> Optional[list[str]]:
    """Return SSOT display labels for an Enum metric's categories, in property order.

    Used as a fallback when the backend's own `labels` field is null — some
    boolean-shaped Enum metrics (e.g. WAKEUP_STROKE) return a single caseCount
    with no labels array, rather than one entry per category.
    """
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))
    labels = meta.get("labels")
    if isinstance(labels, list) and labels:
        typed_labels = cast(list[Any], labels)
        return [str(label) for label in typed_labels]
    return None


def is_score_metric(metric_code: str) -> bool:
    """Return True when the metric's SSOT numeric unit is a discrete score."""
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))
    numeric_block = _mapping_to_dict(meta.get("numeric"))
    unit = meta.get("unit") or numeric_block.get("unit")
    return str(unit or "").strip().lower() == "score"


def is_minutes_metric(metric_code: str) -> bool:
    """Return True when the metric's SSOT numeric unit is minutes."""
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))
    numeric_block = _mapping_to_dict(meta.get("numeric"))
    unit = meta.get("unit") or numeric_block.get("unit")
    return str(unit or "").strip().lower() == "minutes"


def is_enum_metric(metric_code: str) -> bool:
    """Return True if the metric's SSOT data_type is Enum (categorical, not numeric).

    Enumeration-type metrics (e.g. SEX, HOSPITALIZED_IN) reject the backend's
    numeric kpi(kpiOptions/distribution) query shape; they must be requested
    via the bare kpi + labels shape instead (see MetricRequest.with_categorical).
    """
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))
    return str(meta.get("data_type") or "").strip().lower() == "enum"


def get_distribution_defaults(metric_code: str) -> tuple[int, int, int]:
    """Return (bins, min_value, max_value) from SSOT metadata.

    Falls back to safe per-metric ranges, then to a general default of (20, 0, 200).
    All parameters are derived from data; no runtime inference is performed.
    """
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))

    bins_any: Any = meta.get("distribution_default_buckets")
    numeric_block = _mapping_to_dict(meta.get("numeric"))
    bins = bins_any or numeric_block.get("default_buckets") or 20

    rmin: Any = meta.get("range_min")
    rmax: Any = meta.get("range_max")
    if rmin is None or rmax is None:
        rmin = rmin if rmin is not None else numeric_block.get("range_min")
        rmax = rmax if rmax is not None else numeric_block.get("range_max")

    if rmin is None or rmax is None:
        known_ranges: dict[str, tuple[int, int]] = {
            "AGE": (18, 95),
            "ADMISSION_NIHSS": (0, 42),
            "DTN": (0, 120),
        }
        if code in known_ranges:
            rmin, rmax = known_ranges[code]
        else:
            rmin = rmin if rmin is not None else 0
            rmax = rmax if rmax is not None else 200

    try:
        bins = int(bins)
    except Exception:
        logger.debug("[ssot_metric_defaults] Could not parse bin count for %s; using 20", code)
        bins = 20
    try:
        rmin = int(rmin)
        rmax = int(rmax)
    except Exception:
        logger.debug("[ssot_metric_defaults] Could not parse range for %s; using 0-200", code)
        rmin, rmax = 0, 200

    if rmin > rmax:
        rmin, rmax = rmax, rmin

    if is_minutes_metric(code):
        rmax = _round_up_to_step(rmax, _MINUTES_BIN_WIDTH)

    return bins, rmin, rmax


def resolve_minutes_distribution_layout(
    lower_bound: int,
    upper_bound: int,
) -> PrettyDistributionBins:
    """Return a fixed-width 5-minute layout for minute-unit metrics."""
    resolved_lower = int(lower_bound)
    resolved_upper = int(upper_bound)
    if resolved_lower > resolved_upper:
        resolved_lower, resolved_upper = resolved_upper, resolved_lower

    snapped_lower = _round_down_to_step(resolved_lower, _MINUTES_BIN_WIDTH)
    snapped_upper = _round_up_to_step(resolved_upper, _MINUTES_BIN_WIDTH)
    if snapped_upper <= snapped_lower:
        snapped_upper = snapped_lower + _MINUTES_BIN_WIDTH

    bin_count = max(1, int((snapped_upper - snapped_lower) / _MINUTES_BIN_WIDTH))
    return PrettyDistributionBins(
        lower_bound=float(snapped_lower),
        upper_bound=float(snapped_upper),
        bin_width=float(_MINUTES_BIN_WIDTH),
        bin_count=bin_count,
    )


def get_histogram_axes(
    metric_code: str, x_min: int, x_max: int
) -> tuple[ChartAxis, ChartAxis]:
    """Return (x_axis, y_axis) for a histogram chart from SSOT metadata."""
    code = (metric_code or "").upper()
    meta = _mapping_to_dict(_METRIC_METADATA.get(code))

    display = _normalize_axis_display_label(get_metric_display_name(code))

    unit_any: Any = meta.get("unit") or _mapping_to_dict(meta.get("numeric")).get("unit")
    unit: Optional[str] = cast(Optional[str], unit_any)

    x_label = f"{display} ({unit})" if unit else display
    return ChartAxis(label=x_label, min_value=x_min, max_value=x_max), ChartAxis(label="Cases")
