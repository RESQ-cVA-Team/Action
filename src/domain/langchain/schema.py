from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.domain.graphql.ssot_enums import (
    BooleanPropertyType as BooleanType,
)
from src.domain.graphql.ssot_enums import (
    GroupByType as CanonicalGroupByField,
)
from src.domain.graphql.ssot_enums import (
    MetricType,
    SexType,
    StrokeType,
)
from src.domain.graphql.ssot_enums import (
    Operator as OperatorType,
)


def _deep_freeze(value: Any) -> Any:
    """Recursively convert dict/list/set structures into hashable tuples.

    Ensures a stable, order-independent representation for dictionaries by
    sorting keys, while preserving list order (which aligns with equality semantics).
    """
    if isinstance(value, dict):
        mapping: Dict[str, Any] = cast(Dict[str, Any], value)
        return tuple((k, _deep_freeze(v)) for k, v in sorted(mapping.items(), key=lambda kv: kv[0]))
    if isinstance(value, list):
        seq: List[Any] = cast(List[Any], value)
        return tuple(_deep_freeze(v) for v in seq)
    if isinstance(value, set):
        s: set[Any] = cast(set[Any], value)
        return tuple(sorted((_deep_freeze(v) for v in s), key=lambda x: str(x)))
    return value


class HashableBaseModel(BaseModel):
    """BaseModel with a content-derived hash compatible with Pydantic equality.

    - Does not freeze/lock instances (avoids breaking existing mutations).
    - Hash is computed from a normalized dump of the model (mode='json').
    """

    def __hash__(self) -> int:  # type: ignore[override]
        data = self.model_dump(mode="json")
        return hash(_deep_freeze(data))


def _enum_allowed_values(enum_cls: Any) -> Set[str]:
    """Return a set of canonical string values for a dynamic Enum.

    Works with str-subclass Enums created via SSOT loader; avoids EnumMeta __contains__ pitfalls.
    """
    try:
        members = list(enum_cls)
    except Exception:
        return set()

    allowed_values: Set[str] = set()
    for member in members:
        allowed_values.add(str(getattr(member, "value", member)))
    return allowed_values


def _extract_canonical(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        cast_entry: Dict[str, Any] = cast(Dict[str, Any], entry)
        val = cast_entry.get("canonical")
        if isinstance(val, str):
            return val
    return None


def _load_chart_or_test_enum(filename: str) -> List[str]:
    path = Path(__file__).resolve().parents[2] / "shared" / "SSOT" / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw_any: Any = yaml.safe_load(f)
    if not isinstance(raw_any, list):
        return []
    out: List[str] = []
    for entry in cast(List[Any], raw_any):
        canonical = _extract_canonical(entry)
        if canonical:
            out.append(canonical)
    return out


ChartType = _load_chart_or_test_enum("ChartType.yml")
StatisticalTestType = _load_chart_or_test_enum("StatisticalTestType.yml")


class DateFilter(BaseModel):
    """
    Filter for date fields using an operator and an ISO 8601 date string.

    Attributes:
        operator: The comparison operator (e.g., 'GE', 'LE', etc.), must be in OperatorType.
        value: The date value to compare, as an ISO 8601 string.
    """

    operator: str
    type: Literal["DateFilter"] = "DateFilter"
    value: str  # ISO 8601 date string

    @field_validator("operator")
    def validate_operator_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(OperatorType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid OperatorType. Allowed: {sorted(allowed)}")
        return v_norm

    @field_validator("value")
    def validate_date_value(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid ISO 8601 date or datetime string.")
        return v


class AgeFilter(BaseModel):
    """
    Filter for patient age using an operator and a numeric value.

    Attributes:
        operator: The comparison operator (e.g., 'GE', 'LE', etc.), must be in OperatorType.
        value: The age value to compare (float).
    """

    operator: str
    type: Literal["AgeFilter"] = "AgeFilter"
    value: float

    @field_validator("operator")
    def validate_operator_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(OperatorType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid OperatorType. Allowed: {sorted(allowed)}")
        return v_norm


class NIHSSFilter(BaseModel):
    """
    Filter for NIHSS score using an operator and a numeric value.

    Attributes:
        operator: The comparison operator (e.g., 'GE', 'LE', etc.), must be in OperatorType.
        value: The NIHSS score to compare (float).
    """

    operator: str
    type: Literal["NIHSSFilter"] = "NIHSSFilter"
    value: float

    @field_validator("operator")
    def validate_operator_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(OperatorType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid OperatorType. Allowed: {sorted(allowed)}")
        return v_norm


class AndFilter(BaseModel):
    """
    Logical AND of multiple filter nodes.

    Attributes:
        and_: List of filter nodes to combine with AND logic.
    """

    type: Literal["AndFilter"] = "AndFilter"
    and_: List["FilterNode"]


class OrFilter(BaseModel):
    """
    Logical OR of multiple filter nodes.

    Attributes:
        or_: List of filter nodes to combine with OR logic.
    """

    type: Literal["OrFilter"] = "OrFilter"
    or_: List["FilterNode"]


class NotFilter(BaseModel):
    """
    Logical NOT of a filter node.

    Attributes:
        not_: The filter node to negate.
    """

    type: Literal["NotFilter"] = "NotFilter"
    not_: "FilterNode"


class SexFilter(BaseModel):
    """
    Filter for patient sex.

    Attributes:
        value: The sex value to filter by (must be in SexType).
    """

    type: Literal["SexFilter"] = "SexFilter"
    value: str  # Should be a value from SexType

    @field_validator("value")
    def validate_sex_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(SexType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid SexType. Allowed: {sorted(allowed)}")
        return v_norm


class StrokeFilter(BaseModel):
    """
    Filter for stroke type.

    Attributes:
        value: The stroke type to filter by (must be in StrokeType).
    """

    type: Literal["StrokeFilter"] = "StrokeFilter"
    value: str  # Should be a value from StrokeType

    @field_validator("value")
    def validate_stroke_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(StrokeType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid StrokeType. Allowed: {sorted(allowed)}")
        return v_norm


class BooleanFilter(BaseModel):
    """
    Filter for boolean fields.

    Attributes:
        boolean_type: The boolean field to filter by (must be in BooleanType).
        value: The boolean value to match (True/False).
    """

    type: Literal["BooleanFilter"] = "BooleanFilter"
    boolean_type: str  # Should be a value from BooleanType
    value: bool

    @field_validator("boolean_type")
    def validate_boolean_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(BooleanType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid BooleanType. Allowed: {sorted(allowed)}")
        return v_norm


FilterNode = Union[
    AndFilter,
    OrFilter,
    NotFilter,
    DateFilter,
    AgeFilter,
    NIHSSFilter,
    SexFilter,
    StrokeFilter,
    BooleanFilter,
]


def _normalize_filter_node(value: Any) -> Any:
    """Accept a bare JSON list of filters as an equivalent, unambiguous
    alternate encoding of a single FilterNode.

    The planner LLM sometimes emits "filters": [{"type": "SexFilter", ...}]
    instead of wrapping multiple filters in an explicit AndFilter (or just
    the bare object for a single filter) -- a reasonable, lossless way to
    write "these filters" that FilterNode's schema doesn't accept as-is,
    which previously caused a validation error (and downstream "I need a bit
    more detail" clarify) for a plan the LLM had already correctly decided.
    This reshapes the same, unmodified filter content into the schema's
    canonical envelope; it never invents or drops a filter the LLM specified.
    """
    if not isinstance(value, list):
        return value
    items = cast(List[Any], value)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return {"type": "AndFilter", "and_": items}


class GroupBySex(HashableBaseModel):
    """
    Grouping by patient sex.

    Attributes:
        categories: List of sex categories to group by (must be in SexType). None = all.
    """

    categories: Optional[List[str]] = Field(default=None, description="List of sex categories to group by. None = all.")

    @field_validator("categories")
    def validate_categories(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            allowed = _enum_allowed_values(SexType)
            out: List[str] = []
            for val in v:
                val_norm = val.upper()
                if val_norm not in allowed:
                    raise ValueError(f"{val} is not a valid SexType. Allowed: {sorted(allowed)}")
                out.append(val_norm)
            return out
        return v


class Bucket(BaseModel):
    """
    Represents a bucket for grouping (e.g., age or NIHSS score range).

    Attributes:
        min: Minimum value of the bucket (inclusive).
        max: Maximum value of the bucket (inclusive).
    """

    min: int
    max: int


class GroupByAge(HashableBaseModel):
    """
    Grouping by age buckets.

    Attributes:
        buckets: List of age buckets (each a Bucket object).
    """

    buckets: List[Bucket] = Field(description="List of age buckets.")


class GroupByNIHSS(HashableBaseModel):
    """
    Grouping by NIHSS score buckets.

    Attributes:
        buckets: List of NIHSS score buckets (each a Bucket object).
    """

    buckets: List[Bucket] = Field(description="List of NIHSS score buckets.")


class GroupByStrokeType(HashableBaseModel):
    """
    Grouping by stroke type.

    Attributes:
        categories: List of stroke types to group by (must be in StrokeType). None = all.
    """

    categories: Optional[List[str]] = Field(default=None, description="List of stroke types to group by. None = all.")

    @field_validator("categories")
    def validate_categories(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            allowed = _enum_allowed_values(StrokeType)
            out: List[str] = []
            for val in v:
                val_norm = val.upper()
                if val_norm not in allowed:
                    raise ValueError(f"{val} is not a valid StrokeType. Allowed: {sorted(allowed)}")
                out.append(val_norm)
            return out
        return v


TIME_INTERVALS: Set[str] = {"DAY", "WEEK", "BIWEEK", "MONTH", "QUARTER", "YEAR"}
TIME_ALIGNMENT: Set[str] = {"CALENDAR", "FISCAL"}


class TimeWindow(BaseModel):
    """
    Relative time window specification.

    Attributes:
        last_n: Positive integer count of units to include (e.g., 6 for last 6 months).
        unit: Unit for the window (DAY, WEEK, MONTH, YEAR).
    """

    last_n: int = Field(gt=0, description="Positive count for the relative window.")
    unit: str = Field(description="Time unit for the window. One of DAY, WEEK, BIWEEK, MONTH, QUARTER, YEAR.")

    @field_validator("unit")
    def validate_unit(cls, v: str) -> str:
        v_norm = v.upper()
        if v_norm not in TIME_INTERVALS:
            raise ValueError(f"{v} is not a valid time unit. Allowed: {sorted(TIME_INTERVALS)}")
        return v_norm


class TimeRange(BaseModel):
    """
    Absolute time range specification.

    Attributes:
        start_date: ISO 8601 start date (inclusive).
        end_date: ISO 8601 end date (inclusive).
    """

    start_date: str = Field(description="Start date (inclusive), ISO 8601 format.")
    end_date: str = Field(description="End date (inclusive), ISO 8601 format.")

    @field_validator("start_date", "end_date")
    def validate_iso_date(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid ISO 8601 date or datetime string.")
        return v


class GroupByTime(HashableBaseModel):
    """Grouping by time buckets.

    Simplified for now to just grain + window; timezone and
    fiscal-year alignment are commented out until we know how
    to drive them from configuration.

    Attributes:
        grain: Aggregation grain (DAY, WEEK, BIWEEK, MONTH, QUARTER, YEAR).
        window: Optional relative time window (e.g., last 6 months). If omitted, executor may use defaults or chart-level filters.
        include_partial: Whether to include the current, incomplete bucket.
    """

    grain: str = Field(description="Time aggregation grain.")
    window: Optional[Union[TimeWindow, TimeRange]] = Field(default=None, description="Optional relative or absolute time window.")
    periods: Optional[List[TimeRange]] = Field(default=None, description="Explicit, possibly non-adjacent periods, one bucket each. Takes precedence over window.")
    include_partial: Optional[bool] = Field(default=None, description="Whether to include the current, incomplete bucket.")

    @field_validator("grain")
    def validate_grain(cls, v: str) -> str:
        v_norm = v.upper()
        if v_norm not in TIME_INTERVALS:
            raise ValueError(f"{v} is not a valid time grain. Allowed: {sorted(TIME_INTERVALS)}")
        return v_norm


class GroupByBoolean(HashableBaseModel):
    """
    Grouping by boolean field.

    Attributes:
        boolean_type: The boolean field to group by (must be in BooleanType).
        values: List of boolean values to group by. None = all.
    """

    boolean_type: str = Field(description="The boolean field to group by. Should be a value from BooleanType.")
    values: Optional[List[bool]] = Field(default=None, description="Boolean values to group by. None = all.")

    @field_validator("boolean_type")
    def validate_boolean_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(BooleanType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid BooleanType. Allowed: {sorted(allowed)}")
        return v_norm


class GroupByCanonicalField(HashableBaseModel):
    """
    Grouping by a canonical field from SSOT/GraphQL.

    Attributes:
        field: The canonical field name (must be in CanonicalGroupByField).
        values: List of values to group by. None = all.
    """

    field: str = Field(description="Canonical field name, should be a value from CanonicalGroupByField.")
    values: Optional[List[str]] = Field(default=None, description="Values to group by. None = all.")

    @field_validator("field")
    def validate_field(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(CanonicalGroupByField)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid CanonicalGroupByField. Allowed: {sorted(allowed)}")
        return v_norm


class CustomGroup(HashableBaseModel):
    """
    Custom group defined by filters.

    Attributes:
        label: Label for the custom group.
        filters: List of filters defining this group.
    """

    label: str = Field(description="Label for the custom group.")
    filters: List[FilterNode] = Field(description="Filters defining this group.")


GroupBySpec = Union[
    GroupBySex,
    GroupByAge,
    GroupByNIHSS,
    GroupByStrokeType,
    GroupByBoolean,
    GroupByCanonicalField,
    GroupByTime,
    CustomGroup,
]


AnalysisIntentType = [
    "DISTRIBUTION",
    "TREND",
    "COMPARISON",
    "RELATIONSHIP",
    "RANK",
]

MeasureType = [
    "DISTRIBUTION",
    "COUNT",
    "MEAN",
    "MEDIAN",
    "PERCENTILE",
    "SUM",
    "MIN",
    "MAX",
    "RATE",
]

SplitKindType = [
    "SEX",
    "STROKE_TYPE",
    "AGE",
    "NIHSS",
    "BOOLEAN",
    "CANONICAL",
    "CUSTOM",
]

AxisRoleType = [
    "METRIC_VALUE",
    "TIME",
    "COUNT",
    "AGGREGATE_VALUE",
    "CATEGORY",
]


class MeasureSemanticsSpec(BaseModel):
    """Semantic description of what value should be computed for chart output."""

    type: str
    percentile: Optional[float] = None

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        v_norm = (v or "").strip().upper()
        if v_norm not in MeasureType:
            raise ValueError(f"{v} is not a valid measure type. Allowed: {MeasureType}")
        return v_norm

    @model_validator(mode="after")
    def validate_percentile(self) -> "MeasureSemanticsSpec":
        if self.type == "PERCENTILE":
            if self.percentile is None:
                raise ValueError("measure.percentile is required when measure.type is PERCENTILE")
            if not (0.0 < self.percentile <= 100.0):
                raise ValueError("measure.percentile must be > 0 and <= 100")
            return self
        if self.percentile is not None:
            raise ValueError("measure.percentile is only allowed when measure.type is PERCENTILE")
        return self


class TimeSemanticsSpec(BaseModel):
    """Semantic time context for analysis, independent of backend retrieval strategy."""

    grain: Optional[str] = None
    window: Optional[Union[TimeWindow, TimeRange]] = None
    include_partial: Optional[bool] = None
    # Several specific, possibly non-adjacent periods ("Q1 2023, Q3 2025 and
    # Q2 2026"), one bucket each. A single date range stays a TimeRange window.
    periods: Optional[List[TimeRange]] = None

    @field_validator("periods")
    def validate_periods(cls, v: Optional[List[TimeRange]]) -> Optional[List[TimeRange]]:
        if v is None:
            return v
        if not v:
            return None
        for period in v:
            if period.start_date > period.end_date:
                raise ValueError(f"time.periods entry has start_date after end_date: {period.start_date} > {period.end_date}")
        return v

    @field_validator("grain")
    def validate_grain(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v_norm = (v or "").strip().upper()
        if v_norm not in TIME_INTERVALS:
            raise ValueError(f"{v} is not a valid time grain. Allowed: {sorted(TIME_INTERVALS)}")
        return v_norm


class SplitSpec(BaseModel):
    """Semantic split/cohort instruction for analysis.

    This is planner-facing semantics. Backend groupBy selection is handled by
    retrieval compilation and must not leak here.
    """

    kind: str
    field: Optional[str] = None
    categories: Optional[List[str]] = None
    buckets: Optional[List[Bucket]] = Field(
        default=None,
        description="Explicit numeric buckets for AGE/NIHSS splits, e.g. [{min:0,max:10}, {min:10,max:20}].",
    )

    @field_validator("kind")
    def validate_kind(cls, v: str) -> str:
        v_norm = (v or "").strip().upper()
        if v_norm not in SplitKindType:
            raise ValueError(f"{v} is not a valid split kind. Allowed: {SplitKindType}")
        return v_norm

    @model_validator(mode="after")
    def validate_field_requirements(self) -> "SplitSpec":
        if self.kind in {"BOOLEAN", "CANONICAL"}:
            if not isinstance(self.field, str) or not self.field.strip():
                raise ValueError(f"split.field is required when split.kind is {self.kind}")
            self.field = self.field.strip().upper()
        elif self.field is not None and not self.field.strip():
            self.field = None

        if self.kind in {"AGE", "NIHSS"}:
            if not self.buckets:
                raise ValueError(f"split.buckets is required when split.kind is {self.kind}")
            for bucket in self.buckets:
                if bucket.min >= bucket.max:
                    raise ValueError("Each split.buckets entry requires min < max.")
        elif self.buckets is not None and not self.buckets:
            self.buckets = None
        return self


class AxisSemanticsSpec(BaseModel):
    """Optional advanced axis override for planner output semantics."""

    role: str
    metric: Optional[str] = None
    label: Optional[str] = None
    unit: Optional[str] = None
    aggregation: Optional[str] = None
    grain: Optional[str] = None

    @field_validator("role")
    def validate_role(cls, v: str) -> str:
        v_norm = (v or "").strip().upper()
        if v_norm not in AxisRoleType:
            raise ValueError(f"{v} is not a valid axis role. Allowed: {AxisRoleType}")
        return v_norm

    @field_validator("aggregation")
    def validate_aggregation(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v_norm = (v or "").strip().upper()
        if v_norm not in MeasureType:
            raise ValueError(f"{v} is not a valid aggregation. Allowed: {MeasureType}")
        return v_norm

    @field_validator("grain")
    def validate_grain(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v_norm = (v or "").strip().upper()
        if v_norm not in TIME_INTERVALS:
            raise ValueError(f"{v} is not a valid axis grain. Allowed: {sorted(TIME_INTERVALS)}")
        return v_norm

    @field_validator("metric")
    def validate_metric(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v_norm = (v or "").strip().upper()
        allowed_values = _enum_allowed_values(MetricType)
        if v_norm not in allowed_values:
            raise ValueError(f"{v} is not a valid MetricType. Allowed: {sorted(allowed_values)}")
        return v_norm


class AnalysisSemanticsSpec(BaseModel):
    """Planner-facing semantic analysis contract for a chart request."""

    intent: str
    measure: Optional[MeasureSemanticsSpec] = None
    splits: Optional[List[SplitSpec]] = None
    time: Optional[TimeSemanticsSpec] = None
    x_axis: Optional[AxisSemanticsSpec] = Field(default=None, alias="xAxis")
    y_axis: Optional[AxisSemanticsSpec] = Field(default=None, alias="yAxis")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("intent")
    def validate_intent(cls, v: str) -> str:
        v_norm = (v or "").strip().upper()
        if v_norm not in AnalysisIntentType:
            raise ValueError(f"{v} is not a valid analysis intent. Allowed: {AnalysisIntentType}")
        return v_norm


class DataOriginSpec(BaseModel):
    """Data origin scope for a metric/chart query."""

    model_config = ConfigDict(populate_by_name=True)

    provider_id: Optional[List[int]] = Field(default=None, alias="providerId", description="Provider IDs to query.")
    provider_group_id: Optional[List[int]] = Field(
        default=None,
        alias="providerGroupId",
        description="Provider group IDs to query.",
    )

    @field_validator("provider_id", "provider_group_id")
    def validate_positive_ids(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is None:
            return v
        out: List[int] = []
        for item in v:
            value = int(item)
            if value <= 0:
                raise ValueError("Data origin IDs must be positive integers.")
            out.append(value)
        return out

    @model_validator(mode="after")
    def validate_origin(self) -> "DataOriginSpec":
        if not self.provider_id and not self.provider_group_id:
            raise ValueError("DataOriginSpec requires providerId or providerGroupId.")
        return self


class OriginScopeSpec(BaseModel):
    """Semantic data-origin reference resolved at execution time.

    This allows planner outputs to remain user-intent oriented (e.g., "mine",
    "country code", "hospital name") while execution resolves concrete IDs.
    """

    model_config = ConfigDict(populate_by_name=True)

    scope_type: str = Field(alias="scopeType")
    value: Optional[Any] = None
    label: Optional[str] = None
    country_code: Optional[str] = Field(default=None, alias="countryCode")

    @field_validator("scope_type")
    def validate_scope_type(cls, v: str) -> str:
        raw = (v or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "hospital_name": "provider_name",
            "provider": "provider_name",
            "group_id": "provider_group_id",
            "group_name": "provider_group_name",
            "country": "country_code",
            "all": "all_accessible",
        }
        normalized = aliases.get(raw, raw)
        allowed = {
            "mine",
            "provider_id",
            "provider_name",
            "provider_group_id",
            "provider_group_name",
            "country_code",
            "country_average",
            "all_accessible",
        }
        if normalized not in allowed:
            raise ValueError(f"{v} is not a valid OriginScopeSpec.scopeType. Allowed: {sorted(allowed)}")
        return normalized

    @field_validator("country_code")
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        token = v.strip().upper()
        if not token:
            return None
        if len(token) != 2 or not token.isalpha():
            raise ValueError("countryCode must be a 2-letter ISO country code")
        return token


class MetricSpec(BaseModel):
    """
    Specification for a metric to be analyzed or visualized.

    Attributes:
        metric: The metric type (must be in MetricType).
    """

    metric: str  # Should be a value from MetricType
    data_origin: Optional[DataOriginSpec] = Field(default=None, alias="dataOrigin")
    origin_scope: Optional[OriginScopeSpec] = Field(default=None, alias="originScope")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("metric")
    def validate_metric_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed_values = _enum_allowed_values(MetricType)
        if v_norm not in allowed_values:
            raise ValueError(f"{v} is not a valid MetricType. Allowed: {sorted(allowed_values)}")
        return v_norm


class NumericValueDomainSpec(BaseModel):
    """Optional numeric domain override for chart metric requests."""

    model_config = ConfigDict(populate_by_name=True)

    lower_bound: Optional[int] = Field(default=None, alias="lowerBound")
    upper_bound: Optional[int] = Field(default=None, alias="upperBound")

    @model_validator(mode="after")
    def validate_bounds(self) -> "NumericValueDomainSpec":
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound >= self.upper_bound:
            raise ValueError("numericResolution.valueDomain requires lowerBound < upperBound.")
        return self


class NumericBucketingSpec(BaseModel):
    """Optional bucketing override for chart metric requests."""

    model_config = ConfigDict(populate_by_name=True)

    bucket_count: Optional[int] = Field(default=None, alias="bucketCount")
    bucket_size: Optional[int] = Field(default=None, alias="bucketSize")

    @model_validator(mode="after")
    def validate_bucketing(self) -> "NumericBucketingSpec":
        if self.bucket_count is None and self.bucket_size is None:
            raise ValueError("numericResolution.bucketing requires bucketCount or bucketSize.")
        if self.bucket_count is not None and self.bucket_count <= 0:
            raise ValueError("numericResolution.bucketing.bucketCount must be > 0.")
        if self.bucket_size is not None and self.bucket_size <= 0:
            raise ValueError("numericResolution.bucketing.bucketSize must be > 0.")
        return self


class NumericResolutionSpec(BaseModel):
    """Chart-level numeric request controls shared by all metrics in the chart."""

    model_config = ConfigDict(populate_by_name=True)

    value_domain: Optional[NumericValueDomainSpec] = Field(default=None, alias="valueDomain")
    bucketing: Optional[NumericBucketingSpec] = None


class ChartSpec(BaseModel):
    """
    Specification for a chart to be generated.

    Attributes:
        chart_type: The chart type (must be in ChartType).
        filters: Optional chart-level filters applied to all metrics/series.
        semantics: Explicit semantic intent, splits, time, and measure metadata.
        metrics: List of metrics to include in the chart.
    """

    chart_type: str  # Should be a value from ChartType
    semantics: Optional[AnalysisSemanticsSpec] = None
    filters: Optional[FilterNode] = None
    metrics: List[MetricSpec]
    numeric_resolution: Optional[NumericResolutionSpec] = Field(default=None, alias="numericResolution")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("filters", mode="before")
    def normalize_filters(cls, v: Any) -> Any:
        return _normalize_filter_node(v)

    @field_validator("chart_type")
    def validate_chart_type(cls, v: str) -> str:
        v_norm = v.upper()
        if v_norm not in ChartType:
            raise ValueError(f"{v} is not a valid ChartType. Allowed: {ChartType}")
        return v_norm

    @model_validator(mode="after")
    def validate_chart_level_semantics(self) -> "ChartSpec":
        if self.semantics is None:
            return self
        if self.semantics.measure is None:
            raise ValueError("Chart semantics.measure is required when semantics is present.")
        return self


class StatisticalTestSpec(BaseModel):
    """
    Specification for a statistical test to be performed.

    Attributes:
        test_type: The statistical test type (must be in StatisticalTestType).
        metrics: List of metrics to include in the test.
    """

    test_type: str  # Should be a value from StatisticalTestType
    metrics: List[MetricSpec]
    group_by: Optional[List[GroupBySpec]] = None
    filters: Optional[FilterNode] = None

    @field_validator("filters", mode="before")
    def normalize_filters(cls, v: Any) -> Any:
        return _normalize_filter_node(v)

    @field_validator("test_type")
    def validate_test_type(cls, v: str) -> str:
        v_norm = v.upper()
        if v_norm not in StatisticalTestType:
            raise ValueError(f"{v} is not a valid StatisticalTestType. Allowed: {StatisticalTestType}")
        return v_norm

    @model_validator(mode="after")
    def validate_test_groupby(self) -> "StatisticalTestSpec":
        """Statistical tests must use explicit metric cohorts, not chart-like grouping semantics."""
        gb = self.group_by or []
        if not gb:
            return self

        raise ValueError(
            "Statistical tests do not support group_by. Define explicit cohorts via metric-level origins/filters."
        )
        return self


class AnalysisPlan(BaseModel):
    """
    The top-level plan object returned by the planner.

    Attributes:
        charts: List of chart specifications to generate.
        statistical_tests: List of statistical test specifications to perform.
    """

    charts: Optional[List[ChartSpec]] = None
    statistical_tests: Optional[List[StatisticalTestSpec]] = None
