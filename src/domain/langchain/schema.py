from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

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

    and_: List["FilterNode"]


class OrFilter(BaseModel):
    """
    Logical OR of multiple filter nodes.

    Attributes:
        or_: List of filter nodes to combine with OR logic.
    """

    or_: List["FilterNode"]


class NotFilter(BaseModel):
    """
    Logical NOT of a filter node.

    Attributes:
        not_: The filter node to negate.
    """

    not_: "FilterNode"


class SexFilter(BaseModel):
    """
    Filter for patient sex.

    Attributes:
        value: The sex value to filter by (must be in SexType).
    """

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

    boolean_type: str  # Should be a value from BooleanType
    value: bool

    @field_validator("boolean_type")
    def validate_boolean_type(cls, v: str) -> str:
        v_norm = v.upper()
        allowed = _enum_allowed_values(BooleanType)
        if v_norm not in allowed:
            raise ValueError(f"{v} is not a valid BooleanType. Allowed: {sorted(allowed)}")
        return v_norm


FilterNode = Union[AndFilter, OrFilter, NotFilter, DateFilter, AgeFilter, NIHSSFilter, SexFilter, StrokeFilter, BooleanFilter]


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


class DataOriginSpec(BaseModel):
    """Data origin scope for a metric/chart query."""

    model_config = ConfigDict(populate_by_name=True)

    provider_id: Optional[List[int]] = Field(default=None, alias="providerId", description="Provider IDs to query.")
    provider_group_id: Optional[List[int]] = Field(default=None, alias="providerGroupId", description="Provider group IDs to query.")

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


class DistributionSpec(BaseModel):
    """
    Specification for a distribution to be computed.

    Attributes:
        num_buckets: Number of buckets in the distribution.
        min_value: Minimum value of the distribution range.
        max_value: Maximum value of the distribution range.
    """

    num_buckets: int = Field(description="Number of buckets in the distribution.")
    min_value: int = Field(description="Minimum value of the distribution range.")
    max_value: int = Field(description="Maximum value of the distribution range.")


class MetricSpec(BaseModel):
    """
    Specification for a metric to be analyzed or visualized.

    Attributes:
        metric: The metric type (must be in MetricType).
        distribution: Optional distribution specification for this metric.
    """

    metric: str  # Should be a value from MetricType
    distribution: Optional[DistributionSpec] = None
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


class ChartSpec(BaseModel):
    """
    Specification for a chart to be generated.

    Attributes:
        chart_type: The chart type (must be in ChartType).
        filters: Optional chart-level filters applied to all metrics/series.
        group_by: Optional chart-level groupings applied to all metrics/series.
        metrics: List of metrics to include in the chart.
    """

    chart_type: str  # Should be a value from ChartType
    filters: Optional[FilterNode] = None
    group_by: Optional[List[GroupBySpec]] = None
    metrics: List[MetricSpec]

    @field_validator("chart_type")
    def validate_chart_type(cls, v: str) -> str:
        v_norm = v.upper()
        if v_norm not in ChartType:
            raise ValueError(f"{v} is not a valid ChartType. Allowed: {ChartType}")
        return v_norm

    @model_validator(mode="after")
    def validate_chart_level_groupby(self) -> "ChartSpec":
        """Validate chart-level group_by and filters.

        Rules enforced:
        - No duplicate GroupBy of the same exact spec.
        - At most one GroupBySex and one GroupByStrokeType.
        - At most one GroupByAge and one GroupByNIHSS.
        - GroupByBoolean: at most one per boolean_type.
        - GroupByCanonicalField: no duplicates of the same field (multiple distinct canonical fields are allowed).
        """
        gb = self.group_by or []
        if not gb:
            return self

        seen_specs: set[GroupBySpec] = set()
        sex_count = 0
        stroke_count = 0
        age_count = 0
        nihss_count = 0
        time_count = 0
        boolean_by_type: dict[str, int] = {}
        canonical_fields: set[str] = set()

        for g in gb:
            if g in seen_specs:
                raise ValueError("Duplicate groupBy spec detected in chart.group_by; remove duplicates.")
            seen_specs.add(g)

            if isinstance(g, GroupBySex):
                sex_count += 1
                if sex_count > 1:
                    raise ValueError("Only one GroupBySex is allowed per chart.")
            elif isinstance(g, GroupByStrokeType):
                stroke_count += 1
                if stroke_count > 1:
                    raise ValueError("Only one GroupByStrokeType is allowed per chart.")
            elif isinstance(g, GroupByAge):
                age_count += 1
                if age_count > 1:
                    raise ValueError("Only one GroupByAge is allowed per chart.")
            elif isinstance(g, GroupByNIHSS):
                nihss_count += 1
                if nihss_count > 1:
                    raise ValueError("Only one GroupByNIHSS is allowed per chart.")
            elif isinstance(g, GroupByTime):
                time_count += 1
                if time_count > 1:
                    raise ValueError("Only one GroupByTime is allowed per chart.")
            elif isinstance(g, GroupByBoolean):
                boolean_by_type[g.boolean_type] = boolean_by_type.get(g.boolean_type, 0) + 1
            elif isinstance(g, GroupByCanonicalField):
                if g.field in canonical_fields:
                    raise ValueError("Duplicate GroupByCanonicalField for the same field is not allowed.")
                canonical_fields.add(g.field)

        for btype, count in boolean_by_type.items():
            if count > 1:
                raise ValueError(f"Only one GroupByBoolean per boolean_type is allowed (duplicate for '{btype}').")

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

    @field_validator("test_type")
    def validate_test_type(cls, v: str) -> str:
        v_norm = v.upper()
        if v_norm not in StatisticalTestType:
            raise ValueError(f"{v} is not a valid StatisticalTestType. Allowed: {StatisticalTestType}")
        return v_norm

    @model_validator(mode="after")
    def validate_test_groupby(self) -> "StatisticalTestSpec":
        """Ensure no duplicate group_by specs and single-instance constraints similar to charts.

        - Only one GroupBySex / GroupByStrokeType / GroupByAge / GroupByNIHSS per test.
        - GroupByBoolean: only one per boolean_type.
        - Allow multiple distinct GroupByCanonicalField but no duplicates of same field.
        """
        gb = self.group_by or []
        if not gb:
            return self

        seen: set[GroupBySpec] = set()
        sex = stroke = age = nihss = time = 0
        boolean_types: dict[str, int] = {}
        canonical_fields: set[str] = set()
        for g in gb:
            if g in seen:
                raise ValueError("Duplicate group_by spec in statistical test.")
            seen.add(g)
            if isinstance(g, GroupBySex):
                sex += 1
                if sex > 1:
                    raise ValueError("Only one GroupBySex allowed in a statistical test.")
            elif isinstance(g, GroupByStrokeType):
                stroke += 1
                if stroke > 1:
                    raise ValueError("Only one GroupByStrokeType allowed in a statistical test.")
            elif isinstance(g, GroupByAge):
                age += 1
                if age > 1:
                    raise ValueError("Only one GroupByAge allowed in a statistical test.")
            elif isinstance(g, GroupByNIHSS):
                nihss += 1
                if nihss > 1:
                    raise ValueError("Only one GroupByNIHSS allowed in a statistical test.")
            elif isinstance(g, GroupByTime):
                time += 1
                if time > 1:
                    raise ValueError("Only one GroupByTime allowed in a statistical test.")
            elif isinstance(g, GroupByBoolean):
                boolean_types[g.boolean_type] = boolean_types.get(g.boolean_type, 0) + 1
            elif isinstance(g, GroupByCanonicalField):
                if g.field in canonical_fields:
                    raise ValueError("Duplicate GroupByCanonicalField for same field in statistical test.")
                canonical_fields.add(g.field)
        for bt, ct in boolean_types.items():
            if ct > 1:
                raise ValueError(f"Duplicate GroupByBoolean for boolean_type '{bt}' in statistical test.")
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
