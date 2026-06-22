from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union, cast

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
from src.shared.ssot_loader import get_metric_metadata, get_metric_text_lookup


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


class PredicateFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["predicate"] = "predicate"
    field: str
    operator: str
    value: Optional[Union[str, int, float, bool]] = None
    values: Optional[List[Union[str, int, float, bool]]] = None

    @field_validator("field")
    def validate_field(cls, v: str) -> str:
        token = (v or "").strip().upper()
        if not token:
            raise ValueError("Predicate filter field must be non-empty.")
        return token

    @field_validator("operator")
    def validate_operator(cls, v: str) -> str:
        token = (v or "").strip().upper()
        allowed = _enum_allowed_values(OperatorType)
        if token not in allowed:
            raise ValueError(f"{v} is not a valid OperatorType. Allowed: {sorted(allowed)}")
        return token

    @model_validator(mode="after")
    def validate_payload(self) -> "PredicateFilter":
        has_value = self.value is not None
        has_values = bool(self.values)
        if has_value == has_values:
            raise ValueError("Predicate filter requires exactly one of value or values.")
        return self


class NotFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["not"] = "not"
    clause: "FilterNode"


class AndFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["and"] = "and"
    clauses: List["FilterNode"] = Field(min_length=2)


class OrFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["or"] = "or"
    clauses: List["FilterNode"] = Field(min_length=2)


FilterNode = Annotated[
    Union[PredicateFilter, NotFilter, AndFilter, OrFilter],
    Field(discriminator="op"),
]


class GroupBySex(HashableBaseModel):
    """
    Grouping by patient sex.

    Attributes:
        categories: List of sex categories to group by (must be in SexType). None = all.
    """

    categories: List[str] = Field(min_length=1, description="List of sex categories to group by.")

    @field_validator("categories")
    def validate_categories(cls, v: List[str]) -> List[str]:
        allowed = _enum_allowed_values(SexType)
        out: List[str] = []
        for val in v:
            val_norm = val.upper()
            if val_norm not in allowed:
                raise ValueError(f"{val} is not a valid SexType. Allowed: {sorted(allowed)}")
            out.append(val_norm)
        return out


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

    categories: List[str] = Field(min_length=1, description="List of stroke types to group by.")

    @field_validator("categories")
    def validate_categories(cls, v: List[str]) -> List[str]:
        allowed = _enum_allowed_values(StrokeType)
        out: List[str] = []
        for val in v:
            val_norm = val.upper()
            if val_norm not in allowed:
                raise ValueError(f"{val} is not a valid StrokeType. Allowed: {sorted(allowed)}")
            out.append(val_norm)
        return out


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
        normalized = raw
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


_Y_STATISTICS = {"MEAN", "MEDIAN", "P25", "P75", "COUNT", "PERCENT", "CASE_COUNT"}


def _metric_data_shape(metric_code: str) -> tuple[Optional[str], Optional[str]]:
    meta = get_metric_metadata().get(metric_code, {})
    data_type_raw = meta.get("data_type")
    data_type = str(data_type_raw).strip().upper() if isinstance(data_type_raw, str) else None

    unit_raw = meta.get("unit")
    unit = str(unit_raw).strip().lower() if isinstance(unit_raw, str) and str(unit_raw).strip() else None

    if unit is None:
        numeric_block = meta.get("numeric")
        if isinstance(numeric_block, dict):
            nested_unit = numeric_block.get("unit")
            if isinstance(nested_unit, str) and nested_unit.strip():
                unit = nested_unit.strip().lower()

    if data_type is None or unit is None:
        text_meta = get_metric_text_lookup().get((metric_code or "").strip().lower(), {})
        if data_type is None:
            fallback_data_type = text_meta.get("data_type")
            if isinstance(fallback_data_type, str) and fallback_data_type.strip():
                data_type = fallback_data_type.strip().upper()
        if unit is None:
            fallback_unit = text_meta.get("unit")
            if isinstance(fallback_unit, str) and fallback_unit.strip():
                unit = fallback_unit.strip().lower()

    return data_type, unit


def _metric_unit(metric_code: str) -> Optional[str]:
    _, unit = _metric_data_shape(metric_code)
    return unit


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



class StatisticalTestSpec(BaseModel):
    test_type: str
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


# --- Plan axis and chart types ---


class TimeXAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["time"] = "time"
    grain: str
    window: Optional[Union[TimeWindow, TimeRange]] = None
    include_partial: Optional[bool] = Field(default=None, alias="includePartial")

    @field_validator("grain")
    def validate_grain(cls, v: str) -> str:
        token = (v or "").strip().upper()
        if token not in TIME_INTERVALS:
            raise ValueError(f"{v} is not a valid time grain. Allowed: {sorted(TIME_INTERVALS)}")
        return token


class CategoryXAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["category"] = "category"
    group_by: GroupBySpec = Field(alias="groupBy")
    order: Optional[str] = None


class NumericMetricXAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["numeric_metric"] = "numeric_metric"
    metric: str
    bins: Optional[int] = Field(default=None, gt=0)
    min_value: Optional[int] = Field(default=None, alias="minValue")
    max_value: Optional[int] = Field(default=None, alias="maxValue")

    @field_validator("metric")
    def validate_metric(cls, v: str) -> str:
        token = (v or "").strip().upper()
        allowed = _enum_allowed_values(MetricType)
        if token not in allowed:
            raise ValueError(f"{v} is not a valid MetricType. Allowed: {sorted(allowed)}")
        return token


XAxisSpec = Annotated[
    Union[TimeXAxis, CategoryXAxis, NumericMetricXAxis],
    Field(discriminator="kind"),
]


class MetricValueAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["metric_value"] = "metric_value"
    statistic: str = "MEAN"
    unit: Optional[str] = None

    @field_validator("statistic")
    def validate_statistic(cls, v: str) -> str:
        token = (v or "").strip().upper()
        if token not in _Y_STATISTICS:
            raise ValueError(f"{v} is not a valid y-axis statistic. Allowed: {sorted(_Y_STATISTICS)}")
        return token


class CountAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["count"] = "count"


YAxisSpec = Annotated[Union[MetricValueAxis, CountAxis], Field(discriminator="kind")]


class LineSeries(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metric: str
    x_axis: str = Field(alias="xAxis")
    y_axis: str = Field(alias="yAxis")
    label: Optional[str] = None
    filters: Optional[FilterNode] = None
    data_origin: Optional[DataOriginSpec] = Field(default=None, alias="dataOrigin")
    origin_scope: Optional[OriginScopeSpec] = Field(default=None, alias="originScope")

    @field_validator("metric")
    def validate_metric(cls, v: str) -> str:
        token = (v or "").strip().upper()
        allowed = _enum_allowed_values(MetricType)
        if token not in allowed:
            raise ValueError(f"{v} is not a valid MetricType. Allowed: {sorted(allowed)}")
        return token


class LineChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    chart_type: Literal["LINE"] = Field(alias="chartType")
    x_axes: Dict[str, XAxisSpec] = Field(alias="xAxes", min_length=1)
    y_axes: Dict[str, YAxisSpec] = Field(alias="yAxes", min_length=1)
    series: List[LineSeries] = Field(min_length=1)
    series_split: Optional[GroupBySpec] = Field(default=None, alias="seriesSplit")
    filters: Optional[FilterNode] = None
    title: Optional[str] = None

    @model_validator(mode="after")
    def validate_axis_references(self) -> "LineChartSpec":
        used_x: set[str] = set()
        used_y: set[str] = set()
        for item in self.series:
            if item.x_axis not in self.x_axes:
                raise ValueError(f"Unknown x-axis key '{item.x_axis}' in line series.")
            if item.y_axis not in self.y_axes:
                raise ValueError(f"Unknown y-axis key '{item.y_axis}' in line series.")
            used_x.add(item.x_axis)
            used_y.add(item.y_axis)

        orphan_x = sorted(set(self.x_axes.keys()) - used_x)
        orphan_y = sorted(set(self.y_axes.keys()) - used_y)
        if orphan_x:
            raise ValueError(f"Unused x-axis keys are not allowed: {orphan_x}")
        if orphan_y:
            raise ValueError(f"Unused y-axis keys are not allowed: {orphan_y}")

        if self.series_split is not None:
            if len(used_x) != 1:
                raise ValueError("LINE charts with seriesSplit must reference exactly one x-axis.")
            for x_key in used_x:
                if isinstance(self.x_axes[x_key], CategoryXAxis):
                    raise ValueError("LINE charts with seriesSplit cannot use a category x-axis.")
        else:
            for x_key in used_x:
                if not isinstance(self.x_axes[x_key], CategoryXAxis):
                    continue
                series_on_axis = [item for item in self.series if item.x_axis == x_key]
                if len(series_on_axis) > 1:
                    raise ValueError(
                        "LINE charts with multiple series on a category x-axis must declare seriesSplit explicitly."
                    )
        return self

    @model_validator(mode="after")
    def validate_semantics(self) -> "LineChartSpec":
        for x_key, x_axis in self.x_axes.items():
            if not isinstance(x_axis, NumericMetricXAxis):
                continue

            referencing_series = [s for s in self.series if s.x_axis == x_key]
            if not referencing_series:
                raise ValueError(
                    f"LINE numeric_metric x-axis ('{x_key}') is not referenced by any series."
                )

            for series in referencing_series:
                y_axis = self.y_axes.get(series.y_axis)
                if not isinstance(y_axis, CountAxis):
                    raise ValueError(
                        f"LINE numeric_metric x-axis ('{x_key}') requires count y-axis. "
                        "Use metric_value with time/category x-axis for trends or comparisons."
                    )
                if (series.metric or "").strip().upper() != (x_axis.metric or "").strip().upper():
                    raise ValueError(
                        f"LINE distribution series metric '{series.metric}' must match numeric_metric x-axis metric '{x_axis.metric}'."
                    )

        if self.series_split is None and len(self.series) >= 2:
            x_keys = {series.x_axis for series in self.series}
            y_keys = {series.y_axis for series in self.series}
            if len(x_keys) == 1 and len(y_keys) == 1:
                legacy_x_key = next(iter(x_keys))
                legacy_y_key = next(iter(y_keys))
                legacy_x_axis = self.x_axes.get(legacy_x_key)
                legacy_y_axis = self.y_axes.get(legacy_y_key)

                if isinstance(legacy_x_axis, NumericMetricXAxis) and isinstance(legacy_y_axis, CountAxis):
                    first = self.series[0]
                    metric = (first.metric or "").strip().upper()
                    if metric:
                        seen_categories: set[str] = set()
                        legacy_shape = True
                        for series in self.series:
                            if (series.metric or "").strip().upper() != metric:
                                legacy_shape = False
                                break
                            if series.x_axis != legacy_x_key or series.y_axis != legacy_y_key:
                                legacy_shape = False
                                break
                            if series.data_origin != first.data_origin or series.origin_scope != first.origin_scope:
                                legacy_shape = False
                                break

                            filt = series.filters
                            if not isinstance(filt, PredicateFilter):
                                legacy_shape = False
                                break
                            if (filt.field or "").strip().upper() != "SEX":
                                legacy_shape = False
                                break
                            if (filt.operator or "").strip().upper() != "EQ":
                                legacy_shape = False
                                break
                            if not isinstance(filt.value, str) or filt.values:
                                legacy_shape = False
                                break
                            seen_categories.add(filt.value.strip().upper())

                        if legacy_shape and len(seen_categories) >= 2:
                            raise ValueError(
                                "LINE charts splitting by sex must declare seriesSplit explicitly."
                            )

        units_by_y: Dict[str, List[str]] = {}
        for series in self.series:
            unit = _metric_unit(series.metric)
            if unit is None:
                continue
            units = units_by_y.setdefault(series.y_axis, [])
            if unit not in units:
                units.append(unit)

        for y_key, units in units_by_y.items():
            if len(units) > 1:
                raise ValueError(
                    f"y-axis '{y_key}' mixes metric units {sorted(units)}. "
                    "Use separate y-axes or separate charts for different units."
                )

        return self


class HistogramChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    chart_type: Literal["HISTOGRAM"] = Field(alias="chartType")
    x_axis: NumericMetricXAxis = Field(alias="xAxis")
    y_axis: CountAxis = Field(default_factory=CountAxis, alias="yAxis")
    filters: Optional[FilterNode] = None
    title: Optional[str] = None
    data_origin: Optional[DataOriginSpec] = Field(default=None, alias="dataOrigin")
    origin_scope: Optional[OriginScopeSpec] = Field(default=None, alias="originScope")


ChartSpec = Annotated[
    Union[LineChartSpec, HistogramChartSpec],
    Field(discriminator="chart_type"),
]


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    charts: Optional[List[ChartSpec]] = Field(default=None, min_length=1)
    statistical_tests: Optional[List[StatisticalTestSpec]] = Field(default=None, alias="statisticalTests")

    @model_validator(mode="after")
    def validate_filter_complexity(self) -> "AnalysisPlan":
        if not self.charts and not self.statistical_tests:
            raise ValueError("AnalysisPlan must have at least one chart or statistical test.")

        def _walk(node: FilterNode, depth: int) -> int:
            if depth > 3:
                raise ValueError("Filter nesting depth exceeds max_depth=3.")
            if isinstance(node, PredicateFilter):
                return 1
            if isinstance(node, NotFilter):
                return 1 + _walk(node.clause, depth + 1)
            if isinstance(node, (AndFilter, OrFilter)):
                return 1 + sum(_walk(child, depth + 1) for child in node.clauses)
            return 1

        total_nodes = 0
        for chart in self.charts or []:
            chart_filter = getattr(chart, "filters", None)
            if chart_filter is not None:
                total_nodes += _walk(chart_filter, 1)

            if isinstance(chart, LineChartSpec):
                for item in chart.series:
                    if item.filters is not None:
                        total_nodes += _walk(item.filters, 1)

        if total_nodes > 30:
            raise ValueError("Filter complexity exceeds max_nodes=30 across the plan.")
        return self


NotFilter.model_rebuild()
AndFilter.model_rebuild()
OrFilter.model_rebuild()
CustomGroup.model_rebuild()
LineSeries.model_rebuild()
LineChartSpec.model_rebuild()
HistogramChartSpec.model_rebuild()
AnalysisPlan.model_rebuild()
StatisticalTestSpec.model_rebuild()
MetricSpec.model_rebuild()
