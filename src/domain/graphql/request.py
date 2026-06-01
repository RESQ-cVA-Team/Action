"""
Pydantic-based GraphQL query input models that mirror the output structure.
Clean, type-safe alternative to the builder pattern approach.
Uses SSOT-based enums for consistency across the system.
"""

from typing import Any, List, Literal, Mapping, Optional, Union, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from src.domain.graphql.ssot_enums import BooleanPropertyType, GroupByType, MetricType, Operator, SexType, StrokeType
from src.shared.ssot_loader import get_metric_metadata


def _mapping_to_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def _numeric_filter_properties() -> set[str]:
    out: set[str] = set()
    metadata = get_metric_metadata()
    for meta in metadata.values():
        meta_dict = _mapping_to_dict(meta)
        props = meta_dict.get("properties")
        if isinstance(props, list):
            for prop in cast(List[object], props):
                if isinstance(prop, str) and prop.strip():
                    out.add(prop.strip().upper())
    # Safe fallback for legacy environments with partial metadata.
    if not out:
        out = {
            "AGE",
            "ADMISSION_NIHSS",
            "GLUCOSE",
            "CHOLESTEROL",
            "SYSTOLIC_PRESSURE",
            "DIASTOLIC_PRESSURE",
        }
    return out


_NUMERIC_FILTER_PROPERTIES = _numeric_filter_properties()


class IntegerFilter(BaseModel):
    """Filter for integer/numeric values"""

    property: str
    operator: Operator
    value: int

    @field_validator("property")
    def validate_property(cls, value: str) -> str:
        normalized = (value or "").strip().upper()
        if normalized not in _NUMERIC_FILTER_PROPERTIES:
            allowed = sorted(_NUMERIC_FILTER_PROPERTIES)
            raise ValueError(f"{value} is not a valid numeric filter property. Allowed: {allowed}")
        return normalized


class BooleanFilter(BaseModel):
    """Filter for boolean values - uses SSOT BooleanPropertyType"""

    property: BooleanPropertyType
    value: bool


class SexFilter(BaseModel):
    """Filter for sex/gender"""

    sex_type: SexType = Field(alias="sexType")
    contains: bool = True


class StrokeFilter(BaseModel):
    """Filter for stroke type"""

    stroke_type: StrokeType = Field(alias="strokeType")
    contains: bool = True


class DateFilter(BaseModel):
    """Filter for date values"""

    property: Literal["DISCHARGE_DATE", "ADMISSION_DATE", "ONSET_DATE"]
    operator: Operator
    value: str


class LogicalFilter(BaseModel):
    """Logical combination of filters (AND, OR, NOT)"""

    operator: Literal["AND", "OR", "NOT"]
    children: List[Union["LogicalFilter", IntegerFilter, BooleanFilter, SexFilter, StrokeFilter, DateFilter]]


LogicalFilter.model_rebuild()


FilterType = Union[LogicalFilter, IntegerFilter, BooleanFilter, SexFilter, StrokeFilter, DateFilter]


class MetricOptions(BaseModel):
    """Options for metric calculation"""

    lower_boundary: Optional[int] = Field(default=None, alias="lowerBoundary")
    upper_boundary: Optional[int] = Field(default=None, alias="upperBoundary")


class DistributionOptions(BaseModel):
    """Options for distribution calculation"""

    bin_count: int = Field(default=20, alias="binCount")
    lower_bound: int = Field(alias="lowerBound")
    upper_bound: int = Field(alias="upperBound")


class MetricRequest(BaseModel):
    """Request for a specific metric with options"""

    metric_type: MetricType = Field(alias="metricType")
    alias: Optional[str] = None

    include_stats: bool = Field(default=False, alias="includeStats")
    include_distribution: bool = Field(default=False, alias="includeDistribution")
    include_grouping: bool = Field(default=False, alias="includeGrouping")

    metric_options: Optional[MetricOptions] = Field(default=None, alias="metricOptions")
    distribution_options: Optional[DistributionOptions] = Field(default=None, alias="distributionOptions")

    def with_stats(self) -> "MetricRequest":
        """Builder method: include statistical measures"""
        self.include_stats = True
        return self

    def with_distribution(self, bin_count: int = 20, lower: int = 0, upper: int = 100) -> "MetricRequest":
        """Builder method: include distribution data"""
        self.include_distribution = True
        self.distribution_options = DistributionOptions(binCount=bin_count, lowerBound=lower, upperBound=upper)
        if not self.metric_options:
            self.metric_options = MetricOptions()
        self.metric_options.lower_boundary = lower
        self.metric_options.upper_boundary = upper
        return self

    def with_bounds(self, lower: int, upper: int) -> "MetricRequest":
        """Builder method: set metric boundaries"""
        if not self.metric_options:
            self.metric_options = MetricOptions()
        self.metric_options.lower_boundary = lower
        self.metric_options.upper_boundary = upper
        return self


class TimePeriod(BaseModel):
    """Time period for the query"""

    start_date: Optional[str] = Field(default="2022-01-01", alias="startDate")
    end_date: Optional[str] = Field(default="2024-12-31", alias="endDate")

    @model_validator(mode="after")
    def _fill_none_bounds(self):
        """If start/end are explicitly None, replace with defaults.

        This allows callers to pass None to mean "use the implicit min/max".
        """
        if self.start_date is None:
            self.start_date = "2022-01-01"
        if self.end_date is None:
            self.end_date = "2024-12-31"
        return self


class DataOrigin(BaseModel):
    """Data source configuration"""

    provider_group_id: Optional[List[int]] = Field(default=None, alias="providerGroupId")
    provider_id: Optional[List[int]] = Field(default=None, alias="providerId")

    @model_validator(mode="after")
    def validate_origin(self):
        if not self.provider_group_id and not self.provider_id:
            raise ValueError("DataOrigin requires at least one of providerGroupId or providerId")
        return self


class GraphQLQueryRequest(BaseModel):
    """Main GraphQL query request model"""

    metrics: List[MetricRequest]
    time_period: TimePeriod | List[TimePeriod] = Field(default_factory=TimePeriod, alias="timePeriod")
    data_origin: DataOrigin = Field(alias="dataOrigin")

    case_filter: Optional[FilterType] = Field(default=None, alias="caseFilter")
    group_by: Optional[GroupByType] = Field(default=None, alias="groupBy")
    include_general_stats: bool = Field(default=False, alias="includeGeneralStats")

    @model_validator(mode="after")
    def set_grouping_on_metrics(self):
        """Automatically enable grouping on all metrics if group_by is specified"""
        if self.group_by:
            for metric in self.metrics:
                metric.include_grouping = True
        if isinstance(self.time_period, list) and len(self.time_period) == 0:
            raise ValueError("timePeriod list cannot be empty")
        return self

    def to_graphql_string(self) -> str:
        """Convert this request to a GraphQL query string"""
        return GraphQLQueryGenerator.generate(self)


class GraphQLQueryGenerator:
    """Generates GraphQL query strings from Pydantic models"""

    @staticmethod
    def generate(request: GraphQLQueryRequest) -> str:
        """Generate GraphQL query string from request model"""

        filter_args: List[str] = []
        filter_args.append(GraphQLQueryGenerator._generate_time_period_arg(request.time_period))
        filter_args.append(GraphQLQueryGenerator._generate_data_origin_arg(request.data_origin))

        if request.case_filter:
            filter_args.append(f"caseFilter: {GraphQLQueryGenerator._generate_filter(request.case_filter)}")

        filter_string = ", ".join(filter_args)

        query_args = [f"filter: {{{filter_string}}}"]

        if request.group_by:
            query_args.append(f"groupBy: {request.group_by.value}")

        metric_fields: List[str] = []
        for metric in request.metrics:
            metric_fields.append(GraphQLQueryGenerator._generate_metric_field(metric))

        if request.include_general_stats:
            metric_fields.append("""
                generalStatsGroup {
                    generalStatistics {
                        casesInPeriod
                        filteredCasesInPeriod
                    }
                }
            """)

        query = f"""
        query {{
            getMetrics({", ".join(query_args)}) {{
                {" ".join(metric_fields)}
            }}
        }}
        """

        return GraphQLQueryGenerator._clean_query(query)

    @staticmethod
    def _generate_time_period_arg(time_period: TimePeriod | List[TimePeriod]) -> str:
        def render_one(tp: TimePeriod) -> str:
            return f'{{ startDate: "{tp.start_date}", endDate: "{tp.end_date}" }}'

        if isinstance(time_period, list):
            parts = ", ".join(render_one(tp) for tp in time_period)
            return f"timePeriod: [{parts}]"
        return f"timePeriod: {render_one(time_period)}"

    @staticmethod
    def _generate_data_origin_arg(data_origin: DataOrigin) -> str:
        parts: List[str] = []
        if data_origin.provider_group_id:
            provider_group_ids = ", ".join(str(id) for id in data_origin.provider_group_id)
            parts.append(f"providerGroupId: [{provider_group_ids}]")
        if data_origin.provider_id:
            provider_ids = ", ".join(str(id) for id in data_origin.provider_id)
            parts.append(f"providerId: [{provider_ids}]")
        return f"dataOrigin: {{{', '.join(parts)}}}"

    @staticmethod
    def _generate_filter(filter_obj: FilterType) -> str:
        """Generate filter GraphQL from filter models"""

        match filter_obj:
            case LogicalFilter():
                children_str = ", ".join([GraphQLQueryGenerator._generate_filter(child) for child in filter_obj.children])
                return f"""{{
                    node: {{
                        logicalOperator: {filter_obj.operator},
                        children: [{children_str}]
                    }}
                }}"""

            case IntegerFilter():
                return f'''{{
                    leaf: {{
                        integerCaseFilter: {{
                            property: "{filter_obj.property}",
                            operator: "{filter_obj.operator.value}",
                            value: {filter_obj.value}
                        }}
                    }}
                }}'''

            case BooleanFilter():
                return f'''{{
                    leaf: {{
                        booleanCaseFilter: {{
                            property: "{filter_obj.property}",
                            value: {str(filter_obj.value).lower()}
                        }}
                    }}
                }}'''

            case SexFilter():
                return f"""{{
                    leaf: {{
                        enumCaseFilter: {{
                            sexType: {{
                                values: [{filter_obj.sex_type.value}],
                                contains: {str(filter_obj.contains).lower()}
                            }}
                        }}
                    }}
                }}"""

            case StrokeFilter():
                return f"""{{
                    leaf: {{
                        enumCaseFilter: {{
                            strokeType: {{
                                values: [{filter_obj.stroke_type.value}],
                                contains: {str(filter_obj.contains).lower()}
                            }}
                        }}
                    }}
                }}"""

            case DateFilter():
                return f'''{{
                    leaf: {{
                        dateCaseFilter: {{
                            property: {filter_obj.property},
                            operator: {filter_obj.operator.value},
                            value: "{filter_obj.value}"
                        }}
                    }}
                }}'''

            case _:
                return "{}"

    @staticmethod
    def _generate_metric_field(metric: MetricRequest) -> str:
        """Generate GraphQL field for a metric request"""

        alias = metric.alias or f"metric_{metric.metric_type.value}"

        kpi_fields = ["caseCount"]

        if metric.include_stats:
            kpi_fields.extend(["percents", "normalizedPercents", "cohortSize", "normalizedCohortSize", "median", "mean", "variance", "confidenceIntervalMean", "confidenceIntervalMedian", "interquartileRange", "quartiles"])

        if metric.include_distribution and metric.distribution_options:
            kpi_fields.append(f"""
                d1: distribution(binCount: {metric.distribution_options.bin_count}) {{
                    edges
                    caseCount
                    percents
                    normalizedPercents
                }}
            """)

        kpi_options = ""
        if metric.metric_options:
            options: List[str] = []
            if metric.metric_options.lower_boundary is not None:
                options.append(f"lowerBoundary: {metric.metric_options.lower_boundary}")
            if metric.metric_options.upper_boundary is not None:
                options.append(f"upperBoundary: {metric.metric_options.upper_boundary}")

            if options:
                kpi_options = f"kpiOptions: {{{', '.join(options)}}}"

        kpi_call = f"kpi({kpi_options})" if kpi_options else "kpi"
        kpi_group_fields = [
            f"""
            kpi1: {kpi_call} {{
                {" ".join(kpi_fields)}
            }}
        """
        ]

        if metric.include_grouping:
            kpi_group_fields.append("""
                groupedBy {
                    groupItemName
                }
            """)

        return f"""
            {alias}: metric(metricId: {metric.metric_type.value}) {{
                kpiGroup {{
                    {" ".join(kpi_group_fields)}
                }}
            }}
        """

    @staticmethod
    def _clean_query(query: str) -> str:
        """Clean up the generated query string"""
        import re

        query = re.sub(r"\s+", " ", query)
        query = re.sub(r"\s*{\s*", " { ", query)
        query = re.sub(r"\s*}\s*", " } ", query)
        query = query.strip()

        return query


def create_age_filter(operator: Operator, value: int) -> IntegerFilter:
    """Create an age filter"""
    return IntegerFilter(property="AGE", operator=operator, value=value)


def create_sex_filter(sex: SexType, contains: bool = True) -> SexFilter:
    """Create a sex filter"""
    return SexFilter(sexType=sex, contains=contains)


def create_stroke_filter(stroke_type: StrokeType, contains: bool = True) -> StrokeFilter:
    """Create a stroke type filter"""
    return StrokeFilter(strokeType=stroke_type, contains=contains)


def create_and_filter(*filters: FilterType) -> LogicalFilter:
    """Create an AND logical filter"""
    return LogicalFilter(operator="AND", children=list(filters))


def create_or_filter(*filters: FilterType) -> LogicalFilter:
    """Create an OR logical filter"""
    return LogicalFilter(operator="OR", children=list(filters))


def create_not_filter(filter_obj: FilterType) -> LogicalFilter:
    """Create a NOT logical filter"""
    return LogicalFilter(operator="NOT", children=[filter_obj])
