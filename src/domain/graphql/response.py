from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# -----------------------------
# Base Components
# -----------------------------


class GeneralStatistics(BaseModel):
    cases_in_period: int = Field(..., alias="casesInPeriod")
    filtered_cases_in_period: int = Field(..., alias="filteredCasesInPeriod")

    model_config = ConfigDict(populate_by_name=True)


class GeneralStatsGroup(BaseModel):
    general_statistics: GeneralStatistics = Field(..., alias="generalStatistics")

    model_config = ConfigDict(populate_by_name=True)


class GroupedBy(BaseModel):
    group_item_name: str = Field(..., alias="groupItemName")

    model_config = ConfigDict(populate_by_name=True)


class TimePeriodRef(BaseModel):
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")

    model_config = ConfigDict(populate_by_name=True)


class DataOriginRef(BaseModel):
    provider_group_id: Optional[int] = Field(default=None, alias="providerGroupId")
    provider_id: Optional[int] = Field(default=None, alias="providerId")

    model_config = ConfigDict(populate_by_name=True)


class D1(BaseModel):
    edges: List[float]
    case_count: List[int] = Field(..., alias="caseCount")
    percents: List[Optional[float]]
    normalized_percents: List[Optional[float]] = Field(..., alias="normalizedPercents")

    model_config = ConfigDict(populate_by_name=True)


# -----------------------------
# Unified KPI + Group Models
# -----------------------------


class Kpi1(BaseModel):
    case_count: List[int] = Field(..., alias="caseCount")
    # Upstream may emit null placeholders in numeric arrays.
    percents: Optional[List[Optional[float]]] = None
    normalized_percents: Optional[List[Optional[float]]] = Field(default=None, alias="normalizedPercents")
    cohort_size: Optional[int] = Field(default=None, alias="cohortSize")
    normalized_cohort_size: Optional[List[int]] = Field(default=None, alias="normalizedCohortSize")
    median: Optional[float] = None
    mean: Optional[float] = None
    variance: Optional[float] = None
    confidence_interval_mean: Optional[List[Optional[float]]] = Field(default=None, alias="confidenceIntervalMean")
    confidence_interval_median: Optional[List[Optional[float]]] = Field(default=None, alias="confidenceIntervalMedian")
    interquartile_range: Optional[float] = Field(default=None, alias="interquartileRange")
    quartiles: Optional[List[Optional[float]]] = None
    d1: Optional[D1] = None

    model_config = ConfigDict(populate_by_name=True)


class MetricKpiGroup(BaseModel):
    kpi1: Optional[Kpi1] = None
    grouped_by: Optional[GroupedBy] = Field(default=None, alias="groupedBy")
    time_period: Optional[TimePeriodRef] = Field(default=None, alias="timePeriod")
    data_origin: Optional[DataOriginRef] = Field(default=None, alias="dataOrigin")

    model_config = ConfigDict(populate_by_name=True)


class Metric(BaseModel):
    kpi_group: List[MetricKpiGroup] = Field(..., alias="kpiGroup")

    model_config = ConfigDict(populate_by_name=True)


# -----------------------------
# Top-Level Structures
# -----------------------------


class GetMetrics(BaseModel):
    # API returns a single generalStatsGroup object; accept either object or list for robustness
    general_stats_group: Optional[GeneralStatsGroup | List[GeneralStatsGroup]] = Field(default=None, alias="generalStatsGroup")
    metrics: Optional[Dict[str, Metric]] = Field(default=None, alias="metrics")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    def collect_metrics(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        keys_to_remove: List[str] = []
        for k, v in values.items():
            if k.startswith("metric_"):
                metrics[k] = v
                keys_to_remove.append(k)
        for k in keys_to_remove:
            values.pop(k)
        if metrics:
            values["metrics"] = metrics
        return values


class Data(BaseModel):
    get_metrics: GetMetrics = Field(..., alias="getMetrics")

    model_config = ConfigDict(populate_by_name=True)


class MetricsQueryResponse(BaseModel):
    # GraphQL may return data: null alongside errors; make this optional to surface errors cleanly
    data: Optional[Data] = None
    errors: Optional[List[Dict[str, Any]]] = None  # GraphQL typically uses structured errors

    model_config = ConfigDict(populate_by_name=True)
