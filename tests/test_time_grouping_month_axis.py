from types import SimpleNamespace

from src.domain.langchain import schema as S
from src.executors.mapping.series_mapper import map_metrics_payload_to_series
from src.executors.planning.query_compiler import compile_chart_grouping


def test_compile_chart_grouping_derives_month_periods_from_date_filters() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=[S.GroupByTime(grain="MONTH")],
        filters=S.AndFilter(
            and_=[
                S.DateFilter(type="DateFilter", operator="GE", value="2022-10-01"),
                S.DateFilter(type="DateFilter", operator="LE", value="2023-09-30"),
            ]
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 12
    assert batch.batched_time_periods[0].start_date == "2022-10-01"
    assert batch.batched_time_periods[-1].end_date == "2023-09-30"


def test_map_metrics_payload_to_series_uses_month_label_for_time_periods() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=14.0,
            case_count=[],
            d1=None,
        ),
        grouped_by=None,
        time_period=SimpleNamespace(start_date="2023-02-01", end_date="2023-02-28"),
        data_origin=None,
    )
    metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field=None,
        add_time_period_labels=True,
    )

    assert len(series) == 1
    assert len(series[0].data) == 1
    assert series[0].data[0].x == "2023-02"


def test_map_metrics_payload_to_series_emits_missing_time_point_as_null() -> None:
    kpi = SimpleNamespace(
        kpi1=SimpleNamespace(
            median=None,
            mean=None,
            case_count=[],
            d1=None,
        ),
        grouped_by=None,
        time_period=SimpleNamespace(start_date="2023-03-01", end_date="2023-03-31"),
        data_origin=None,
    )
    metric_payload = {"metric_DTN": SimpleNamespace(kpi_group=[kpi])}

    series = map_metrics_payload_to_series(
        metrics_payload=metric_payload,
        label_parts=[],
        include_metric_alias=True,
        group_by_field=None,
        add_time_period_labels=True,
    )

    assert len(series) == 1
    assert len(series[0].data) == 1
    assert series[0].data[0].x == "2023-03"
    assert series[0].data[0].y is None


def test_compile_chart_grouping_month_and_sex_defaults_to_recent_12_months() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        group_by=[
            S.GroupByTime(grain="MONTH"),
            S.GroupBySex(categories=["MALE", "FEMALE"]),
        ],
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 12
    # Split-by-sex remains as filter-driven series fan-out on one chart.
    assert len(batch.combos_list) == 2
