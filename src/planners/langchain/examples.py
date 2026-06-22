import json
from typing import Dict, List, Tuple

from src.domain.langchain.schema import (
    AnalysisPlan,
    AndFilter,
    CategoryXAxis,
    CountAxis,
    GroupByCanonicalField,
    GroupBySex,
    GroupByStrokeType,
    HistogramChartSpec,
    LineChartSpec,
    LineSeries,
    MetricSpec,
    MetricValueAxis,
    NotFilter,
    NumericMetricXAxis,
    OriginScopeSpec,
    PredicateFilter,
    StatisticalTestSpec,
    TimeWindow,
    TimeXAxis,
)


def _assistant(plan: AnalysisPlan) -> str:
    return plan.model_dump_json(indent=2, by_alias=True)


def _line_plan(
    *,
    metric: str,
    x_axis: TimeXAxis | CategoryXAxis,
    chart_type: str = "LINE",
    filters: PredicateFilter | AndFilter | NotFilter | None = None,
    series: List[LineSeries] | None = None,
) -> AnalysisPlan:
    chart_series = series or [LineSeries(metric=metric, xAxis="x1", yAxis="y1")]
    return AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType=chart_type,
                xAxes={"x1": x_axis},
                yAxes={"y1": MetricValueAxis(kind="metric_value", statistic="MEAN")},
                series=chart_series,
                filters=filters,
            )
        ],
    )


def _histogram_plan(metric: str) -> AnalysisPlan:
    return AnalysisPlan(
        charts=[
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric=metric),
                yAxis=CountAxis(kind="count"),
            )
        ],
    )


def example_dtn_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": NumericMetricXAxis(kind="numeric_metric", metric="DTN")},
                yAxes={"y1": CountAxis(kind="count")},
                series=[LineSeries(metric="DTN", xAxis="x1", yAxis="y1")],
                seriesSplit=GroupBySex(categories=["MALE", "FEMALE"]),
            )
        ],
    )
    user = f"USER_UTTERANCE:\nShow me a line graph of DTN for males and females\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected)}"
    return user, _assistant(plan)


def example_dtn_by_first_contact_place() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["LINE"], "group_by": ["FIRST_CONTACT_PLACE"]}
    plan = _line_plan(metric="DTN", x_axis=CategoryXAxis(kind="category", groupBy=GroupByCanonicalField(field="FIRST_CONTACT_PLACE")))
    user = (
        "USER_UTTERANCE:\nShow me a line graph of DTN grouped by first contact place\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected)
    )
    return user, _assistant(plan)


def example_dtn_over_time() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"]}
    plan = _histogram_plan("DTN")
    user = "USER_UTTERANCE:\nShow me DTN over time\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_metric_only_default() -> Tuple[str, str]:
    detected = {"metric": ["DTN"]}
    plan = _histogram_plan("DTN")
    user = "USER_UTTERANCE:\nShow me DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_monthly_trend() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"], "time_window": ["MONTH"]}
    plan = _histogram_plan("DTN")
    user = "USER_UTTERANCE:\nShow me the monthly trend of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_line_chart_no_grouping() -> Tuple[str, str]:
    """Explicit LINE entity without grouping -> LINE over metric distribution."""
    detected = {"metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": NumericMetricXAxis(kind="numeric_metric", metric="DTN")},
                yAxes={"y1": CountAxis(kind="count")},
                series=[LineSeries(metric="DTN", xAxis="x1", yAxis="y1")],
            )
        ],
    )
    user = "USER_UTTERANCE:\nShow me a line chart of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_bar_chart_no_grouping() -> Tuple[str, str]:
    """Explicit BAR entity without grouping -> closest supported shape (LINE over time)."""
    detected = {"metric": ["DTN"], "chart_type": ["BAR"]}
    plan = _line_plan(metric="DTN", x_axis=TimeXAxis(kind="time", grain="MONTH"))
    user = "USER_UTTERANCE:\nShow me a bar chart of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_distribution_by_month() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"], "group_by": ["month"]}
    plan = _histogram_plan("DTN")
    user = "USER_UTTERANCE:\nShow me the distribution of DTN by month\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_box_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["BOX"]}
    plan = AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": NumericMetricXAxis(kind="numeric_metric", metric="DTN")},
                yAxes={"y1": CountAxis(kind="count")},
                series=[LineSeries(metric="DTN", xAxis="x1", yAxis="y1")],
                seriesSplit=GroupBySex(categories=["MALE", "FEMALE"]),
            )
        ],
    )
    user = "USER_UTTERANCE:\nShow me a box plot of DTN by sex\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_males_only_filter() -> Tuple[str, str]:
    detected = {"sex": ["MALE"], "metric": ["DTN"], "chart_type": ["HISTOGRAM"]}
    plan = AnalysisPlan(
        charts=[
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                filters=PredicateFilter(op="predicate", field="SEX", operator="EQ", value="MALE"),
            )
        ],
    )
    user = "USER_UTTERANCE:\nShow me DTN for males only\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_females_only_filter() -> Tuple[str, str]:
    detected = {"sex": ["FEMALE"], "metric": ["DTN"], "chart_type": ["HISTOGRAM"]}
    plan = AnalysisPlan(
        charts=[
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                filters=PredicateFilter(op="predicate", field="SEX", operator="EQ", value="FEMALE"),
            )
        ],
    )
    user = "USER_UTTERANCE:\nShow me DTN for females only\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_by_sex_and_stroke() -> Tuple[str, str]:
    detected = {
        "sex": ["MALE", "FEMALE"],
        "metric": ["DTN"],
        "chart_type": ["BAR"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
    }
    plan = AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": NumericMetricXAxis(kind="numeric_metric", metric="DTN")},
                yAxes={"y1": CountAxis(kind="count")},
                series=[
                    LineSeries(
                        metric="DTN",
                        xAxis="x1",
                        yAxis="y1",
                        label="ISCHEMIC",
                        filters=PredicateFilter(op="predicate", field="STROKE_TYPE", operator="EQ", value="ISCHEMIC"),
                    ),
                    LineSeries(
                        metric="DTN",
                        xAxis="x1",
                        yAxis="y1",
                        label="INTRACEREBRAL_HEMORRHAGE",
                        filters=PredicateFilter(
                            op="predicate",
                            field="STROKE_TYPE",
                            operator="EQ",
                            value="INTRACEREBRAL_HEMORRHAGE",
                        ),
                    ),
                ],
                seriesSplit=GroupBySex(categories=["MALE", "FEMALE"]),
            )
        ],
    )
    user = f"USER_UTTERANCE:\nShow me a bar chart of DTN for males and females, grouped by stroke type\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected)}"
    return user, _assistant(plan)


def example_one_graph_cross_split() -> Tuple[str, str]:
    detected = {
        "sex": ["MALE", "FEMALE"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
    }
    plan = example_dtn_by_sex_and_stroke()[1]
    user = "USER_UTTERANCE:\nShow DTN in one graph split by sex and stroke type\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, plan


def example_two_separate_charts() -> Tuple[str, str]:
    detected = {
        "sex": ["MALE", "FEMALE"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
        "metric": ["DTN"],
        "chart_type": ["LINE", "BAR"],
    }
    plan = AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": CategoryXAxis(kind="category", groupBy=GroupBySex(categories=["MALE", "FEMALE"]))},
                yAxes={"y1": MetricValueAxis(kind="metric_value", statistic="MEAN")},
                series=[LineSeries(metric="DTN", xAxis="x1", yAxis="y1")],
            ),
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": CategoryXAxis(kind="category", groupBy=GroupByStrokeType(categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"]))},
                yAxes={"y1": MetricValueAxis(kind="metric_value", statistic="MEAN")},
                series=[LineSeries(metric="DTN", xAxis="x1", yAxis="y1")],
            ),
        ],
    )
    user = "USER_UTTERANCE:\nCreate two charts: DTN by sex and DTN by stroke type\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_last_6_months_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"], "time_window": ["LAST_6_MONTHS"]}
    plan = AnalysisPlan(
        charts=[
            LineChartSpec(
                chartType="LINE",
                xAxes={"x1": TimeXAxis(kind="time", grain="MONTH", window=TimeWindow(last_n=6, unit="MONTH"))},
                yAxes={"y1": MetricValueAxis(kind="metric_value", statistic="MEAN")},
                series=[LineSeries(metric="DTN", xAxis="x1", yAxis="y1")],
                seriesSplit=GroupBySex(categories=["MALE", "FEMALE"]),
            )
        ],
    )
    user = "USER_UTTERANCE:\nShow me DTN by sex over the last 6 months, monthly\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_statistical_test_dtn_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "statistical_test_type": ["MANN_WHITNEY_U_TEST"]}
    plan = AnalysisPlan(
        statisticalTests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[MetricSpec(metric="DTN"), MetricSpec(metric="DTN")],
            )
        ],
    )
    user = f"USER_UTTERANCE:\nRun a Mann-Whitney U test comparing DTN between male and female patients\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected)}"
    return user, _assistant(plan)


def example_dtn_my_hospital_vs_country_average() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"], "scope": ["mine", "country_average"], "country": ["IT"]}
    mine = OriginScopeSpec(scopeType="mine", label="My hospital")
    country = OriginScopeSpec(scopeType="country_average", countryCode="IT", label="Italy national average")
    plan = AnalysisPlan(
        charts=[
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                originScope=mine,
            ),
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                originScope=country,
            ),
        ],
    )
    user = "USER_UTTERANCE:\nCompare DTN for my hospital against the Italy national average\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_my_hospital_vs_provider_group_name() -> Tuple[str, str]:
    detected = {
        "metric": ["DTN"],
        "chart_type": ["HISTOGRAM"],
        "scope": ["mine", "provider_group_name"],
        "provider_group_name": ["Nordic Stroke Network"],
    }
    mine = OriginScopeSpec(scopeType="mine", label="My hospital")
    group = OriginScopeSpec(scopeType="provider_group_name", value="Nordic Stroke Network", label="Nordic Stroke Network")
    plan = AnalysisPlan(
        charts=[
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                originScope=mine,
            ),
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                originScope=group,
            ),
        ],
    )
    user = "USER_UTTERANCE:\nShow DTN for my hospital versus provider group Nordic Stroke Network\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_mw_my_hospital_vs_national() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "scope": ["mine", "country_average"], "statistical_test_type": ["MANN_WHITNEY_U_TEST"]}
    plan = AnalysisPlan(
        statisticalTests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=[
                    MetricSpec(metric="DTN", originScope=OriginScopeSpec(scopeType="mine", label="My hospital")),
                    MetricSpec(metric="DTN", originScope=OriginScopeSpec(scopeType="country_average", label="National mean")),
                ],
            )
        ],
    )
    user = "USER_UTTERANCE:\nCompare DTN between my hospital and the national mean\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_year_filter() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"], "date": ["2026"]}
    plan = AnalysisPlan(
        charts=[
            HistogramChartSpec(
                chartType="HISTOGRAM",
                xAxis=NumericMetricXAxis(kind="numeric_metric", metric="DTN"),
                yAxis=CountAxis(kind="count"),
                filters=AndFilter(
                    op="and",
                    clauses=[
                        PredicateFilter(op="predicate", field="DISCHARGE_DATE", operator="GE", value="2026-01-01"),
                        PredicateFilter(op="predicate", field="DISCHARGE_DATE", operator="LE", value="2026-12-31"),
                    ],
                ),
            )
        ],
    )
    user = "USER_UTTERANCE:\nShow me a bar chart of DTN for patients in 2026\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_mw_hospital_vs_hospital() -> Tuple[str, str]:
    detected = {
        "metric": ["DTN"],
        "scope": ["provider_name"],
        "provider_name": ["City Stroke Center", "University Hospital"],
        "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
    }
    plan = AnalysisPlan(
        statisticalTests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=[
                    MetricSpec(metric="DTN", originScope=OriginScopeSpec(scopeType="provider_name", value="City Stroke Center", label="City Stroke Center")),
                    MetricSpec(metric="DTN", originScope=OriginScopeSpec(scopeType="provider_name", value="University Hospital", label="University Hospital")),
                ],
            )
        ],
    )
    user = "USER_UTTERANCE:\nIs there a significant difference in DTN between City Stroke Center and University Hospital?\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def get_few_shot_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for user, assistant in [
        example_mw_my_hospital_vs_national(),
        example_mw_hospital_vs_hospital(),
        example_dtn_my_hospital_vs_country_average(),
        example_dtn_my_hospital_vs_provider_group_name(),
        example_one_graph_cross_split(),
        example_two_separate_charts(),
        example_dtn_last_6_months_by_sex(),
        example_dtn_line_chart_no_grouping(),
        example_dtn_bar_chart_no_grouping(),
        example_dtn_metric_only_default(),
        example_dtn_over_time(),
        example_dtn_monthly_trend(),
        example_dtn_distribution_by_month(),
        example_dtn_box_by_sex(),
        example_dtn_males_only_filter(),
        example_dtn_females_only_filter(),
        example_dtn_by_first_contact_place(),
        example_dtn_by_sex(),
        example_dtn_by_sex_and_stroke(),
        example_statistical_test_dtn_by_sex(),
        example_dtn_year_filter(),
    ]:
        examples.append({"user": user, "assistant": assistant})
    return examples
