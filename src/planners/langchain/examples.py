import json
from typing import Dict, List, Optional, Tuple

from src.domain.langchain.schema import (
    AnalysisPlan,
    CategoryXAxis,
    ChartSpec,
    DateFilter,
    GroupByCanonicalField,
    GroupBySex,
    GroupByStrokeType,
    MetricSpec,
    NumericXAxis,
    OriginScopeSpec,
    ScopeXAxis,
    SeriesSpec,
    SexFilter,
    StatisticalTestSpec,
    TimeWindow,
    TimeXAxis,
    YAxisSpec,
)


def _assistant(plan: AnalysisPlan) -> str:
    return plan.model_dump_json(indent=2)


def _single_metric_yaxis(metric: str, statistic: str = "MEAN", scope: Optional[OriginScopeSpec] = None) -> YAxisSpec:
    return YAxisSpec(metrics=[MetricSpec(metric=metric, originScope=scope)], statistic=statistic)


def example_dtn_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=CategoryXAxis(groupBy=GroupBySex(categories=["MALE", "FEMALE"])),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
    )
    user = f"USER_UTTERANCE:\nShow me a line graph of DTN for males and females\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected)}"
    return user, _assistant(plan)


def example_dtn_by_first_contact_place() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["LINE"], "group_by": ["FIRST_CONTACT_PLACE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=CategoryXAxis(groupBy=GroupByCanonicalField(field="FIRST_CONTACT_PLACE")),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
    )
    user = (
        "USER_UTTERANCE:\nShow me a line graph of DTN grouped by first contact place\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected)
    )
    return user, _assistant(plan)


def example_dtn_over_time() -> Tuple[str, str]:
    detected = {"metric": ["DTN"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=TimeXAxis(grain="MONTH"),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me DTN over time\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_monthly_trend() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["LINE"], "time_window": ["MONTH"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=TimeXAxis(grain="MONTH"),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me the monthly trend of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_distribution_by_month() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"], "group_by": ["month"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="HISTOGRAM",
                xAxis=NumericXAxis(metric="DTN", bins=20),
                yAxes=[_single_metric_yaxis("DTN", statistic="COUNT")],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me the distribution of DTN by month\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_box_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["BOX"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BOX",
                xAxis=CategoryXAxis(groupBy=GroupBySex(categories=["MALE", "FEMALE"])),
                yAxes=[_single_metric_yaxis("DTN", statistic="MEDIAN")],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me a box plot of DTN by sex\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_males_only_filter() -> Tuple[str, str]:
    detected = {"sex": ["MALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=TimeXAxis(grain="MONTH"),
                yAxes=[_single_metric_yaxis("DTN")],
                filters=SexFilter(value="MALE"),
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me a line graph of DTN for males only\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_females_only_filter() -> Tuple[str, str]:
    detected = {"sex": ["FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=TimeXAxis(grain="MONTH"),
                yAxes=[_single_metric_yaxis("DTN")],
                filters=SexFilter(value="FEMALE"),
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me a line graph of DTN for females only\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
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
            ChartSpec(
                chart_type="BAR",
                xAxis=CategoryXAxis(groupBy=GroupBySex(categories=["MALE", "FEMALE"])),
                seriesBy=SeriesSpec(splitBy=GroupByStrokeType()),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
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
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=CategoryXAxis(groupBy=GroupBySex(categories=["MALE", "FEMALE"])),
                seriesBy=SeriesSpec(splitBy=GroupByStrokeType(categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"])),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow DTN in one graph split by sex and stroke type\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_two_separate_charts() -> Tuple[str, str]:
    detected = {
        "sex": ["MALE", "FEMALE"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
        "metric": ["DTN"],
        "chart_type": ["LINE", "BAR"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=CategoryXAxis(groupBy=GroupBySex(categories=["MALE", "FEMALE"])),
                yAxes=[_single_metric_yaxis("DTN")],
            ),
            ChartSpec(
                chart_type="BAR",
                xAxis=CategoryXAxis(groupBy=GroupByStrokeType(categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"])),
                yAxes=[_single_metric_yaxis("DTN")],
            ),
        ]
    )
    user = "USER_UTTERANCE:\nCreate two charts: DTN by sex and DTN by stroke type\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_last_6_months_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"], "time_window": ["LAST_6_MONTHS"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                xAxis=TimeXAxis(grain="MONTH", window=TimeWindow(last_n=6, unit="MONTH")),
                seriesBy=SeriesSpec(splitBy=GroupBySex(categories=["MALE", "FEMALE"])),
                yAxes=[_single_metric_yaxis("DTN")],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow me DTN by sex over the last 6 months, monthly\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_statistical_test_dtn_by_sex() -> Tuple[str, str]:
    detected = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "statistical_test_type": ["MANN_WHITNEY_U_TEST"]}
    plan = AnalysisPlan(
        charts=None,
        statistical_tests=[
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
    detected = {"metric": ["DTN"], "chart_type": ["BAR"], "scope": ["mine", "country_average"], "country": ["IT"]}
    mine = OriginScopeSpec(scopeType="mine", label="My hospital")
    country = OriginScopeSpec(scopeType="country_average", countryCode="IT", label="Italy national average")
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                xAxis=ScopeXAxis(scopes=[mine, country]),
                yAxes=[
                    YAxisSpec(
                        metrics=[
                            MetricSpec(metric="DTN", originScope=mine),
                            MetricSpec(metric="DTN", originScope=country),
                        ],
                        statistic="MEAN",
                    )
                ],
            )
        ]
    )
    user = "USER_UTTERANCE:\nCompare DTN for my hospital against the Italy national average\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_dtn_my_hospital_vs_provider_group_name() -> Tuple[str, str]:
    detected = {
        "metric": ["DTN"],
        "chart_type": ["BAR"],
        "scope": ["mine", "provider_group_name"],
        "provider_group_name": ["Nordic Stroke Network"],
    }
    mine = OriginScopeSpec(scopeType="mine", label="My hospital")
    group = OriginScopeSpec(scopeType="provider_group_name", value="Nordic Stroke Network", label="Nordic Stroke Network")
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                xAxis=ScopeXAxis(scopes=[mine, group]),
                yAxes=[
                    YAxisSpec(
                        metrics=[
                            MetricSpec(metric="DTN", originScope=mine),
                            MetricSpec(metric="DTN", originScope=group),
                        ],
                        statistic="MEAN",
                    )
                ],
            )
        ]
    )
    user = "USER_UTTERANCE:\nShow DTN for my hospital versus provider group Nordic Stroke Network\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected)
    return user, _assistant(plan)


def example_mw_my_hospital_vs_national() -> Tuple[str, str]:
    detected = {"metric": ["DTN"], "scope": ["mine", "country_average"], "statistical_test_type": ["MANN_WHITNEY_U_TEST"]}
    plan = AnalysisPlan(
        charts=None,
        statistical_tests=[
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
    detected = {"metric": ["DTN"], "chart_type": ["BAR"], "date": ["2026"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                xAxis=TimeXAxis(grain="MONTH", window=TimeWindow(last_n=12, unit="MONTH")),
                yAxes=[_single_metric_yaxis("DTN")],
                filters=DateFilter(operator="GE", value="2026-01-01"),
            )
        ]
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
        charts=None,
        statistical_tests=[
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
