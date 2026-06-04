import json
from typing import Dict, List, Tuple

from src.domain.langchain.schema import (
    AnalysisPlan,
    ChartSpec,
    DateFilter,
    GroupByCanonicalField,
    GroupBySex,
    GroupByStrokeType,
    GroupByTime,
    MetricSpec,
    OriginScopeSpec,
    SexFilter,
    StatisticalTestSpec,
    TimeWindow,
)


def example_dtn_by_sex() -> Tuple[str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="COMPARISON",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[
                    MetricSpec(
                        metric="DTN",
                    )
                ],
            )
        ],
        statistical_tests=None,
    )
    user = f"USER_UTTERANCE:\nShow me a line graph of DTN for males and females\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_by_first_contact_place() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["LINE"],
        "group_by": ["FIRST_CONTACT_PLACE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="COMPARISON",
                group_by=[GroupByCanonicalField(field="FIRST_CONTACT_PLACE")],
                metrics=[
                    MetricSpec(
                        metric="DTN",
                    )
                ],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line graph of DTN grouped by first contact place\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_over_time() -> Tuple[str, str]:
    """Plain 'over time' request without explicit grain → LINE/TIME_SERIES with monthly grouping."""
    detected_entities = {"metric": ["DTN"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="TIME_SERIES",
                group_by=[GroupByTime(grain="MONTH")],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nShow me DTN over time\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_monthly_trend() -> Tuple[str, str]:
    detected_entities = {"metric": ["DTN"], "chart_type": ["LINE"], "time_window": ["MONTH"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="TIME_SERIES",
                group_by=[GroupByTime(grain="MONTH")],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nShow me the monthly trend of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_distribution_by_month() -> Tuple[str, str]:
    """'Distribution of DTN by month' \u2192 flat DISTRIBUTION (no GroupByTime; value spread over the period)."""
    detected_entities = {"metric": ["DTN"], "chart_type": ["HISTOGRAM"], "group_by": ["month"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="HISTOGRAM",
                analysis_mode="DISTRIBUTION",
                group_by=None,
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nShow me the distribution of DTN by month\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_box_by_sex() -> Tuple[str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["BOX"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BOX",
                analysis_mode="SUMMARY",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[
                    MetricSpec(
                        metric="DTN",
                    )
                ],
            )
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nShow me a box plot of DTN by sex\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_males_only_filter() -> Tuple[str, str]:
    detected_entities = {"sex": ["MALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="COMPARISON",
                filters=SexFilter(value="MALE"),
                group_by=None,
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line graph of DTN for males only\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_females_only_filter() -> Tuple[str, str]:
    detected_entities = {"sex": ["FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="COMPARISON",
                filters=SexFilter(value="FEMALE"),
                group_by=None,
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line graph of DTN for females only\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_by_sex_and_stroke() -> Tuple[str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "metric": ["DTN"],
        "chart_type": ["BAR"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                analysis_mode="COMPARISON",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"]), GroupByStrokeType()],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = f"USER_UTTERANCE:\nShow me a bar chart of DTN for males and females, grouped by stroke type\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_one_graph_cross_split() -> Tuple[str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="COMPARISON",
                group_by=[
                    GroupBySex(categories=["MALE", "FEMALE"]),
                    GroupByStrokeType(
                        categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"]
                    ),
                ],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow DTN in one graph split by sex and stroke type\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_two_separate_charts() -> Tuple[str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"],
        "metric": ["DTN"],
        "chart_type": ["LINE", "BAR"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="COMPARISON",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[MetricSpec(metric="DTN")],
            ),
            ChartSpec(
                chart_type="BAR",
                analysis_mode="COMPARISON",
                group_by=[GroupByStrokeType(categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"])],
                metrics=[MetricSpec(metric="DTN")],
            ),
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nCreate two charts: DTN by sex and DTN by stroke type\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_last_6_months_by_sex() -> Tuple[str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
        "time_window": ["LAST_6_MONTHS"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                analysis_mode="TIME_SERIES",
                group_by=[
                    GroupByTime(
                        grain="MONTH", window=TimeWindow(last_n=6, unit="MONTH")
                    ),
                    GroupBySex(categories=["MALE", "FEMALE"]),
                ],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me DTN by sex over the last 6 months, monthly\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_statistical_test_dtn_by_sex() -> Tuple[str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "metric": ["DTN"],
        "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
    }
    plan = AnalysisPlan(
        charts=None,
        statistical_tests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[
                    MetricSpec(
                        metric="DTN",
                    ),
                    MetricSpec(
                        metric="DTN",
                    ),
                ],
            )
        ],
    )
    user = f"USER_UTTERANCE:\nRun a Mann-Whitney U test comparing DTN between male and female patients\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_my_hospital_vs_country_average() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["BAR"],
        "scope": ["mine", "country_average"],
        "country": ["IT"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                analysis_mode="COMPARISON",
                group_by=None,
                metrics=[
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="mine", label="My hospital"
                        ),
                    ),
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="country_average",
                            countryCode="IT",
                            label="Italy national average",
                        ),
                    ),
                ],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nCompare DTN for my hospital against the Italy national average\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_my_hospital_vs_provider_group_name() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["BAR"],
        "scope": ["mine", "provider_group_name"],
        "provider_group_name": ["Nordic Stroke Network"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                analysis_mode="COMPARISON",
                group_by=None,
                metrics=[
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="mine", label="My hospital"
                        ),
                    ),
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="provider_group_name",
                            value="Nordic Stroke Network",
                            label="Nordic Stroke Network",
                        ),
                    ),
                ],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow DTN for my hospital versus provider group Nordic Stroke Network\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_mw_my_hospital_vs_national() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "scope": ["mine", "country_average"],
        "statistical_test_type": ["MANN_WHITNEY_U_TEST"],
    }
    plan = AnalysisPlan(
        charts=None,
        statistical_tests=[
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                group_by=None,
                metrics=[
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="mine", label="My hospital"
                        ),
                    ),
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="country_average", label="National mean"
                        ),
                    ),
                ],
            )
        ],
    )
    user = (
        "USER_UTTERANCE:\nCompare DTN between my hospital and the national mean\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_year_filter() -> Tuple[str, str]:
    detected_entities = {"metric": ["DTN"], "chart_type": ["BAR"], "date": ["2026"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
                analysis_mode="SUMMARY",
                filters=DateFilter(operator="GE", value="2026-01-01"),
                group_by=None,
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a bar chart of DTN for patients in 2026\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_mw_hospital_vs_hospital() -> Tuple[str, str]:
    detected_entities = {
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
                group_by=None,
                metrics=[
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="provider_name",
                            value="City Stroke Center",
                            label="City Stroke Center",
                        ),
                    ),
                    MetricSpec(
                        metric="DTN",
                        originScope=OriginScopeSpec(
                            scopeType="provider_name",
                            value="University Hospital",
                            label="University Hospital",
                        ),
                    ),
                ],
            )
        ],
    )
    user = (
        "USER_UTTERANCE:\nIs there a significant difference in DTN between City Stroke Center and University Hospital?\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


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
        examples.append(
            {
                "user": user,
                "assistant": assistant,
            }
        )
    return examples
