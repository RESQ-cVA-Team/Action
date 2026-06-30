import json
from typing import Dict, List, Tuple

from src.domain.langchain.schema import (
    AnalysisPlan,
    AndFilter,
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
    StrokeFilter,
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


def example_dtn_distribution_line() -> Tuple[str, str]:
    detected_entities = {"metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                group_by=None,
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
        "USER_UTTERANCE:\nShow me a line graph of DTN\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_males_only_filter() -> Tuple[str, str]:
    detected_entities = {"sex": ["MALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
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
                group_by=[
                    GroupBySex(categories=["MALE", "FEMALE"]),
                    GroupByStrokeType(),
                ],
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
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[MetricSpec(metric="DTN")],
            ),
            ChartSpec(
                chart_type="BAR",
                group_by=[
                    GroupByStrokeType(
                        categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"]
                    )
                ],
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


def example_filter_time_and_sex() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["LINE"],
        "group_by": ["QUARTER"],
        "stroke_type": ["ISCHEMIC"],
        "sex": ["MALE", "FEMALE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                filters=StrokeFilter(value="ISCHEMIC"),
                group_by=[GroupByTime(grain="QUARTER"), GroupBySex(categories=None)],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nshow DTN per quarter for ischemic patients grouped by sex\n\n"
        "ENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    )
    return user, plan.model_dump_json(indent=2)


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


def example_dtn_quarterly() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["LINE"],
        "group_by": ["quarter"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                group_by=[GroupByTime(grain="QUARTER")],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line chart of DTN per quarter\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_monthly() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["LINE"],
        "group_by": ["month"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                group_by=[GroupByTime(grain="MONTH")],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line chart of DTN per month\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_ischemic_only_filter() -> Tuple[str, str]:
    detected_entities = {
        "stroke_type": ["ISCHEMIC"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                filters=StrokeFilter(value="ISCHEMIC"),
                group_by=None,
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line graph of DTN for ischemic strokes only\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_group_by_stroke_type_bare() -> Tuple[str, str]:
    detected_entities = {
        "stroke_type": ["stroke type"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                group_by=[GroupByStrokeType(categories=None)],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nShow me a line chart of DTN grouped by stroke type\n\nENTITIES_DETECTED(JSON):\n"
        + json.dumps(detected_entities)
    )
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_ischemic_and_female_filter() -> Tuple[str, str]:
    detected_entities = {"sex": ["FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
                filters=AndFilter(
                    and_=[
                        StrokeFilter(value="ISCHEMIC"),
                        SexFilter(value="FEMALE"),
                    ]
                ),
                group_by=[GroupByTime(grain="QUARTER")],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = (
        "USER_UTTERANCE:\nPrevious chart plan (carry over everything except what the user explicitly changes):\n"
        '{"charts": [{"chart_type": "LINE", "filters": {"type": "StrokeFilter", "value": "ISCHEMIC"}, '
        '"group_by": [{"kind": "time", "grain": "QUARTER"}], "metrics": [{"metric": "DTN"}]}]}\n\n'
        "Conversation context (oldest to newest user turns):\nfilter for female patients\n\n"
        "ENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
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


def example_dtn_grouped_by_sex_bare() -> Tuple[str, str]:
    detected_entities = {
        "metric": ["DTN"],
        "chart_type": ["LINE"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="LINE",
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
    user = f"USER_UTTERANCE:\nShow DTN grouped by sex\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
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
        example_dtn_distribution_line(),
        example_dtn_males_only_filter(),
        example_dtn_females_only_filter(),
        example_dtn_ischemic_only_filter(),
        example_dtn_by_first_contact_place(),
        example_dtn_by_sex(),
        example_dtn_by_sex_and_stroke(),
        example_statistical_test_dtn_by_sex(),
        example_dtn_year_filter(),
        example_dtn_quarterly(),
        example_dtn_monthly(),
        example_dtn_ischemic_and_female_filter(),
        example_group_by_stroke_type_bare(),
        example_filter_time_and_sex(),
        example_dtn_grouped_by_sex_bare(),
    ]:
        examples.append(
            {
                "user": user,
                "assistant": assistant,
            }
        )
    return examples
