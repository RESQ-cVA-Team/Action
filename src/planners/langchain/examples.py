import json
from typing import Dict, List, Tuple

from src.domain.langchain.schema import AnalysisPlan, ChartSpec, GroupByCanonicalField, GroupBySex, GroupByStrokeType, GroupByTime, MetricSpec, StatisticalTestSpec, TimeWindow


def example_dtn_by_sex() -> Tuple[str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
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
    detected_entities = {"metric": ["DTN"], "chart_type": ["LINE"], "group_by": ["FIRST_CONTACT_PLACE"]}
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
    user = "USER_UTTERANCE:\nShow me a line graph of DTN grouped by first contact place\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
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
    user = "USER_UTTERANCE:\nShow me a line graph of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_dtn_by_sex_and_stroke() -> Tuple[str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["BAR"], "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                chart_type="BAR",
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
                group_by=[
                    GroupBySex(categories=["MALE", "FEMALE"]),
                    GroupByStrokeType(categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"]),
                ],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nShow DTN in one graph split by sex and stroke type\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
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
                group_by=[GroupByStrokeType(categories=["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"])],
                metrics=[MetricSpec(metric="DTN")],
            ),
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nCreate two charts: DTN by sex and DTN by stroke type\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
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
                    GroupByTime(grain="MONTH", window=TimeWindow(last_n=6, unit="MONTH")),
                    GroupBySex(categories=["MALE", "FEMALE"]),
                ],
                metrics=[MetricSpec(metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    user = "USER_UTTERANCE:\nShow me DTN by sex over the last 6 months, monthly\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return user, assistant


def example_statistical_test_dtn_by_sex() -> Tuple[str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "statistical_test_type": ["MANN_WHITNEY_U_TEST"]}
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


def get_few_shot_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for user, assistant in [
        example_one_graph_cross_split(),
        example_two_separate_charts(),
        example_dtn_last_6_months_by_sex(),
        example_dtn_distribution_line(),
        example_dtn_by_first_contact_place(),
        example_dtn_by_sex(),
        example_dtn_by_sex_and_stroke(),
        example_statistical_test_dtn_by_sex(),
    ]:
        examples.append(
            {
                "user": user,
                "assistant": assistant,
            }
        )
    return examples
