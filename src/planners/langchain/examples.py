import json
from typing import Dict, List, Tuple

from src.domain.langchain.schema import AnalysisPlan, ChartSpec, GroupByCanonicalField, GroupBySex, GroupByStrokeType, GroupByTime, MetricSpec, StatisticalTestSpec, TimeWindow


def example_dtn_by_sex() -> Tuple[str, str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                title="DTN by Sex",
                description="Line graph of Door-to-Needle Time for males and females.",
                chart_type="LINE",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[
                    MetricSpec(
                        title="DTN for Males",
                        description="Average DTN for male patients",
                        metric="DTN",
                    )
                ],
            )
        ],
        statistical_tests=None,
    )
    desc = "Line graph of DTN for males and females."
    user = f"USER_UTTERANCE:\nShow me a line graph of DTN for males and females\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
    assistant = plan.model_dump_json(indent=2)
    return desc, user, assistant


def example_dtn_by_first_contact_place() -> Tuple[str, str, str]:
    detected_entities = {"metric": ["DTN"], "chart_type": ["LINE"], "group_by": ["FIRST_CONTACT_PLACE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                title="DTN by First Contact Place",
                description="Line graph of Door-to-Needle Time grouped by first contact place.",
                chart_type="LINE",
                group_by=[GroupByCanonicalField(field="FIRST_CONTACT_PLACE")],
                metrics=[
                    MetricSpec(
                        title="DTN",
                        description="Average Door-to-Needle Time across different first contact places",
                        metric="DTN",
                    )
                ],
            )
        ],
        statistical_tests=None,
    )
    desc = "Line graph of DTN grouped by first contact place."
    user = "USER_UTTERANCE:\nShow me a line graph of DTN grouped by first contact place\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return desc, user, assistant


def example_dtn_distribution_line() -> Tuple[str, str, str]:
    detected_entities = {"metric": ["DTN"], "chart_type": ["LINE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                title="DTN Distribution",
                description="Line visualization of the distribution of Door-to-Needle Time.",
                chart_type="LINE",
                group_by=None,
                metrics=[
                    MetricSpec(
                        title="DTN",
                        description="Distribution of DTN",
                        metric="DTN",
                    )
                ],
            )
        ],
        statistical_tests=None,
    )
    desc = "Line visualization of the distribution of DTN (no explicit time axis)."
    user = "USER_UTTERANCE:\nShow me a line graph of DTN\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return desc, user, assistant


def example_dtn_by_sex_and_stroke() -> Tuple[str, str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "chart_type": ["BAR"], "stroke_type": ["ISCHEMIC", "INTRACEREBRAL_HEMORRHAGE"]}
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                title="DTN by Sex and Stroke Type",
                description="Bar chart of Door-to-Needle Time for males and females, grouped by stroke type.",
                chart_type="BAR",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"]), GroupByStrokeType()],
                metrics=[MetricSpec(title="DTN", description="DTN across sex and stroke types", metric="DTN")],
            )
        ],
        statistical_tests=None,
    )
    desc = "Bar chart of DTN for males and females, grouped by stroke type."
    user = f"USER_UTTERANCE:\nShow me a bar chart of DTN for males and females, grouped by stroke type\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
    assistant = plan.model_dump_json(indent=2)
    return desc, user, assistant


def example_dtn_last_6_months_by_sex() -> Tuple[str, str, str]:
    detected_entities = {
        "sex": ["MALE", "FEMALE"],
        "metric": ["DTN"],
        "chart_type": ["LINE"],
        "time_window": ["LAST_6_MONTHS"],
    }
    plan = AnalysisPlan(
        charts=[
            ChartSpec(
                title="DTN by Sex over Last 6 Months",
                description="Line graph of DTN for males and females over the last 6 months, monthly.",
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
    desc = "Line graph of DTN by sex over the last 6 months."
    user = "USER_UTTERANCE:\nShow me DTN by sex over the last 6 months, monthly\n\nENTITIES_DETECTED(JSON):\n" + json.dumps(detected_entities)
    assistant = plan.model_dump_json(indent=2)
    return desc, user, assistant


def example_statistical_test_dtn_by_sex() -> Tuple[str, str, str]:
    detected_entities = {"sex": ["MALE", "FEMALE"], "metric": ["DTN"], "statistical_test_type": ["T_TEST"]}
    plan = AnalysisPlan(
        charts=None,
        statistical_tests=[
            StatisticalTestSpec(
                title="T-Test for DTN by Sex",
                description="T-Test comparing DTN between male and female patients.",
                test_type="T_TEST",
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
                metrics=[
                    MetricSpec(
                        title="DTN for Males",
                        description="DTN for male patients",
                        metric="DTN",
                    ),
                    MetricSpec(
                        title="DTN for Females",
                        description="DTN for female patients",
                        metric="DTN",
                    ),
                ],
            )
        ],
    )
    desc = "T-Test comparing DTN between male and female patients."
    user = f"USER_UTTERANCE:\nRun a t-test comparing DTN between male and female patients\n\nENTITIES_DETECTED(JSON):\n{json.dumps(detected_entities)}"
    assistant = plan.model_dump_json(indent=2)
    return desc, user, assistant


def get_few_shot_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for desc, user, assistant in [
        example_dtn_last_6_months_by_sex(),
        example_dtn_distribution_line(),
        example_dtn_by_first_contact_place(),
        example_dtn_by_sex(),
        example_dtn_by_sex_and_stroke(),
        example_statistical_test_dtn_by_sex(),
    ]:
        examples.append(
            {
                "description": desc,
                "user": user,
                "assistant": assistant,
            }
        )
    return examples
