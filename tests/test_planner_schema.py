from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.pipeline import SCHEMA_DESCRIPTION, plan_prompt


def test_the_planner_sees_every_object_of_the_plan_once() -> None:
    names = [line.split(" {", 1)[0] for line in SCHEMA_DESCRIPTION.splitlines() if " { " in line]
    assert sorted(names) == sorted(["AnalysisPlan", *AnalysisPlan.model_json_schema()["$defs"]])


def test_the_planner_sees_required_fields_and_allowed_values() -> None:
    lines = SCHEMA_DESCRIPTION.splitlines()
    assert 'NIHSSFilter { operator: EQ|GE|GT|LE|LT|NE; type: "NIHSSFilter"; value: number }' in lines
    assert "DataOriginSpec { provider_id?: [integer]; provider_group_id?: [integer] }" in lines
    assert any(line.startswith("MetricSpec {") and "categories?: [METRIC_OPTIONS key]" in line for line in lines)
    assert any(line.startswith("SplitSpec { kind: AGE|BOOLEAN|CANONICAL|CUSTOM|NIHSS|SEX|STROKE_TYPE; field?: string") for line in lines)
    assert any(line.startswith("ChartSpec { chart_type: ") and "filters?: Filter;" in line for line in lines)


def test_the_schema_reaches_the_planner_prompt_as_written() -> None:
    inputs = {name: "" for name in plan_prompt.input_variables}
    prompt = "\n".join(str(message.content) for message in plan_prompt.format_messages(**inputs))
    assert SCHEMA_DESCRIPTION in prompt
