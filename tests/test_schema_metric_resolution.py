import pytest
from pydantic import ValidationError

from src.domain.langchain.schema import AxisSemanticsSpec, MetricSpec


def test_metric_code_is_kept() -> None:
    assert MetricSpec(metric="dtn").metric == "DTN"


def test_metric_synonym_resolves_to_its_code() -> None:
    assert MetricSpec(metric="door to needle time").metric == "DTN"


def test_correctly_spelled_name_resolves_to_the_misspelled_code() -> None:
    # The analytics backend spells these codes this way; the English synonym is spelled correctly.
    assert MetricSpec(metric="INTRAVENTRICULAR_HEMORRHAGE").metric == "INTRAVENTICULAR_HEMORRHAGE"
    assert AxisSemanticsSpec(role="METRIC_VALUE", metric="INTRAVENTRICULAR_HEMORRHAGE").metric == "INTRAVENTICULAR_HEMORRHAGE"


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MetricSpec(metric="NOT_A_METRIC")
