import unittest

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.langchain.schema import AndFilter, ChartSpec, DateFilter, GroupBySex, GroupByTime, MetricSpec, StrokeFilter
from src.executors.mapping.chart_builder import _derive_title, build_chart_dto
from src.executors.planning.query_compiler import Dimension


def _chart(metric: str = "DTN", chart_type: str = "BAR", filters=None) -> ChartSpec:
    return ChartSpec(chart_type=chart_type, metrics=[MetricSpec(metric=metric)], filters=filters)


class ChartTitleTests(unittest.TestCase):
    def test_plain_metric_without_dimensions(self) -> None:
        self.assertEqual(_derive_title(_chart(), []), "Door to needle")

    def test_distribution_carries_its_value_range(self) -> None:
        title = _derive_title(_chart(), [], value_range="30 to 290 minutes", is_distribution=True)
        self.assertEqual(title, "Door to needle distribution, 30 to 290 minutes")

    def test_time_grouping_reads_per_grain(self) -> None:
        title = _derive_title(_chart(chart_type="LINE"), [Dimension(GroupByTime(grain="QUARTER"))], sampled_period_override="2024-08-24 to 2026-08-24")
        self.assertEqual(title, "Door to needle per quarter, 2024-08-24 to 2026-08-24")

    def test_split_and_filter_are_told_apart(self) -> None:
        title = _derive_title(_chart(filters=StrokeFilter(value="ISCHEMIC")), [Dimension(GroupBySex())])
        self.assertEqual(title, "Door to needle by sex type, filtered by stroke type = ischemic")

    def test_date_filters_become_the_period_not_a_filter(self) -> None:
        filters = AndFilter(and_=[DateFilter(operator="GE", value="2025-01-01"), DateFilter(operator="LE", value="2025-12-31")])
        self.assertEqual(_derive_title(_chart(filters=filters), []), "Door to needle, 2025-01-01 to 2025-12-31")

    def test_built_distribution_chart_states_the_range_actually_rendered(self) -> None:
        series = [ChartSeries(name="DTN", data=[ChartPoint(x=30, y=4), ChartPoint(x=60, y=7), ChartPoint(x=90, y=1)])]
        chart = build_chart_dto(plan_chart=_chart(), dimensions=[], series=series, derived_axes=None, apply_numeric_tail_trim=False)
        self.assertEqual(chart.metadata.title, "Door to needle distribution, 30 to 120 minutes")


if __name__ == "__main__":
    unittest.main()
