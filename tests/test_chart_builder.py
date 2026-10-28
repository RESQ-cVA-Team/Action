import unittest

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.langchain.schema import ChartSpec, GroupBySex, GroupByStrokeType, MetricSpec
from src.executors.mapping.chart_builder import (
    _axis_label_for_dimension,
    _dimension_label,
    _format_operator,
    build_chart_dto,
)
from src.executors.planning.query_compiler import Dimension
from src.executors.planning.ssot_metric_defaults import get_histogram_axes
from src.shared.ssot_loader import get_canonical_display_name


class ChartBuilderTests(unittest.TestCase):
    def test_bar_distribution_without_dimensions_uses_metric_and_cases_axes(
        self,
    ) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            metrics=[MetricSpec(metric="SYSTOLIC_PRESSURE")],
        )
        series = [
            ChartSeries(
                name="SYSTOLIC_PRESSURE",
                data=[
                    ChartPoint(x=0, y=0),
                    ChartPoint(x=30, y=15),
                    ChartPoint(x=60, y=18),
                ],
            )
        ]

        chart = build_chart_dto(
            plan_chart=plan_chart,
            dimensions=[],
            series=series,
            derived_axes=None,
        )

        self.assertIsNotNone(chart.metadata.x_axis)
        self.assertIsNotNone(chart.metadata.y_axis)
        x_axis = chart.metadata.x_axis
        y_axis = chart.metadata.y_axis
        if x_axis is None or y_axis is None:
            self.fail("Expected both axes to be present")
        self.assertEqual(x_axis.label, "initial systolic blood pressure (mmHg)")
        self.assertEqual(x_axis.type.value, "linear")
        self.assertEqual(y_axis.label, "Cases")
        self.assertEqual(y_axis.type.value, "linear")
        self.assertNotIn("angels_awards", chart.model_dump(mode="json")["metadata"])

    def test_line_distribution_without_dimensions_uses_metric_and_cases_axes(
        self,
    ) -> None:
        plan_chart = ChartSpec(
            chart_type="LINE",
            metrics=[MetricSpec(metric="SYSTOLIC_PRESSURE")],
        )
        series = [
            ChartSeries(
                name="SYSTOLIC_PRESSURE",
                data=[
                    ChartPoint(x=0, y=0),
                    ChartPoint(x=30, y=15),
                    ChartPoint(x=60, y=18),
                ],
            )
        ]

        chart = build_chart_dto(
            plan_chart=plan_chart,
            dimensions=[],
            series=series,
            derived_axes=None,
        )

        self.assertIsNotNone(chart.metadata.x_axis)
        self.assertIsNotNone(chart.metadata.y_axis)
        x_axis = chart.metadata.x_axis
        y_axis = chart.metadata.y_axis
        if x_axis is None or y_axis is None:
            self.fail("Expected both axes to be present")
        self.assertEqual(x_axis.label, "initial systolic blood pressure (mmHg)")
        self.assertEqual(x_axis.type.value, "linear")
        self.assertEqual(y_axis.label, "Cases")
        self.assertEqual(y_axis.type.value, "linear")

    def test_angels_awards_thresholds_are_in_serialized_chart_metadata(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            metrics=[MetricSpec(metric="AA_DTN_LE60")],
        )

        chart = build_chart_dto(
            plan_chart=plan_chart,
            dimensions=[],
            series=[ChartSeries(name="Door-to-needle", data=[ChartPoint(x="yes", y=50)])],
            derived_axes=None,
        )

        self.assertEqual(
            chart.model_dump(mode="json")["metadata"]["angels_awards"],
            {"AA_DTN_LE60": {"gold": {"min_percent": 50}, "platinum": {"min_percent": 75}, "diamond": {"min_percent": 75}}},
        )

    def test_angels_awards_levels_without_a_requirement_are_left_out(self) -> None:
        plan_chart = ChartSpec(
            chart_type="BAR",
            metrics=[
                MetricSpec(metric="AA_DTN_LE45"),
                MetricSpec(metric="AA_DTG_LE90"),
                MetricSpec(metric="AA_STROKE_UNIT"),
            ],
        )

        chart = build_chart_dto(
            plan_chart=plan_chart,
            dimensions=[],
            series=[ChartSeries(name="Award metric", data=[ChartPoint(x="yes", y=50)])],
            derived_axes=None,
        )

        self.assertEqual(
            chart.model_dump(mode="json")["metadata"]["angels_awards"],
            {
                "AA_DTN_LE45": {"diamond": {"min_percent": 50}},
                "AA_DTG_LE90": {"diamond": {"min_percent": 50}},
                "AA_STROKE_UNIT": {"diamond": {"required": True}},
            },
        )


class OperatorSymbolTests(unittest.TestCase):
    def test_format_operator_matches_ssot(self) -> None:
        expected = {"GE": ">=", "LE": "<=", "LT": "<", "GT": ">", "EQ": "=", "NE": "!="}
        for code, symbol in expected.items():
            self.assertEqual(_format_operator(code), symbol)


class DimensionLabelSsotConsistencyTests(unittest.TestCase):
    # Sex/StrokeType axis labels used to be hardcoded ("Sex", "Stroke Type");
    # they now come from SSOT like every other dimension label in this module
    # (NIHSS, Age, generic canonical fields already did). This just pins the
    # label to whatever SSOT currently says, so a future SSOT wording change
    # is a deliberate, visible test update rather than a silent surprise.
    def test_axis_label_for_sex_dimension_comes_from_ssot(self) -> None:
        dimension = Dimension(GroupBySex(categories=None))
        expected = get_canonical_display_name("SEX_TYPE")
        self.assertEqual(_axis_label_for_dimension(dimension), expected)
        self.assertEqual(_dimension_label(dimension), expected)

    def test_axis_label_for_stroke_type_dimension_comes_from_ssot(self) -> None:
        dimension = Dimension(GroupByStrokeType(categories=None))
        expected = get_canonical_display_name("STROKE_TYPE")
        self.assertEqual(_axis_label_for_dimension(dimension), expected)
        self.assertEqual(_dimension_label(dimension), expected)


class HistogramAxesTests(unittest.TestCase):
    def test_dtn_histogram_axis_no_longer_needs_a_hardcoded_override(self) -> None:
        x_axis, y_axis = get_histogram_axes("DTN", x_min=0, x_max=520)
        self.assertIn("minutes", x_axis.label)
        self.assertEqual(y_axis.label, "Cases")


if __name__ == "__main__":
    unittest.main()
