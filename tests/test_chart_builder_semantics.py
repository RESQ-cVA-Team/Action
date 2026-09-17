import unittest

from src.domain.dto.charts.types import ChartPoint, ChartSeries
from src.domain.langchain import schema as S
from src.executors.mapping.chart_builder import build_chart_dto
from src.executors.planning.query_compiler import compile_chart_grouping


class ChartBuilderSemanticsTests(unittest.TestCase):
    def test_line_chart_with_semantic_time_is_not_smoothed(self) -> None:
        chart = S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="TREND",
                measure=S.MeasureSemanticsSpec(type="MEAN"),
                time=S.TimeSemanticsSpec(grain="MONTH"),
            ),
            filters=S.AndFilter(
                and_=[
                    S.DateFilter(type="DateFilter", operator="GE", value="2023-01-01"),
                    S.DateFilter(type="DateFilter", operator="LE", value="2023-12-31"),
                ]
            ),
        )

        compiled = compile_chart_grouping(chart)
        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=compiled.dimensions,
            series=[ChartSeries(name="DTN", data=[])],
            derived_axes=None,
        )

        self.assertFalse(dto.smooth)

    def test_line_chart_without_time_dimension_is_smoothed(self) -> None:
        chart = S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="COMPARISON",
                measure=S.MeasureSemanticsSpec(type="MEAN"),
                splits=[S.SplitSpec(kind="SEX", categories=["MALE", "FEMALE"])],
            ),
        )

        compiled = compile_chart_grouping(chart)
        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=compiled.dimensions,
            series=[ChartSeries(name="DTN", data=[])],
            derived_axes=None,
        )

        self.assertTrue(dto.smooth)

    def test_unsupported_chart_type_is_rejected_by_schema(self) -> None:
        with self.assertRaises(ValueError):
            S.ChartSpec(
                chart_type="UNSUPPORTED_SHAPE",
                metrics=[S.MetricSpec(metric="DTN")],
                semantics=S.AnalysisSemanticsSpec(
                    intent="COMPARISON",
                    measure=S.MeasureSemanticsSpec(type="MEAN"),
                ),
            )

    def test_histogram_chart_raises_on_non_numeric_bin_values(self) -> None:
        chart = S.ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="DISTRIBUTION",
                measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            ),
        )

        with self.assertRaises(ValueError) as err:
            build_chart_dto(
                plan_chart=chart,
                dimensions=[],
                series=[
                    ChartSeries(
                        name="DTN",
                        data=[ChartPoint.model_construct(x="bad-x", y="bad-y")],
                    )
                ],
                derived_axes=None,
            )

        self.assertIn("Expected numeric chart value", str(err.exception))

    def test_histogram_uses_consecutive_bin_intervals_for_final_bin(self) -> None:
        chart = S.ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="DISTRIBUTION",
                measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            ),
        )

        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=[],
            series=[
                ChartSeries(
                    name="DTN",
                    data=[
                        ChartPoint.model_construct(x=30, y=4),
                        ChartPoint.model_construct(x=33, y=5),
                        ChartPoint.model_construct(x=36, y=6),
                        ChartPoint.model_construct(x=39, y=8),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual(len(dto.data), 4)
        self.assertEqual(dto.data[-1].range_start, 39.0)
        self.assertEqual(dto.data[-1].range_end, 42.0)
        self.assertEqual(dto.bin_width, 3.0)

    def test_histogram_trims_leading_and_trailing_zero_bins_but_preserves_original_bin_count(self) -> None:
        chart = S.ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="DISTRIBUTION",
                measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            ),
        )

        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=[],
            series=[
                ChartSeries(
                    name="DTN",
                    data=[
                        ChartPoint.model_construct(x=27, y=0),
                        ChartPoint.model_construct(x=30, y=4),
                        ChartPoint.model_construct(x=33, y=5),
                        ChartPoint.model_construct(x=36, y=0),
                        ChartPoint.model_construct(x=39, y=0),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual(len(dto.data), 2)
        self.assertEqual(dto.bin_count, 5)
        self.assertEqual(dto.data[0].range_start, 30.0)
        self.assertEqual(dto.data[-1].range_start, 33.0)

    def test_histogram_all_zero_tail_trims_to_empty_data(self) -> None:
        chart = S.ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="DISTRIBUTION",
                measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            ),
        )

        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=[],
            series=[
                ChartSeries(
                    name="DTN",
                    data=[
                        ChartPoint.model_construct(x=30, y=0),
                        ChartPoint.model_construct(x=33, y=0),
                        ChartPoint.model_construct(x=36, y=0),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual(dto.data, [])
        self.assertEqual(dto.bin_count, 3)

    def test_histogram_single_surviving_bin_preserves_original_bin_width_and_range(self) -> None:
        chart = S.ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="DISTRIBUTION",
                measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            ),
        )

        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=[],
            series=[
                ChartSeries(
                    name="DTN",
                    data=[
                        ChartPoint.model_construct(x=0, y=0),
                        ChartPoint.model_construct(x=3, y=0),
                        ChartPoint.model_construct(x=6, y=9),
                        ChartPoint.model_construct(x=9, y=0),
                        ChartPoint.model_construct(x=12, y=0),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual(dto.bin_count, 5)
        self.assertEqual(dto.bin_width, 3.0)
        self.assertEqual(len(dto.data), 1)
        self.assertEqual(dto.data[0].range_start, 6.0)
        self.assertEqual(dto.data[0].range_end, 9.0)
        self.assertEqual(dto.data[0].frequency, 9.0)

    def test_bar_distribution_trims_zero_edges(self) -> None:
        chart = S.ChartSpec(
            chart_type="BAR",
            metrics=[S.MetricSpec(metric="SYSTOLIC_PRESSURE")],
        )

        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=[],
            series=[
                ChartSeries(
                    name="SYSTOLIC_PRESSURE",
                    data=[
                        ChartPoint.model_construct(x=20, y=0),
                        ChartPoint.model_construct(x=30, y=5),
                        ChartPoint.model_construct(x=40, y=3),
                        ChartPoint.model_construct(x=50, y=0),
                        ChartPoint.model_construct(x=60, y=0),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual([point.x for point in dto.series[0].data], [30, 40])

    def test_histogram_preserves_internal_zero_bins(self) -> None:
        chart = S.ChartSpec(
            chart_type="HISTOGRAM",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="DISTRIBUTION",
                measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            ),
        )

        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=[],
            series=[
                ChartSeries(
                    name="DTN",
                    data=[
                        ChartPoint.model_construct(x=20, y=0),
                        ChartPoint.model_construct(x=30, y=4),
                        ChartPoint.model_construct(x=40, y=0),
                        ChartPoint.model_construct(x=50, y=2),
                        ChartPoint.model_construct(x=60, y=0),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual([bin_item.range_start for bin_item in dto.data], [30.0, 40.0, 50.0])
        self.assertEqual([bin_item.frequency for bin_item in dto.data], [4.0, 0.0, 2.0])

    def test_line_with_categorical_dimension_does_not_trim_tail(self) -> None:
        chart = S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="COMPARISON",
                measure=S.MeasureSemanticsSpec(type="MEAN"),
                splits=[S.SplitSpec(kind="SEX", categories=["MALE", "FEMALE"])],
            ),
        )

        compiled = compile_chart_grouping(chart)
        dto = build_chart_dto(
            plan_chart=chart,
            dimensions=compiled.dimensions,
            series=[
                ChartSeries(
                    name="DTN",
                    data=[
                        ChartPoint.model_construct(x="Male", y=8),
                        ChartPoint.model_construct(x="Female", y=0),
                    ],
                )
            ],
            derived_axes=None,
        )

        self.assertEqual([point.x for point in dto.series[0].data], ["Male", "Female"])

    def test_pie_chart_raises_on_non_numeric_slice_value(self) -> None:
        chart = S.ChartSpec(
            chart_type="PIE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(
                intent="COMPARISON",
                measure=S.MeasureSemanticsSpec(type="COUNT"),
            ),
        )

        with self.assertRaises(ValueError) as err:
            build_chart_dto(
                plan_chart=chart,
                dimensions=[],
                series=[
                    ChartSeries(
                        name="DTN",
                        data=[ChartPoint.model_construct(x="A", y="not-a-number")],
                    )
                ],
                derived_axes=None,
            )

        self.assertIn("Expected numeric chart value", str(err.exception))


if __name__ == "__main__":
    unittest.main()
