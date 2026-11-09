import unittest

from pydantic import ValidationError

from src.domain.langchain.schema import ChartSpec, GroupBySex, MetricSpec, StatisticalTestSpec


class SemanticIntentSchemaTests(unittest.TestCase):
    def test_metric_can_request_explicit_category_detail(self) -> None:
        metric = MetricSpec.model_validate({"metric": "AA_DTN_LE60", "includeCategoryDetail": True})

        self.assertTrue(metric.include_category_detail)

    def test_accepts_semantic_distribution_with_split(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "LINE",
                "metrics": [{"metric": "DTN"}],
                "semantics": {
                    "intent": "distribution",
                    "measure": {"type": "distribution"},
                    "splits": [
                        {
                            "kind": "sex",
                            "categories": ["MALE", "FEMALE"],
                        }
                    ],
                    "xAxis": {
                        "role": "metric_value",
                        "metric": "DTN",
                        "unit": "minutes",
                    },
                    "yAxis": {
                        "role": "count",
                    },
                },
            }
        )

        self.assertIsNotNone(chart.semantics)
        semantics = chart.semantics
        self.assertIsNotNone(semantics)
        self.assertEqual(semantics.intent, "DISTRIBUTION")
        self.assertIsNotNone(semantics.measure)
        self.assertEqual(semantics.measure.type, "DISTRIBUTION")
        self.assertIsNotNone(semantics.splits)
        self.assertEqual(semantics.splits[0].kind, "SEX")
        self.assertIsNotNone(semantics.x_axis)
        self.assertEqual(semantics.x_axis.metric, "DTN")

    def test_rejects_chart_group_by_field(self) -> None:
        with self.assertRaises(ValidationError):
            ChartSpec.model_validate(
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "group_by": [{"grain": "MONTH"}],
                    "semantics": {
                        "intent": "TREND",
                        "measure": {"type": "MEAN"},
                        "time": {"grain": "MONTH"},
                    },
                }
            )

    def test_rejects_invalid_intent(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {
                        "intent": "WHATEVER",
                    },
                }
            )

    def test_rejects_percentile_measure_without_percentile(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {
                        "intent": "TREND",
                        "measure": {"type": "PERCENTILE"},
                    },
                }
            )

    def test_rejects_boolean_split_without_field(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "BAR",
                    "metrics": [{"metric": "DTN"}],
                    "semantics": {
                        "intent": "COMPARISON",
                        "splits": [
                            {
                                "kind": "BOOLEAN",
                            }
                        ],
                    },
                }
            )

    def test_statistical_test_rejects_group_by(self) -> None:
        with self.assertRaises(ValidationError) as err:
            StatisticalTestSpec(
                test_type="MANN_WHITNEY_U_TEST",
                metrics=[MetricSpec(metric="DTN")],
                group_by=[GroupBySex(categories=["MALE", "FEMALE"])],
            )

        self.assertIn("do not support group_by", str(err.exception))


if __name__ == "__main__":
    unittest.main()
