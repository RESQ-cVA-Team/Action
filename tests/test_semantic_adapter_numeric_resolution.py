import unittest
from typing import Any, cast

from src.domain.langchain.schema import AnalysisPlan
from src.planners.langchain.semantic_adapter import normalize_analysis_plan, to_analysis_plan, to_semantic_plan


class SemanticAdapterNumericResolutionTests(unittest.TestCase):
    def test_roundtrip_preserves_chart_numeric_resolution(self) -> None:
        planner_payload = {
            "charts": [
                {
                    "chart_type": "HISTOGRAM",
                    "metrics": [{"metric": "DTN"}],
                    "numericResolution": {
                        "valueDomain": {"lowerBound": 5, "upperBound": 155},
                        "bucketing": {"bucketCount": 15},
                    },
                }
            ]
        }
        input_plan = AnalysisPlan.model_validate(planner_payload)

        semantic = to_semantic_plan(input_plan)
        self.assertEqual(len(semantic.charts), 1)
        self.assertIsNotNone(semantic.charts[0].numeric_resolution)

        output_plan = to_analysis_plan(semantic)
        self.assertIsNotNone(output_plan.charts)
        charts = output_plan.charts
        self.assertIsNotNone(charts)
        charts_value = cast(Any, charts)
        chart = charts_value[0]
        self.assertIsNotNone(chart.numeric_resolution)
        numeric_resolution = chart.numeric_resolution
        self.assertIsNotNone(numeric_resolution)
        numeric_resolution_value = numeric_resolution
        self.assertEqual(numeric_resolution_value.value_domain.lower_bound, 5)
        self.assertEqual(numeric_resolution_value.value_domain.upper_bound, 155)
        self.assertEqual(numeric_resolution_value.bucketing.bucket_count, 15)

        output_json = output_plan.model_dump(by_alias=True, exclude_none=True)
        chart_json = output_json["charts"][0]
        self.assertIn("numericResolution", chart_json)
        self.assertEqual(chart_json["numericResolution"]["valueDomain"]["lowerBound"], 5)
        self.assertEqual(chart_json["numericResolution"]["valueDomain"]["upperBound"], 155)
        self.assertEqual(chart_json["numericResolution"]["bucketing"]["bucketCount"], 15)

    def test_normalize_analysis_plan_keeps_numeric_resolution(self) -> None:
        input_plan = AnalysisPlan.model_validate(
            {
                "charts": [
                    {
                        "chart_type": "line",
                        "metrics": [{"metric": "dtn"}],
                        "numericResolution": {
                            "valueDomain": {"upperBound": 120},
                            "bucketing": {"bucketSize": 10},
                        },
                    }
                ]
            }
        )

        normalized = normalize_analysis_plan(input_plan)
        self.assertIsNotNone(normalized.charts)
        charts = normalized.charts
        self.assertIsNotNone(charts)
        charts_value = cast(Any, charts)
        chart = charts_value[0]
        self.assertEqual(chart.chart_type, "LINE")
        self.assertEqual(chart.metrics[0].metric, "DTN")
        self.assertIsNotNone(chart.numeric_resolution)
        numeric_resolution = chart.numeric_resolution
        self.assertIsNotNone(numeric_resolution)
        numeric_resolution_value = numeric_resolution
        self.assertEqual(numeric_resolution_value.value_domain.upper_bound, 120)
        self.assertEqual(numeric_resolution_value.bucketing.bucket_size, 10)


if __name__ == "__main__":
    unittest.main()
