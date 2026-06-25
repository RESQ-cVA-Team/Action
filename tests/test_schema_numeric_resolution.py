import unittest
from typing import Any, cast

from src.domain.langchain.schema import ChartSpec


class NumericResolutionSchemaTests(unittest.TestCase):
    def test_accepts_numeric_resolution_alias_shape(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "HISTOGRAM",
                "metrics": [{"metric": "DTN"}],
                "numericResolution": {
                    "valueDomain": {"lowerBound": 0, "upperBound": 180},
                    "bucketing": {"bucketCount": 12},
                },
            }
        )

        self.assertIsNotNone(chart.numeric_resolution)
        numeric_resolution = chart.numeric_resolution
        self.assertIsNotNone(numeric_resolution)
        numeric_resolution_value = cast(Any, numeric_resolution)
        self.assertIsNotNone(numeric_resolution_value.value_domain)
        self.assertEqual(numeric_resolution_value.value_domain.lower_bound, 0)
        self.assertEqual(numeric_resolution_value.value_domain.upper_bound, 180)
        self.assertIsNotNone(numeric_resolution_value.bucketing)
        self.assertEqual(numeric_resolution_value.bucketing.bucket_count, 12)

    def test_rejects_invalid_value_domain_bounds(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "LINE",
                    "metrics": [{"metric": "DTN"}],
                    "numericResolution": {
                        "valueDomain": {"lowerBound": 100, "upperBound": 100},
                    },
                }
            )

    def test_rejects_empty_bucketing_object(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "HISTOGRAM",
                    "metrics": [{"metric": "DTN"}],
                    "numericResolution": {
                        "bucketing": {},
                    },
                }
            )

    def test_rejects_non_positive_bucket_count(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "HISTOGRAM",
                    "metrics": [{"metric": "DTN"}],
                    "numericResolution": {
                        "bucketing": {"bucketCount": 0},
                    },
                }
            )

    def test_rejects_non_positive_bucket_size(self) -> None:
        with self.assertRaises(ValueError):
            ChartSpec.model_validate(
                {
                    "chart_type": "HISTOGRAM",
                    "metrics": [{"metric": "DTN"}],
                    "numericResolution": {
                        "bucketing": {"bucketSize": -2},
                    },
                }
            )

    def test_accepts_snake_case_numeric_resolution(self) -> None:
        chart = ChartSpec.model_validate(
            {
                "chart_type": "BAR",
                "metrics": [{"metric": "DTN"}],
                "numeric_resolution": {
                    "value_domain": {"upper_bound": 130},
                    "bucketing": {"bucket_size": 10},
                },
            }
        )

        self.assertIsNotNone(chart.numeric_resolution)
        numeric_resolution = chart.numeric_resolution
        self.assertIsNotNone(numeric_resolution)
        numeric_resolution_value = cast(Any, numeric_resolution)
        self.assertEqual(numeric_resolution_value.value_domain.upper_bound, 130)
        self.assertEqual(numeric_resolution_value.bucketing.bucket_size, 10)


if __name__ == "__main__":
    unittest.main()
