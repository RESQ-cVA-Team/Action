import importlib.util
import sys
import unittest
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


query_compiler = load_module("action_query_compiler_groupby_test", "src/executors/planning/query_compiler.py")


class QueryCompilerGroupByTests(unittest.TestCase):
    def test_series_split_sex_compiles_to_single_groupby_query(self) -> None:
        chart = query_compiler.S.LineChartSpec.model_validate(
            {
                "chartType": "LINE",
                "xAxes": {"x1": {"kind": "numeric_metric", "metric": "DTN", "bins": 52}},
                "yAxes": {"y1": {"kind": "count"}},
                "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                "seriesSplit": {"categories": ["MALE", "FEMALE"]},
            }
        )

        compiled = query_compiler.compile_chart_grouping(chart)

        self.assertEqual(compiled.total_requests, 1)
        self.assertEqual(len(compiled.batches), 1)
        self.assertEqual(compiled.batches[0].server_groupby, "SEX")
        self.assertEqual(compiled.batches[0].combos_list, [tuple()])

    def test_time_axis_with_series_split_keeps_single_request_per_batch(self) -> None:
        chart = query_compiler.S.LineChartSpec.model_validate(
            {
                "chartType": "LINE",
                "xAxes": {
                    "x1": {
                        "kind": "time",
                        "grain": "MONTH",
                        "window": {"last_n": 12, "unit": "MONTH"},
                    }
                },
                "yAxes": {"y1": {"kind": "metric_value", "statistic": "MEAN"}},
                "series": [{"metric": "DTN", "xAxis": "x1", "yAxis": "y1"}],
                "seriesSplit": {"categories": ["MALE", "FEMALE"]},
            }
        )

        compiled = query_compiler.compile_chart_grouping(chart)

        self.assertEqual(compiled.total_requests, 1)
        self.assertEqual(compiled.batches[0].server_groupby, "SEX")
        self.assertTrue(compiled.batches[0].batched_time_enabled)
        self.assertGreater(len(compiled.batches[0].batched_time_periods), 0)


if __name__ == "__main__":
    unittest.main()
