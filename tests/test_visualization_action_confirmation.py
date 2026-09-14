import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")
from src.domain.langchain import schema as S


def _load_confirmation_builder():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "actions" / "visualization_action.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_confirmation_message"
    )
    isolated_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {"lang_schema": S}
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return namespace["_build_confirmation_message"]


class VisualizationActionConfirmationTests(unittest.TestCase):
    def test_confirmation_message_uses_semantic_time_grain(self) -> None:
        build_confirmation_message = _load_confirmation_builder()
        plan = S.AnalysisPlan(
            charts=[
                S.ChartSpec(
                    chart_type="LINE",
                    metrics=[S.MetricSpec(metric="DTN")],
                    semantics=S.AnalysisSemanticsSpec(
                        intent="TREND",
                        measure=S.MeasureSemanticsSpec(type="MEAN"),
                        time=S.TimeSemanticsSpec(grain="MONTH"),
                    ),
                )
            ]
        )

        message = build_confirmation_message(plan, is_update=False)

        self.assertEqual(message, "Here's your line chart of DTN per month.")

    def test_confirmation_message_for_stats_only_plan(self) -> None:
        build_confirmation_message = _load_confirmation_builder()
        plan = S.AnalysisPlan(
            statistical_tests=[
                S.StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        S.MetricSpec(metric="DTN"),
                        S.MetricSpec(metric="DTN"),
                    ],
                )
            ]
        )

        message = build_confirmation_message(plan, is_update=False)
        self.assertEqual(message, "Completed statistical analysis: 1 test result is ready.")

    def test_confirmation_message_for_multiple_stats_only_plan(self) -> None:
        build_confirmation_message = _load_confirmation_builder()
        plan = S.AnalysisPlan(
            statistical_tests=[
                S.StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        S.MetricSpec(metric="DTN"),
                        S.MetricSpec(metric="DTN"),
                    ],
                ),
                S.StatisticalTestSpec(
                    test_type="MANN_WHITNEY_U_TEST",
                    metrics=[
                        S.MetricSpec(metric="DIDO"),
                        S.MetricSpec(metric="DIDO"),
                    ],
                ),
            ]
        )

        message = build_confirmation_message(plan, is_update=True)
        self.assertEqual(message, "Updated statistical analysis: 2 test results are ready.")


if __name__ == "__main__":
    unittest.main()