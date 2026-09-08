import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")

from src.domain.langchain import schema as S


def _load_helpers():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "actions" / "visualization_action.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "_is_missing_temporal_bounds_error",
        "_build_empty_plan_clarification",
    }
    selected = [
        node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    isolated_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    namespace = {
        "lang_schema": S,
        "Optional": __import__("typing").Optional,
        # Translation passthrough for isolated tests.
        "translate": lambda _key, language=None, default=None, params=None: default,
    }
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return namespace


class VisualizationActionClarificationHelpersTests(unittest.TestCase):
    def test_detects_missing_temporal_bounds_error(self) -> None:
        ns = _load_helpers()
        matcher = ns["_is_missing_temporal_bounds_error"]

        err = ValueError(
            "Semantic time grouping requires explicit time window/range or date-filter bounds for retrieval compilation"
        )
        self.assertTrue(matcher(err))
        self.assertFalse(matcher(ValueError("other error")))

    def test_build_empty_plan_clarification_when_plan_has_no_charts_or_tests(self) -> None:
        ns = _load_helpers()
        builder = ns["_build_empty_plan_clarification"]

        empty_plan = S.AnalysisPlan(charts=None, statistical_tests=None)
        msg = builder(empty_plan, "en")

        self.assertIsInstance(msg, str)
        self.assertIn("executable analysis plan", msg)

    def test_no_empty_plan_clarification_when_plan_has_content(self) -> None:
        ns = _load_helpers()
        builder = ns["_build_empty_plan_clarification"]

        non_empty = S.AnalysisPlan(
            charts=[
                S.ChartSpec(
                    chart_type="LINE",
                    metrics=[S.MetricSpec(metric="DTN")],
                    semantics=S.AnalysisSemanticsSpec(
                        intent="TREND",
                        measure=S.MeasureSemanticsSpec(type="MEAN"),
                    ),
                )
            ]
        )

        self.assertIsNone(builder(non_empty, "en"))


if __name__ == "__main__":
    unittest.main()
