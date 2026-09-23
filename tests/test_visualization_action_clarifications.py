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
        "_build_metric_clarification_prompt",
    }
    selected = [
        node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    isolated_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated_module)

    namespace = {
        "lang_schema": S,
        "Dict": __import__("typing").Dict,
        "List": __import__("typing").List,
        "Optional": __import__("typing").Optional,
        # Translation passthrough for isolated tests.
        "translate": lambda _key, language=None, default=None, params=None: (
            default if default is not None else f"Which metric did you mean: {params['options']}?"
        ),
        "ssot_loader": type(
            "_StubSsotLoader",
            (),
            {
                "get_metric_metadata": staticmethod(
                    lambda: {
                        "DTN": {"display_name": "Door-to-needle time"},
                        "DTG": {"display_name": "Door-to-groin time"},
                    }
                ),
                "get_metric_display_name": staticmethod(
                    lambda code: {
                        "DTN": "Door-to-needle time",
                        "DTG": "Door-to-groin time",
                    }[code]
                ),
            },
        ),
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

    def test_metric_clarification_prompt_uses_display_names_and_canonical_payloads(self) -> None:
        ns = _load_helpers()
        builder = ns["_build_metric_clarification_prompt"]

        message, buttons = builder("metric", ["DTN", "DTG"], "en", "fallback")

        self.assertEqual(message, "Which metric did you mean: Door-to-needle time, Door-to-groin time?")
        self.assertEqual(
            buttons,
            [
                {"title": "Door-to-needle time", "payload": '/clarify_visualization{"metric":"DTN"}'},
                {"title": "Door-to-groin time", "payload": '/clarify_visualization{"metric":"DTG"}'},
            ],
        )

    def test_metric_clarification_prompt_falls_back_for_non_metric_or_unknown_options(self) -> None:
        ns = _load_helpers()
        builder = ns["_build_metric_clarification_prompt"]

        message, buttons = builder("metric", ["DTN", "UNKNOWN"], "en", "fallback")
        self.assertEqual(message, "fallback")
        self.assertEqual(buttons, [])

        message, buttons = builder("analysis_plan", ["DTN", "DTG"], "en", "fallback")
        self.assertEqual(message, "fallback")
        self.assertEqual(buttons, [])


if __name__ == "__main__":
    unittest.main()
