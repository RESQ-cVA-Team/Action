import ast
from pathlib import Path
from typing import Any, Dict, List, cast
import unittest


def _load_serialize_plan_for_frontend():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "helpers" / "visualization.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "SupportsModelDump",
        "_mapping_to_dict",
        "_maybe_model_dump_dict",
        "serialize_plan_for_frontend",
    }

    selected = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in required
    ]

    prelude = ast.parse(
        "from typing import Any, Dict, List, Mapping, Optional, Protocol, cast, runtime_checkable"
    )
    isolated_module = ast.Module(body=[*prelude.body, *selected], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace: Dict[str, Any] = {}
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return cast(Any, namespace["serialize_plan_for_frontend"])


class VisualizationPlanPreviewSerializerTests(unittest.TestCase):
    def test_serialize_plan_for_frontend_emits_minimal_chart_preview(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "charts": [
                {
                    "chart_type": "line",
                    "metrics": [
                        {
                            "metric": "dtn",
                            "title": "remove",
                        }
                    ],
                    "semantics": {
                        "intent": "trend",
                        "measure": {"type": "mean"},
                        "time": {"grain": "month"},
                        "splits": [
                            {
                                "kind": "sex",
                                "categories": ["male", "female"],
                            }
                        ],
                    },
                    "filters": {"operator": "GE", "value": "2023-01-01"},
                    "title": "remove",
                    "description": "remove",
                }
            ],
            "metadata": {"trace_id": "internal"},
        }

        payload = serialize_plan_for_frontend(plan)

        self.assertEqual(list(payload.keys()), ["charts"])
        self.assertIsInstance(payload["charts"], list)
        self.assertEqual(len(payload["charts"]), 1)

        chart = payload["charts"][0]
        self.assertEqual(chart["chart_type"], "LINE")
        self.assertEqual(chart["metrics"], [{"metric": "DTN"}])
        self.assertEqual(chart["semantics"]["intent"], "TREND")
        self.assertEqual(chart["semantics"]["measure"], {"type": "MEAN"})
        self.assertEqual(chart["semantics"]["time"], {"grain": "MONTH"})
        self.assertEqual(
            chart["semantics"]["splits"],
            [{"kind": "SEX", "categories": ["MALE", "FEMALE"]}],
        )

        # A filter node without a type is dropped; raw/internal fields are not leaked.
        self.assertIsNone(chart["filters"])
        self.assertNotIn("title", chart)
        self.assertNotIn("description", chart)
        self.assertNotIn("metadata", payload)

    def test_serialize_plan_for_frontend_drops_empty_or_invalid_entries(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "charts": [
                {
                    "chart_type": "",
                    "metrics": [{"metric": ""}, {}],
                    "semantics": {
                        "intent": "",
                        "measure": {"type": ""},
                        "time": {"grain": ""},
                        "splits": [{"kind": "", "categories": ["a"]}],
                    },
                },
                {
                    "chart_type": "bar",
                    "metrics": [{"metric": " dido "}],
                },
            ]
        }

        payload = serialize_plan_for_frontend(plan)

        self.assertIn("charts", payload)
        charts = cast(List[Dict[str, Any]], payload["charts"])
        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0], {"chart_type": "BAR", "metrics": [{"metric": "DIDO"}], "filters": None})

    def test_serialize_plan_for_frontend_emits_statistical_test_preview(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "statistical_tests": [
                {
                    "test_type": "mann_whitney_u_test",
                    "metrics": [{"metric": "dtn"}, {"metric": "dtn"}],
                    "group_by": [{"kind": "sex", "categories": ["male", "female"]}],
                    "title": "remove",
                }
            ]
        }

        payload = serialize_plan_for_frontend(plan)

        self.assertIn("statistical_tests", payload)
        tests = cast(List[Dict[str, Any]], payload["statistical_tests"])
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0]["test_type"], "MANN_WHITNEY_U_TEST")
        self.assertEqual(tests[0]["metrics"], [{"metric": "DTN"}, {"metric": "DTN"}])
        self.assertEqual(
            tests[0]["group_by"],
            [{"kind": "SEX", "categories": ["MALE", "FEMALE"]}],
        )

    def test_serialize_plan_for_frontend_keeps_statistical_test_cohort_origin(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "statistical_tests": [
                {
                    "test_type": "MANN_WHITNEY_U_TEST",
                    "metrics": [
                        {
                            "metric": "DTN",
                            "originScope": {
                                "scopeType": "provider_name",
                                "label": "My hospital",
                                "value": "my",
                            },
                        },
                        {
                            "metric": "DTN",
                            "originScope": {
                                "scopeType": "provider_name",
                                "label": "Army Alhama de Murcia Hospital",
                                "value": "Army Alhama de Murcia Hospital",
                            },
                        },
                    ],
                }
            ]
        }

        payload = serialize_plan_for_frontend(plan)

        tests = cast(List[Dict[str, Any]], payload["statistical_tests"])
        self.assertEqual(
            tests[0]["metrics"],
            [
                {
                    "metric": "DTN",
                    "origin_scope": {
                        "scope_type": "PROVIDER_NAME",
                        "label": "My hospital",
                        "value": "my",
                    },
                },
                {
                    "metric": "DTN",
                    "origin_scope": {
                        "scope_type": "PROVIDER_NAME",
                        "label": "Army Alhama de Murcia Hospital",
                        "value": "Army Alhama de Murcia Hospital",
                    },
                },
            ],
        )

    def test_serialize_plan_for_frontend_keeps_statistical_test_filters(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "statistical_tests": [
                {
                    "test_type": "MANN_WHITNEY_U_TEST",
                    "metrics": [{"metric": "DTN"}, {"metric": "DTN"}],
                    "filters": {
                        "type": "AndFilter",
                        "and_": [
                            {"type": "DateFilter", "operator": "GE", "value": "2025-10-01"},
                            {"type": "DateFilter", "operator": "LE", "value": "2025-12-31"},
                        ],
                    },
                }
            ]
        }

        payload = serialize_plan_for_frontend(plan)

        tests = cast(List[Dict[str, Any]], payload["statistical_tests"])
        self.assertEqual(
            tests[0]["filters"],
            {
                "type": "ANDFILTER",
                "and": [
                    {"type": "DATEFILTER", "operator": "GE", "value": "2025-10-01"},
                    {"type": "DATEFILTER", "operator": "LE", "value": "2025-12-31"},
                ],
            },
        )

    def test_serialize_plan_for_frontend_keeps_chart_filters(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "charts": [
                {
                    "chart_type": "line",
                    "metrics": [{"metric": "DTN"}],
                    "filters": {
                        "type": "AndFilter",
                        "and_": [
                            {"type": "StrokeFilter", "value": "ischemic"},
                            {"type": "SexFilter", "value": "female"},
                            {"type": "BooleanFilter", "boolean_type": "hypertension", "value": True},
                        ],
                    },
                }
            ]
        }

        payload = serialize_plan_for_frontend(plan)

        charts = cast(List[Dict[str, Any]], payload["charts"])
        self.assertEqual(
            charts[0]["filters"],
            {
                "type": "ANDFILTER",
                "and": [
                    {"type": "STROKEFILTER", "value": "ischemic"},
                    {"type": "SEXFILTER", "value": "female"},
                    {"type": "BOOLEANFILTER", "boolean_type": "HYPERTENSION", "value": True},
                ],
            },
        )

    def test_serialize_plan_for_frontend_keeps_statistical_tests_when_charts_absent(self) -> None:
        serialize_plan_for_frontend = _load_serialize_plan_for_frontend()

        plan: Dict[str, Any] = {
            "statistical_tests": [
                {
                    "test_type": "MANN_WHITNEY_U_TEST",
                    "metrics": [{"metric": "DTN"}, {"metric": "DTN"}],
                }
            ]
        }

        payload = serialize_plan_for_frontend(plan)

        self.assertNotIn("charts", payload)
        self.assertIn("statistical_tests", payload)
        tests = cast(List[Dict[str, Any]], payload["statistical_tests"])
        self.assertEqual(tests[0], {"test_type": "MANN_WHITNEY_U_TEST", "metrics": [{"metric": "DTN"}, {"metric": "DTN"}]})


if __name__ == "__main__":
    unittest.main()
