import ast
import unittest
from pathlib import Path
from typing import Any, Dict, cast


class _TranslatorStub:
    @staticmethod
    def translate(_key: str, language: str | None = None, params: Dict[str, Any] | None = None, default: str | None = None) -> str:
        if default is None:
            return ""
        if params:
            return default.format(**params)
        return default


def _load_format_execution_summary():
    source_path = Path(__file__).resolve().parents[1] / "src" / "actions" / "helpers" / "visualization.py"
    source = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(source_path))

    required = {
        "SupportsModelDump",
        "_mapping_to_dict",
        "_maybe_model_dump_dict",
        "format_execution_summary",
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

    namespace: Dict[str, Any] = {
        "translate": _TranslatorStub.translate,
    }
    exec(compile(isolated_module, filename=str(source_path), mode="exec"), namespace)
    return cast(Any, namespace["format_execution_summary"])


class VisualizationSummaryStatsTests(unittest.TestCase):
    def test_stats_only_summary_does_not_mention_charts(self) -> None:
        formatter = _load_format_execution_summary()
        summary = {
            "chart_count": 0,
            "stats_count": 1,
        }

        msg = formatter(summary, language="en")

        self.assertEqual(msg, "Here's your statistical comparison.")

    def test_mixed_summary_reports_a_single_natural_line_no_technical_detail(self) -> None:
        formatter = _load_format_execution_summary()
        summary = {
            "trace_id": "abc123",
            "chart_count": 1,
            "stats_count": 3,
            "estimated_queries": 4,
            "actual_queries": 4,
        }

        msg = formatter(summary, language="en")

        self.assertEqual(msg, "Here's your chart and statistical comparison.")
        # None of the developer diagnostics this used to dump into the chat
        # should ever appear -- see CVaLab for that detail instead.
        self.assertNotIn("Trace ID", msg)
        self.assertNotIn("queried", msg)
        self.assertNotIn("cache", msg.lower())

    def test_stats_errors_take_priority_over_skipped_count(self) -> None:
        formatter = _load_format_execution_summary()
        summary = {
            "chart_count": 1,
            "stats_count": 3,
            "stats_skipped": 1,
            "stats_errors": 1,
        }

        msg = formatter(summary, language="en")

        self.assertIn("One statistical test couldn't be completed.", msg)
        self.assertNotIn("skipped", msg.lower())

    def test_many_charts_and_skipped_stats_use_plural_phrasing(self) -> None:
        formatter = _load_format_execution_summary()
        summary = {
            "chart_count": 2,
            "stats_count": 2,
            "stats_skipped": 2,
        }

        msg = formatter(summary, language="en")

        self.assertIn("Here's your chart and statistical comparison.", msg)
        self.assertIn("2 statistical tests were skipped.", msg)

    def test_opening_line_can_be_suppressed_to_avoid_duplicate_confirmation(self) -> None:
        # visualization_action.py's one-shot flow already sends its own
        # specific confirmation (_build_confirmation_message) before this is
        # called -- this must return empty (not "Here's your chart.") when
        # there's nothing else to add, so callers don't send a second,
        # redundant chat bubble.
        formatter = _load_format_execution_summary()

        msg = formatter({"chart_count": 1, "stats_count": 0}, language="en", include_opening_line=False)
        self.assertEqual(msg, "")

        msg = formatter({}, language="en", include_opening_line=False)
        self.assertEqual(msg, "")

    def test_opening_line_suppressed_still_reports_a_genuine_caveat(self) -> None:
        formatter = _load_format_execution_summary()

        msg = formatter(
            {"chart_count": 1, "stats_count": 1, "stats_errors": 1},
            language="en",
            include_opening_line=False,
        )

        self.assertEqual(msg, "One statistical test couldn't be completed.")

    def test_no_emoji_anywhere_in_output(self) -> None:
        formatter = _load_format_execution_summary()
        for summary in (
            {"chart_count": 1, "stats_count": 0},
            {"chart_count": 0, "stats_count": 1},
            {"chart_count": 2, "stats_count": 2, "stats_errors": 1},
            {},
        ):
            msg = formatter(summary, language="en")
            for ch in msg:
                self.assertLess(ord(ch), 0x1F300, f"unexpected emoji-range character in: {msg!r}")


if __name__ == "__main__":
    unittest.main()
