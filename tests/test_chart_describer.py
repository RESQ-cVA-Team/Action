import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from src.domain.dto.charts.pie import PieChart, PieSlice
from src.domain.dto.charts.types import ChartMetadata, ChartMetric, ChartPoint, ChartSeries
from src.domain.langchain.schema import AnalysisPlan
from src.executors.mapping.chart_builder import build_chart_dto
from src.planners.langchain import chart_describer


def _line_chart():
    plan = AnalysisPlan.model_validate(
        {"charts": [{"chart_type": "LINE", "metrics": [{"metric": "DTN"}], "semantics": {"intent": "TREND", "measure": {"type": "MEDIAN"}, "time": {"grain": "QUARTER"}}}]}
    )
    series = [ChartSeries(name="My Hospital", data=[ChartPoint(x="2026-Q1", y=42.5), ChartPoint(x="2026-Q2", y=57.25)])]
    return build_chart_dto(plan_chart=plan.charts[0], dimensions=[], series=series, derived_axes=None)


class ChartMetadataTests(unittest.TestCase):
    def test_a_chart_names_its_metrics_and_statistic(self) -> None:
        metadata = _line_chart().metadata
        self.assertEqual((metadata.metrics, metadata.statistic), ([ChartMetric(metric="DTN")], "MEDIAN"))


class ChartFactsTests(unittest.TestCase):
    def test_facts_describe_the_chart_in_the_users_language_without_its_values(self) -> None:
        facts = chart_describer.chart_facts(_line_chart(), "cs")
        self.assertEqual(facts["metrics"][0]["name"], "door-to-needle")
        self.assertEqual(facts["statistic"], "medián")
        self.assertEqual(facts["series"], ["My Hospital"])
        self.assertEqual(facts["x_values"], {"count": 2, "first": "2026-Q1", "last": "2026-Q2"})
        text = json.dumps(facts)
        self.assertNotIn("42.5", text)
        self.assertNotIn("57.25", text)

    def test_a_pie_chart_gives_its_categories_but_not_their_values(self) -> None:
        pie = PieChart(metadata=ChartMetadata(title="Stroke type", metrics=[ChartMetric(metric="STROKE_TYPE")], statistic="RATE"), data=[PieSlice(label="ischemic", value=61.0), PieSlice(label="tia", value=12.0)])
        facts = chart_describer.chart_facts(pie, "en")
        self.assertEqual(facts["categories"], ["ischemic", "tia"])
        self.assertEqual(facts["statistic"], "share of cases (%)")
        self.assertNotIn("61.0", json.dumps(facts))


    def test_angels_awards_bands_are_a_note_only_when_drawn(self) -> None:
        self.assertNotIn("notes", chart_describer.chart_facts(_line_chart(), "en"))
        chart = _line_chart()
        chart.metadata.angels_awards = {"AA_DTN_LE60": {"gold": {"min_percent": 50}}}
        self.assertEqual(chart_describer.chart_facts(chart, "el")["notes"], [chart_describer._BANDS_NOTE["el"]])

    def test_options_counted_together_are_named_in_the_users_language(self) -> None:
        chart = _line_chart()
        chart.metadata.metrics = [ChartMetric(metric="STROKE_TYPE", categories=["ischemic"])]
        chart.metadata.statistic = "RATE"
        self.assertEqual(chart_describer.chart_facts(chart, "cs")["metrics"][0]["option_counted"], "ischemický iktus")
        self.assertNotIn("notes", chart_describer.chart_facts(chart, "cs"))
        chart.metadata.metrics = [ChartMetric(metric="ARRIVAL_MODE", categories=["ems_from_home_scene", "ems_from_gp"])]
        self.assertEqual(
            chart_describer.chart_facts(chart, "en")["notes"],
            ["ems from home or scene, ems from general practitioner are counted together as one group"],
        )

    def test_a_count_leaves_out_the_metrics_own_meaning(self) -> None:
        chart = _line_chart()
        chart.metadata.statistic = "COUNT"
        self.assertEqual(chart_describer.chart_facts(chart, "en")["metrics"], [{"name": "door to needle"}])


    def test_many_series_are_counted_instead_of_named(self) -> None:
        chart = _line_chart()
        chart.series = [ChartSeries(name=f"group {i}", data=[ChartPoint(x="2026-Q1", y=1.0)]) for i in range(5)]
        self.assertEqual(chart_describer.chart_facts(chart, "en")["series"], {"count": 5})


class DescribeChartsTests(unittest.TestCase):
    def test_each_chart_gets_its_description_in_order(self) -> None:
        def answer(prompt):
            facts = json.loads(prompt.messages[-1].content.split("CHART_FACTS_JSON: ", 1)[1])
            return SimpleNamespace(content=f"About {facts['title']}.")

        charts = [_line_chart(), _line_chart()]
        charts[1].metadata.title = "Second"
        with patch.object(chart_describer, "_get_llm", return_value=RunnableLambda(answer)):
            descriptions = asyncio.run(chart_describer.describe_charts(charts, "en"))
        self.assertEqual(descriptions, [f"About {charts[0].metadata.title}.", "About Second."])

    def test_a_failed_description_is_left_out_and_the_others_kept(self) -> None:
        def answer(prompt):
            if "Second" in prompt.messages[-1].content:
                raise RuntimeError("model error")
            return SimpleNamespace(content="About the first chart.")

        charts = [_line_chart(), _line_chart()]
        charts[1].metadata.title = "Second"
        with patch.object(chart_describer, "_get_llm", return_value=RunnableLambda(answer)):
            descriptions = asyncio.run(chart_describer.describe_charts(charts, "en"))
        self.assertEqual(descriptions, ["About the first chart."])


if __name__ == "__main__":
    unittest.main()
