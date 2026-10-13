import unittest

from src.planners.langchain.request_orchestrator import _normalize_entities_for_question


class ChartRequestEntityNormalizationTests(unittest.TestCase):
    def test_pagination_entities_are_dropped_for_chart_requests(self) -> None:
        # Live: "Show me dtn from 0-500 in a bar chart" arrived with limit/offset
        # from the hospital-list regexes.
        entities = {"metric": "DTN", "chart_type": "BAR", "limit": "500", "offset": ["0", "500"], "sort": "asc"}
        self.assertEqual(_normalize_entities_for_question("Show me dtn from 0-500 in a bar chart", entities), {"metric": "DTN", "chart_type": "BAR"})

    def test_everything_else_passes_through_untouched(self) -> None:
        entities = {"metric": ["DTN"], "age": ["30", "60"], "age_lower": "30", "age_upper": "60", "hospital_name": "X"}
        self.assertEqual(_normalize_entities_for_question("q", entities), entities)

    def test_returns_a_copy(self) -> None:
        entities = {"metric": "DTN"}
        self.assertIsNot(_normalize_entities_for_question("q", entities), entities)


if __name__ == "__main__":
    unittest.main()
