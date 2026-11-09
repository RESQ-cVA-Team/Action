import unittest

from src.planners.langchain.request_orchestrator import _normalize_entities_for_question, _validate_group_by_support

_QUESTION = "Make a line chart of percentage of ischemic strokes per quarter"


class GroupByPrefixWordsTests(unittest.TestCase):
    def test_per_quarter_becomes_the_quarter_grouping(self) -> None:
        # Hosted dev: group_by arrived as the words "per quarter" and the request was refused.
        normalized = _normalize_entities_for_question(_QUESTION, {"chart_type": "LINE", "stroke_type": "ISCHEMIC", "group_by": "per quarter"})

        self.assertEqual(normalized["group_by"], "QUARTER")
        self.assertIsNone(_validate_group_by_support(_QUESTION, normalized))

    def test_the_raw_words_are_still_refused_without_normalisation(self) -> None:
        outcome = _validate_group_by_support(_QUESTION, {"chart_type": "LINE", "group_by": "per quarter"})

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.reason, "unsupported_group_by_dimension")

    def test_prefixed_synonyms_resolve_to_their_canonical(self) -> None:
        normalized = _normalize_entities_for_question("dtn by sex and stroke type", {"group_by": ["by sex", "stroke type", "grouped by month"]})

        self.assertEqual(normalized["group_by"], ["SEX_TYPE", "STROKE_TYPE", "MONTH"])

    def test_canonical_and_unknown_values_are_left_alone(self) -> None:
        normalized = _normalize_entities_for_question("dtn", {"group_by": ["QUARTER", "age in 10-year buckets"]})

        self.assertEqual(normalized["group_by"], ["QUARTER", "age in 10-year buckets"])

    def test_entities_without_group_by_are_untouched(self) -> None:
        entities = {"metric": "DTN", "chart_type": "LINE"}

        self.assertEqual(_normalize_entities_for_question("dtn", entities), entities)


if __name__ == "__main__":
    unittest.main()
