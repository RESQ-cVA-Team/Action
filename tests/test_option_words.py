import unittest

from src.planners.langchain.request_orchestrator import _metric_from_option_words, _validate_option_words_name_one_metric


class OptionWordsTests(unittest.TestCase):
    def test_options_of_one_metric_name_that_metric(self) -> None:
        self.assertEqual(_metric_from_option_words("How many of my patients had CT imaging?", {}, None), {"metric": "IMAGING_TYPE"})
        self.assertEqual(_metric_from_option_words("How many patients received alteplase?", {}, None), {"metric": "THROMBOLYSIS_DRUGS"})

    def test_a_metric_name_wins_over_the_option_names_in_it(self) -> None:
        # "stroke mimics" is a STROKE_TYPE option; "craniectomy" also an ICH and SAH treatment type.
        self.assertEqual(_metric_from_option_words("Show me stroke mimics diagnosis as a bar chart", {"chart_type": "BAR"}, None), {"chart_type": "BAR", "metric": "STROKE_MIMICS_DIAGNOSIS"})
        self.assertEqual(_metric_from_option_words("How many patients had a craniectomy?", {}, None), {"metric": "CRANIECTOMY"})
        self.assertIsNone(_validate_option_words_name_one_metric("How many patients had a craniectomy?", {}, None))

    def test_the_counted_option_is_not_also_a_filter(self) -> None:
        share = _metric_from_option_words("Make a line chart of percentage of ischemic strokes per quarter", {"chart_type": "LINE", "stroke_type": "ISCHEMIC"}, None)
        self.assertEqual(share, {"chart_type": "LINE", "metric": "STROKE_TYPE"})
        self.assertEqual(_metric_from_option_words("Percentage of female patients", {"sex": "FEMALE"}, None), {"metric": "SEX"})

    def test_a_named_metric_is_kept(self) -> None:
        entities = {"metric": "DTN", "stroke_type": "ISCHEMIC"}
        self.assertEqual(_metric_from_option_words("DTN for ischemic strokes", entities, None), entities)

    def test_only_the_users_words_are_read(self) -> None:
        edit = 'Previous chart plan (carry over everything except what the user explicitly changes):\n{"categories": ["ct"]}\n\nUser\'s newest instruction:\nShow it per quarter'
        self.assertEqual(_metric_from_option_words(edit, {}, None), {})
        exchange = "Show me patients\nThe assistant asked: Did you mean CT imaging?\nThe user answered: no"
        self.assertEqual(_metric_from_option_words(exchange, {}, None), {})

    def test_names_are_read_in_english_and_the_users_language(self) -> None:
        # Czech "ambulance" is an outpatient clinic: only a Czech request reads it as one.
        self.assertEqual(_metric_from_option_words("How many patients arrived by ambulance?", {}, None), {"metric": "ARRIVAL_MODE"})
        clarification = _validate_option_words_name_one_metric("How many patients arrived by ambulance?", {}, "cs")
        assert clarification is not None
        self.assertEqual(clarification.clarification_options, ["ARRIVAL_MODE", "FIRST_CONTACT_PLACE"])
        self.assertEqual(_metric_from_option_words("Kolik pacientů mělo ischemický iktus?", {}, "cs"), {"metric": "STROKE_TYPE"})

    def test_options_of_several_metrics_ask_which_metric(self) -> None:
        question = "How many patients came from another hospital?"
        self.assertEqual(_metric_from_option_words(question, {}, None), {})
        outcome = _validate_option_words_name_one_metric(question, {}, None)
        assert outcome is not None
        self.assertEqual((outcome.decision, outcome.clarification_type), ("clarify", "metric"))
        self.assertEqual(outcome.clarification_options, ["ARRIVAL_MODE", "DISCHARGE_DESTINATION"])
        self.assertIsNone(_validate_option_words_name_one_metric(question, {"metric": "ARRIVAL_MODE"}, None))


if __name__ == "__main__":
    unittest.main()
