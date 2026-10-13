import unittest

from src.planners.langchain.request_orchestrator import _validate_scope_grouping_cohorts


class ScopeGroupingCohortValidationTests(unittest.TestCase):
    def _assert_clarifies(self, question: str, entities: dict, grouping: str) -> None:
        outcome = _validate_scope_grouping_cohorts(question, entities)
        self.assertIsNotNone(outcome, f"expected a clarification for: {question!r}")
        assert outcome is not None
        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_scope_grouping_cohorts")
        self.assertIn(grouping, (outcome.message or "").lower())
        self.assertEqual(outcome.missing_fields, [])

    def test_bare_group_by_hospital_asks_for_hospitals(self) -> None:
        self._assert_clarifies(
            "Show me a bar chart of dtn grouped by hospital",
            {"chart_type": "BAR", "metric": "DTN", "group_by": "HOSPITAL"},
            "hospital",
        )

    def test_grouping_cue_counts_without_a_group_by_entity(self) -> None:
        # "hospitals" (plural) does not hit the lookup table, so no entity.
        self._assert_clarifies("Show dtn across hospitals", {"metric": "DTN"}, "hospital")
        self._assert_clarifies("Can I break the data down by hospital?", {"metric": "DTN"}, "hospital")

    def test_metric_name_containing_hospital_is_not_a_grouping_request(self) -> None:
        outcome = _validate_scope_grouping_cohorts(
            "Show me a bar chart of imaging at first hospital",
            {"chart_type": "BAR", "metric": "AA_IMAGING", "group_by": "HOSPITAL"},
        )
        self.assertIsNone(outcome)

    def test_named_hospitals_satisfy_hospital_grouping(self) -> None:
        outcome = _validate_scope_grouping_cohorts(
            "Compare my dtn per hospital with army alhama de murcia hospital using a line chart",
            {
                "hospital_scope_reference": ["my"],
                "metric": ["DTN"],
                "hospital_name": ["Army Alhama de Murcia Hospital"],
                "chart_type": ["LINE"],
                "group_by": ["QUARTER", "HOSPITAL"],
            },
        )
        self.assertIsNone(outcome)

    def test_all_hospitals_scope_is_not_a_set_of_cohorts(self) -> None:
        self._assert_clarifies(
            "Show dtn for all hospitals grouped by hospital",
            {"metric": "DTN", "hospital_scope_reference": "all hospitals", "group_by": "HOSPITAL"},
            "hospital",
        )

    def test_hospital_group_scope_is_not_hospital_grouping(self) -> None:
        outcome = _validate_scope_grouping_cohorts(
            "Show dtn by hospital group Alpha",
            {"metric": "DTN", "provider_group_name": "Alpha"},
        )
        self.assertIsNone(outcome)

    def test_statistical_test_requests_are_left_to_cohort_validation(self) -> None:
        outcome = _validate_scope_grouping_cohorts(
            "Run a mann-whitney test on dtn per hospital",
            {"metric": "DTN", "statistical_test_type": "MANN_WHITNEY"},
        )
        self.assertIsNone(outcome)

    def test_chart_request_is_validated_even_when_wording_trips_stat_test_scan(self) -> None:
        self._assert_clarifies(
            "Compare dtn per hospital using a bar chart",
            {"chart_type": "BAR", "metric": "DTN"},
            "hospital",
        )

    def test_bare_group_by_country_asks_for_countries(self) -> None:
        self._assert_clarifies(
            "Show me a bar chart of dtn grouped by country",
            {"chart_type": "BAR", "metric": "DTN"},
            "country",
        )

    def test_named_country_scope_is_not_a_grouping_request(self) -> None:
        outcome = _validate_scope_grouping_cohorts("show dtn from Czech Republic", {"metric": ["DTN"], "country_code": ["CZ"]})
        self.assertIsNone(outcome)

    def test_named_countries_satisfy_country_grouping(self) -> None:
        outcome = _validate_scope_grouping_cohorts(
            "Compare dtn by country for Czech Republic and Spain",
            {"metric": "DTN", "country_code": ["CZ", "ES"]},
        )
        self.assertIsNone(outcome)

    def test_country_average_satisfies_country_grouping(self) -> None:
        outcome = _validate_scope_grouping_cohorts(
            "Show my dtn per country average",
            {"metric": "DTN", "country_average": True},
        )
        self.assertIsNone(outcome)

    def test_czech_and_greek_grouping_cues(self) -> None:
        self._assert_clarifies("Ukaž DTN podle nemocnice", {"metric": "DTN"}, "hospital")
        self._assert_clarifies("Δείξε DTN ανά χώρα", {"metric": "DTN"}, "country")

    def test_no_grouping_cue_is_a_no_op(self) -> None:
        self.assertIsNone(_validate_scope_grouping_cohorts("Show me a bar chart of dtn", {"chart_type": "BAR", "metric": "DTN"}))
        self.assertIsNone(_validate_scope_grouping_cohorts("", {"metric": "DTN"}))


if __name__ == "__main__":
    unittest.main()
