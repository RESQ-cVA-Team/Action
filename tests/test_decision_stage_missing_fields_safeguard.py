import unittest
from unittest.mock import patch

from src.planners.langchain.request_orchestrator import (
    _decision_stage,
    _drop_falsely_missing_fields,
    _entity_present,
)


class DropFalselyMissingFieldsTests(unittest.TestCase):
    def test_drops_metric_when_present_as_string(self) -> None:
        result = _drop_falsely_missing_fields(["metric"], {"metric": "DTN"})
        self.assertEqual(result, [])

    def test_drops_metric_when_present_as_list(self) -> None:
        result = _drop_falsely_missing_fields(["metric"], {"metric": ["DTN"]})
        self.assertEqual(result, [])

    def test_keeps_metric_when_genuinely_absent(self) -> None:
        result = _drop_falsely_missing_fields(["metric"], {"chart_type": "LINE"})
        self.assertEqual(result, ["metric"])

    def test_keeps_metric_when_present_but_blank(self) -> None:
        result = _drop_falsely_missing_fields(["metric"], {"metric": "   "})
        self.assertEqual(result, ["metric"])

    def test_only_drops_the_field_thats_actually_present(self) -> None:
        result = _drop_falsely_missing_fields(["metric", "chart_type"], {"metric": "DTN"})
        self.assertEqual(result, ["chart_type"])

    def test_does_not_touch_fields_outside_the_self_verifiable_set(self) -> None:
        # statistical_cohorts needs two distinct cohorts, not just "some value
        # exists" -- this safeguard must never paper over that with a bare
        # presence check.
        result = _drop_falsely_missing_fields(
            ["statistical_cohorts"],
            {"statistical_cohorts": ["Hospital A"]},
        )
        self.assertEqual(result, ["statistical_cohorts"])

    def test_entity_present_rejects_empty_list(self) -> None:
        self.assertFalse(_entity_present({"metric": []}, "metric"))
        self.assertFalse(_entity_present({"metric": [""]}, "metric"))


class DecisionStageSafeguardTests(unittest.TestCase):
    def test_overrides_clarify_when_llm_falsely_claims_metric_missing(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_metric",
                "missing_fields": ["metric"],
                "message": "What metric would you like?",
            },
        ):
            outcome = _decision_stage(
                question="line chart please",
                entities={"metric": "DTN", "chart_type": "LINE", "country_code": "CZ"},
                language="en",
                conversation_history=["show dtn from Czech Republic", "line chart please"],
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "all_required_fields_present")
        self.assertEqual(outcome.missing_fields, [])

    def test_overrides_reject_when_llm_falsely_claims_metric_missing(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "missing_metric",
                "missing_fields": ["metric"],
                "message": "Please specify a metric.",
            },
        ):
            outcome = _decision_stage(
                question="line chart please",
                entities={"metric": "DTN", "chart_type": "LINE", "hospital_name": "Hospital X"},
                language="en",
            )

        self.assertEqual(outcome.decision, "proceed")

    def test_keeps_genuine_missing_field_untouched(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_chart_type",
                "missing_fields": ["chart_type"],
                "message": "What chart type would you like?",
            },
        ):
            outcome = _decision_stage(
                question="show dtn from Czech Republic",
                entities={"metric": "DTN", "country_code": "CZ"},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["chart_type"])

    def test_narrows_missing_fields_to_only_the_genuine_one(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_fields",
                "missing_fields": ["metric", "chart_type"],
                "message": "What metric and chart type?",
            },
        ):
            outcome = _decision_stage(
                question="show me something",
                entities={"metric": "DTN", "country_code": "CZ"},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["chart_type"])

    def test_coerces_reject_to_clarify_when_reason_is_missing_required_fields(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "missing_required_fields",
                "missing_fields": ["metric"],
                "message": "What specific metric do you want to visualize?",
            },
        ):
            outcome = _decision_stage(
                question="show me a line graph",
                entities={"chart_type": "LINE"},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_required_fields")
        self.assertEqual(outcome.missing_fields, ["metric"])


class DecisionStageOutOfScopeSafeguardTests(unittest.TestCase):
    def test_reclassifies_out_of_scope_reject_with_missing_chart_type_as_clarify(self) -> None:
        """Regression test for a reported CVaLab failure: 'Make a graph for
        dtn for male ischemic patients from 2025 to 2026' -- a well-formed
        chart request with a real metric, filters, and a date range -- was
        rejected as out_of_scope, when the LLM's own message ("Chart requests
        must specify a chart type") shows the actual gap is just a missing
        chart_type. A present, valid metric is proof the request is in scope
        (Rasa's own intent routing already established that before this
        action ever runs), so this can never legitimately be a scope
        rejection.
        """
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "out_of_scope",
                "missing_fields": [],
                "message": "Chart requests must specify a chart type.",
            },
        ):
            outcome = _decision_stage(
                question="Make a graph for dtn for male ischemic patients from 2025 to 2026",
                entities={"metric": "DTN", "sex": "MALE", "stroke_type": "ISCHEMIC", "date": ["2025-01-01", "2026-12-31"]},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["chart_type"])
        self.assertEqual(outcome.reason, "missing_required_fields")

    def test_reclassifies_out_of_scope_reject_as_proceed_when_all_fields_present(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "out_of_scope",
                "missing_fields": [],
                "message": "Not a visualization request.",
            },
        ):
            outcome = _decision_stage(
                question="dtn line chart",
                entities={"metric": "DTN", "chart_type": "LINE"},
                language="en",
            )

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "all_required_fields_present")

    def test_leaves_genuine_out_of_scope_reject_untouched_when_no_metric(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "out_of_scope",
                "missing_fields": [],
                "message": "This doesn't look like a visualization request.",
            },
        ):
            outcome = _decision_stage(
                question="what's the weather like today",
                entities={},
                language="en",
            )

        self.assertEqual(outcome.decision, "reject")
        self.assertEqual(outcome.reason, "out_of_scope")

    def test_reclassifies_out_of_scope_reject_for_bare_age_filter_with_no_metric(self) -> None:
        """Regression test for a reported CVaLab failure: 'show only patients
        older than 60' (a bare age filter, no metric stated yet) was rejected
        as out_of_scope with the message 'This request is not related to
        clinical stroke-care analytics' -- objectively false, since Rasa only
        ever extracts an "age" entity within this domain. The old safeguard
        only recognized a present "metric" as scope-proof, missing this case
        entirely (no metric here at all). Also checks the corrected message
        doesn't repeat the false "not related to..." claim.
        """
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "out_of_scope",
                "missing_fields": [],
                "message": "This request is not related to clinical stroke-care analytics.",
            },
        ):
            outcome = _decision_stage(
                question="show only patients older than 60",
                entities={"age": "60"},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "missing_required_fields")
        self.assertIn("metric", outcome.missing_fields)
        self.assertNotIn("not related", (outcome.message or "").lower())

    def test_does_not_touch_a_reject_for_an_unrelated_data_validity_reason(self) -> None:
        """Regression test for a bug this safeguard itself introduced (caught
        by CVaLab's webapp_negative_invalid_time_period scenario): a reject
        for a genuinely invalid date ("2023-13-40" -- month 13 doesn't exist)
        was being waved through to "proceed" just because a metric happened
        to be present too. The safeguard's premise (a present metric
        disproves an out-of-scope claim) says nothing about date validity --
        it must only fire for reasons that are actually about scope.
        """
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "invalid_date_format",
                "missing_fields": [],
                "message": "The date format is incorrect. Please provide valid dates.",
            },
        ):
            outcome = _decision_stage(
                question="Show me a line graph of DTN from 2023-13-01 to 2023-13-40",
                entities={"chart_type": "LINE", "metric": "DTN", "date": ["2023-13-01", "2023-13-40"]},
                language="en",
            )

        self.assertEqual(outcome.decision, "reject")
        self.assertEqual(outcome.reason, "invalid_date_format")


class DecisionStageInvalidDecisionRetryTests(unittest.TestCase):
    """Regression tests for a real caught exception (CVaLab: 'show a line
    graph of age grouped by sex' -> ValueError: Invalid decision from
    decision stage -> generic 'orchestrator_failed' clarify for a perfectly
    well-formed request). Unlike generate_analysis_plan in pipeline.py, this
    call previously had zero retry on a malformed response.
    """

    def test_retries_once_and_succeeds_on_second_attempt(self) -> None:
        responses = [
            {"decision": "maybe", "reason": "unsure", "missing_fields": []},
            {"decision": "proceed", "reason": "all_required_fields_present", "missing_fields": []},
        ]
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            side_effect=responses,
        ) as mocked:
            outcome = _decision_stage(
                question="show a line graph of age grouped by sex",
                entities={"metric": "AGE", "chart_type": "LINE", "group_by": "SEX"},
                language="en",
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(outcome.decision, "proceed")

    def test_first_attempt_success_does_not_retry(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "proceed", "reason": "all_required_fields_present", "missing_fields": []},
        ) as mocked:
            _decision_stage(question="dtn line chart", entities={"metric": "DTN", "chart_type": "LINE"}, language="en")

        self.assertEqual(mocked.call_count, 1)

    def test_raises_with_the_bad_value_after_two_failed_attempts(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "maybe", "reason": "unsure", "missing_fields": []},
        ) as mocked:
            with self.assertRaises(ValueError) as ctx:
                _decision_stage(question="dtn line chart", entities={"metric": "DTN", "chart_type": "LINE"}, language="en")

        self.assertEqual(mocked.call_count, 2)
        self.assertIn("maybe", str(ctx.exception))

    def test_recovers_when_llm_puts_a_reason_value_in_the_decision_field(self) -> None:
        """Regression test for the exact live CVaLab failure: 'show a line
        graph of age grouped by sex' got {"decision": "ambiguous_request",
        ...} on both attempts (not one-off noise -- same wrong value twice in
        a row), which the plain retry alone couldn't recover. Since each
        reason value in the closed taxonomy unambiguously implies its
        decision, this is losslessly recoverable on the first attempt.
        """
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "ambiguous_request", "missing_fields": []},
        ) as mocked:
            outcome = _decision_stage(
                question="show a line graph of age grouped by sex",
                entities={"metric": "AGE", "group_by": "SEX"},
                language="en",
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "ambiguous_request")

    def test_recovers_out_of_scope_reason_swapped_into_decision_as_reject(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={"decision": "out_of_scope", "missing_fields": []},
        ):
            outcome = _decision_stage(question="what's the weather", entities={}, language="en")

        self.assertEqual(outcome.decision, "reject")
        self.assertEqual(outcome.reason, "out_of_scope")
