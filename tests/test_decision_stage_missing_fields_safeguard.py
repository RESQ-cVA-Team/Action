import unittest
from unittest.mock import patch

from src.planners.langchain.request_orchestrator import (
    _decision_stage,
    _drop_falsely_missing_fields,
    _entity_present,
    _normalize_entities_for_question,
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

    def test_drops_a_range_bound_claim_when_the_range_was_given(self) -> None:
        # "over 50 and under 50" arrived as age plus age_upper; the model asked for age_lower.
        self.assertEqual(_drop_falsely_missing_fields(["age_lower"], {"metric": "DTN", "age": "50", "age_upper": "50"}), [])
        self.assertEqual(_drop_falsely_missing_fields(["age_upper"], {"metric": "DTN", "age": "60"}), [])
        self.assertEqual(_drop_falsely_missing_fields(["nihss_lower"], {"metric": "DTN", "nihss_upper": "10"}), [])

    def test_keeps_a_range_bound_claim_when_no_bound_was_given(self) -> None:
        self.assertEqual(_drop_falsely_missing_fields(["age_lower"], {"metric": "DTN"}), ["age_lower"])


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
                "reason": "missing_metric",
                "missing_fields": ["metric"],
                "message": "What metric would you like?",
            },
        ):
            outcome = _decision_stage(
                question="show me a line chart from Czech Republic",
                entities={"chart_type": "LINE", "country_code": "CZ"},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["metric"])

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

        # Narrowed to chart_type only, and a missing chart type alone is a
        # default chart, not a question.
        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.missing_fields, [])

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
    def test_reclassifies_out_of_scope_reject_with_missing_chart_type_as_proceed(self) -> None:
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

        # Live, after the default-chart rule: this path still asked "Please
        # specify the chart type you'd like to use" whenever the model called
        # the request out of scope, which it does intermittently.
        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.missing_fields, [])

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

    def test_normalizes_ambiguous_metric_request_to_metric_clarification_options(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "reject",
                "reason": "ambiguous_request",
                "clarification_type": None,
                "clarification_options": None,
                "message": "Please clarify which metric you want to visualize.",
                "missing_fields": [],
            },
        ):
            outcome = _decision_stage(
                question="Show me a line graph of ivt dose",
                entities={"chart_type": "LINE", "metric": ["THROMBOLYSIS_DRUG_DOSE", "THROMBOLYSIS"]},
                language="en",
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.reason, "ambiguous_request")
        self.assertEqual(outcome.clarification_type, "metric")
        self.assertEqual(outcome.clarification_options, ["THROMBOLYSIS_DRUG_DOSE", "THROMBOLYSIS"])


class PendingClarificationSafeguardTests(unittest.TestCase):
    """Covers the "stay on the field we just asked about" safeguard -- see
    its docstring in request_orchestrator.py for the live bug this closes:
    a user's free-text reply that Rasa's NLU failed to extract an entity
    from left the LLM free to ask about a *different* still-missing field
    instead of re-asking the one it had just requested."""

    _PENDING_METRIC_CLARIFICATION = {
        "decision": "clarify",
        "reason": "missing_required_fields",
        "clarification_type": "metric",
        "clarification_options": [],
        "message": "Which NIHSS metric would you like to visualize over time?",
        "missing_fields": ["metric"],
    }

    def test_overrides_when_llm_drifts_to_a_different_missing_field(self) -> None:
        # NLU extraction failed for the user's reply (e.g. "AdmissionNihss"
        # standalone) -- entities still has no metric, only the group_by
        # entity carried forward from the prior turn. The LLM nonetheless
        # asks about chart_type instead of re-asking about metric.
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_required_fields",
                "clarification_type": "chart_type",
                "clarification_options": ["LINE", "BAR"],
                "message": "What type of chart would you like for the NIHSS data?",
                "missing_fields": ["chart_type"],
            },
        ):
            outcome = _decision_stage(
                question="AdmissionNihss",
                entities={"group_by": "NIHSS"},
                language="en",
                pending_clarification=self._PENDING_METRIC_CLARIFICATION,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["metric"])
        self.assertEqual(outcome.clarification_type, "metric")
        self.assertEqual(outcome.message, "Which NIHSS metric would you like to visualize over time?")

    def test_does_not_override_when_the_reply_actually_resolves_it(self) -> None:
        # The reply DID extract a metric this time -- the LLM's own next
        # question (here a genuine ambiguity between two NIHSS metrics; a
        # missing chart type alone would be a default chart, not a question)
        # should stand.
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "ambiguous_request",
                "clarification_type": "metric",
                "clarification_options": ["ADMISSION_NIHSS", "DISCHARGE_NIHSS"],
                "message": "Admission or discharge NIHSS?",
                "missing_fields": [],
            },
        ):
            outcome = _decision_stage(
                question="nihss",
                entities={"metric": ["ADMISSION_NIHSS", "DISCHARGE_NIHSS"]},
                language="en",
                pending_clarification=self._PENDING_METRIC_CLARIFICATION,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.clarification_type, "metric")
        self.assertEqual(outcome.message, "Admission or discharge NIHSS?")

    def test_no_pending_clarification_leaves_outcome_untouched(self) -> None:
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_required_fields",
                "clarification_type": "metric",
                "clarification_options": [],
                "message": "What metric would you like to see?",
                "missing_fields": ["metric"],
            },
        ):
            outcome = _decision_stage(
                question="show me a line chart",
                entities={"chart_type": "LINE"},
                language="en",
                pending_clarification=None,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.missing_fields, ["metric"])

    def test_overrides_when_llm_only_covers_part_of_two_pending_fields(self) -> None:
        # Live-caught regression: when the pending clarification was asking
        # about BOTH metric and chart_type, a new response that only re-asks
        # about chart_type (silently dropping metric) must not be mistaken
        # for a resolution just because the two sets overlap at all.
        pending_both = {
            "decision": "clarify",
            "reason": "missing_required_fields",
            "clarification_type": "request_missing_fields",
            "clarification_options": [],
            "message": "What metric and chart type would you like to use?",
            "missing_fields": ["metric", "chart_type"],
        }
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_required_fields",
                "clarification_type": "chart_type",
                "clarification_options": ["LINE", "BAR"],
                "message": "What type of chart would you like for AdmissionNihss?",
                "missing_fields": ["chart_type"],
            },
        ):
            outcome = _decision_stage(
                question="AdmissionNihss",
                entities={},
                language="en",
                pending_clarification=pending_both,
            )

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(set(outcome.missing_fields or []), {"metric", "chart_type"})
        self.assertEqual(outcome.message, "What metric and chart type would you like to use?")

    def test_ignores_a_pending_payload_that_was_not_itself_a_clarify(self) -> None:
        # Defensive: only a prior clarify should ever pin the field -- a
        # proceed/reject payload has no "field we were waiting on" at all.
        with patch(
            "src.planners.langchain.request_orchestrator._invoke_chain",
            return_value={
                "decision": "clarify",
                "reason": "missing_required_fields",
                "clarification_type": "chart_type",
                "clarification_options": [],
                "message": "What type of chart?",
                "missing_fields": ["chart_type"],
            },
        ):
            outcome = _decision_stage(
                question="something",
                entities={},
                language="en",
                pending_clarification={"decision": "proceed", "missing_fields": ["metric"]},
            )

        self.assertEqual(outcome.missing_fields, ["chart_type"])


class RangeEntityMissingFieldTests(unittest.TestCase):
    def test_age_claimed_missing_is_dropped_when_a_bound_companion_is_present(self) -> None:
        result = _drop_falsely_missing_fields(["age"], {"metric": "DTN", "age_lower": "50", "age_upper": "50"})
        self.assertEqual(result, [])

    def test_age_claimed_missing_is_dropped_when_the_flat_key_is_present(self) -> None:
        result = _drop_falsely_missing_fields(["age"], {"metric": "DTN", "age": "50"})
        self.assertEqual(result, [])

    def test_age_claimed_missing_is_kept_when_no_bound_was_given(self) -> None:
        result = _drop_falsely_missing_fields(["age"], {"metric": "DTN"})
        self.assertEqual(result, ["age"])


class MetricClarificationWithoutRealOptionsTests(unittest.TestCase):
    _INVENTED = {
        "decision": "clarify",
        "reason": "ambiguous_request",
        "missing_fields": None,
        "clarification_type": "metric",
        "clarification_options": ["ISHEMIC_STROKES", "PERCENT_ISCHEMIC_STROKES"],
        "message": "Which metric do you want: ischemic strokes or the percentage of ischemic strokes?",
    }

    def _run(self, response, entities):
        with patch("src.planners.langchain.request_orchestrator._invoke_chain", return_value=response):
            return _decision_stage(
                question="Make a line chart of percentage of ischemic strokes per quarter",
                entities=entities,
                language="en",
            )

    def test_invented_options_with_one_resolved_metric_proceed(self) -> None:
        outcome = self._run(self._INVENTED, {"chart_type": "LINE", "stroke_type": "ISCHEMIC", "group_by": "QUARTER", "metric": "STROKE_TYPE"})

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "all_required_fields_present")

    def test_invented_options_without_a_type_but_with_an_ambiguous_reason_proceed(self) -> None:
        response = dict(self._INVENTED, clarification_type=None)
        outcome = self._run(response, {"chart_type": "LINE", "stroke_type": "ISCHEMIC", "metric": "STROKE_TYPE"})

        self.assertEqual(outcome.decision, "proceed")

    def test_real_metric_options_still_ask(self) -> None:
        response = dict(self._INVENTED, clarification_options=["ADMISSION_NIHSS", "DISCHARGE_NIHSS"])
        outcome = self._run(response, {"metric": "ADMISSION_NIHSS", "chart_type": "LINE"})

        self.assertEqual(outcome.decision, "clarify")
        self.assertEqual(outcome.clarification_options, ["ADMISSION_NIHSS", "DISCHARGE_NIHSS"])

    def test_metric_options_written_as_words_count_as_real(self) -> None:
        response = dict(self._INVENTED, clarification_options=["admission nihss", "discharge nihss"])
        outcome = self._run(response, {"metric": "ADMISSION_NIHSS", "chart_type": "LINE"})

        self.assertEqual(outcome.decision, "clarify")

    def test_without_a_resolved_metric_the_question_stands(self) -> None:
        outcome = self._run(self._INVENTED, {"chart_type": "LINE", "stroke_type": "ISCHEMIC"})

        self.assertEqual(outcome.decision, "clarify")

    def test_a_question_about_something_else_stands(self) -> None:
        response = dict(self._INVENTED, clarification_type="filter", clarification_options=["ischemic only", "all strokes"])
        outcome = self._run(response, {"metric": "STROKE_TYPE", "stroke_type": "ISCHEMIC"})

        self.assertEqual(outcome.decision, "clarify")


class RangeClarificationTests(unittest.TestCase):
    _ASKS_WHICH_SIDE = {
        "decision": "clarify",
        "reason": "ambiguous_request",
        "missing_fields": None,
        "clarification_type": "age_group",
        "clarification_options": ["over 50", "under 50"],
        "message": "Please specify if you want data for patients over 50 or under 50.",
    }
    _BOTH_SIDES = {"chart_type": "BAR", "metric": "DTN", "age": "50", "age_lower": "50", "age_upper": "50"}

    def _run(self, response, entities):
        with patch("src.planners.langchain.request_orchestrator._invoke_chain", return_value=response):
            return _decision_stage(question="Show me a bar chart of dtn for patients over 50 and under 50", entities=entities, language="en")

    def test_asking_which_side_of_a_given_bound_proceeds(self) -> None:
        outcome = self._run(self._ASKS_WHICH_SIDE, self._BOTH_SIDES)

        self.assertEqual(outcome.decision, "proceed")
        self.assertEqual(outcome.reason, "all_required_fields_present")

    def test_options_made_of_the_given_bound_count_without_a_type(self) -> None:
        outcome = self._run(dict(self._ASKS_WHICH_SIDE, clarification_type=None), self._BOTH_SIDES)

        self.assertEqual(outcome.decision, "proceed")

    def test_a_single_given_bound_is_enough(self) -> None:
        response = dict(self._ASKS_WHICH_SIDE, clarification_type="age_range", clarification_options=["18-50", "50-120"])
        outcome = self._run(response, {"metric": "DTN", "age": "60", "age_lower": "60"})

        self.assertEqual(outcome.decision, "proceed")

    def test_a_range_question_with_no_bound_given_stands(self) -> None:
        outcome = self._run(self._ASKS_WHICH_SIDE, {"chart_type": "BAR", "metric": "DTN"})

        self.assertEqual(outcome.decision, "clarify")

    def test_a_type_that_merely_contains_the_letters_is_not_a_range_question(self) -> None:
        response = dict(self._ASKS_WHICH_SIDE, clarification_type="percentage_type", clarification_options=["share", "count"])
        outcome = self._run(response, self._BOTH_SIDES)

        self.assertEqual(outcome.decision, "clarify")


class NormalizeEntitiesForQuestionTests(unittest.TestCase):
    def test_kpi_annotation_is_dropped(self) -> None:
        normalized = _normalize_entities_for_question(
            "percentage of ischemic strokes per quarter",
            {"kpi": "percent", "stroke_type": "ISCHEMIC", "chart_type": "LINE"},
        )

        self.assertEqual(normalized, {"stroke_type": "ISCHEMIC", "chart_type": "LINE"})


if __name__ == "__main__":
    unittest.main()
