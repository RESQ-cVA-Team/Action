from src.actions.helpers.visualization import question_with_resolved_metrics


def _diet(entity, value, start, end):
    return {"entity": entity, "value": value, "start": start, "end": end, "extractor": "DIETClassifier", "extractors": [{"extractor": "DIETClassifier"}]}


def _regex(entity, value, start, end):
    return {"entity": entity, "value": value, "start": start, "end": end, "extractor": "RegexEntityExtractor", "extractors": [{"extractor": "RegexEntityExtractor"}]}


def test_metric_words_are_written_as_the_resolved_metric():
    text = "Show me a line graph of antiplatelets at discharge without atrial fibrillation"
    message = {
        "text": text,
        "entities": [_diet("chart_type", "LINE", 10, 14), _diet("metric", "DISCHARGE_ANTIPLATELETS_NO_AFIB", 24, len(text))],
    }
    assert question_with_resolved_metrics(message) == "Show me a line graph of DISCHARGE_ANTIPLATELETS_NO_AFIB"


def test_only_metric_spans_change_and_the_rest_of_the_sentence_stays():
    text = "Display a chart of door to needle time for ischemic patients per quarter"
    message = {
        "text": text,
        "entities": [_diet("metric", "DTN", 19, 38), _diet("stroke_type", "ISCHEMIC", 43, 51), _diet("group_by", "QUARTER", 65, 72)],
    }
    assert question_with_resolved_metrics(message) == "Display a chart of DTN for ischemic patients per quarter"


def test_the_metric_written_is_the_one_extraction_keeps():
    # "ivt dose" is an SSOT synonym: the wider regex span wins over DIET's "ivt".
    synonym = {
        "text": "Show me a line graph of ivt dose",
        "entities": [_regex("metric", "THROMBOLYSIS_DRUG_DOSE", 24, 32), _diet("metric", "THROMBOLYSIS", 24, 27)],
    }
    assert question_with_resolved_metrics(synonym) == "Show me a line graph of THROMBOLYSIS_DRUG_DOSE"
    # "ivt dosage" is not: DIET's "ivt" is kept and the rest of the words stay.
    other = {
        "text": "Show me a line graph of ivt dosage",
        "entities": [_regex("metric", "THROMBOLYSIS_DRUG_DOSE", 24, 34), _diet("metric", "THROMBOLYSIS", 24, 27)],
    }
    assert question_with_resolved_metrics(other) == "Show me a line graph of THROMBOLYSIS dosage"


def test_a_message_without_metrics_is_unchanged():
    assert question_with_resolved_metrics({"text": "as a bar chart", "entities": [_diet("chart_type", "BAR", 5, 8)]}) == "as a bar chart"
    assert question_with_resolved_metrics({"text": "hello"}) == "hello"
