import pytest
from pydantic import ValidationError

from src.domain.langchain import schema as S
from src.executors.planning.query_compiler import compile_chart_grouping


def test_compile_chart_grouping_prefers_semantics_splits_and_time() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(grain="MONTH"),
            splits=[S.SplitSpec(kind="SEX", categories=["MALE", "FEMALE"])],
        ),
        filters=S.AndFilter(
            and_=[
                S.DateFilter(type="DateFilter", operator="GE", value="2023-01-01"),
                S.DateFilter(type="DateFilter", operator="LE", value="2023-12-31"),
            ]
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1

    batch = compiled.batches[0]
    assert batch.batched_time_enabled is True
    assert len(batch.batched_time_periods) == 12
    # Compiled from semantic split SEX, not fallback group_by stroke type.
    assert len(batch.combos_list) == 2


def test_compile_chart_grouping_falls_back_to_default_bounds_without_explicit_time_range() -> None:
    """Charts and statistical tests must behave the same when no explicit time
    window/range is given: fall back to default_time_bounds() rather than
    hard-failing and forcing the user to restate an explicit range."""
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(grain="MONTH"),
        ),
    )

    result = compile_chart_grouping(chart)
    assert result.total_requests > 0


def test_compile_chart_grouping_quarter_grain_falls_back_to_default_bounds() -> None:
    """Regression test for a reported bug: 'quarterly over the past 2 years' with
    no window populated must not hard-fail asking for an explicit range -- QUARTER
    grain needs the same default-bounds fallback MONTH grain already has."""
    chart = S.ChartSpec(
        chart_type="BAR",
        metrics=[S.MetricSpec(metric="STROKE_TYPE")],
        semantics=S.AnalysisSemanticsSpec(
            intent="DISTRIBUTION",
            measure=S.MeasureSemanticsSpec(type="DISTRIBUTION"),
            time=S.TimeSemanticsSpec(grain="QUARTER"),
        ),
    )

    result = compile_chart_grouping(chart)
    assert result.total_requests > 0


def test_compile_chart_grouping_year_grain_falls_back_to_default_bounds() -> None:
    """YEAR grain previously wasn't bucketable at all (no branch in
    Dimension.categories() for it), so any yearly-trend chart hard-failed
    regardless of whether a window was given."""
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(grain="YEAR"),
        ),
    )

    result = compile_chart_grouping(chart)
    assert result.total_requests > 0


def test_compile_chart_grouping_year_grain_with_relative_window() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(grain="YEAR", window=S.TimeWindow(last_n=3, unit="YEAR")),
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1
    assert len(compiled.batches[0].batched_time_periods) == 3


def test_compile_chart_grouping_year_grain_with_explicit_range() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEAN"),
            time=S.TimeSemanticsSpec(
                grain="YEAR",
                window=S.TimeRange(start_date="2022-06-15", end_date="2025-01-10"),
            ),
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert len(compiled.batches) == 1
    periods = compiled.batches[0].batched_time_periods
    assert [p.start_date for p in periods] == ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
    assert [p.end_date for p in periods] == ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]


def test_compile_chart_grouping_day_grain_falls_back_to_default_bounds() -> None:
    """DAY grain should use default bounds when planner omits an explicit
    window/range, mirroring MONTH/QUARTER/YEAR fallback behavior."""
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DAY_2_TEMPERATURE_CHECKS")],
        semantics=S.AnalysisSemanticsSpec(
            intent="TREND",
            measure=S.MeasureSemanticsSpec(type="MEDIAN"),
            time=S.TimeSemanticsSpec(grain="DAY"),
        ),
    )

    compiled = compile_chart_grouping(chart)
    assert compiled.total_requests > 0
    assert compiled.batches[0].batched_time_enabled is True
    assert len(compiled.batches[0].batched_time_periods) > 0


def test_compile_chart_grouping_rejects_unsupported_custom_split() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="COMPARISON",
            measure=S.MeasureSemanticsSpec(type="MEDIAN"),
            splits=[S.SplitSpec(kind="CUSTOM")],
        ),
    )

    try:
        compile_chart_grouping(chart)
        assert False, "Expected ValueError for unsupported CUSTOM split"
    except ValueError as exc:
        assert "CUSTOM" in str(exc)


def test_compile_chart_grouping_requires_semantics() -> None:
    chart = S.ChartSpec(
        chart_type="LINE",
        metrics=[S.MetricSpec(metric="DTN")],
    )

    try:
        compile_chart_grouping(chart)
        assert False, "Expected ValueError when semantics are missing"
    except ValueError as exc:
        assert "semantics" in str(exc)


def test_chart_spec_requires_semantic_measure() -> None:
    with pytest.raises(ValidationError, match="semantics.measure is required"):
        S.ChartSpec(
            chart_type="LINE",
            metrics=[S.MetricSpec(metric="DTN")],
            semantics=S.AnalysisSemanticsSpec(intent="TREND", time=S.TimeSemanticsSpec(grain="MONTH")),
        )


def test_canonical_split_rejects_unknown_field() -> None:
    chart = S.ChartSpec(
        chart_type="BAR",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="COMPARISON",
            measure=S.MeasureSemanticsSpec(type="COUNT"),
            splits=[S.SplitSpec(kind="CANONICAL", field="FOO", categories=[])],
        ),
    )

    with pytest.raises(ValueError, match="FOO is not a valid CanonicalGroupByField"):
        compile_chart_grouping(chart)


def test_canonical_split_without_categories_groups_server_side() -> None:
    chart = S.ChartSpec(
        chart_type="BAR",
        metrics=[S.MetricSpec(metric="DTN")],
        semantics=S.AnalysisSemanticsSpec(
            intent="COMPARISON",
            measure=S.MeasureSemanticsSpec(type="COUNT"),
            splits=[S.SplitSpec(kind="CANONICAL", field="EMS_PRENOTIFICATION", categories=[])],
        ),
    )

    compiled = compile_chart_grouping(chart)

    assert [batch.server_groupby for batch in compiled.batches] == ["EMS_PRENOTIFICATION"]


def test_time_range_rejects_non_iso_bounds() -> None:
    with pytest.raises(ValidationError, match="not a valid ISO 8601 date"):
        S.TimeRange(start_date="not-a-date", end_date="2023-12-31")
