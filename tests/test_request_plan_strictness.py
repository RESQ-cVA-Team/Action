import pytest

from src.domain.graphql.request import DataOrigin, MetricRequest, MetricType
from src.executors.planning.request_plan import CategoryView, build_primary_request_specs


def _metric_request() -> MetricRequest:
    return MetricRequest(metricType=MetricType.DTN).with_stats()


def test_build_primary_request_specs_requires_explicit_chart_data_origin() -> None:
    try:
        build_primary_request_specs(
            metric_requests=[_metric_request()],
            metric_data_origins=None,
            chart_filter=None,
            filter_dims=[],
            combos_list=[tuple()],
            batched_time_enabled=False,
            batched_time_periods=[],
            include_metric_alias=False,
            group_by_field=None,
            data_origin=None,
            include_general_stats=True,
        )
        assert False, "Expected ValueError for missing chart data origin"
    except ValueError as exc:
        assert "explicit chart data origin" in str(exc)


def test_build_primary_request_specs_uses_explicit_chart_data_origin() -> None:
    specs = build_primary_request_specs(
        metric_requests=[_metric_request()],
        metric_data_origins=None,
        chart_filter=None,
        filter_dims=[],
        combos_list=[tuple()],
        batched_time_enabled=False,
        batched_time_periods=[],
        include_metric_alias=False,
        group_by_field=None,
        data_origin=DataOrigin(providerGroupId=[99]),
        include_general_stats=True,
    )

    assert len(specs) == 1
    request = specs[0].req
    assert request.data_origin.provider_group_id == [99]
    assert request.include_general_stats is True


def test_build_primary_request_specs_keeps_category_detail_out_of_graphql_request() -> None:
    aa_request = MetricRequest(metricType=MetricType.AA_DTN_LE60).with_stats().with_categorical()
    specs = build_primary_request_specs(
        metric_requests=[aa_request],
        metric_data_origins=None,
        chart_filter=None,
        filter_dims=[],
        combos_list=[tuple()],
        batched_time_enabled=False,
        batched_time_periods=[],
        include_metric_alias=False,
        group_by_field=None,
        data_origin=DataOrigin(providerGroupId=[99]),
        include_general_stats=True,
        category_views=[CategoryView(include_no=True)],
    )

    assert specs[0].category_views == {"AA_DTN_LE60": CategoryView(include_no=True)}
    assert "categoryDetail" not in specs[0].req.to_graphql_string()


def test_build_primary_request_specs_requires_explicit_per_metric_origin() -> None:
    try:
        build_primary_request_specs(
            metric_requests=[_metric_request()],
            metric_data_origins=[None],
            chart_filter=None,
            filter_dims=[],
            combos_list=[tuple()],
            batched_time_enabled=False,
            batched_time_periods=[],
            include_metric_alias=False,
            group_by_field=None,
            data_origin=None,
            include_general_stats=False,
        )
        assert False, "Expected ValueError for missing per-metric data origin"
    except ValueError as exc:
        assert "explicit chart data origin" in str(exc)


def test_build_primary_request_specs_uses_explicit_per_metric_origin() -> None:
    specs = build_primary_request_specs(
        metric_requests=[_metric_request()],
        metric_data_origins=[DataOrigin(providerId=[7])],
        chart_filter=None,
        filter_dims=[],
        combos_list=[tuple()],
        batched_time_enabled=False,
        batched_time_periods=[],
        include_metric_alias=False,
        group_by_field=None,
        data_origin=None,
        include_general_stats=False,
    )

    assert len(specs) == 1
    request = specs[0].req
    assert request.data_origin.provider_id == [7]
    assert request.include_general_stats is False


def test_build_primary_request_specs_keeps_category_detail_per_metric_origin() -> None:
    aa_requests = [
        MetricRequest(metricType=MetricType.AA_DTN_LE60).with_stats().with_categorical()
        for _ in range(2)
    ]
    specs = build_primary_request_specs(
        metric_requests=aa_requests,
        metric_data_origins=[DataOrigin(providerId=[7]), DataOrigin(providerId=[8])],
        chart_filter=None,
        filter_dims=[],
        combos_list=[tuple()],
        batched_time_enabled=False,
        batched_time_periods=[],
        include_metric_alias=False,
        group_by_field=None,
        include_general_stats=False,
        category_views=[CategoryView(include_no=True), CategoryView()],
    )

    assert [spec.category_views for spec in specs] == [{"AA_DTN_LE60": CategoryView(include_no=True)}, {}]


def test_build_primary_request_specs_rejects_one_metric_plotted_two_ways_in_one_request() -> None:
    request = MetricRequest(metricType=MetricType.IMAGING_TYPE).with_stats().with_categorical()
    with pytest.raises(ValueError, match="two ways"):
        build_primary_request_specs(
            metric_requests=[request, request],
            metric_data_origins=None,
            chart_filter=None,
            filter_dims=[],
            combos_list=[tuple()],
            batched_time_enabled=False,
            batched_time_periods=[],
            include_metric_alias=False,
            group_by_field=None,
            data_origin=DataOrigin(providerGroupId=[99]),
            include_general_stats=True,
            category_views=[CategoryView(combined=("ct",)), CategoryView(combined=("mr_dwi_flair",))],
        )
