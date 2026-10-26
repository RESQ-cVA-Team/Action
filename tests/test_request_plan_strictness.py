from src.domain.graphql.request import DataOrigin, MetricRequest, MetricType
from src.executors.planning.request_plan import build_primary_request_specs


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
        category_detail_metric_codes=["AA_DTN_LE60"],
    )

    assert specs[0].category_detail_metric_codes == ["AA_DTN_LE60"]
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
