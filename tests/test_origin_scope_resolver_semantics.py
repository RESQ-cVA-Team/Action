import unittest
from unittest.mock import patch

from src.domain.langchain import schema as S
from src.executors.planning import origin_scope_resolver


class OriginScopeResolverSemanticsTests(unittest.TestCase):
    def test_list_accessible_providers_does_not_apply_user_filter(self) -> None:
        calls = []

        class FakeClient:
            def list_providers(self, **kwargs):
                calls.append(kwargs)
                return {
                    "results": [
                        {
                            "id": 1853,
                            "nameEnglish": "Army Alhama de Murcia Hospital",
                        }
                    ],
                    "count": 1,
                }

        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            providers = origin_scope_resolver._list_accessible_providers(
                user_sub="0a709c3b-2c71-4c5b-85d6-66454da5c9d7:thread:4",
                job_id="job-1",
                trace_id="trace-1",
            )

        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["id"], 1853)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("user", calls[0])

    def test_resolve_plan_metric_origins_preserves_chart_semantics(self) -> None:
        plan = S.AnalysisPlan(
            charts=[
                S.ChartSpec(
                    chart_type="LINE",
                    metrics=[S.MetricSpec(metric="DTN")],
                    semantics=S.AnalysisSemanticsSpec(
                        intent="TREND",
                        measure=S.MeasureSemanticsSpec(type="MEAN"),
                        time=S.TimeSemanticsSpec(grain="MONTH"),
                    ),
                )
            ]
        )

        with patch.object(origin_scope_resolver, "_resolve_metric_origin", side_effect=lambda metric, **_: metric):
            resolved = origin_scope_resolver.resolve_plan_metric_origins(
                plan=plan,
                user_sub="user-1",
                job_id="job-1",
                trace_id="trace-1",
            )

        self.assertIsNotNone(resolved.charts)
        resolved_chart = resolved.charts[0]
        self.assertIsNotNone(resolved_chart.semantics)
        self.assertEqual(resolved_chart.semantics.intent, "TREND")
        self.assertIsNotNone(resolved_chart.semantics.measure)
        self.assertEqual(resolved_chart.semantics.measure.type, "MEAN")
        self.assertIsNotNone(resolved_chart.semantics.time)
        self.assertEqual(resolved_chart.semantics.time.grain, "MONTH")

    def test_resolve_scope_provider_id_rejects_inaccessible_provider(self) -> None:
        class FakeClient:
            def list_providers(self, **kwargs):
                return {"results": [{"id": 1853, "nameEnglish": "Hospital A"}], "count": 1}

        scope = S.OriginScopeSpec(scopeType="provider_id", value=9999)
        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            with self.assertRaises(origin_scope_resolver.OriginScopeResolutionError):
                origin_scope_resolver._resolve_scope(
                    scope=scope,
                    user_sub="test-provider-id-reject",
                    job_id="job-1",
                    trace_id="trace-provider-id-reject",
                )

    def test_resolve_scope_provider_id_accepts_accessible_provider(self) -> None:
        class FakeClient:
            def list_providers(self, **kwargs):
                return {"results": [{"id": 1853, "nameEnglish": "Hospital A"}], "count": 1}

        scope = S.OriginScopeSpec(scopeType="provider_id", value=1853)
        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            resolved = origin_scope_resolver._resolve_scope(
                scope=scope,
                user_sub="test-provider-id-accept",
                job_id="job-1",
                trace_id="trace-provider-id-accept",
            )

        self.assertEqual(resolved.provider_id, [1853])

    def test_resolve_scope_provider_group_id_rejects_inaccessible_group(self) -> None:
        class FakeClient:
            def list_provider_groups(self, **kwargs):
                return {"results": [{"id": 42, "name": "Group A"}], "count": 1}

        scope = S.OriginScopeSpec(scopeType="provider_group_id", value=7777)
        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            with self.assertRaises(origin_scope_resolver.OriginScopeResolutionError):
                origin_scope_resolver._resolve_scope(
                    scope=scope,
                    user_sub="test-group-id-reject",
                    job_id="job-1",
                    trace_id="trace-group-id-reject",
                )

    def test_resolve_scope_provider_group_id_accepts_accessible_group(self) -> None:
        class FakeClient:
            def list_provider_groups(self, **kwargs):
                return {"results": [{"id": 42, "name": "Group A"}], "count": 1}

        scope = S.OriginScopeSpec(scopeType="provider_group_id", value=42)
        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            resolved = origin_scope_resolver._resolve_scope(
                scope=scope,
                user_sub="test-group-id-accept",
                job_id="job-1",
                trace_id="trace-group-id-accept",
            )

        self.assertEqual(resolved.provider_group_id, [42])

    def test_search_accessible_providers_by_name_stops_after_exact_match(self) -> None:
        calls = []

        class FakeClient:
            def list_providers(self, **kwargs):
                calls.append(kwargs)
                return {
                    "results": [
                        {
                            "id": 1853,
                            "nameEnglish": "Army Alhama de Murcia Hospital",
                        }
                    ],
                    "count": 1,
                }

        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=FakeClient()):
            providers = origin_scope_resolver._search_accessible_providers_by_name(
                requested_names=["Army Alhama de Murcia Hospital"],
                user_sub="user-1",
                job_id="job-1",
                trace_id="trace-2",
            )

        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["id"], 1853)
        self.assertEqual(len(calls), 1)


class ProviderNameSearchCachingTests(unittest.TestCase):
    def setUp(self) -> None:
        origin_scope_resolver._PROVIDER_LIST_CACHE.clear()

    def tearDown(self) -> None:
        origin_scope_resolver._PROVIDER_LIST_CACHE.clear()

    def _client(self, calls):
        class FakeClient:
            def list_providers(self, **kwargs):
                calls.append(kwargs)
                return {
                    "results": [
                        {"id": 1853, "nameEnglish": "Army Alhama de Murcia Hospital"},
                        {"id": 279, "nameEnglish": "My Hospital"},
                    ],
                    "count": 2,
                }

        return FakeClient()

    def test_name_that_matches_nothing_scans_once_and_is_cached(self) -> None:
        # Live: a hallucinated name paged through every provider, then the
        # catalog fallback paged through them all again, and the whole
        # resolution hit the 25s origin-scope timeout.
        calls = []
        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=self._client(calls)):
            first = origin_scope_resolver._search_accessible_providers_by_name(requested_names=["first hospital"], user_sub="user-1", job_id="job-1", trace_id="t")
            catalog = origin_scope_resolver._list_all_providers_catalog(user_sub="user-1", job_id="job-1", trace_id="t")
            second = origin_scope_resolver._search_accessible_providers_by_name(requested_names=["My Hospital"], user_sub="user-1", job_id="job-1", trace_id="t")

        self.assertEqual(first, [])
        self.assertEqual(len(catalog), 2)
        self.assertEqual([p["id"] for p in second], [279])
        self.assertEqual(len(calls), 1)

    def test_cache_is_per_user(self) -> None:
        calls = []
        with patch.object(origin_scope_resolver, "get_analytics_center_client", return_value=self._client(calls)):
            origin_scope_resolver._search_accessible_providers_by_name(requested_names=["My Hospital"], user_sub="user-1", job_id="job-1", trace_id="t")
            origin_scope_resolver._search_accessible_providers_by_name(requested_names=["My Hospital"], user_sub="user-2", job_id="job-1", trace_id="t")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()