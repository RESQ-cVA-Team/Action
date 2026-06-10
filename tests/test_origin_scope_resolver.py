import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


resolver = load_module("action_origin_scope_resolver_under_test", "src/executors/planning/origin_scope_resolver.py")


class _FakeAnalyticsClient:
    def __init__(self) -> None:
        self._default_scope = {"provider_id": 999}

    def resolve_my_default_scope(self, **kwargs):
        return self._default_scope

    def list_providers(self, **kwargs):
        return {
            "results": [
                {
                    "id": 279,
                    "nameEnglish": "Aalborg University Hospital",
                    "countryCode": "DK",
                }
            ],
            "count": 1,
            "limit": 200,
            "offset": 0,
        }

    def list_provider_groups(self, **kwargs):
        return {
            "results": [
                {
                    "id": 7,
                    "name": "Aalborg Group",
                    "countryCode": "DK",
                }
            ],
            "count": 1,
            "limit": 200,
            "offset": 0,
        }

    def resolve_country_code(self, **kwargs):
        country_input = str(kwargs.get("country_input", "")).strip().upper()
        return country_input if len(country_input) == 2 else None


def _single_metric_plan(origin_scope: resolver.S.OriginScopeSpec) -> resolver.S.AnalysisPlan:
    return resolver.S.AnalysisPlan(
        charts=[
            resolver.S.ChartSpec(
                chart_type="LINE",
                xAxis={"grain": "MONTH"},
                yAxes=[
                    resolver.S.YAxisSpec(
                        metrics=[resolver.S.MetricSpec(metric="DTN", originScope=origin_scope)],
                        statistic="MEAN",
                    )
                ],
            )
        ]
    )


class OriginScopeResolverTests(unittest.TestCase):
    def test_mine_falls_back_to_single_accessible_provider(self) -> None:
        plan = _single_metric_plan(resolver.S.OriginScopeSpec(scopeType="mine"))
        fake_client = _FakeAnalyticsClient()

        with patch.object(resolver, "get_analytics_center_client", return_value=fake_client):
            resolved = resolver.resolve_plan_metric_origins(plan, user_sub="user-1", trace_id="trace-1")

        metric = resolved.charts[0].y_axes[0].metrics[0]
        self.assertIsNotNone(metric.data_origin)
        self.assertEqual(metric.data_origin.provider_id, [279])
        self.assertIsNone(metric.data_origin.provider_group_id)

    def test_inaccessible_provider_group_id_is_rejected_without_fallback(self) -> None:
        plan = _single_metric_plan(resolver.S.OriginScopeSpec(scopeType="provider_group_id", value=1))
        fake_client = _FakeAnalyticsClient()

        with patch.object(resolver, "get_analytics_center_client", return_value=fake_client):
            with self.assertRaises(resolver.OriginScopeResolutionError) as ctx:
                resolver.resolve_plan_metric_origins(plan, user_sub="user-1", trace_id="trace-2")

        self.assertEqual(ctx.exception.reason, "unauthorized_origin")
        self.assertEqual(ctx.exception.clarification_type, "provider_group_id")
        self.assertIn("did not automatically switch", str(ctx.exception).lower())
        self.assertTrue(any("group 7" in option for option in ctx.exception.clarification_options))

    def test_inaccessible_provider_id_is_rejected_without_fallback(self) -> None:
        plan = _single_metric_plan(resolver.S.OriginScopeSpec(scopeType="provider_id", value=1))
        fake_client = _FakeAnalyticsClient()

        with patch.object(resolver, "get_analytics_center_client", return_value=fake_client):
            with self.assertRaises(resolver.OriginScopeResolutionError) as ctx:
                resolver.resolve_plan_metric_origins(plan, user_sub="user-1", trace_id="trace-3")

        self.assertEqual(ctx.exception.reason, "unauthorized_origin")
        self.assertEqual(ctx.exception.clarification_type, "provider_id")
        self.assertIn("did not automatically switch", str(ctx.exception).lower())
        self.assertTrue(any("provider 279" in option for option in ctx.exception.clarification_options))


if __name__ == "__main__":
    unittest.main()