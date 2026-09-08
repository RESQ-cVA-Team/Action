import asyncio
import os
import unittest
from uuid import uuid4

from src.domain.langchain.schema import (
    AnalysisPlan,
    AnalysisSemanticsSpec,
    ChartSpec,
    DataOriginSpec,
    DateFilter,
    AndFilter,
    MeasureSemanticsSpec,
    MetricSpec,
    TimeRange,
    TimeSemanticsSpec,
)
from src.executors.orchestration.plan_executor import execute_plan_async
from src.executors.orchestration.plan_executor import VisualizationExecutionError
from src.planners.langchain.request_orchestrator import orchestrate_visualization_request
from src.executors.graphql.client import GraphQLProxyClient


def _env_enabled(name: str) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _has_required_env() -> bool:
    required = [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "RASA_PROXY_URL",
        "KEYCLOAK_ISSUER",
        "ACTION_SERVICE_CLIENT_ID",
        "ACTION_SERVICE_CLIENT_SECRET",
        "RASA_PROXY_GRAPHQL_TARGET",
    ]
    return all(bool((os.getenv(key) or "").strip()) for key in required)


def _strict_external_e2e() -> bool:
    return _env_enabled("RUN_EXTERNAL_API_E2E_STRICT")


@unittest.skipUnless(
    _env_enabled("RUN_EXTERNAL_API_E2E") and _has_required_env(),
    "Set RUN_EXTERNAL_API_E2E=1 and required env vars to run real external API E2E smoke tests.",
)
class ExternalApiE2ESmokeTests(unittest.TestCase):
    @staticmethod
    def _graph_client() -> GraphQLProxyClient:
        return GraphQLProxyClient(
            proxy_url=os.environ["RASA_PROXY_URL"],
            target=os.environ.get("RASA_PROXY_GRAPHQL_TARGET") or "graphql",
        )

    @staticmethod
    def _has_graphql_token_cache(user_sub: str) -> bool:
        client = ExternalApiE2ESmokeTests._graph_client()
        probe = client.query_raw(
            query_str="query { __typename }",
            user_sub=user_sub,
            trace_id=f"external-e2e-probe-{uuid4().hex[:8]}",
            variables={},
            raise_on_error=False,
        )
        return probe is not None

    @staticmethod
    def _require_graphql_token_cache_or_skip(user_sub: str) -> None:
        if ExternalApiE2ESmokeTests._has_graphql_token_cache(user_sub):
            return
        raise unittest.SkipTest(
            "Skipping GraphQL executor E2E: no cached proxy user access token for senderId in this environment. "
            "Run one authenticated /api/rasa request first to seed token cache, then re-run with RUN_EXTERNAL_API_E2E=1."
        )

    def test_orchestrator_decision_stage_calls_live_llm(self) -> None:
        outcome = orchestrate_visualization_request(
            question="Show me a line graph of DTN",
            entities={"metric": "DTN", "chart_type": "LINE"},
            include_plan=False,
            trace_id="external-e2e-orchestrator",
        )

        self.assertEqual(outcome.decision, "proceed")
        self.assertTrue(bool((outcome.reason or "").strip()))

    def test_plan_executor_calls_live_graphql_proxy(self) -> None:
        user_sub = "external-e2e-user"
        self._require_graphql_token_cache_or_skip(user_sub)

        plan = AnalysisPlan(
            charts=[
                ChartSpec(
                    chart_type="LINE",
                    metrics=[MetricSpec(metric="DTN", data_origin=DataOriginSpec(providerGroupId=[289]))],
                    filters=AndFilter(
                        and_=[
                            DateFilter(operator="GE", value="2023-01-01"),
                            DateFilter(operator="LE", value="2023-12-31"),
                        ]
                    ),
                    semantics=AnalysisSemanticsSpec(
                        intent="TREND",
                        measure=MeasureSemanticsSpec(type="MEAN"),
                        time=TimeSemanticsSpec(
                            grain="MONTH",
                            window=TimeRange(start_date="2023-01-01", end_date="2023-12-31"),
                        ),
                    ),
                )
            ]
        )

        try:
            response = asyncio.run(
                execute_plan_async(
                    plan=plan,
                    user_sub=user_sub,
                    trace_id="external-e2e-executor",
                )
            )
        except VisualizationExecutionError:
            if (not _strict_external_e2e()) and (not self._has_graphql_token_cache(user_sub)):
                raise unittest.SkipTest(
                    "Skipping GraphQL executor E2E: proxy user token cache became unavailable during execution. "
                    "Set RUN_EXTERNAL_API_E2E_STRICT=1 to fail instead of skip."
                )
            raise

        # The smoke assertion is transport-level: the request executed and returned a valid envelope.
        self.assertEqual(response.type, "visualization_response")
        self.assertEqual(response.schema_version, 1)
        self.assertIsNotNone(response.trace_id)


if __name__ == "__main__":
    unittest.main()