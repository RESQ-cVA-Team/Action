import asyncio
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("RASA_PROXY_URL", "http://localhost")
os.environ.setdefault("RASA_PROXY_GRAPHQL_TARGET", "http://localhost/graphql")

from src.executors.transport.request_runner import run_graphql_request


class _FakeRequest:
    def to_graphql_string(self) -> str:
        return "query { getMetrics { metrics } }"


class _FakeClient:
    def __init__(self, response: object):
        self._response = response

    def query(self, **kwargs: object) -> object:
        return self._response


class RequestRunnerStrictnessTests(unittest.TestCase):
    def test_raises_when_metric_kpi_group_is_not_list(self) -> None:
        bad_metric = SimpleNamespace(kpi_group="not-a-list")
        response = SimpleNamespace(
            data=SimpleNamespace(get_metrics=SimpleNamespace(metrics={"metric_DTN": bad_metric})),
            errors=None,
        )
        client = _FakeClient(response)

        with self.assertRaises(ValueError) as err:
            asyncio.run(
                run_graphql_request(
                    req=_FakeRequest(),
                    label_parts=["DTN"],
                    include_metric_alias=True,
                    group_by_field=None,
                    add_time_period_labels=False,
                    request_failures=[],
                    client=client,
                    user_sub="user-1",
                    trace_id="trace-1",
                    semaphore=asyncio.Semaphore(1),
                )
            )

        self.assertIn("non-list kpi_group", str(err.exception))


if __name__ == "__main__":
    unittest.main()
