import unittest
from unittest.mock import Mock, patch

from src.executors.graphql import client as graphql_client


class GraphQLProxyClientAuthHeaderTests(unittest.TestCase):
    def _make_client(self) -> graphql_client.GraphQLProxyClient:
        return graphql_client.GraphQLProxyClient(
            proxy_url="http://webapp:3000/api/rasa-proxy",
            retry_attempts=0,
        )

    def _mock_response(self, status_code: int = 502) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = {}
        response.text = ""
        return response

    def test_attaches_keycloak_bearer_token_and_job_id(self) -> None:
        client = self._make_client()
        with patch.object(
            graphql_client, "get_service_account_token_or_raise", return_value="fresh-keycloak-token"
        ):
            with patch.object(graphql_client.requests, "post", return_value=self._mock_response()) as post:
                client.query(
                    query_str="{ __typename }",
                    user_sub="user-1",
                    trace_id="trace-1",
                    job_id="job-1",
                )

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer fresh-keycloak-token")
        self.assertNotIn("x-action-server-token", headers)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["jobId"], "job-1")
        self.assertNotIn("senderId", body)

    def test_raises_when_job_id_is_missing(self) -> None:
        """Synchronous/shell execution has no callback URL and therefore no
        jobId -- this must fail loudly, not silently drop identity."""
        client = self._make_client()
        with patch.object(graphql_client.requests, "post") as post:
            with self.assertRaises(graphql_client.GraphQLProxyError):
                client.query(
                    query_str="{ __typename }",
                    user_sub="user-1",
                    trace_id="trace-1",
                    job_id=None,
                )

        post.assert_not_called()

    def test_query_raw_also_requires_job_id(self) -> None:
        client = self._make_client()
        with patch.object(graphql_client.requests, "post") as post:
            with self.assertRaises(graphql_client.GraphQLProxyError):
                client.query_raw(
                    query_str="{ __typename }",
                    user_sub="user-1",
                    trace_id="trace-1",
                    job_id=None,
                )

        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
