import unittest
from unittest.mock import Mock, patch

from src.executors.analytics_center import client as analytics_center_client


class AnalyticsCenterClientAuthHeaderTests(unittest.TestCase):
    def _make_client(self) -> analytics_center_client.AnalyticsCenterClient:
        return analytics_center_client.AnalyticsCenterClient(
            proxy_url="http://webapp:3000/api/rasa-proxy",
            action_server_token="static-service-secret",
        )

    def _mock_response(self) -> Mock:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": [], "count": 0}
        return response

    def test_attaches_keycloak_bearer_token_when_available(self) -> None:
        client = self._make_client()
        with patch.object(analytics_center_client, "get_service_account_token", return_value="fresh-keycloak-token"):
            with patch.object(analytics_center_client.requests, "post", return_value=self._mock_response()) as post:
                client._request_via_proxy(
                    user_sub="user-1",
                    path="/providers",
                    query={},
                    request_name="list_providers",
                    trace_id="trace-1",
                )

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer fresh-keycloak-token")
        self.assertEqual(headers["x-action-server-token"], "static-service-secret")

    def test_falls_back_to_static_token_when_no_service_account_configured(self) -> None:
        client = self._make_client()
        with patch.object(analytics_center_client, "get_service_account_token", return_value=None):
            with patch.object(analytics_center_client.requests, "post", return_value=self._mock_response()) as post:
                client._request_via_proxy(
                    user_sub="user-1",
                    path="/providers",
                    query={},
                    request_name="list_providers",
                    trace_id="trace-1",
                )

        headers = post.call_args.kwargs["headers"]
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["x-action-server-token"], "static-service-secret")


if __name__ == "__main__":
    unittest.main()
