import unittest
from unittest.mock import Mock, patch

from src.executors.analytics_center import client as analytics_center_client


class AnalyticsCenterClientAuthHeaderTests(unittest.TestCase):
    def _make_client(self) -> analytics_center_client.AnalyticsCenterClient:
        return analytics_center_client.AnalyticsCenterClient(
            proxy_url="http://webapp:3000/api/rasa-proxy",
        )

    def _mock_response(self) -> Mock:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": [], "count": 0}
        return response

    def test_attaches_keycloak_bearer_token(self) -> None:
        client = self._make_client()
        with patch.object(
            analytics_center_client, "get_service_account_token_or_raise", return_value="fresh-keycloak-token"
        ):
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
        self.assertNotIn("x-action-server-token", headers)

    def test_raises_instead_of_sending_unauthenticated_request(self) -> None:
        client = self._make_client()
        with patch.object(
            analytics_center_client,
            "get_service_account_token_or_raise",
            side_effect=RuntimeError("Could not obtain a Keycloak service-account token"),
        ):
            with patch.object(analytics_center_client.requests, "post") as post:
                with self.assertRaises(RuntimeError):
                    client._request_via_proxy(
                        user_sub="user-1",
                        path="/providers",
                        query={},
                        request_name="list_providers",
                        trace_id="trace-1",
                    )

        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
