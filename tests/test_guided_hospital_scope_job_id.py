import unittest
from unittest.mock import patch

from src.actions import guided_visualization_validation as gv


class _Dispatcher:
    def __init__(self) -> None:
        self.messages = []

    def utter_message(self, **kwargs):
        self.messages.append(kwargs)


class _Tracker:
    sender_id = "0a709c3b-2c71-4c5b-85d6-66454da5c9d7:thread:4"
    slots = {}

    def __init__(self, entities):
        self.latest_message = {
            "text": "",
            "entities": entities,
            "metadata": {"callback_url": "http://webapp:3000/api/rasa/long-task-callback?jobId=job-123&traceId=trace-1", "trace_id": "trace-1"},
        }

    def get_slot(self, name):
        return None


class _Client:
    def __init__(self):
        self.calls = []

    def list_providers(self, **kwargs):
        self.calls.append(("list_providers", kwargs))
        return {"results": [{"id": 1853, "nameEnglish": "Army Alhama de Murcia Hospital"}], "count": 1}

    def resolve_country_code(self, **kwargs):
        self.calls.append(("resolve_country_code", kwargs))
        return "ES"

    def resolve_my_default_scope(self, **kwargs):
        self.calls.append(("resolve_my_default_scope", kwargs))
        return {"provider_id": 279}


class GuidedHospitalScopeJobIdTests(unittest.TestCase):
    # Live: the form validation crashed with "list_providers() missing 1
    # required positional argument: 'job_id'", so the guided flow went silent
    # at the hospital-scope step.

    def test_named_hospital_lookup_carries_the_job_id(self) -> None:
        client = _Client()
        with patch.object(gv, "get_analytics_center_client", return_value=client):
            result = gv.validate_guided_hospital_scope(
                slot_value="Army Alhama de Murcia Hospital",
                dispatcher=_Dispatcher(),
                tracker=_Tracker([{"entity": "hospital_name", "value": "Army Alhama de Murcia Hospital"}]),
            )
        self.assertIsNotNone(result.get("guided_hospital_scope"))
        self.assertTrue(client.calls)
        for name, kwargs in client.calls:
            self.assertEqual(kwargs.get("job_id"), "job-123", name)

    def test_mine_scope_lookup_carries_the_job_id(self) -> None:
        client = _Client()
        with patch.object(gv, "get_analytics_center_client", return_value=client):
            gv.validate_guided_hospital_scope(
                slot_value="my hospital",
                dispatcher=_Dispatcher(),
                tracker=_Tracker([{"entity": "hospital_scope_reference", "value": "my hospital"}]),
            )
        self.assertTrue(client.calls)
        for name, kwargs in client.calls:
            self.assertEqual(kwargs.get("job_id"), "job-123", name)

    def test_country_lookup_carries_the_job_id(self) -> None:
        client = _Client()
        with patch.object(gv, "get_analytics_center_client", return_value=client):
            gv.validate_guided_hospital_scope(
                slot_value="ES",
                dispatcher=_Dispatcher(),
                tracker=_Tracker([{"entity": "country_code", "value": "ES"}]),
            )
        self.assertEqual(client.calls[0][0], "resolve_country_code")
        self.assertEqual(client.calls[0][1].get("job_id"), "job-123")


if __name__ == "__main__":
    unittest.main()
