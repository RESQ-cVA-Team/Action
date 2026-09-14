import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[1]
_ensure_package("src", _ROOT / "src")
_ensure_package("src.actions", _ROOT / "src/actions")
_ensure_package("src.actions.long_action", _ROOT / "src/actions/long_action")

rasa_sdk_module = types.ModuleType("rasa_sdk")


class _StubAction:
    pass


rasa_sdk_module.Action = _StubAction
sys.modules.setdefault("rasa_sdk", rasa_sdk_module)

_load_module("src.actions.long_action.long_action_registry", "src/actions/long_action/long_action_registry.py")
long_action_context_module = _load_module(
    "src.actions.long_action.long_action_context",
    "src/actions/long_action/long_action_context.py",
)
long_action_module = _load_module(
    "src.actions.long_action.long_action",
    "src/actions/long_action/long_action.py",
)

LongAction = long_action_module.LongAction
LongActionContext = long_action_context_module.LongActionContext


class _FakeLongAction(LongAction):
    def name(self) -> str:
        return "fake_long_action"

    async def work(self, ctx: LongActionContext) -> Any:
        return None


class LongActionCallbackPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action = _FakeLongAction()

    def test_build_callback_payload_wraps_user_visible_message_as_tracker_event(self) -> None:
        ctx = LongActionContext(sender_id="u1:thread:7", tracker_snapshot={})

        payload = self.action._build_callback_payload(
            ctx,
            {
                "text": "Hello there",
                "custom": {"progress": "working"},
                "buttons": [{"title": "Retry", "payload": "/retry"}],
            },
            trace_id="trace-123",
        )

        self.assertEqual(payload["controls"], [])
        self.assertEqual(len(payload["events"]), 1)

        event = payload["events"][0]
        self.assertEqual(event["event"], "bot")
        self.assertEqual(event["text"], "Hello there")
        self.assertEqual(event["metadata"]["source"], "long-task-callback")
        self.assertEqual(event["metadata"]["trace_id"], "trace-123")
        self.assertEqual(event["data"]["custom"], {"progress": "working"})
        self.assertEqual(event["data"]["buttons"], [{"title": "Retry", "payload": "/retry"}])

    def test_build_callback_payload_emits_lock_as_control_signal(self) -> None:
        ctx = LongActionContext(sender_id="u1:thread:7", tracker_snapshot={})
        ctx._job_id = "job-abc"

        payload = self.action._build_callback_payload(
            ctx,
            {"type": "lock"},
            trace_id="trace-123",
        )

        self.assertEqual(payload["events"], [])
        self.assertEqual(
            payload["controls"],
            [{
                "type": "lock",
                "jobId": "job-abc",
                "scope": "long_action",
                "source": "long-task-callback",
                "traceId": "trace-123",
            }],
        )


if __name__ == "__main__":
    unittest.main()