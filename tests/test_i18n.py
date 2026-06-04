import importlib.util
import unittest
from pathlib import Path
from unittest import mock


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


i18n = load_module("action_i18n_under_test", "src/actions/i18n.py")
DEFAULT_LANGUAGE = i18n.DEFAULT_LANGUAGE
normalize_language_code = i18n.normalize_language_code
resolve_language = i18n.resolve_language
resolve_language_from_tracker = i18n.resolve_language_from_tracker


class I18nTests(unittest.TestCase):
    def test_normalize_language_code_accepts_locale_variants(self) -> None:
        self.assertEqual(normalize_language_code("en-US,en;q=0.9"), "en")
        self.assertEqual(normalize_language_code("cs"), "cs")
        self.assertIsNone(normalize_language_code("fr"))

    def test_resolve_language_prefers_metadata_then_slots_then_tracker(self) -> None:
        tracker = mock.Mock()
        tracker.current_state.return_value = {"slots": {"language": "cs"}}

        self.assertEqual(resolve_language(metadata={"language": "el"}, slots={"language": "cs"}, tracker=tracker), "el")
        self.assertEqual(resolve_language(slots={"language": "cs"}, tracker=tracker), "cs")
        self.assertEqual(resolve_language(tracker=tracker), "cs")
        self.assertEqual(resolve_language(metadata={"language": "fr"}), DEFAULT_LANGUAGE)

    def test_resolve_language_from_tracker_uses_latest_message_metadata(self) -> None:
        tracker = mock.Mock()
        tracker.latest_message = {"metadata": {"language": "el"}}
        tracker.current_state.return_value = {"slots": {"language": "cs"}}

        self.assertEqual(resolve_language_from_tracker(tracker), "el")


if __name__ == "__main__":
    unittest.main()