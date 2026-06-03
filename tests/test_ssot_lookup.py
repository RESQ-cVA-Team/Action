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


ssot_lookup = load_module("action_ssot_lookup_under_test", "src/actions/ssot_lookup.py")


class SsotLookupTests(unittest.TestCase):
    def test_normalize_text_collapses_spacing_and_punctuation(self) -> None:
        self.assertEqual(ssot_lookup.normalize_text("  Heart-Rate__Value  "), "heart rate value")

    def test_resolve_metric_candidates_uses_exact_match_first(self) -> None:
        with mock.patch.object(ssot_lookup, "normalize_metric_text_key", return_value="heart rate"):
            with mock.patch.object(
                ssot_lookup,
                "get_metric_text_lookup",
                return_value={"heart rate": {"canonical": "HEART_RATE"}},
            ):
                self.assertEqual(ssot_lookup.resolve_metric_candidates("heart rate"), ["HEART_RATE"])

    def test_resolve_catalog_candidates_uses_exact_and_fuzzy_matches(self) -> None:
        items = [
            {"canonical": "heart_rate", "synonyms": ["pulse"]},
            {"canonical": "blood_pressure", "synonyms": ["bp"]},
        ]

        with mock.patch.object(ssot_lookup, "get_ssot_items", return_value=items):
            self.assertEqual(ssot_lookup.resolve_catalog_candidates("metrics.yml", "pulse"), ["HEART_RATE"])
            self.assertEqual(ssot_lookup.resolve_catalog_candidates("metrics.yml", "blood"), ["BLOOD_PRESSURE"])


if __name__ == "__main__":
    unittest.main()