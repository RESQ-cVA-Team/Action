import importlib.util
import sys
import unittest
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


guided_scope = load_module("action_guided_scope_under_test", "src/actions/guided_scope.py")


class GuidedScopeParserTests(unittest.TestCase):
    def test_provider_group_id_entity_takes_precedence_over_slot_text(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="1",
            entities={"provider_group_id": "1"},
        )
        self.assertEqual(intent.kind, "provider_group_id")
        self.assertEqual(intent.value, 1)

    def test_legacy_group_id_noise_does_not_override_provider_id(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="provider 1",
            entities={"provider_id": "1", "group_id": "1"},
        )
        self.assertEqual(intent.kind, "provider_id")
        self.assertEqual(intent.value, 1)

    def test_provider_id_entity_maps_to_provider_id(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="",
            entities={"provider_id": "289"},
        )
        self.assertEqual(intent.kind, "provider_id")
        self.assertEqual(intent.value, 289)

    def test_scope_kind_provider_group_maps_to_provider_group_mine(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="",
            entities={"scope_kind": "provider_group"},
        )
        self.assertEqual(intent.kind, "provider_group_mine")

    def test_region_entity_is_explicitly_unsupported(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="",
            entities={"region": "emea"},
        )
        self.assertEqual(intent.kind, "region_unsupported")

    def test_bare_numeric_without_entity_is_ambiguous(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="1",
            entities={},
        )
        self.assertEqual(intent.kind, "numeric_ambiguous")

    def test_keyword_all_maps_to_all_accessible(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="all hospitals",
            entities={},
        )
        self.assertEqual(intent.kind, "all_accessible")

    def test_free_text_without_structured_entity_requires_clarification(self) -> None:
        intent = guided_scope.parse_guided_scope_intent(
            slot_value="st anna",
            entities={},
        )
        self.assertEqual(intent.kind, "missing_structured_scope")


if __name__ == "__main__":
    unittest.main()
