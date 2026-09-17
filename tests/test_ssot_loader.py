import unittest
from typing import Any, Dict, List
from unittest import mock

from src.shared import ssot_loader


class SsotLoaderTests(unittest.TestCase):
    def test_metric_text_lookup_supports_localized_synonyms_and_descriptions(
        self,
    ) -> None:
        items: List[Dict[str, Any]] = [
            {
                "canonical": "THROMBECTOMY",
                "synonyms": {
                    "en": ["clot removal", "mechanical thrombectomy"],
                    "el": ["αφαίρεση θρόμβου"],
                },
                "descriptions": {
                    "en": "Description for THROMBECTOMY",
                    "el": "Περιγραφή για ΘΡΟΜΒΕΚΤΟΜΙΑ",
                },
                "data_type": "Enum",
            }
        ]

        ssot_loader.get_metric_text_lookup.cache_clear()
        with mock.patch.object(ssot_loader, "_load_yaml", return_value=items):
            lookup = ssot_loader.get_metric_text_lookup()

        self.assertEqual(lookup["clot removal"]["canonical"], "THROMBECTOMY")
        self.assertEqual(lookup["αφαίρεση θρόμβου"]["canonical"], "THROMBECTOMY")
        self.assertEqual(
            lookup["clot removal"]["descriptions"]["el"],
            "Περιγραφή για ΘΡΟΜΒΕΚΤΟΜΙΑ",
        )

    def test_metric_metadata_has_real_descriptions_from_disk(self) -> None:
        # Regression guard: MetricType.yml uses "descriptions" (plural), unlike
        # every other SSOT file which uses "description" (singular). A mock-only
        # test can't catch a mismatch against the real key -- this reads the
        # actual on-disk file to make sure descriptions are non-empty.
        ssot_loader.get_metric_metadata.cache_clear()
        metadata = ssot_loader.get_metric_metadata()
        self.assertIn("DTN", metadata)
        self.assertIn("descriptions", metadata["DTN"])
        self.assertTrue(metadata["DTN"]["descriptions"].get("en"))

    def test_metric_text_lookup_prefers_canonical_key_over_synonym_collision(self) -> None:
        items: List[Dict[str, Any]] = [
            {
                "canonical": "ICH_TREATMENT_TYPE",
                "synonyms": {"en": ["ich treatment type"]},
                "Enum": [
                    {
                        "key": "ich_treatment_craniectomy",
                        "synonyms": {"en": ["craniectomy"]},
                    }
                ],
            },
            {
                "canonical": "CRANIECTOMY",
                "synonyms": {"en": ["decompressive craniectomy performed"]},
            },
        ]

        ssot_loader.get_metric_text_lookup.cache_clear()
        with mock.patch.object(ssot_loader, "_load_yaml", return_value=items):
            lookup = ssot_loader.get_metric_text_lookup()

        self.assertEqual(lookup["craniectomy"]["canonical"], "CRANIECTOMY")

    def tearDown(self) -> None:
        ssot_loader.get_metric_text_lookup.cache_clear()
        ssot_loader.get_metric_metadata.cache_clear()


class ResolveScopeTests(unittest.TestCase):
    def test_resolves_all_variants(self) -> None:
        for value in ["all", "All Hospitals", "all sites", "ALL PROVIDERS"]:
            self.assertEqual(ssot_loader.resolve_scope(value), "ALL")

    def test_resolves_mine_variants(self) -> None:
        for value in ["mine", "my hospital", "our centre", "MY SITE"]:
            self.assertEqual(ssot_loader.resolve_scope(value), "MINE")

    def test_resolves_non_english_variants(self) -> None:
        # These had no locale coverage at all before ScopeType.yml existed.
        self.assertEqual(ssot_loader.resolve_scope("moje nemocnice"), "MINE")
        self.assertEqual(ssot_loader.resolve_scope("όλα"), "ALL")

    def test_unknown_scope_returns_none(self) -> None:
        self.assertIsNone(ssot_loader.resolve_scope("St. Mary's Hospital"))


class GetOperatorSymbolTests(unittest.TestCase):
    def test_returns_real_symbols_from_disk(self) -> None:
        # Regression guard: OperatorType.yml's GE/LE/EQ entries must keep the
        # symbol as their first synonym, since get_operator_symbol (like every
        # other SSOT display accessor) just returns the first synonym.
        expected = {"GE": ">=", "LE": "<=", "LT": "<", "GT": ">", "EQ": "=", "NE": "!="}
        for code, symbol in expected.items():
            self.assertEqual(ssot_loader.get_operator_symbol(code), symbol)

    def test_falls_back_to_code_when_unknown(self) -> None:
        self.assertEqual(ssot_loader.get_operator_symbol("NOT_REAL"), "NOT_REAL")


class ResolveCountryCodeTests(unittest.TestCase):
    _ITEMS: List[Dict[str, Any]] = [
        {
            "canonical": "CZ",
            "synonyms": {"en": ["Czechia", "Czech Republic"], "cs": ["Česko"]},
        },
        {
            "canonical": "ES",
            "synonyms": {"en": ["Spain", "Espana", "España"], "cs": ["Španělsko"]},
        },
    ]

    def setUp(self) -> None:
        ssot_loader._canonical_lookup.cache_clear()
        self._patcher = mock.patch.object(ssot_loader, "_load_yaml", return_value=self._ITEMS)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        ssot_loader._canonical_lookup.cache_clear()

    def test_resolves_bare_iso_code(self) -> None:
        self.assertEqual(ssot_loader.resolve_country_code("cz"), "CZ")

    def test_resolves_english_synonym(self) -> None:
        self.assertEqual(ssot_loader.resolve_country_code("Czech Republic"), "CZ")
        self.assertEqual(ssot_loader.resolve_country_code("Czechia"), "CZ")

    def test_resolves_diacritic_variant(self) -> None:
        self.assertEqual(ssot_loader.resolve_country_code("España"), "ES")
        self.assertEqual(ssot_loader.resolve_country_code("Espana"), "ES")

    def test_unknown_country_returns_none(self) -> None:
        self.assertIsNone(ssot_loader.resolve_country_code("Narnia"))


if __name__ == "__main__":
    unittest.main()
