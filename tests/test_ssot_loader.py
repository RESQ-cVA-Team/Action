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
                "description": {
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

    def tearDown(self) -> None:
        ssot_loader.get_metric_text_lookup.cache_clear()


if __name__ == "__main__":
    unittest.main()
