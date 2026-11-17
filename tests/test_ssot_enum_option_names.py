import unittest

from src.executors.planning.ssot_metric_defaults import get_enum_labels, get_enum_option_key
from src.shared import ssot_loader


class EnumOptionNameTests(unittest.TestCase):
    def test_options_use_their_ssot_names(self) -> None:
        # SSOT gives option names per language; they were read as missing and
        # replaced by title-cased keys ("Ct Cta").
        self.assertEqual(get_enum_labels("IMAGING_TYPE")[:3], ["ct", "ct with cta", "ct with cta and perfusion"])
        self.assertEqual(get_enum_option_key("IMAGING_TYPE", "CT with CTA"), "ct_cta")

    def test_loading_leaves_the_ssot_items_unchanged(self) -> None:
        ssot_loader.get_metric_metadata()
        items = ssot_loader.get_ssot_items("MetricType.yml")
        risk_factors = next(item for item in items if item.get("canonical") == "RISK_FACTORS_TYPE")
        self.assertTrue(all(isinstance(entry.get("synonyms"), dict) for entry in risk_factors["Enum"]))


if __name__ == "__main__":
    unittest.main()
