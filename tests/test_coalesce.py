import unittest

from src.util.coalesce import coalesce


class CoalesceTests(unittest.TestCase):
    def test_returns_original_value_when_present(self) -> None:
        self.assertEqual(coalesce("value", "fallback"), "value")

    def test_returns_default_for_none(self) -> None:
        self.assertEqual(coalesce(None, "fallback"), "fallback")

    def test_returns_none_when_both_inputs_are_none(self) -> None:
        self.assertIsNone(coalesce(None))


if __name__ == "__main__":
    unittest.main()