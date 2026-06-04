import os
import unittest
from unittest import mock

from src.util import env


class EnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_loaded = env._loaded
        env._loaded = True

    def tearDown(self) -> None:
        env._loaded = self.original_loaded

    def test_parse_bool_accepts_truthy_and_falsy_values(self) -> None:
        self.assertTrue(env._parse_bool("yes", default=False))
        self.assertFalse(env._parse_bool("no", default=True))
        self.assertTrue(env._parse_bool("maybe", default=True))

    def test_is_production_like_env_detects_common_keys(self) -> None:
        with mock.patch.dict(os.environ, {"NODE_ENV": "production"}, clear=False):
            self.assertTrue(env._is_production_like_env())

    def test_env_flag_uses_default_for_missing_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertTrue(env.env_flag("MISSING_FLAG", default=True))

    def test_require_all_env_raises_when_missing(self) -> None:
        with mock.patch.dict(os.environ, {"ACTION_ONE": "one"}, clear=False):
            with self.assertRaises(OSError):
                env.require_all_env("ACTION_ONE", "ACTION_TWO")

    def test_require_any_env_requires_at_least_one_value(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            with self.assertRaises(OSError):
                env.require_any_env("ACTION_ONE", "ACTION_TWO")


if __name__ == "__main__":
    unittest.main()