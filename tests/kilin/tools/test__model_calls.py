"""Validate tool-owned thinking defaults and deterministic resolution.

The tests use real packaged policies and temporary JSON files. No models,
network services or GUI components are required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.tools import TOOL_REGISTRY
from kilin.tools._model_calls import (
    ModelCallSettings,
    load_model_call_settings,
)


class TestModelCallSettings(unittest.TestCase):
    """Keep inheritance explicit and every override reversible."""

    def test_default_inherits_for_every_budget(self) -> None:
        for tokens in (None, 1, 120, 512, 8192):
            with self.subTest(tokens=tokens):
                self.assertIsNone(
                    ModelCallSettings().thinking_override(tokens)
                )

    def test_short_override_uses_inclusive_requested_bound(self) -> None:
        policy = ModelCallSettings("inherit", "off", 512)
        for tokens, expected in [(1, False), (512, False), (513, None),
                                 (8192, None), (None, None)]:
            with self.subTest(tokens=tokens):
                self.assertIs(policy.thinking_override(tokens), expected)

    def test_forced_policy_wins_over_short_reply_rule(self) -> None:
        for forced, expected in [("off", False), ("on", True)]:
            policy = ModelCallSettings(forced, "off", 512)
            for tokens in (None, 120, 1000):
                with self.subTest(forced=forced, tokens=tokens):
                    self.assertIs(policy.thinking_override(tokens), expected)

    def test_zero_or_inherit_disables_short_override(self) -> None:
        for policy in (ModelCallSettings("inherit", "off", 0),
                       ModelCallSettings("inherit", "inherit", 512)):
            self.assertIsNone(policy.thinking_override(120))
        self.assertTrue(
            ModelCallSettings("inherit", "on", 120).thinking_override(120)
        )

    def test_invalid_values_are_rejected(self) -> None:
        for value in (True, False, None, "auto", "ON", 1, []):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ModelCallSettings(thinking=value)
                with self.assertRaises(ValueError):
                    ModelCallSettings(short_reply_thinking=value)
        for value in (True, -1, 100001, 512.0, "512", None):
            with self.subTest(bound=value):
                with self.assertRaises(ValueError):
                    ModelCallSettings(short_reply_max_tokens=value)
        for value in (True, 0, -1, 1.5, "120"):
            with self.subTest(tokens=value):
                with self.assertRaises(ValueError):
                    ModelCallSettings().thinking_override(value)

    def test_json_validation_and_cache(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "defaults.json"
            path.write_text(json.dumps({"MODEL_CALLS": {
                "THINKING": "inherit", "SHORT_REPLY_THINKING": "off",
                "SHORT_REPLY_MAX_TOKENS": 512,
            }}), encoding="utf-8")
            policy = load_model_call_settings(path)
            self.assertEqual(policy, ModelCallSettings("inherit", "off", 512))
            self.assertIs(load_model_call_settings(path), policy)
            # Existing defaults are deliberately cached until restart.
            path.write_text("not json", encoding="utf-8")
            self.assertIs(load_model_call_settings(path), policy)

    def test_json_rejects_missing_unknown_and_wrong_types(self) -> None:
        valid = {
            "THINKING": "inherit", "SHORT_REPLY_THINKING": "off",
            "SHORT_REPLY_MAX_TOKENS": 512,
        }
        bad_sections = [
            {}, {**valid, "THINKNG": "on"}, {**valid, "THINKING": True},
            {**valid, "SHORT_REPLY_MAX_TOKENS": True},
            {**valid, "SHORT_REPLY_THINKING": "auto"}, [],
        ]
        with TemporaryDirectory() as directory:
            for index, section in enumerate(bad_sections):
                with self.subTest(section=section):
                    path = Path(directory) / f"defaults_{index}.json"
                    path.write_text(json.dumps({"MODEL_CALLS": section}),
                                    encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_model_call_settings(path)
            path = Path(directory) / "missing.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_call_settings(path)

    def test_all_shipped_policies_match_owned_configuration(self) -> None:
        off = {"batch_classify", "find_passages", "filter_results"}
        short = {"batch_review", "find_fact", "generate_paper_summary",
                 "reference_check", "suggest_reference"}
        self.assertEqual(len(TOOL_REGISTRY), 12)
        for name, tool in TOOL_REGISTRY.items():
            with self.subTest(tool=name):
                config = importlib.import_module(
                    f"kilin.tools.{name}._configuration"
                )
                self.assertEqual(tool.model_calls, config.MODEL_CALLS)
                expected = (
                    ModelCallSettings("off") if name in off else
                    ModelCallSettings("inherit", "off", 512)
                    if name in short else ModelCallSettings()
                )
                self.assertEqual(tool.model_calls, expected)


if __name__ == "__main__":
    unittest.main()
