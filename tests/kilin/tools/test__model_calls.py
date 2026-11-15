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
        self.assertIsNone(ModelCallSettings().thinking_override())
        self.assertEqual(list(ModelCallSettings.__dataclass_fields__),
                         ["thinking"])

    def test_removed_short_override_fields_are_rejected(self) -> None:
        for field in ("short_reply_thinking", "short_reply_max_tokens"):
            with self.subTest(field=field), self.assertRaises(TypeError):
                ModelCallSettings(**{field: "off"})

    def test_forced_policy_is_explicit(self) -> None:
        for forced, expected in [("off", False), ("on", True)]:
            self.assertIs(ModelCallSettings(forced).thinking_override(),
                          expected)

    def test_inherit_resets_a_forced_policy(self) -> None:
        for value in ("inherit", "off", "on", "inherit"):
            result = ModelCallSettings(value).thinking_override()
            self.assertIs(
                result, None if value == "inherit" else value == "on"
            )

    def test_invalid_values_are_rejected(self) -> None:
        for value in (True, False, None, "auto", "ON", 1, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModelCallSettings(thinking=value)

    def test_json_validation_and_cache(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "defaults.json"
            path.write_text(json.dumps({"MODEL_CALLS": {
                "THINKING": "inherit",
            }}), encoding="utf-8")
            policy = load_model_call_settings(path)
            self.assertEqual(policy, ModelCallSettings("inherit"))
            self.assertIs(load_model_call_settings(path), policy)
            # Existing defaults are deliberately cached until restart.
            path.write_text("not json", encoding="utf-8")
            self.assertIs(load_model_call_settings(path), policy)

    def test_json_rejects_missing_unknown_and_wrong_types(self) -> None:
        valid = {
            "THINKING": "inherit",
        }
        bad_sections = [
            {}, {**valid, "THINKNG": "on"}, {**valid, "THINKING": True},
            {**valid, "SHORT_REPLY_MAX_TOKENS": 512},
            {**valid, "SHORT_REPLY_THINKING": "inherit"}, [],
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
        off = {
            "batch_classify", "batch_review", "filter_results",
            "find_passages", "generate_paper_summary", "writing_check",
        }
        self.assertEqual(len(TOOL_REGISTRY), 13)
        for name, tool in TOOL_REGISTRY.items():
            with self.subTest(tool=name):
                config = importlib.import_module(
                    f"kilin.tools.{name}._configuration"
                )
                self.assertEqual(tool.model_calls, config.MODEL_CALLS)
                expected = (
                    ModelCallSettings("off") if name in off else
                    ModelCallSettings()
                )
                self.assertEqual(tool.model_calls, expected)


if __name__ == "__main__":
    unittest.main()
