"""Verify descriptive setting metadata without initializing models or Qt.

The catalog is user-interface independent. These tests check schema
validation, canonical names, cache isolation, package resources and coverage
of every public project profile, intent and registered tool setting.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import kilin
from kilin.project import ProjectProfile
from kilin.project_behavior import (
    _CAPABILITY_FIELDS,
    _GLOBAL_FIELDS,
    _INTENT_FIELDS,
)
from kilin.settings_info import (
    _load_resource,
    get_setting_info,
    get_settings_info,
    get_tool_settings_info,
)
from kilin.tools._registry import TOOL_CHOICES


class TestSettingsInfo(unittest.TestCase):
    """Check an independent core catalog and registered tool resources."""

    def test_public_api_has_descriptive_only_fields(self):
        catalog = kilin.get_settings_info()
        self.assertEqual(catalog["format"], "settings-info-v1")
        self.assertGreater(len(catalog["settings"]), 100)
        for key, value in catalog["settings"].items():
            with self.subTest(key=key):
                self.assertEqual(set(value), {
                    "title", "summary", "description",
                })
                self.assertTrue(all(
                    isinstance(text, str) and text.strip()
                    for text in value.values()
                ))
        self.assertEqual(
            kilin.get_setting_info("project.profile.temperature")[
                "title"
            ], "Temperature",
        )
        self.assertIsNone(kilin.get_setting_info("not.a.setting"))
        with self.assertRaises(TypeError):
            get_setting_info(None)

    def test_defensive_copies_and_cache_isolation(self):
        catalog = get_settings_info()
        catalog["settings"]["project.profile.temperature"][
            "summary"
        ] = "Incorrect"
        self.assertNotEqual(
            get_setting_info("project.profile.temperature")["summary"],
            "Incorrect",
        )
        tool = get_tool_settings_info("reference_check")
        tool["max_online"]["title"] = "Changed"
        self.assertNotEqual(
            get_tool_settings_info("reference_check")["max_online"][
                "title"
            ], "Changed",
        )
        with self.assertRaisesRegex(ValueError, "Unknown KíLín tool"):
            get_tool_settings_info("some_unknown_tool")

    def test_every_project_profile_setting_has_metadata(self):
        keys = set(get_settings_info()["settings"])
        missing = {
            f"project.profile.{field.name}"
            for field in fields(ProjectProfile)
        } - keys
        self.assertEqual(missing, set())

    def test_all_editable_behavior_fields_have_metadata(self):
        keys = set(get_settings_info()["settings"])
        expected = {
            "project_behavior.global." + key
            for key in _GLOBAL_FIELDS - {"history"}
        } | {
            "project_behavior.global.history.recent_turns"
        } | {
            "project_behavior.intents." + key
            for key in _INTENT_FIELDS - {"history", "retrieval"}
        } | {
            "project_behavior.intents.history.policy"
        } | {
            "project_behavior.capabilities." + key
            for key in _CAPABILITY_FIELDS
        }
        from kilin.project_behavior import _RETRIEVAL_FIELDS

        expected |= {
            "project_behavior.intents.retrieval." + key
            for key in _RETRIEVAL_FIELDS
        }
        self.assertFalse(expected - keys)

    def test_every_tool_declared_setting_is_described(self):
        from importlib.resources import files

        catalog = get_settings_info()["settings"]
        for name in TOOL_CHOICES:
            with self.subTest(tool=name):
                root = files(f"kilin.tools.{name}")
                data = json.loads(root.joinpath("defaults.json")
                                  .read_text(encoding="utf-8"))
                expected = {key.lower() for key in data["SETTINGS"]}
                actual = set(get_tool_settings_info(name))
                self.assertEqual(actual, expected)
                self.assertEqual({
                    f"tools.{name}.{key}" for key in expected
                } - set(catalog), set())

    def test_core_model_and_source_card_settings_are_described(self):
        from kilin.model_settings import SourceCardGenerationSettings
        from kilin.source_card_policy import (
            SourceCardOverride, SourceCardPolicy,
        )

        keys = set(get_settings_info()["settings"])
        for prefix, klass in (
            ("source_cards.policy", SourceCardPolicy),
            ("source_cards.source", SourceCardOverride),
            ("source_cards.generation", SourceCardGenerationSettings),
        ):
            for field in fields(klass):
                self.assertIn(f"{prefix}.{field.name}", keys)

    def test_trace_and_per_answer_settings_are_described(self):
        """Keep public execution options independent of GUI coverage."""

        from kilin.trace_logging import TraceConfig

        keys = set(get_settings_info()["settings"])
        for field in fields(TraceConfig):
            self.assertIn(f"trace.{field.name}", keys)
        for field in (
            "retrieval_count", "answer_guidance", "retrieval_query",
            "reuse_evidence", "use_router",
        ):
            self.assertIn(f"answer.{field}", keys)
        for field in ("temperature", "thinking", "max_new_tokens"):
            self.assertIn(f"answer.generation_settings.{field}", keys)

    def test_invalid_json_and_duplicate_keys_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "settings_info.json"
            for raw in (
                "not json",
                '{"format":"settings-info-v1","settings":{},'
                '"settings":{}}',
                '{"format":"unexpected","settings":{"x":{}}}',
                '{"format":"settings-info-v1","settings":{}}',
                '{"format":"settings-info-v1","settings":'
                '{"x":{"title":"Hi","summary":"Yes"}}}',
                '{"format":"settings-info-v1","settings":'
                '{"x":{"title":"Hi","summary":" ",'
                '"description":"Details"}}}',
                '{"format":"settings-info-v1","settings":'
                '{"x":{"title":"Hi","summary":"Yes",'
                '"description":"Details","default":10}}}',
            ):
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    with patch("kilin.settings_info.files", return_value=root):
                        with self.assertRaises(ValueError):
                            _load_resource("test", "settings_info.json")

    def test_invalid_metadata_does_not_affect_core_operations(self):
        with patch("kilin.settings_info._load_resource",
                   side_effect=ValueError("Bad metadata")):
            from kilin.project_type import project_type
            from kilin.tools._registry import TOOL_REGISTRY

            self.assertEqual(project_type("literature").id, "literature")
            self.assertIn("reference_check", TOOL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
