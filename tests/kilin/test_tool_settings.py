"""Test listing and changing a project's tool settings.

Use real projects in temporary folders and the packaged project types and
tool defaults; no models are loaded.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import MagicMock

import kilin
from kilin.project_type import project_type
from kilin.tool_settings import (
    project_tool_settings,
    save_tool_settings,
    with_project_settings,
)


class ToolSettingsFixture(unittest.TestCase):

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "project"
        kilin.initialize_project(self.root, "literature")

    def tool(self, name: str) -> dict[str, object]:
        """Return one tool's record from the project's tool list."""

        return next(
            item for item in project_tool_settings(self.root)
            if item["name"] == name
        )


class TestProjectToolSettings(ToolSettingsFixture):

    def test_every_tool_of_the_project_type_is_listed_once(self):
        intents = project_type("literature").intents.values()
        used = list(dict.fromkeys(
            intent.tool.name for intent in intents if intent.tool
        ))
        tools = project_tool_settings(self.root)
        self.assertEqual([tool["name"] for tool in tools], used)
        for tool in tools:
            with self.subTest(tool=tool["name"]):
                self.assertTrue(tool["summary"])
                self.assertTrue(tool["intents"])

    def test_settings_come_from_the_tools_defaults(self):
        """The changeable settings are the defaults' SETTINGS keys."""

        tool = self.tool("find_fact")
        defaults = json.loads(
            (Path(kilin.__file__).parent / "tools" / "find_fact"
             / "defaults.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(tool["settings"]),
            {key.lower() for key in defaults["SETTINGS"]},
        )
        self.assertEqual(tool["settings"], tool["type_settings"])
        self.assertEqual(tool["intents"], ["fact_finding"])

    def test_settings_each_use_sets_differently_are_not_shared(self):
        tool = self.tool("generate_paper_summary")
        self.assertNotIn("template", tool["settings"])
        self.assertEqual(
            tool["per_intent"]["research_briefing"],
            {"template": "research_briefing"},
        )
        # Four fixed templates and templated_report (any template).
        self.assertEqual(len(tool["intents"]), 5)
        self.assertEqual(
            tool["per_intent"]["templated_report"], {"template": "choose"}
        )


class TestSaveToolSettings(ToolSettingsFixture):

    def test_only_changed_values_are_saved_and_shown(self):
        trace = MagicMock()
        tools = save_tool_settings(
            self.root, "find_fact",
            {"part_tokens": 8000, "max_model_calls": 200}, trace=trace,
        )
        tool = next(item for item in tools if item["name"] == "find_fact")
        self.assertEqual(tool["settings"]["part_tokens"], 8000)
        self.assertEqual(tool["type_settings"]["part_tokens"], 6000)
        profile = (self.root / "profile.ini").read_text(encoding="utf-8")
        self.assertIn("[tool.find_fact]\npart_tokens = 8000\n", profile)
        self.assertNotIn("max_model_calls", profile)
        trace.emit.assert_called_once_with(
            "project.tool_settings.saved",
            component="project",
            data={"tool": "find_fact", "changed": {"part_tokens": 8000}},
        )
        # The rest of the profile still opens unchanged.
        self.assertEqual(
            kilin.open_project(self.root).profile.type_id, "literature"
        )

    def test_later_saves_keep_other_changed_values(self):
        save_tool_settings(self.root, "find_fact", {"part_tokens": 8000})
        save_tool_settings(self.root, "find_fact", {"confirmation": "never"})
        self.assertEqual(self.tool("find_fact")["settings"], {
            **self.tool("find_fact")["type_settings"],
            "part_tokens": 8000,
            "confirmation": "never",
        })

    def test_the_project_types_values_reset_the_tool(self):
        save_tool_settings(self.root, "find_fact", {"part_tokens": 8000})
        save_tool_settings(
            self.root, "find_fact", self.tool("find_fact")["type_settings"]
        )
        profile = (self.root / "profile.ini").read_text(encoding="utf-8")
        self.assertNotIn("[tool.find_fact]", profile)

    def test_values_are_checked_by_the_tool_before_saving(self):
        before = (self.root / "profile.ini").read_bytes()
        for name, settings, message in (
            ("find_fact", {"part_tokens": 5}, "part_tokens"),
            ("find_fact", {"confirmation": "maybe"}, "confirmation"),
            ("generate_paper_summary", {"template": "x"}, "cannot be"),
            ("unknown_tool", {}, "does not use"),
        ):
            with self.subTest(name=name, settings=settings):
                with self.assertRaisesRegex(ValueError, message):
                    save_tool_settings(self.root, name, settings)
                self.assertEqual(
                    (self.root / "profile.ini").read_bytes(), before
                )

    def test_text_values_with_percent_signs_are_kept(self):
        save_tool_settings(
            self.root, "batch_classify",
            {"categories": ["100% match", "partial"]},
        )
        self.assertEqual(
            self.tool("batch_classify")["settings"]["categories"],
            ["100% match", "partial"],
        )


class TestWithProjectSettings(ToolSettingsFixture):

    def test_saved_values_apply_to_the_intents_tool(self):
        use = project_type("literature").intents["fact_finding"].tool
        self.assertIs(with_project_settings(self.root, use), use)
        save_tool_settings(self.root, "find_fact", {"part_tokens": 8000})
        changed = with_project_settings(self.root, use)
        self.assertEqual(changed.settings.part_tokens, 8000)
        self.assertEqual(
            changed.settings.max_model_calls, use.settings.max_model_calls
        )

    def test_suggest_reference_discovery_override_is_project_owned(self):
        use = project_type("literature").intents["suggest_reference"].tool
        self.assertEqual(use.settings.discovery_strategy, "hybrid")
        tool = self.tool("suggest_reference")
        self.assertEqual(tool["settings"]["discovery_tokens"], 8000)
        saved = save_tool_settings(self.root, "suggest_reference", {
            "discovery_strategy": "source_first",
            "discovery_tokens": 12000,
        })
        configured = next(
            item for item in saved
            if item["name"] == "suggest_reference"
        )
        self.assertEqual(
            configured["settings"]["discovery_strategy"], "source_first"
        )
        changed = with_project_settings(self.root, use)
        self.assertEqual(changed.settings.discovery_tokens, 12000)
        self.assertEqual(changed.settings.files, use.settings.files)
        self.assertEqual(use.settings.discovery_strategy, "hybrid")
        save_tool_settings(self.root, "suggest_reference", {
            "discovery_strategy": "hybrid",
            "discovery_tokens": 8000,
        })
        self.assertIs(with_project_settings(self.root, use), use)

    def test_invalid_saved_values_are_rejected_before_a_run(self):
        use = project_type("literature").intents["fact_finding"].tool
        with (self.root / "profile.ini").open("a", encoding="utf-8") as file:
            file.write("\n[tool.find_fact]\npart_tokens = 5\n")
        with self.assertRaisesRegex(ValueError, "part_tokens"):
            with_project_settings(self.root, use)
        with (self.root / "profile.ini").open("a", encoding="utf-8") as file:
            file.write("confirmation = not json\n")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            with_project_settings(self.root, use)


if __name__ == "__main__":
    unittest.main()
