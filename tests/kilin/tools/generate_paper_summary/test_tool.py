"""Test the packaged paper-summary tool without language models.

Verify per-tool defaults, project-type overrides, report parsing,
workspace resource declarations and predictable failure paths.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.project_type import project_type
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_GENERATE_PAPER_SUMMARY, TOOL_REGISTRY
from kilin.tools.generate_paper_summary import (
    SHIPPED_TEMPLATES_DIR,
    SummarySettings,
    check_report,
    choose_template,
    load_template,
    parse_settings,
    parse_template,
    run,
    template_names,
    workspace_resources,
)
from kilin.tools.generate_paper_summary import _configuration as config
from kilin.workspace_resources import templates_folder


class TestGeneratePaperSummary(unittest.TestCase):
    """Verify one independently configured tool package."""

    def setUp(self) -> None:
        """Create an empty project root for isolated tool tests.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = templates_folder(self.root)
        self.folder.mkdir()

    def test_tool_registry_points_to_package(self) -> None:
        """Resolve parser, runner and resource declarations by identity.

        """

        tool = TOOL_REGISTRY[TOOL_GENERATE_PAPER_SUMMARY]
        self.assertIs(tool.parse_settings, parse_settings)
        self.assertIs(tool.run, run)
        self.assertIs(tool.workspace_resources, workspace_resources)

    def test_packaged_settings_defaults(self) -> None:
        """Preserve every existing default value after migration.

        """

        defaults = SummarySettings()
        self.assertEqual(defaults, parse_settings({}, "example"))
        self.assertEqual(
            defaults,
            SummarySettings(
                template="research_briefing",
                part_tokens=6000,
                note_tokens=500,
                section_tokens=600,
                repair_attempts=1,
                max_model_calls=20,
                confirmation="when_large",
                large_batches=3,
            ),
        )

    def test_project_type_overrides_tool_defaults(self) -> None:
        """Allow project settings to replace a subset of tool defaults.

        """

        changes = {
            "template": "technical_summary",
            "part_tokens": 8000,
            "repair_attempts": 2,
            "confirmation": "always",
        }
        settings = parse_settings(changes, "literature.custom")
        self.assertEqual(settings.template, "technical_summary")
        self.assertEqual(settings.part_tokens, 8000)
        self.assertEqual(settings.repair_attempts, 2)
        self.assertEqual(settings.confirmation, "always")
        self.assertEqual(settings.note_tokens, SummarySettings().note_tokens)

        names = [
            intent.tool.settings.template
            for intent in project_type("literature").intents.values()
            if intent.tool is not None
            and intent.tool.name == TOOL_GENERATE_PAPER_SUMMARY
        ]
        self.assertEqual(
            set(names),
            {
                "research_briefing",
                "popular_science_summary",
                "technical_summary",
                "experimental_protocol",
            },
        )

    def test_bad_project_settings(self) -> None:
        """Reject traversal, invalid values, policies and unknown keys.

        """

        for changes in (
            {"template": "../foo"},
            {"template": "a/b"},
            {"template": ""},
            {"note_tokens": True},
            {"part_tokens": 100},
            {"repair_attempts": -1},
            {"confirmation": "sometimes"},
            {"unknown": "value"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    parse_settings(changes, "example")

    def test_complete_packaged_defaults_have_documentation(self) -> None:
        """Require a documentation entry for each configured tool key.

        """

        folder = SHIPPED_TEMPLATES_DIR.parent
        data = json.loads(
            (folder / "defaults.json").read_text(encoding="utf-8")
        )
        readme = (folder / "README.md").read_text(encoding="utf-8")
        for name, group in data.items():
            with self.subTest(section=name):
                self.assertIn(f"`{name}`", readme)
            for key in group:
                with self.subTest(key=f"{name}.{key}"):
                    self.assertIn(f"`{key}`", readme)
        self.assertEqual(
            {field.name for field in fields(SummarySettings)},
            {
                "template", "part_tokens", "note_tokens",
                "section_tokens", "repair_attempts", "max_model_calls",
                "confirmation", "large_batches",
            },
        )

    def test_existing_numeric_and_language_defaults(self) -> None:
        """Protect the original algorithmic and estimate parameters.

        """

        self.assertEqual(config.BEGINNING_PASSAGES, 3)
        self.assertEqual(config.BEGINNING_CHARACTERS, 1500)
        self.assertEqual(config.REFERENCE_CHARACTERS, 40000)
        self.assertEqual(config.NUMBERED_ENTRIES_MIN, 5)
        self.assertEqual(config.DEFAULT_REFERENCE_ITEMS, (1, 3))
        self.assertEqual(config.EARLIER_SECTION_WORDS, 30)
        self.assertEqual(config.HEADLINE_TOKENS, 60)
        self.assertEqual(config.REFERENCES_TOKENS, 40)
        self.assertEqual(config.LIMIT_TOLERANCE, 0.1)
        self.assertEqual(config.MIN_TOTAL_WORDS, 50)
        self.assertEqual(config.PROMPT_TOKENS_PER_SECOND, 700.0)
        self.assertEqual(config.REPLY_TOKENS_PER_SECOND, 25.0)
        self.assertEqual(config.REPLY_SHARE, 0.6)
        self.assertIn("consortium", config.ORGANISATION_WORDS)
        self.assertIn("paper", config.COMMON_WORDS)

    def test_shipped_resources_belong_to_the_tool_package(self) -> None:
        """Install exactly the configured template without file writes.

        """

        bundled = set(template_names(SHIPPED_TEMPLATES_DIR))
        self.assertEqual(
            bundled,
            {
                "research_briefing", "popular_science_summary",
                "technical_summary", "experimental_protocol",
            },
        )
        self.assertFalse(list(self.folder.iterdir()))
        settings = SummarySettings(template="technical_summary")
        resources = workspace_resources(settings)
        self.assertEqual(len(resources), 1)
        self.assertEqual(
            resources[0].destination, "templates/technical_summary.md"
        )
        self.assertEqual(
            resources[0].source,
            SHIPPED_TEMPLATES_DIR / "technical_summary.md",
        )
        self.assertFalse(list(self.folder.iterdir()))

    def test_custom_template_has_no_packaged_resource(self) -> None:
        """Leave manually supplied formats untouched and uninstalled.

        """

        self.assertEqual(
            workspace_resources(SummarySettings(template="my_format")),
            (),
        )

    def test_template_is_read_from_the_project_not_the_package(self) -> None:
        """Honor manual template removal and edits without fallback.

        """

        name = "research_briefing"
        path = self.folder / f"{name}.md"
        path.write_bytes((SHIPPED_TEMPLATES_DIR / path.name).read_bytes())
        original = choose_template("summarize it", self.folder, name)
        self.assertEqual(original.name, name)
        self.assertEqual(original.title, "Research Briefing")

        path.unlink()
        with self.assertRaisesRegex(ValueError, "Unknown report template"):
            choose_template("summarize it", self.folder, name)

    def test_explicitly_named_template_takes_precedence(self) -> None:
        """Choose a custom project-owned template by its title or name.

        """

        format_text = "# Sample\n\n## Findings (max 100 words)\nWrite.\n"
        (self.folder / "sample.md").write_text(format_text, encoding="utf-8")
        chosen = choose_template("Make a sample report", self.folder,
                                 "research_briefing")
        self.assertEqual(chosen.name, "sample")
        self.assertEqual(chosen.sections[0].name, "Findings")

    def test_template_parser_and_report_checks(self) -> None:
        """Check deterministic template structure without model calls.

        """

        template = parse_template(
            "# Example\n\n## Summary (2-4 words)\nWrite a summary.\n",
            "example",
        )
        self.assertFalse(check_report("# Example\n", template).valid)
        report = "# Example\n\n## Summary\nTwo clear words.\n"
        self.assertTrue(check_report(report, template).valid)
        self.assertFalse(
            check_report("# Example\n\n## Summary\nOne.\n", template)
            .valid
        )
        with self.assertRaises(ValueError):
            load_template("../secret", SHIPPED_TEMPLATES_DIR)

    def test_templates_need_an_explicit_directory(self) -> None:
        """Avoid accidental fallback to packaged templates at runtime.

        """

        with self.assertRaises(TypeError):
            load_template("research_briefing")
        with self.assertRaises(TypeError):
            template_names()
        self.assertEqual(template_names(self.folder), [])
        with self.assertRaisesRegex(ValueError, "Unknown report template"):
            load_template("research_briefing", self.folder)

    def test_linked_template_is_not_read_or_listed(self) -> None:
        """Reject links from project templates to files outside the folder.

        """

        source = SHIPPED_TEMPLATES_DIR / "research_briefing.md"
        link = self.folder / "foreign.md"
        try:
            link.symlink_to(source)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        self.assertEqual(template_names(self.folder), [])
        with self.assertRaisesRegex(ValueError, "Unknown report template"):
            load_template("foreign", self.folder)

    def test_linked_template_directory_is_rejected(self) -> None:
        """Do not follow symbolic links in the project template folder.

        """

        linked = self.root / "linked"
        try:
            linked.symlink_to(
                SHIPPED_TEMPLATES_DIR, target_is_directory=True
            )
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        with self.assertRaisesRegex(ValueError, "folder cannot be a link"):
            template_names(linked)
        with self.assertRaisesRegex(ValueError, "folder cannot be a link"):
            load_template("research_briefing", linked)

    def test_run_missing_template_stops_before_models(self) -> None:
        """Return a helpful error without selecting papers or models.

        """

        done = Mock()
        trace = SimpleNamespace(emit=Mock())
        context = SimpleNamespace(
            project_root=self.root, done=done, trace=trace,
        )
        request = ToolRequest("Make a paper summary", "Make a paper summary")
        with patch(
            "kilin.tools.generate_paper_summary.tool.choose_paper"
        ) as choose_paper:
            result = run(SummarySettings(), request, context)
        choose_paper.assert_not_called()
        done.assert_called_once()
        self.assertEqual(result.evidence, ())
        self.assertIn("Unknown report template", result.answer)
        trace.emit.assert_not_called()

    def test_run_when_paper_is_not_found(self) -> None:
        """Read selected project template but stop if paper unavailable.

        """

        source = SHIPPED_TEMPLATES_DIR / "research_briefing.md"
        (self.folder / source.name).write_bytes(source.read_bytes())
        done = Mock()
        context = SimpleNamespace(
            project_root=self.root, done=done, progress=Mock(),
            trace=SimpleNamespace(emit=Mock()),
        )
        request = ToolRequest("Summarize that paper", "Summarize that paper")
        with patch(
            "kilin.tools.generate_paper_summary.tool.choose_paper",
            return_value="No matching paper",
        ) as choose_paper:
            result = run(SummarySettings(), request, context)
        choose_paper.assert_called_once()
        self.assertEqual(result.answer, "No matching paper")
        self.assertEqual(result.evidence, ())
        done.assert_called_once()


if __name__ == "__main__":
    unittest.main()
