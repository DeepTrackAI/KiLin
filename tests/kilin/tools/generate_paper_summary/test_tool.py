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

from kilin.chunking import Chunk
from kilin.project_type import project_type
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_GENERATE_PAPER_SUMMARY, TOOL_REGISTRY
from kilin.tools._shared.references import ReferenceLayout
from kilin.tools.generate_paper_summary import (
    REFERENCE_LAYOUT,
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
from kilin.tools.generate_paper_summary.tool import choose_paper
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
        self.assertFalse(hasattr(settings, "note_tokens"))

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
                "choose",
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
                "template", "part_tokens",
                "repair_attempts", "max_model_calls",
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
        self.assertEqual(config.HEADLINE_TOKENS, 128)
        self.assertEqual(config.REFERENCES_TOKENS, 128)
        self.assertEqual(config.LIMIT_TOLERANCE, 0.1)
        self.assertEqual(config.MIN_TOTAL_WORDS, 50)
        self.assertEqual(config.PROMPT_TOKENS_PER_SECOND, 700.0)
        self.assertEqual(config.REPLY_TOKENS_PER_SECOND, 25.0)
        self.assertEqual(config.REPLY_SHARE, 0.6)
        self.assertIn("consortium", config.ORGANISATION_WORDS)

    def test_reference_layout_comes_from_tool_defaults(self) -> None:
        """Pass this tool's thresholds to the shared extractor.

        """

        self.assertEqual(
            REFERENCE_LAYOUT,
            ReferenceLayout(
                max_characters=config.REFERENCE_CHARACTERS,
                numbered_entries_min=config.NUMBERED_ENTRIES_MIN,
                running_line_repeats=config.RUNNING_LINE_REPEATS,
                running_line_words=config.RUNNING_LINE_WORDS,
                entry_min_words=config.ENTRY_MIN_WORDS,
            ),
        )
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


class TestChoosePaper(unittest.TestCase):
    """Choose saved outputs only when the question names them exactly."""

    def setUp(self) -> None:
        """Prepare an indexed source and an unrelated saved report.

        """

        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "outputs").mkdir()
        (self.root / "outputs" / "smith20_reference_check.md").write_text(
            "# Reference check: smith20.pdf\n\nThree references.\n",
            encoding="utf-8",
        )
        source = Chunk("s1", Path("smith20.pdf"), "Sleep study.", 1, 2,
                       "pdf_page", None, "Page 1")
        self.context = SimpleNamespace(
            project_root=self.root,
            chunks=[source],
            retrieval_count=8,
            choose=None,
        )
        chosen = patch(
            "kilin.tools.generate_paper_summary.tool.choose_document",
            return_value=(Path("smith20.pdf"), "named in the question",
                          False),
        )
        self.choose_document = chosen.start()
        self.addCleanup(chosen.stop)

    def _choose(self, question: str):
        """Run paper selection with the test's fixed indexed source.

        """

        return choose_paper(ToolRequest(question, question), self.context)

    def test_source_named_by_file_name_is_not_a_saved_report(self) -> None:
        """Never substitute a report sharing the original paper's stem.

        """

        for question in ("Prepare a research briefing for smith20.pdf.",
                         "Summarize my paper smith20.pdf for the public."):
            with self.subTest(question=question):
                path, passages, _, _ = self._choose(question)
                self.assertEqual(path, Path("smith20.pdf"))
                self.assertEqual(len(passages), 1)

    def test_output_named_by_file_name_is_used(self) -> None:
        """Select a report only when its complete name is mentioned.

        """

        path, _, reason, guessed = self._choose(
            "Write a briefing of smith20_reference_check.md."
        )
        self.assertEqual(path.name, "smith20_reference_check.md")
        self.assertEqual((reason, guessed), ("named in the question", False))
        self.choose_document.assert_not_called()

    def test_my_paper_uses_the_only_output_but_this_paper_does_not(
        self,
    ) -> None:
        """Keep the limited my-paper fallback and source precedence.

        """

        for question in ("Summarize my paper.", "Summarize outputs."):
            with self.subTest(question=question):
                path, _, reason, _ = self._choose(question)
                self.assertEqual(path.name, "smith20_reference_check.md")
                self.assertEqual(reason, "the only outputs file")
        path, _, _, _ = self._choose("Summarize this paper.")
        self.assertEqual(path, Path("smith20.pdf"))

    def test_nearby_file_names_do_not_select_a_report(self) -> None:
        """Reject prefixed names, backup extensions and other folders.

        """

        for name in (
            "draft_smith20_reference_check.md",
            "smith20_reference_check.md.bak",
            "smith20_reference_check.md2",
            "other/smith20_reference_check.md",
        ):
            with self.subTest(name=name):
                self.choose_document.reset_mock()
                path, _, _, _ = self._choose(f"Summarize {name}.")
                self.assertEqual(path, Path("smith20.pdf"))
                self.choose_document.assert_called_once()

    def test_my_paper_does_not_bypass_exact_name_matching(self) -> None:
        """Do not turn a mistaken output filename into a sole-file match.

        """

        for name in (
            "draft_smith20_reference_check.md",
            "smith20_reference_check.md.bak",
            "smith20_reference_check.md2",
        ):
            with self.subTest(name=name):
                self.choose_document.reset_mock()
                path, _, _, _ = self._choose(
                    f"Summarize my paper {name}."
                )
                self.assertEqual(path, Path("smith20.pdf"))
                self.choose_document.assert_called_once()

    def test_my_paper_does_not_use_loose_stems_with_many_outputs(
        self,
    ) -> None:
        """Require a complete name instead of the shared stem heuristic.

        """

        (self.root / "outputs" / "other_report.md").write_text(
            "# Other\n\nAnother output.\n", encoding="utf-8"
        )
        path, _, _, _ = self._choose(
            "Summarize my paper smith20_reference_check."
        )
        self.assertEqual(path, Path("smith20.pdf"))
        self.choose_document.assert_called_once()

    def test_output_name_is_case_insensitive_and_can_end_a_sentence(
        self,
    ) -> None:
        """Allow explicit filenames in ordinary sentence punctuation.

        """

        path, _, _, _ = self._choose(
            "Summarize SMITH20_REFERENCE_CHECK.MD."
        )
        self.assertEqual(path, Path("outputs/smith20_reference_check.md"))
        self.choose_document.assert_not_called()

    def test_same_basename_requires_a_relative_path(self) -> None:
        """Do not select arbitrarily between identically named reports.

        """

        for folder in ("drafts", "completed"):
            subfolder = self.root / "outputs" / folder
            subfolder.mkdir()
            (subfolder / "report.md").write_text(
                "# Report\n\nSeparate text for the test.\n",
                encoding="utf-8",
            )

        answer = self._choose("Summarize report.md.")
        self.assertIsInstance(answer, str)
        self.assertIn("relative path", answer)
        self.choose_document.assert_not_called()

        for name in ("drafts/report.md", "outputs/completed/report.md"):
            with self.subTest(name=name):
                path, _, reason, guessed = self._choose(
                    f"Summarize {name}."
                )
                expected = Path("outputs") / name.removeprefix("outputs/")
                self.assertEqual(path, expected)
                self.assertEqual(
                    (reason, guessed), ("named in the question", False)
                )
        self.choose_document.assert_not_called()

    def test_qualified_path_disambiguates_root_and_nested_names(self) -> None:
        """Select the named output, not another with the same basename.

        """

        subfolder = self.root / "outputs" / "archive"
        subfolder.mkdir()
        (subfolder / "smith20_reference_check.md").write_text(
            "# Archive\n\nA second report.\n", encoding="utf-8"
        )
        self.assertIn("relative path", self._choose(
            "Summarize smith20_reference_check.md."
        ))
        for name, expected in (
            ("outputs/smith20_reference_check.md",
             Path("outputs/smith20_reference_check.md")),
            ("archive/smith20_reference_check.md",
             Path("outputs/archive/smith20_reference_check.md")),
        ):
            with self.subTest(name=name):
                path, _, _, _ = self._choose(f"Summarize {name}.")
                self.assertEqual(path, expected)

    def test_multiple_output_files_in_question_are_ambiguous(self) -> None:
        """Require one file even when a full path identifies one of two.

        """

        (self.root / "outputs" / "second.md").write_text(
            "# Second\n\nThe second report.\n", encoding="utf-8"
        )
        answer = self._choose(
            "Summarize outputs/smith20_reference_check.md and second.md."
        )
        self.assertIsInstance(answer, str)
        self.assertIn("Name one saved file", answer)
        self.choose_document.assert_not_called()

    def test_source_selection_uses_the_real_shared_picker(self) -> None:
        """Validate named-source precedence without mocking the picker.

        """

        from kilin.tools._shared.choose_document import choose_document

        with patch(
            "kilin.tools.generate_paper_summary.tool.choose_document",
            choose_document,
        ):
            path, passages, reason, guessed = self._choose(
                "Summarize smith20.pdf."
            )
        self.assertEqual(path, Path("smith20.pdf"))
        self.assertEqual(len(passages), 1)
        self.assertEqual((reason, guessed),
                         ("named in the question", False))

    def test_this_pdf_without_sources_uses_the_only_saved_pdf(self) -> None:
        """Restore the safe PDF fallback in an outputs-only workspace.

        """

        self.context.chunks = []
        (self.root / "outputs" / "paper.pdf").write_bytes(b"placeholder")
        fake = SimpleNamespace(path=Path("outputs/paper.pdf"), passages=())
        with patch(
            "kilin.tools.generate_paper_summary.tool.load_output",
            return_value=fake,
        ) as loader:
            path, passages, reason, guessed = self._choose(
                "Summarize this PDF."
            )
        self.assertEqual(path, Path("outputs/paper.pdf"))
        self.assertEqual(passages, ())
        self.assertEqual((reason, guessed), ("the only outputs PDF", False))
        loader.assert_called_once_with(self.root, Path("paper.pdf"))
        self.choose_document.assert_not_called()

    def test_this_pdf_does_not_override_indexed_sources(self) -> None:
        """Preserve the earlier indexed-paper conversational reference.

        """

        (self.root / "outputs" / "paper.pdf").write_bytes(b"placeholder")
        path, _, _, _ = self._choose("Summarize this PDF.")
        self.assertEqual(path, Path("smith20.pdf"))
        self.choose_document.assert_called_once()

    def test_this_pdf_without_a_saved_pdf_is_not_a_saved_report(
        self,
    ) -> None:
        """Avoid treating a Markdown report as an unnamed PDF.

        """

        from kilin.tools._shared.choose_document import choose_document

        self.context.chunks = []
        with patch(
            "kilin.tools.generate_paper_summary.tool.choose_document",
            choose_document,
        ):
            response = self._choose("Summarize this PDF.")
        self.assertIsInstance(response, str)
        self.assertIn("couldn't tell which file", response)

    def test_this_pdf_with_multiple_saved_pdfs_requests_a_name(
        self,
    ) -> None:
        """Prevent an arbitrary choice among saved PDFs.

        """

        self.context.chunks = []
        for name in ("first.pdf", "second.pdf"):
            (self.root / "outputs" / name).write_bytes(b"placeholder")
        answer = self._choose("Summarize this PDF.")
        self.assertIsInstance(answer, str)
        self.assertIn("Several PDFs", answer)
        self.choose_document.assert_not_called()


if __name__ == "__main__":
    unittest.main()
