"""Test using any report template of a project, by name or by choice.

Check the format guide installed in ``templates/``, the routing built from
a project's templates, choosing a template when the question names none,
and listing and checking templates.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.answering import _cued_intent
from kilin.project_type import project_type, with_project_routing
from kilin.tools._contract import ToolRequest
from kilin.tools.generate_paper_summary import (
    SHIPPED_TEMPLATES_DIR,
    TEMPLATE_CHOOSE,
    SummarySettings,
    choose_template,
    parse_template,
    routing,
    run,
    template_names,
    templates_answer,
    workspace_resources,
)

JOURNAL_CLUB = """---
title: Journal Club Handout
description: One-page handout for a lab journal club discussion.
max_words: 500
---

# Headline (max 90 characters)

**Source paper:** {citation}

## The Question (60-90 words)

What the paper asks.

## Discussion Points (3-5 items)

Points to discuss.
"""
MINUTES = """---
title: Minutes
description: Short notes.
max_words: 300
---

## Notes (50-80 words)

Notes.
"""
LAB_OVERVIEW = """---
title: Lab Overview
description: What all our papers say together.
max_words: 300
scope: project
---

## Main Themes (40-60 words)

The themes.
"""
BROKEN = """---
title: Broken One
description: A template with an unknown placeholder.
max_words: 300
---

## Notes (50-80 words)

Uses {author}.
"""
LAB = """---
title: Lab
description: Short lab report.
max_words: 300
---

## Notes (50-80 words)

Notes.
"""
LAB_NOTES = """---
title: Lab Notes
description: Longer lab report.
max_words: 300
---

## Notes (50-80 words)

Notes.
"""


class _Project(unittest.TestCase):
    """A project with the shipped templates and some of its own."""

    def setUp(self) -> None:
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.templates = self.root / "templates"
        self.templates.mkdir()
        for source in SHIPPED_TEMPLATES_DIR.glob("*.md"):
            shutil.copy(source, self.templates / source.name)
        (self.templates / "journal_club.md").write_text(JOURNAL_CLUB)
        (self.templates / "minutes.md").write_text(MINUTES)
        # A project report, which this tool leaves alone.
        (self.templates / "lab_overview.md").write_text(LAB_OVERVIEW)

    def tearDown(self) -> None:
        self.folder.cleanup()


class TestGuide(unittest.TestCase):

    def test_readme_example_is_a_valid_template(self) -> None:
        guide = (SHIPPED_TEMPLATES_DIR / "README.md").read_text()
        example = re.search(r"```markdown\n(.*?)```", guide, re.S).group(1)
        template = parse_template(example, "journal_club")
        self.assertEqual(template.title, "Journal Club Handout")
        self.assertEqual(
            [section.name for section in template.sections],
            ["Source paper", "In one sentence", "The Question",
             "What They Did", "Discussion Points", "Further Reading"],
        )

    def test_readme_is_not_a_template_and_is_installed(self) -> None:
        self.assertNotIn("README", template_names(SHIPPED_TEMPLATES_DIR))
        shipped = {
            resource.destination
            for resource in workspace_resources(
                SummarySettings(template=TEMPLATE_CHOOSE)
            )
        }
        self.assertIn("templates/README.md", shipped)
        self.assertIn("templates/research_briefing.md", shipped)
        self.assertEqual(len(shipped), 5)
        # A fixed template installs only itself.
        self.assertEqual(
            [r.destination for r in workspace_resources(SummarySettings())],
            ["templates/research_briefing.md"],
        )


class TestRouting(_Project):

    def test_routing_from_the_project_templates(self) -> None:
        self.assertIsNone(routing(SummarySettings(), self.root))
        found = routing(
            SummarySettings(template=TEMPLATE_CHOOSE), self.root,
            (SummarySettings(template="research_briefing"),),
        )
        self.assertIn("Journal Club Handout (journal_club)", found.description)
        self.assertIn("Minutes (minutes): Short notes;", found.description)
        self.assertIn("not a validated standalone laboratory…;",
                      found.description)
        # Templates with their own intent are left to it.
        self.assertNotIn("Research Briefing", found.description)
        self.assertIn("Technical Summary", found.description)
        self.assertNotIn("Lab Overview", found.description)
        cues = [re.compile(cue, re.I) for cue in found.cues]
        for question, cued in (
            ("write a journal club handout of smith20", True),
            ("journal club for smith20", True),
            ("prepare a Journal_Club of patil25", True),
            ("what did we discuss at journal club?", False),
            ("write a research briefing for patil25", False),
            # One-word names are routed only with "template".
            ("write the minutes of smith20", False),
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    any(cue.search(question) for cue in cues), cued
                )

    def test_project_types(self) -> None:
        literature = with_project_routing(project_type("literature"),
                                          self.root)
        writing = with_project_routing(project_type("writing"), self.root)
        for question, in_literature, in_writing in (
            ("write a journal club handout of smith20", "templated_report",
             "templated_report"),
            ("summarize the Smith paper using my journal club template",
             "templated_report", "templated_report"),
            ("use the minutes template for smith20", "templated_report",
             "templated_report"),
            ("which templates do I have?", "templated_report",
             "templated_report"),
            ("check my journal_club template", "templated_report",
             "templated_report"),
            # The fixed intents keep their templates in Literature.
            ("write a research briefing for patil25", None,
             "templated_report"),
            ("what did we discuss at journal club?", None, None),
            ("summarize the Smith paper", "paper_overview", None),
            ("rewrite this abstract", None, "writing"),
        ):
            with self.subTest(question=question):
                self.assertEqual(_cued_intent(literature, question),
                                 in_literature)
                self.assertEqual(_cued_intent(writing, question), in_writing)
        # Without templates, the project type is unchanged.
        empty = Path(self.folder.name) / "empty"
        empty.mkdir()
        literature = project_type("literature")
        self.assertIs(with_project_routing(literature, empty), literature)


class TestChoosing(_Project):

    def test_named_one_or_chosen(self) -> None:
        self.assertEqual(
            choose_template("write a journal club handout of x",
                            self.templates, TEMPLATE_CHOOSE).name,
            "journal_club",
        )
        choose = Mock(return_value=1)
        chosen = choose_template("make a report of smith20",
                                 self.templates, TEMPLATE_CHOOSE, choose)
        options = choose.call_args.args[1]
        self.assertEqual(chosen.name, "journal_club")
        self.assertIn("Journal Club Handout (journal_club)", options)
        self.assertEqual(len(options), 6)
        choose = Mock(return_value=None)
        self.assertIsNone(choose_template(
            "make a report of smith20", self.templates, TEMPLATE_CHOOSE,
            choose,
        ))
        # Nobody to ask: the answer says how to name one.
        with self.assertRaisesRegex(ValueError, "Which template"):
            choose_template("make a report of smith20", self.templates,
                            TEMPLATE_CHOOSE)
        # A project template is not a paper report.
        self.assertIsNone(choose_template(
            "write a lab overview of smith20", self.templates,
            TEMPLATE_CHOOSE, Mock(return_value=None),
        ))
        # A fixed default is used without asking.
        self.assertEqual(
            choose_template("make a report", self.templates,
                            "minutes").name, "minutes",
        )

    def test_only_template_is_used(self) -> None:
        for path in self.templates.glob("*.md"):
            if path.name != "minutes.md":
                path.unlink()
        (self.templates / "broken.md").write_text(BROKEN)
        self.assertEqual(
            choose_template("make a report", self.templates,
                            TEMPLATE_CHOOSE, Mock()).name,
            "minutes",
        )
        (self.templates / "minutes.md").unlink()
        with self.assertRaisesRegex(ValueError, "no usable report template"):
            choose_template("make a report", self.templates, TEMPLATE_CHOOSE)

    def test_overlapping_names_and_boundaries(self) -> None:
        (self.templates / "lab.md").write_text(LAB)
        (self.templates / "lab_notes.md").write_text(LAB_NOTES)
        for question, expected in (
            ("use lab_notes for smith20", "lab_notes"),
            ("use lab notes for smith20", "lab_notes"),
            ("use LAB-NOTES.MD for smith20", "lab_notes"),
            ("use templates/lab_notes.md for smith20", "lab_notes"),
            ("prepare a Lab Notes report", "lab_notes"),
            ("use lab.md for smith20", "lab"),
            ("use lab for smith20", "lab"),
            ("make a collaboration report", "minutes"),
        ):
            with self.subTest(question=question):
                chosen = choose_template(
                    question, self.templates, "minutes"
                )
                self.assertEqual(chosen.name, expected)

    def test_ambiguous_names_do_not_silently_choose(self) -> None:
        (self.templates / "lab.md").write_text(LAB)
        (self.templates / "lab_notes.md").write_text(LAB_NOTES)
        question = "use lab_notes and lab for smith20"
        choose = Mock()
        with self.assertRaisesRegex(
            ValueError, "Several report templates match"
        ) as error:
            choose_template(question, self.templates, "minutes", choose)
        choose.assert_not_called()
        self.assertIn("`lab.md`", str(error.exception))
        self.assertIn("`lab_notes.md`", str(error.exception))

    def test_broken_longer_name_is_reported_not_shorter_one(
        self,
    ) -> None:
        (self.templates / "lab.md").write_text(LAB)
        (self.templates / "lab_notes.md").write_text(BROKEN)
        with self.assertRaisesRegex(ValueError, r"lab_notes.*author"):
            choose_template("use lab_notes.md", self.templates, "lab")
        result = templates_answer(self.templates)
        self.assertIn("`lab_notes.md` cannot be used", result)

    def test_callback_indices_are_validated(self) -> None:
        for value in (-1, 999, True, "1", None):
            with self.subTest(value=value):
                chosen = choose_template(
                    "make a report", self.templates, TEMPLATE_CHOOSE,
                    Mock(return_value=value),
                )
                self.assertIsNone(chosen)


class TestListingAndChecking(_Project):

    def _run(self, question: str, choose=None):
        context = SimpleNamespace(
            project_root=self.root, done=Mock(), progress=Mock(),
            trace=SimpleNamespace(emit=Mock()), trace_turn_id="turn-1",
            choose=choose,
        )
        with patch(
            "kilin.tools.generate_paper_summary.tool.choose_paper"
        ) as choose_paper:
            result = run(SummarySettings(template=TEMPLATE_CHOOSE),
                         ToolRequest(question, question), context)
        return result, choose_paper

    def test_list(self) -> None:
        (self.templates / "broken.md").write_text(BROKEN)
        (self.templates / "README.md").unlink()  # An older project.
        answer = templates_answer(self.templates)
        self.assertIn(
            "- **Journal Club Handout** (`journal_club.md`): One-page "
            "handout for a lab journal club discussion.", answer,
        )
        self.assertIn("**Templates with problems**", answer)
        self.assertIn("- `broken.md` cannot be used:", answer)
        # An older project has no guide: the shipped one is named.
        self.assertIn("the guide shipped with KíLín", answer)
        shutil.copy(SHIPPED_TEMPLATES_DIR / "README.md", self.templates)
        answer = templates_answer(self.templates)
        self.assertIn("folder; `templates/README.md` explains", answer)
        self.assertNotIn("Lab Overview", answer)
        result, choose_paper = self._run("which templates do I have?")
        choose_paper.assert_not_called()
        self.assertEqual(result.answer, answer)
        self.assertNotIn("README", answer.split("**Templates with")[0])
        self.assertTrue(result.summary["listed_templates"])

    def test_check(self) -> None:
        (self.templates / "broken.md").write_text(BROKEN)
        result, choose_paper = self._run("check my journal_club template")
        choose_paper.assert_not_called()
        self.assertIn("**Journal Club Handout** (`journal_club.md`) can be "
                      "used.", result.answer)
        self.assertIn("- The Question (60–90 words)", result.answer)
        self.assertIn("- Discussion Points (3–5 items)", result.answer)
        result, _ = self._run("check the broken template")
        self.assertIn("`broken.md` cannot be used", result.answer)
        self.assertIn("{author}", result.answer)

    def test_cancelled_choice(self) -> None:
        result, choose_paper = self._run("make a report of smith20",
                                         Mock(return_value=None))
        choose_paper.assert_not_called()
        self.assertEqual(result.answer, "No template was chosen.")

    def test_overlapping_names_in_template_check(self) -> None:
        (self.templates / "lab.md").write_text(LAB)
        (self.templates / "lab_notes.md").write_text(LAB_NOTES)
        for question, expected in (
            ("check lab_notes template", "lab_notes.md"),
            ("check lab notes template", "lab_notes.md"),
            ("check lab_notes.md template", "lab_notes.md"),
            ("check lab template", "lab.md"),
        ):
            with self.subTest(question=question):
                result, choose_paper = self._run(question)
                choose_paper.assert_not_called()
                self.assertIn(f"(`{expected}`) can be used", result.answer)
        result, choose_paper = self._run(
            "check lab and lab_notes template"
        )
        choose_paper.assert_not_called()
        self.assertIn("Several report templates match", result.answer)
        self.assertIn("`lab.md`", result.answer)
        self.assertIn("`lab_notes.md`", result.answer)

    def test_choice_is_announced_before_gui_callback(self) -> None:
        seen = []

        def progress(message, **details):
            seen.append(("progress", message, details))

        def choose(question, options):
            self.assertTrue(seen)
            self.assertEqual(seen[0][0], "progress")
            self.assertEqual(seen[0][1], "Waiting for your choice")
            self.assertEqual(
                seen[0][2]["choice"],
                {"question": question, "options": list(options)},
            )
            return 1

        context = SimpleNamespace(
            project_root=self.root,
            done=Mock(),
            progress=progress,
            trace=SimpleNamespace(emit=Mock()),
            trace_turn_id="turn-1",
            choose=choose,
        )
        with patch(
            "kilin.tools.generate_paper_summary.tool.choose_paper"
        ) as choose_paper:
            choose_paper.return_value = "No paper chosen."
            result = run(
                SummarySettings(template=TEMPLATE_CHOOSE),
                ToolRequest("make a report", "make a report"), context,
            )
        self.assertEqual(result.summary["template"], "journal_club")
        self.assertEqual(result.answer, "No paper chosen.")
        self.assertEqual(seen[0][2]["choice"]["options"][1],
                         "Journal Club Handout (journal_club)")
        emitted = [
            call.args[0] for call in context.trace.emit.call_args_list
        ]
        self.assertEqual(emitted,
                         ["tool.choice.proposed", "tool.choice.decided"])
        self.assertEqual(
            context.trace.emit.call_args_list[0].kwargs["turn_id"],
            "turn-1",
        )
        self.assertEqual(
            context.trace.emit.call_args_list[1].kwargs["data"],
            {"index": 1, "asked": True},
        )

    def test_invalid_gui_choice_stops_before_paper_selection(self) -> None:
        for value in (-1, 100, True, "1", None):
            with self.subTest(value=value):
                result, choose_paper = self._run(
                    "make a report of smith20", Mock(return_value=value)
                )
                choose_paper.assert_not_called()
                self.assertEqual(result.answer, "No template was chosen.")

    def test_noninteractive_choice_does_not_guess(self) -> None:
        result, choose_paper = self._run("make a report of smith20")
        choose_paper.assert_not_called()
        self.assertIn("Which template?", result.answer)


if __name__ == "__main__":
    unittest.main()
