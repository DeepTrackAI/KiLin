"""Check Markdown report templates and the deterministic report check."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.tools.report_template import (
    SECTION_SOURCE_CITATION,
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_REFERENCES,
    check_report,
    load_template,
    parse_template,
    template_names,
    templates_folder,
)

_SMALL = """---
title: Lab Summary
max_words: 300
---

# Lab Summary

## Question (max 40 words)
What the paper asks.

## Findings (60–120 words)
What it found, with numbers.

## Read Next (2 items)
From {references}.
"""


class TestTemplateRules(unittest.TestCase):
    """Templates are Markdown; the rules are read, and mistakes named."""

    def test_shipped_briefing(self) -> None:
        self.assertIn("research_briefing", template_names())
        briefing = load_template("research_briefing")
        self.assertEqual(briefing.title, "Research Briefing")
        self.assertEqual(briefing.headline_characters, 90)
        self.assertEqual(briefing.max_total_words, 750)
        self.assertIn("non-specialists", briefing.headline_guidance)
        first, bio, *_, related = briefing.sections
        self.assertEqual(first.name, "This is a summary of")
        self.assertTrue(first.inline)
        self.assertEqual(first.source, SECTION_SOURCE_CITATION)
        self.assertEqual(bio.words, (60, 150))
        self.assertEqual(bio.source, SECTION_SOURCE_NOTES)
        self.assertEqual(related.items, (2, 3))
        self.assertEqual(related.source, SECTION_SOURCE_REFERENCES)
        with self.assertRaisesRegex(ValueError, "research_briefing"):
            load_template("journal_review")

    def test_limits_and_fixed_title(self) -> None:
        template = parse_template(_SMALL, "lab_summary")
        self.assertEqual(template.title, "Lab Summary")
        self.assertIsNone(template.headline_characters)
        question, findings, read = template.sections
        self.assertEqual(question.words, (0, 40))
        self.assertEqual(findings.words, (60, 120))
        self.assertEqual(read.items, (2, 2))
        self.assertEqual(findings.guidance, "What it found, with numbers.")

    def test_mistakes_name_the_line(self) -> None:
        cases = {
            "## A (sixty words)\nx": "line 1: cannot read the limit",
            "## A\nUse {abstract}.": "unknown placeholder {abstract}",
            "## A\nFrom {references}.": "needs an item limit",
            "## A (max 9 characters)\nx": "only the headline",
            "## A\nx\n## A\ny": "appears twice",
            "# Title only\n": "at least one ## section",
            "---\ncolour: red\n---\n## A\n": "line 2: front matter",
            "## A\n# Late headline (max 9 characters)": "before the sections",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, message):
                    parse_template(text, "bad")

    def test_project_folder_starts_with_shipped_templates(self) -> None:
        with TemporaryDirectory() as folder:
            templates = templates_folder(Path(folder))
            self.assertEqual(template_names(templates), template_names())


class TestReportCheck(unittest.TestCase):
    """Python, not the model, decides whether a report follows a template."""

    def setUp(self) -> None:
        self.briefing = load_template("research_briefing")

    def report(self, headline: str = "Plants under stress", **sizes) -> str:
        parts = [f"# {headline}"]
        for section in self.briefing.sections:
            if section.inline:
                parts.append(f"**{section.name}:** Smith, J. et al. A title.")
            elif section.items:
                count = sizes.get("items", 2)
                parts.append(
                    f"## {section.name}\n"
                    + "\n".join(f"- Ref {n}." for n in range(count))
                )
            else:
                low, high = section.words
                words = sizes.get(section.name, (low + high) // 2)
                parts.append(f"## {section.name}\n" + " ".join(["w"] * words))
        return "\n\n".join(parts)

    def test_complete_report_is_valid(self) -> None:
        check = check_report(self.report(), self.briefing)
        self.assertTrue(check.valid, check.problems())

    def test_limits_are_checked(self) -> None:
        long_title = check_report(self.report("x" * 95), self.briefing)
        self.assertIn("95 characters", long_title.headline)
        wordy = check_report(
            self.report(**{"The Reason": 400}), self.briefing
        )
        self.assertIn("The Reason has 400 words", " ".join(wordy.lengths))
        self.assertIn("at most 750", " ".join(wordy.lengths))
        few = check_report(self.report(items=1), self.briefing)
        self.assertIn("lists 1 items", " ".join(few.lengths))
        # Within the 10% tolerance is fine.
        near = check_report(
            self.report(**{"The Future": 108}), self.briefing
        )
        self.assertTrue(near.valid, near.problems())

    def test_structure_problems_are_named(self) -> None:
        names = [s.name for s in self.briefing.sections if not s.inline]
        text = self.report()
        missing = check_report(
            text.replace("## The Future", "## Later"), self.briefing
        )
        self.assertEqual(missing.missing, ("The Future",))
        self.assertEqual(missing.unexpected, ("Later",))
        swapped = text.replace(f"## {names[0]}", "## TEMP").replace(
            f"## {names[1]}", f"## {names[0]}"
        ).replace("## TEMP", f"## {names[1]}")
        self.assertTrue(check_report(swapped, self.briefing).out_of_order)
        self.assertIn(
            "Missing sections:", missing.problems()[0]
        )


if __name__ == "__main__":
    unittest.main()
