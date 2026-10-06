"""Check report templates and the deterministic structure check."""

from __future__ import annotations

import unittest

from qachat.report_templates import (
    check_report,
    load_template,
    report_skeleton,
    template_names,
)


def _report(*names: str, empty: str = "") -> str:
    return "\n".join(
        f"## {name}\n{'' if name == empty else 'Content.'}" for name in names
    )


class TestReportTemplates(unittest.TestCase):
    """Python, not the model, decides whether a report follows a template."""

    def setUp(self) -> None:
        self.template = load_template("research_summary")
        self.names = [section.name for section in self.template.sections]

    def test_packaged_template(self) -> None:
        self.assertIn("research_summary", template_names())
        self.assertEqual(self.names[0], "Citation")
        self.assertEqual(self.names[-1], "Take-Home Message")
        self.assertEqual(len(self.names), 10)
        with self.assertRaisesRegex(ValueError, "research_summary"):
            load_template("journal_review")

    def test_complete_report_is_valid(self) -> None:
        check = check_report(
            "# Research Summary\n" + _report(*self.names), self.template
        )
        self.assertTrue(check.valid, check.problems())

    def test_each_problem_is_named(self) -> None:
        partial = check_report(
            _report("Background", "Key Findings"), self.template
        )
        self.assertFalse(partial.valid)
        self.assertEqual(len(partial.missing), 8)
        self.assertIn("Limitations", partial.missing)
        swapped = [self.names[1], self.names[0], *self.names[2:]]
        self.assertTrue(
            check_report(_report(*swapped), self.template).out_of_order
        )
        empty = check_report(
            _report(*self.names, empty="Limitations"), self.template
        )
        self.assertEqual(empty.empty, ("Limitations",))
        repeated = check_report(
            _report(*self.names, "Strengths", "Extra"), self.template
        )
        self.assertEqual(repeated.duplicated, ("Strengths",))
        self.assertEqual(repeated.unexpected, ("Extra",))
        self.assertIn("Missing sections:", partial.problems()[0])

    def test_briefing_limits_are_checked(self) -> None:
        briefing = load_template("research_briefing")
        self.assertEqual(briefing.headline_characters, 90)
        self.assertEqual(briefing.max_total_words, 750)

        def text(words: int) -> str:
            return " ".join(["word"] * words)

        def report(headline: str, **sizes: int) -> str:
            parts = [f"# {headline}"]
            for section in briefing.sections:
                if section.items:
                    count = sizes.get("items", 2)
                    body = "\n".join(f"- Ref {n}." for n in range(count))
                else:
                    low, high = section.words
                    body = text(sizes.get(section.name, (low + high) // 2))
                parts.append(f"## {section.name}\n{body}")
            return "\n".join(parts)

        good = check_report(report("Plants under stress"), briefing)
        self.assertTrue(good.valid, good.problems())
        long_title = check_report(report("x" * 95), briefing)
        self.assertIn("95 characters", long_title.headline)
        no_title = check_report(report("")[2:], briefing)
        self.assertIn("must start with", no_title.headline)
        wordy = check_report(
            report("T", **{"The Reason": 400}), briefing
        )
        self.assertIn("The Reason has 400 words", " ".join(wordy.lengths))
        self.assertIn("at most 750", " ".join(wordy.lengths))
        few = check_report(report("T", items=1), briefing)
        self.assertIn("lists 1 items", " ".join(few.lengths))
        # Within the 10% tolerance is fine.
        near = check_report(report("T", **{"The Future": 108}), briefing)
        self.assertTrue(near.valid, near.problems())

    def test_skeleton_lists_every_heading(self) -> None:
        skeleton = report_skeleton(self.template)
        self.assertTrue(skeleton.startswith("# Research Summary\n## Citation"))
        self.assertEqual(skeleton.count("\n## "), 10)


if __name__ == "__main__":
    unittest.main()
