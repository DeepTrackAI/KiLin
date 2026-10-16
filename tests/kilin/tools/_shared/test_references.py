"""Unit tests for the shared reference-list extractor.

Check that a paper's reference list is found by its heading and split into
the entries it prints, for numbered lists and for two-column layouts that
print the numbers apart from the entries, independently of any registered
tool's configuration.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from kilin.tools._shared.references import (
    ReferenceLayout,
    reference_body,
    reference_entries,
    reference_list,
)
from tests.kilin.tools._shared._support import chunk

LAYOUT = ReferenceLayout(
    max_characters=40000,
    numbered_entries_min=5,
    running_line_repeats=3,
    running_line_words=12,
    entry_min_words=5,
)

NUMBERED = """Introduction
Some text cites earlier work [1].

References
1. Smith, J. A study of cells in culture. Nature 1, 2-3 (2019).
2. Lee, K. & Kim, H. Another study of tissue. Science 4, 5-6 (2020).
3. Wu, X. A third study about imaging methods. Cell 7, 8-9 (2021).
[4] Rossi, M. A fourth study about microscopy. PNAS 10, 11-12 (2022).
5) Garcia, L. A fifth study about dynamics. eLife 13, 14 (2023).
Acknowledgements We thank the facility.
6. Not a reference, this follows the list.
"""

TWO_COLUMN = """References
1.
2.
Smith, J. A study of cells in
culture. Nature 1, 2-3 (2019).
Lee, K. & Kim, H. Another study of tis-
sue samples. Science 4, 5-6 (2020).
https://doi.org/10.1000/xyz123
Wu, X. A third study about imaging. Cell 7, 8-9 (2021).
"""


class TestReferenceList(unittest.TestCase):
    """Find the reference list by its heading."""

    def test_no_heading_gives_empty_list(self) -> None:
        passages = (chunk("a", "Introduction\nNo list here.\n"),)
        self.assertEqual(reference_list(passages, LAYOUT), "")
        self.assertEqual(reference_entries(passages, LAYOUT), [])

    def test_last_heading_wins_across_passages(self) -> None:
        passages = (
            chunk("a", "References\nCited in the intro.\n"),
            chunk("b", "## Bibliography\nThe real list.\n", line=3),
        )
        self.assertEqual(reference_list(passages, LAYOUT), "The real list.")

    def test_heading_variants(self) -> None:
        for heading in (
            "References", "REFERENCES:", "# Literature cited",
            "Works Cited", "Bibliography",
        ):
            with self.subTest(heading=heading):
                passages = (chunk("a", f"{heading}\nEntry text.\n"),)
                self.assertEqual(
                    reference_list(passages, LAYOUT), "Entry text."
                )

    def test_max_characters_cuts_the_list(self) -> None:
        passages = (chunk("a", "References\n" + "x" * 100),)
        layout = replace(LAYOUT, max_characters=10)
        # The limit counts from the end of the heading, its newline included.
        cut = reference_list(passages, layout)
        self.assertEqual(cut, "x" * 9)


class TestReferenceBody(unittest.TestCase):
    """Return the text before the reference list."""

    def test_cut_before_the_last_heading(self) -> None:
        passages = (
            chunk("a", "Introduction\nCells grow [1].\n"),
            chunk("b", "Results hold [2].\nReferences\n1. Smith\n", line=3),
            chunk("c", "2. Lee\n", line=6),
        )
        body = reference_body(passages)
        self.assertEqual(len(body), 2)
        self.assertEqual(body[1].text, "Results hold [2].\n")
        self.assertEqual((body[1].start_line, body[1].end_line), (3, 4))

    def test_heading_at_passage_start_drops_it(self) -> None:
        passages = (
            chunk("a", "Text [1].\n"),
            chunk("b", "References\n1. Smith\n", line=2),
        )
        self.assertEqual(reference_body(passages), passages[:1])

    def test_no_heading_keeps_everything(self) -> None:
        passages = (chunk("a", "Text only.\n"),)
        self.assertEqual(reference_body(passages), passages)


class TestReferenceEntries(unittest.TestCase):
    """Split the reference list into the entries it prints."""

    def test_numbered_list_splits_at_numbers_and_stops(self) -> None:
        entries = reference_entries((chunk("a", NUMBERED),), LAYOUT)
        self.assertEqual(len(entries), 5)
        self.assertTrue(entries[0].startswith("Smith, J. A study"))
        self.assertTrue(entries[3].startswith("Rossi, M."))
        self.assertTrue(entries[4].startswith("Garcia, L."))
        self.assertFalse(any("Not a reference" in e for e in entries))

    def test_two_column_list_splits_at_entry_ends(self) -> None:
        entries = reference_entries((chunk("a", TWO_COLUMN),), LAYOUT)
        self.assertEqual(
            entries,
            [
                "Smith, J. A study of cells in culture. Nature 1, 2-3 "
                "(2019).",
                "Lee, K. & Kim, H. Another study of tissue samples. "
                "Science 4, 5-6 (2020). https://doi.org/10.1000/xyz123",
                "Wu, X. A third study about imaging. Cell 7, 8-9 (2021).",
            ],
        )

    def test_page_furniture_is_dropped(self) -> None:
        running = "J. Smith et al. Journal 35 (2026)"
        text = (
            f"{running}\nIntro.\n{running}\nReferences\n"
            "Smith, J. A study of cells. Nature 1, 2-3 (2019).\n"
            "12\n"
            f"{running}\n"
            "PLOS ONE | https://doi.org/10.1371/x\n"
            "Lee, K. Another study of tissue. Science 4, 5-6 (2020).\n"
            "3 / 25\n"
            "Too short (2021).\n"
        )
        self.assertEqual(
            reference_entries((chunk("a", text),), LAYOUT),
            [
                "Smith, J. A study of cells. Nature 1, 2-3 (2019).",
                "Lee, K. Another study of tissue. Science 4, 5-6 (2020).",
            ],
        )

    def test_layout_thresholds_are_the_callers(self) -> None:
        # With fewer numbered lines than required, entries split at their
        # ends instead; a high word minimum drops every entry.
        passages = (chunk("a", NUMBERED),)
        strict = replace(LAYOUT, numbered_entries_min=50)
        self.assertEqual(len(reference_entries(passages, strict)), 5)
        long_only = replace(LAYOUT, entry_min_words=50)
        self.assertEqual(reference_entries(passages, long_only), [])

    def test_year_and_pages_end_an_entry(self) -> None:
        # Two-column layout: the lone numbers force splitting at entry
        # ends, and "2016, 160422." or "1995; 289–300." end an entry.
        text = (
            "References\n1.\n2.\n3.\n"
            "Ji GJ, Yu Y. Decreased network efficiency in epilepsy.\n"
            "Radiology 2016, 160422.\n"
            "Liu Y, Liang M. Disrupted small-world networks in\n"
            "schizophrenia. Brain 2008, 131, 945–961. PMID: 18299296\n"
            "Benjamini Y, Hochberg Y. Controlling the false discovery\n"
            "rate. J Royal Stat Soc 1995; 289–300.\n"
        )
        entries = reference_entries((chunk("a", text),), LAYOUT)
        self.assertEqual(len(entries), 3)
        self.assertTrue(entries[1].startswith("Liu Y"))

    def test_layout_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            LAYOUT.entry_min_words = 1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
