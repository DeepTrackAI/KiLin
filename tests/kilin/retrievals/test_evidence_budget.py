"""Verify independent evidence limits and honest source coverage.

The budgeter is pure and requires neither an embedding nor a chat model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.rag import Evidence
from kilin.retrievals.evidence_budget import (
    EvidenceLimits,
    allocate_evidence,
    coverage_report,
    evidence_block,
    evidence_tokens,
)


def _item(source: str, number: int, text: str = "data") -> Evidence:
    """Make an original-chunk evidence record with stable coordinates.

    Parameters
    ----------
    source: str
        Relative source path.
    number: int
        One-based citation and source-line number.
    text: str, optional
        Passage contents. Defaults to ``"data"``.

    Returns
    -------
    Evidence
        A source-backed passage usable in a complete answer.

    """

    return Evidence(number, Path(source), number, number, None, text)


class TestEvidenceLimits(unittest.TestCase):

    def test_reject_invalid_numeric_types_and_coverage(self):
        for value in (0, -1, True, 2.5, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EvidenceLimits(value, 1, 5, 1000)
        with self.assertRaises(ValueError):
            EvidenceLimits(2, 1, 5, 1000, "unknown")

    def test_quota_reserves_first_passage_for_each_source(self):
        items = (
            _item("a.pdf", 1), _item("a.pdf", 2),
            _item("a.pdf", 3), _item("b.pdf", 4),
            _item("b.pdf", 5), _item("c.pdf", 6),
        )
        result = allocate_evidence(
            items, EvidenceLimits(3, 2, 5, 5000),
            priority_sources=(Path("a.pdf"), Path("b.pdf"),
                              Path("c.pdf")),
        )
        self.assertEqual([x.path.name for x in result.items],
                         ["a.pdf", "b.pdf", "c.pdf", "a.pdf", "b.pdf"])
        self.assertEqual(result.omitted, 1)
        self.assertIn("passages", result.limits_hit)

    def test_oversized_evidence_is_skipped_without_truncation(self):
        items = (_item("a.pdf", 1, "a" * 10000),
                 _item("a.pdf", 2, "small"),
                 _item("b.pdf", 3, "small"))
        result = allocate_evidence(
            items, EvidenceLimits(2, 1, 4, 80),
            priority_sources=(Path("a.pdf"), Path("b.pdf")),
        )
        self.assertEqual([x.text for x in result.items],
                         ["small", "small"])
        self.assertIn("tokens", result.limits_hit)
        self.assertLessEqual(result.estimated_tokens, 80)

    def test_headers_and_citation_numbers_are_included(self):
        items = (_item("research-long-name.pdf", 1, "x" * 24),)
        rendered = evidence_block(items[0], 1)
        self.assertIn("research-long-name.pdf:1-1", rendered)
        self.assertIn("x" * 24, rendered)
        expected = evidence_tokens(items)
        with self.assertRaises(ValueError):
            EvidenceLimits(1, 1, 1, 0)
        selected = allocate_evidence(
            items, EvidenceLimits(1, 1, 1, expected)
        )
        self.assertEqual(selected.items, items)
        selected = allocate_evidence(
            items, EvidenceLimits(1, 1, 1, expected - 1)
        )
        self.assertFalse(selected.items)

    def test_related_sources_compete_for_source_slots(self):
        items = (_item("a.pdf", 1), _item("b.pdf", 2),
                 _item("c.pdf", 3))
        selected = allocate_evidence(
            items, EvidenceLimits(2, 1, 3, 300),
            priority_sources=(Path("a.pdf"), Path("b.pdf")),
        )
        self.assertEqual([x.path.name for x in selected.items],
                         ["a.pdf", "b.pdf"])
        self.assertIn("sources", selected.limits_hit)

    def test_coverage_distinguishes_absent_unindexed_source(self):
        requested = (Path("a.pdf"), Path("b.pdf"), Path("c.pdf"))
        result = coverage_report(
            requested, (_item("a.pdf", 1),),
            mode="exhaustive", unavailable=(Path("c.pdf"),),
            limits_hit=("passages",),
        )
        self.assertEqual(result["requested_sources"], 3)
        self.assertEqual(result["represented_sources"], 1)
        self.assertEqual(result["missing_sources"], ["b.pdf", "c.pdf"])
        self.assertEqual(result["unindexed_sources"], ["c.pdf"])
        self.assertFalse(result["complete"])
        self.assertIn("unindexed", result["limits_hit"])

    def test_coverage_never_claims_full_content_review(self):
        result = coverage_report(
            (Path("a.pdf"),), (_item("a.pdf", 1),),
            mode="exhaustive",
        )
        self.assertTrue(result["complete"])
        self.assertIn("not full content review", result["meaning"])


if __name__ == "__main__":
    unittest.main()
