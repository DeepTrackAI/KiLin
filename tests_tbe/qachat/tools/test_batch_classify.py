"""Check batched classification: batches, validation, counts, coverage."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.tools.batch_classify import (
    ClassifySettings,
    Label,
    classification_evidence,
    classification_report,
    item_text,
    parse_labels,
    plan_batches,
    quick_sample,
    validated_categories,
)
from qachat.tools.planning import ToolPlan, needs_confirmation
from qachat.tools.scan_and_list import scan_mentions


def _chunk(path: str, lines: list[str]) -> Chunk:
    return Chunk(
        identifier=path,
        path=Path(path),
        text="".join(f"{line}\n" for line in lines),
        start_line=1,
        end_line=len(lines),
        kind="text",
        symbol=None,
        section=None,
    )


class TestBatchClassify(unittest.TestCase):
    """The model labels; Python validates and counts."""

    def setUp(self) -> None:
        self.scan = scan_mentions(
            [
                _chunk("a.md", ["We wrote it in Python.", "x", "y"]),
                _chunk("b.md", ["Python (ref 3).", "Python again."]),
            ],
            ["Python"],
            context_lines=0,
        )
        self.categories = ("own code", "citation")

    def test_categories_must_be_usable(self) -> None:
        self.assertEqual(
            validated_categories(
                '{"categories": ["own code", "Citation", "unclear"]}'
            ),
            ("own code", "Citation"),
        )
        self.assertEqual(validated_categories('{"categories": ["one"]}'), ())
        self.assertEqual(validated_categories("no json"), ())

    def test_item_text_marks_the_mention(self) -> None:
        text = item_text(1, self.scan.mentions[0], "a.md")
        self.assertEqual(text, "[1] a.md, line 1\nWe wrote it in «Python».")

    def test_batches_respect_the_token_budget(self) -> None:
        texts = ["x" * 400, "y" * 400, "z" * 400]  # About 100 tokens each.
        self.assertEqual(plan_batches(texts, 200), [[0, 1], [2]])
        self.assertEqual(plan_batches(texts, 50), [[0], [1], [2]])

    def test_quick_sample_takes_turns_across_files(self) -> None:
        texts = ["x" * 400] * 3
        self.assertEqual(
            quick_sample(self.scan.mentions, texts, 200), [0, 1]
        )

    def test_only_valid_labels_are_accepted(self) -> None:
        reply = (
            'Sure: [{"id": 1, "category": "Own code", "reason": "ours"}, '
            '{"id": 2, "category": "made up"}, {"id": 9, "category": '
            '"citation"}, {"id": 3, "category": "citation"}, '
            '{"id": 3, "category": "own code"}]'
        )
        labels = parse_labels(reply, [1, 2, 3], self.categories)
        self.assertEqual(
            labels,
            {1: Label("own code", "ours"), 3: Label("citation", "")},
        )
        self.assertEqual(parse_labels("not json", [1], self.categories), {})
        cut_off = (
            '[{"id": 1, "category": "citation", "reason": "ref"}, '
            '{"id": 2, "category": "own c'
        )
        self.assertEqual(
            parse_labels(cut_off, [1, 2], self.categories),
            {1: Label("citation", "ref")},
        )

    def test_report_counts_validated_labels_and_coverage(self) -> None:
        labels = {0: Label("own code", "we wrote"), 1: Label("citation", "")}
        report = classification_report(
            self.scan,
            labels,
            self.categories,
            limit=10,
            classified=[0, 1, 2],
            not_classified=[2],
        )
        self.assertIn("Classified **2 of 3 mentions** of “Python”", report)
        self.assertIn("- own code: 1\n- citation: 1\n- unclear: 0", report)
        self.assertIn("- not classified: 1", report)
        self.assertIn("— we wrote [1]", report)
        evidence = classification_evidence(
            self.scan, labels, self.categories, 10
        )
        self.assertEqual(
            [(e["citation"], str(e["path"])) for e in evidence],
            [(1, "a.md"), (2, "b.md")],
        )
        partial = classification_report(
            self.scan,
            labels,
            self.categories,
            limit=10,
            classified=[0, 1],
            not_classified=[],
        )
        self.assertIn("1 mentions were not sent to the model", partial)
        quick = classification_report(
            self.scan,
            labels,
            self.categories,
            limit=10,
            classified=[0, 1],
            not_classified=[],
            quick=True,
        )
        self.assertIn("Quick sample", quick)
        self.assertNotIn("not sent", quick)

    def test_confirmation_policy(self) -> None:
        plan = ToolPlan("batch_classify", "s", "all", 10, 3, 3, 9, 4, 3)
        self.assertTrue(needs_confirmation("when_large", plan, 2))
        self.assertFalse(needs_confirmation("when_large", plan, 3))
        self.assertTrue(needs_confirmation("always", plan, 99))
        self.assertFalse(needs_confirmation("never", plan, 0))
        empty = ToolPlan("batch_classify", "s", "all", 0, 0, 0, 0, 0, 0)
        self.assertFalse(needs_confirmation("always", empty, 0))
        self.assertEqual(ClassifySettings().confirmation, "when_large")


if __name__ == "__main__":
    unittest.main()
