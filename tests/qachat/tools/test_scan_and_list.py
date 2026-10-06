"""Check exhaustive mention listing: exact counts, locations, coverage."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.tools.scan_and_list import (
    mention_evidence,
    mention_report,
    document_phrase,
    quoted_terms,
    scan_mentions,
    validated_document,
    validated_terms,
)


def _chunk(path, start, lines, section=None, symbol=None, cell=None):
    text = "".join(f"{line}\n" for line in lines)
    return Chunk(
        identifier=f"{path}:{section}:{start}",
        path=Path(path),
        text=text,
        start_line=start,
        end_line=start + len(lines) - 1,
        kind="text",
        symbol=symbol,
        section=section,
        cell=cell,
    )


class TestScan(unittest.TestCase):
    """Every mention is found once, with its exact source location."""

    def test_overlapping_chunks_count_each_mention_once(self) -> None:
        lines = ["intro", "RFE ranks features.", "Then RFE stops.", "end"]
        chunks = [
            _chunk("a.md", 1, lines[:3]),
            _chunk("a.md", 2, lines[1:]),  # Overlaps lines 2-3.
        ]
        scan = scan_mentions(chunks, ["RFE"])
        self.assertEqual(len(scan.mentions), 2)
        self.assertEqual([m.line for m in scan.mentions], [2, 3])
        self.assertEqual(scan.counts(), {Path("a.md"): 2})

    def test_pdf_pages_restart_line_numbers(self) -> None:
        chunks = [
            _chunk("p.pdf", 1, ["DIVA on page one"], section="Page 1"),
            _chunk("p.pdf", 1, ["DIVA again, page two"], section="Page 2"),
        ]
        scan = scan_mentions(chunks, ["diva"])
        self.assertEqual(
            [(m.section, m.line, m.text) for m in scan.mentions],
            [("Page 1", 1, "DIVA"), ("Page 2", 1, "DIVA")],
        )

    def test_phrase_across_a_line_break_and_whole_words(self) -> None:
        chunks = [
            _chunk(
                "b.md",
                10,
                [
                    "We used BRAPH",
                    "2 and BRAPH 2.0.",
                    "BRAPH 20 differs; RFEs",
                ],
            )
        ]
        scan = scan_mentions(chunks, ["BRAPH 2", "RFE"], context_lines=0)
        found = [(m.term, m.line, m.end_line) for m in scan.mentions]
        # "BRAPH 2.0" counts (a period is not a word character); "BRAPH 20"
        # and "RFEs" do not.
        self.assertEqual(found, [("BRAPH 2", 10, 11), ("BRAPH 2", 11, 11)])
        partial = scan_mentions(chunks, ["RFE"], whole_words=False)
        self.assertEqual(len(partial.mentions), 1)
        exact_case = scan_mentions(chunks, ["braph 2"], case_sensitive=True)
        self.assertEqual(exact_case.mentions, ())

    def test_scope_and_evidence_lines(self) -> None:
        chunks = [
            _chunk("x.py", 5, ["a = 1", "fit(a)", "b = 2"], symbol="main"),
            _chunk("y.py", 1, ["fit(b)"]),
        ]
        scan = scan_mentions(chunks, ["fit"], paths=[Path("x.py")])
        self.assertEqual(scan.files_scanned, (Path("x.py"),))
        (item,) = mention_evidence(scan, 10)
        self.assertEqual(
            (item["start_line"], item["end_line"], item["symbol"]),
            (5, 7, "main"),
        )
        # The exact lines, so the reference preview can find them.
        self.assertEqual(item["text"], "a = 1\nfit(a)\nb = 2")

    def test_report_counts_everything_and_states_coverage(self) -> None:
        chunks = [
            _chunk("a.md", 1, [f"RFE {n}" for n in range(1, 6)]),
            _chunk("b.md", 1, ["RFE once"]),
            _chunk("c.md", 1, ["nothing here"]),
        ]
        scan = scan_mentions(chunks, ["RFE"])
        report = mention_report(
            scan, limit=3, unsearchable=[Path("scan.pdf")]
        )
        self.assertIn(
            "Found **6 mentions** of “RFE” in **2 of 3 files**", report
        )
        self.assertIn("- a.md: 5", report)
        self.assertIn("1. a.md, line 1: **RFE** 1 RFE 2 [1]", report)
        self.assertIn("Listed the first 3 of 6 mentions", report)
        self.assertIn("1 source files have no searchable text", report)
        self.assertNotIn("[4]", report)
        self.assertEqual(len(mention_evidence(scan, 3)), 3)


    def test_snippets_show_each_occurrence_and_names_are_unique(self) -> None:
        chunks = [
            _chunk("a/readme.md", 1, ["RFE title", "The RFE pipeline"]),
            _chunk("b/readme.md", 1, ["RFE elsewhere"]),
        ]
        report = mention_report(scan_mentions(chunks, ["RFE"]), limit=5)
        self.assertIn("1. a/readme.md, line 1: **RFE** title The", report)
        self.assertIn("2. a/readme.md, line 2: RFE title The **RFE**", report)
        self.assertIn("- b/readme.md: 1", report)

    def test_single_file_scope_is_named(self) -> None:
        chunks = [_chunk("p.pdf", 1, ["MATLAB v2023b"], section="Page 2")]
        report = mention_report(
            scan_mentions(chunks, ["MATLAB"]),
            limit=5,
            scope_note="Scanned only p.pdf, the best match for “DIVA”.",
        )
        self.assertIn("Found **1 mention** of “MATLAB” in **p.pdf**", report)
        self.assertIn("Scanned only p.pdf", report)
        self.assertIn("1. p.pdf, page 2, line 1:", report)
        self.assertIn("Scanned all indexed text in p.pdf.", report)


class TestTerms(unittest.TestCase):
    """Terms come from the user's own words."""

    def test_quoted_terms(self) -> None:
        self.assertEqual(
            quoted_terms('list "BRAPH 2" and `fit` and “RFE”, DIVA\'s'),
            ("BRAPH 2", "fit", "RFE"),
        )

    def test_model_terms_must_occur_in_the_question(self) -> None:
        text = "list all mentions of braph 2 in the papers"
        self.assertEqual(
            validated_terms('{"terms": ["BRAPH  2", "network"]}', text),
            ("BRAPH 2",),
        )
        self.assertEqual(validated_terms("BRAPH 2", text), ())

    def test_document_phrase_must_occur_in_the_question(self) -> None:
        text = "how many times does the DIVA paper say MATLAB?"
        self.assertEqual(
            validated_document('{"document": "the DIVA paper"}', text),
            "the DIVA paper",
        )
        self.assertIsNone(
            validated_document('{"document": "patil25deep.pdf"}', text)
        )
        self.assertIsNone(validated_document('{"document": null}', text))

    def test_document_phrase_from_the_question(self) -> None:
        self.assertEqual(
            document_phrase("which MRI mentions in the Smith paper are new?"),
            "the Smith paper",
        )
        self.assertEqual(
            document_phrase("count MRI in the 2020 review"), "the 2020 review"
        )
        self.assertIsNone(document_phrase("count MRI in all papers"))


if __name__ == "__main__":
    unittest.main()
