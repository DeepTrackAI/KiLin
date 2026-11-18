"""Unit tests for reading each document of a project report.

Check how documents are chosen and excerpted, the map prompt, and how the
model's notes and per-document checks are parsed and checked.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from kilin.chunking import Chunk
import unittest

from kilin.tools._shared.report_templates import (
    TemplateRules,
    parse_template,
)
from kilin.tools.project_summary._documents import (
    document_excerpt,
    project_documents,
    query_words,
)
from kilin.tools.project_summary._mapping import (
    CHECK_NO,
    CHECK_UNCLEAR,
    CHECK_YES,
    map_prompt,
    parse_notes,
    quote_found,
    template_texts,
)
from tests.kilin.tools._shared._support import chunk

TEMPLATE = parse_template("""---
title: Lab Overview
description: What our papers say together.
max_words: 300
scope: project
---

**Documents:** {documents}

## Main Themes (40-60 words)

The themes the documents share.

## Models (per document)

When: the document trains a deep-learning model.
Do: name the architecture (max 5 words).

## Datasets (per document, table)

When: the document uses a public dataset.
Do: | Dataset | Size |
""", "lab_overview", TemplateRules(0.1, 50))
EXCERPT = (
    "We trained a U-Net on 40 annotated images from the public\n"
    "BBBC dataset."
)


class TestDocuments(unittest.TestCase):

    def setUp(self) -> None:
        self.chunks = [
            chunk("b2", "Results: the networks segment nuclei well.",
                  path="b.pdf", line=20),
            chunk("a1", "Title A. Abstract.", path="a.pdf", line=1),
            chunk("b1", "Title B. Abstract.", path="b.pdf", line=1),
            chunk("b3", "Datasets: we used the public BBBC dataset.",
                  path="b.pdf", line=40),
            chunk("b4", "Acknowledgements and funding.", path="b.pdf",
                  line=60),
        ]

    def test_numbered_with_cards(self) -> None:
        def card(path):
            return {"title": "Study B", "aim": "Segment nuclei."} if (
                path == Path("b.pdf")) else None
        documents = project_documents(self.chunks, card)
        self.assertEqual(
            [(d.number, d.path.name, d.title) for d in documents],
            [(1, "a.pdf", "a.pdf"), (2, "b.pdf", "Study B")],
        )
        self.assertEqual([c.start_line for c in documents[1].passages],
                         [1, 20, 40, 60])
        only = project_documents(self.chunks, None, [Path("b.pdf")])
        self.assertEqual([(d.number, d.title) for d in only],
                         [(1, "b.pdf")])

    def test_notebook_cells_are_ordered_before_cell_local_lines(
        self,
    ) -> None:
        passages = [
            Chunk("a", Path("notes.ipynb"), "Cell one opening", 1, 1,
                  "python", None, None, 1),
            Chunk("b", Path("notes.ipynb"), "Cell one end", 40, 40,
                  "python", None, None, 1),
            Chunk("c", Path("notes.ipynb"), "Cell two opening", 1, 1,
                  "python", None, None, 2),
        ]
        document = project_documents(passages, None)[0]
        self.assertEqual(
            [item.text for item in document.passages],
            [item.text for item in passages],
        )
        excerpt = document_excerpt(document, {"opening"}, 1000)
        self.assertLess(excerpt.index("Cell one end"),
                        excerpt.index("Cell two opening"))

    def test_pdf_pages_are_ordered_before_page_local_lines(self) -> None:
        first = Chunk("p1", Path("paper.pdf"), "Page one conclusion", 40,
                      40, "pdf", None, "Page 1")
        second = Chunk("p2", Path("paper.pdf"), "Page two opening", 1,
                       1, "pdf", None, "Page 2")
        document = project_documents([first, second], None)[0]
        self.assertEqual(document.passages, (first, second))
        excerpt = document_excerpt(document, set(), 1000)
        self.assertLess(excerpt.index("Page one conclusion"),
                        excerpt.index("Page two opening"))

    def test_excerpt(self) -> None:
        def card(path):
            return {"title": "Study B", "aim": "Segment nuclei."}
        document = project_documents(self.chunks, card)[1]
        words = query_words(["Datasets", "the public dataset used"])
        excerpt = document_excerpt(document, words, 1000)
        self.assertTrue(excerpt.startswith(
            "From the document's card:\nTitle: Study B\nAim: Segment "
            "nuclei."
        ))
        self.assertIn("Title B. Abstract.", excerpt)
        self.assertIn("the public BBBC dataset", excerpt)
        self.assertNotIn("Acknowledgements", excerpt)
        # The opening comes first even when the budget is small.
        small = document_excerpt(document, words, 20)
        self.assertIn("Title B. Abstract.", small)
        self.assertNotIn("Datasets: we used", small)


class TestMapping(unittest.TestCase):

    def test_prompt(self) -> None:
        document = project_documents(
            [chunk("a", EXCERPT, path="a.pdf")], None
        )[0]
        system, user = map_prompt(TEMPLATE, document, EXCERPT)
        self.assertIn("Use only the excerpt", system)
        for part in (
            "Report: Lab Overview — What our papers say together.",
            "Document [1]: a.pdf",
            "1. Main Themes: The themes the documents share.",
            "A. Does the document meet this: the document trains a "
            "deep-learning model. If yes, write: name the architecture "
            "(at most 5 words).",
            "B. Does the document meet this: the document uses a public "
            "dataset. If yes, write: the document's Dataset, Size, one per "
            "VALUE line",
            "VALUE B1 Dataset: <for YES only: the document's dataset>",
            "VALUE B2 Size: <for YES only: the document's size>",
            "NOTE 1: <facts, or NONE>", "CHECK B: YES, NO or UNCLEAR",
        ):
            with self.subTest(part=part):
                self.assertIn(part, user)
        self.assertNotIn("Documents", user.split("Topics")[1])
        self.assertTrue(any("Datasets" in t for t in
                            template_texts(TEMPLATE)))

    def test_parse(self) -> None:
        notes = parse_notes(
            "NOTE 1: Segmentation of nuclei with deep learning.\n"
            "CHECK A: YES\nQUOTE A: “trained a U-Net”\n"
            "ENTRY A: U-Net with five levels and skip connections\n"
            "CHECK B: Yes.\nQUOTE B: the public ... BBBC dataset\n"
            "ENTRY B: BBBC | 40 annotated images\n",
            TEMPLATE, 3, EXCERPT +
            "\nThe U-Net with five levels and skip connections was used.",
        )
        self.assertEqual(notes.number, 3)
        self.assertEqual(notes.notes, {
            "Main Themes": "Segmentation of nuclei with deep learning."
        })
        model, data = notes.checks["Models"], notes.checks["Datasets"]
        self.assertEqual((model.answer, model.entry),
                         (CHECK_YES, "U-Net with five levels and…"))
        self.assertEqual(data.cells, ("BBBC", "40 annotated images"))

    def test_table_values(self) -> None:
        for reply, answer, cells in (
            ("VALUE B1 Dataset: BBBC\nVALUE B2 Size: 40 annotated images",
             CHECK_YES, ("BBBC", "40 annotated images")),
            ("VALUE B1: Dataset: BBBC\nVALUE B2 Size: **Size:** 40",
             CHECK_YES, ("BBBC", "40")),
            ("VALUE B1 Dataset: BBBC", CHECK_YES, ("BBBC", "not reported")),
            # The header echoed instead of values.
            ("ENTRY B: Dataset | Size", CHECK_UNCLEAR, ()),
            ("VALUE B1 Dataset: Dataset\nVALUE B2 Size: NONE",
             CHECK_UNCLEAR, ()),
        ):
            with self.subTest(reply=reply):
                check = parse_notes(
                    "CHECK B: YES\nQUOTE B: the public\n" + reply,
                    TEMPLATE, 1, EXCERPT,
                ).checks["Datasets"]
                self.assertEqual((check.answer, check.cells),
                                 (answer, cells))
                if answer == CHECK_UNCLEAR:
                    self.assertEqual(check.why, "no values were written")

    def test_checks_need_evidence(self) -> None:
        for reply, why in (
            ("CHECK A: YES\nQUOTE A: we trained a ResNet\nENTRY A: ResNet",
             "the quoted words are not in the document"),
            ("CHECK A: YES\nQUOTE A: trained a U-Net\nENTRY A: NONE",
             "no entry was written"),
            ("CHECK A: maybe", "no answer"),
            ("", "no answer"),
        ):
            with self.subTest(reply=reply):
                check = parse_notes(reply, TEMPLATE, 1, EXCERPT)
                self.assertEqual(check.checks["Models"].answer,
                                 CHECK_UNCLEAR)
                self.assertEqual(check.checks["Models"].why, why)
        check = parse_notes("NOTE 1: NONE\nCHECK A: NO\nQUOTE A: x",
                            TEMPLATE, 1, EXCERPT)
        self.assertEqual(check.notes, {})
        self.assertEqual(check.checks["Models"].answer, CHECK_UNCLEAR)
        self.assertEqual(
            check.checks["Models"].why,
            "no explicit source-backed negative evidence",
        )
        self.assertEqual(check.checks["Datasets"].answer, CHECK_UNCLEAR)

    def test_quote_found(self) -> None:
        for quote, found in (
            ("trained a U-Net", True),
            ("“We trained a U-Net on 40 annotated images”", True),
            ("we trained a U-Net on 400 images of cells", False),
            ("public ... dataset", True),
            ("a ResNet", False),
            ("", False),
        ):
            with self.subTest(quote=quote):
                self.assertEqual(quote_found(quote, EXCERPT), found)


class TestEvidenceValidation(unittest.TestCase):

    def test_quotes_require_full_ordered_source_matches(self) -> None:
        source = ("Researchers observed a statistically significant "
                  "decrease in measured fluorescence.")
        self.assertFalse(quote_found(
            "Researchers observed a statistically significant increase", source
        ))
        self.assertTrue(quote_found(
            "observed a statistically significant decrease", source
        ))
        self.assertFalse(quote_found("decrease ... researchers", source))
        self.assertTrue(quote_found("Researchers ... fluorescence", source))
        self.assertFalse(quote_found("p < 0.05", "The result had p > 0.05"))
        self.assertTrue(quote_found("p ≤ 0.05", "The result had p <= 0.05"))
        # Numeric signs and percentages must not vanish as punctuation.
        self.assertFalse(quote_found("increased by 40%", "increased by 40"))
        self.assertFalse(quote_found("decreased by -5", "decreased by 5"))
        self.assertFalse(quote_found("loss was −3%", "loss was 3%"))
        self.assertTrue(quote_found("loss was −3 %", "loss was -3%"))
        self.assertFalse(quote_found("rate of 1e-3", "rate of 1e3"))

    def test_unsupported_prose_entry_is_not_accepted(self) -> None:
        result = parse_notes(
            "CHECK A: YES\nQUOTE A: trained a U-Net\n"
            "ENTRY A: FraudNet",
            TEMPLATE, 1, EXCERPT,
        )
        check = result.checks["Models"]
        self.assertEqual(check.answer, CHECK_UNCLEAR)
        self.assertIn("entry's wording", check.why)
        self.assertEqual(check.entry, "")

    def test_table_fabrication_is_not_accepted(self) -> None:
        reply = (
            "CHECK B: YES\nQUOTE B: public BBBC dataset\n"
            "VALUE B1 Dataset: FraudData\n"
            "VALUE B2 Size: 999 million images"
        )
        found = parse_notes(reply, TEMPLATE, 1, EXCERPT)
        self.assertEqual(found.checks["Datasets"].answer, CHECK_UNCLEAR)
        self.assertIn("Dataset, Size", found.checks["Datasets"].why)
        self.assertEqual(found.checks["Datasets"].cells, ())
        # A missing field is allowed, but the other cells must be sourced.
        accepted = parse_notes(
            "CHECK B: YES\nQUOTE B: public BBBC dataset\n"
            "VALUE B1 Dataset: BBBC\nVALUE B2 Size: not reported",
            TEMPLATE, 1, EXCERPT,
        )
        self.assertEqual(
            accepted.checks["Datasets"].cells,
            ("BBBC", "not reported"),
        )

    def test_source_card_alone_cannot_verify_a_quote_or_cell(self) -> None:
        card = ("From the document's card:\nFindings: "
                "FraudData has 999 million images.\n" + EXCERPT)
        reply = (
            "CHECK B: YES\nQUOTE B: FraudData has 999 million images\n"
            "VALUE B1 Dataset: FraudData\n"
            "VALUE B2 Size: 999 million images"
        )
        check = parse_notes(
            reply, TEMPLATE, 1, card, source_passages=(EXCERPT,)
        ).checks["Datasets"]
        self.assertEqual(check.answer, CHECK_UNCLEAR)
        self.assertIn("quoted words", check.why)

    def test_source_passage_boundaries_are_not_spliced(self) -> None:
        source = ("A public BBBC", "dataset of forty images")
        exact = parse_notes(
            "CHECK B: YES\nQUOTE B: public BBBC dataset\n"
            "VALUE B1 Dataset: BBBC\nVALUE B2 Size: forty images",
            TEMPLATE, 1, " ".join(source), source_passages=source,
        )
        self.assertEqual(exact.checks["Datasets"].answer, CHECK_UNCLEAR)
        self.assertIn("quoted words", exact.checks["Datasets"].why)
        # Explicit ellipsis can bridge the two distinct source passages.
        quoted = parse_notes(
            "CHECK B: YES\nQUOTE B: public BBBC ... forty images\n"
            "VALUE B1 Dataset: BBBC\nVALUE B2 Size: forty images",
            TEMPLATE, 1, " ".join(source), source_passages=source,
        )
        self.assertEqual(quoted.checks["Datasets"].answer, CHECK_YES)

    def test_negative_requires_explicit_relevant_source_evidence(self) -> None:
        explicit = "We did not train any deep-learning model."
        report = parse_notes(
            "CHECK A: NO\nQUOTE A: did not train any deep-learning model",
            TEMPLATE, 1, explicit, source_passages=(explicit,),
        )
        self.assertEqual(report.checks["Models"].answer, CHECK_NO)
        for quote in ("did not report the publication year", "a review"):
            source = f"{quote}. A U-Net was trained on an image dataset."
            result = parse_notes(
                f"CHECK A: NO\nQUOTE A: {quote}", TEMPLATE, 1, source
            )
            self.assertEqual(result.checks["Models"].answer, CHECK_UNCLEAR)


if __name__ == "__main__":
    unittest.main()
