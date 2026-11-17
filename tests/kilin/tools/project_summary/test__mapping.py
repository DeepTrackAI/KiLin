"""Unit tests for reading each document of a project report.

Check how documents are chosen and excerpted, the map prompt, and how the
model's notes and per-document checks are parsed and checked.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
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
            "ENTRY B: BBBC | 40 images | extra\n",
            TEMPLATE, 3, EXCERPT,
        )
        self.assertEqual(notes.number, 3)
        self.assertEqual(notes.notes, {
            "Main Themes": "Segmentation of nuclei with deep learning."
        })
        model, data = notes.checks["Models"], notes.checks["Datasets"]
        self.assertEqual((model.answer, model.entry),
                         (CHECK_YES, "U-Net with five levels and…"))
        self.assertEqual(data.cells, ("BBBC", "40 images extra"))

    def test_table_values(self) -> None:
        for reply, answer, cells in (
            ("VALUE B1 Dataset: BBBC\nVALUE B2 Size: 40 images",
             CHECK_YES, ("BBBC", "40 images")),
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
        self.assertEqual(check.checks["Models"].answer, CHECK_NO)
        self.assertEqual(check.checks["Datasets"].answer, CHECK_UNCLEAR)

    def test_quote_found(self) -> None:
        for quote, found in (
            ("trained a U-Net", True),
            ("“We trained a U-Net on 40 annotated images”", True),
            ("we trained a U-Net on 400 images of cells", True),
            ("public ... dataset", True),
            ("a ResNet", False),
            ("", False),
        ):
            with self.subTest(quote=quote):
                self.assertEqual(quote_found(quote, EXCERPT), found)


if __name__ == "__main__":
    unittest.main()
