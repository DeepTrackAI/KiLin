"""Unit tests for checking one paragraph with the model.

The model is a stub. Check what the prompts tell it, how replies are
parsed, which suggested changes are dropped, and how failed calls are
reported.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock

from kilin._model_completion import IncompleteModelResponse
from kilin.tools.writing_check._checks import (
    LEVEL_GRAMMAR,
    LEVEL_STYLE,
    Glossary,
    check_document,
)
from kilin.tools.writing_check._document import read_document
from kilin.tools.writing_check._review import (
    brief_prompt,
    document_brief,
    document_context,
    locate,
    paragraph_prompt,
    parse_findings,
    rejection,
    review_paragraph,
)

LATEX = r"""\begin{document}
\section{Introduction}
Single-particle tracking (SPT) reveals diffusion of molecules in cells.
\section{Methods}
We tracks 120 particles in order to estimate
the diffusion \cite{a}. It was not trivial.
\section{Results}
The MSD grew linearly.
\end{document}
"""
CARD = {"title": "Diffusion in cells", "venue": "Biophys J",
        "aim": "Measure diffusion with SPT.", "year": "2020"}
REPLY = """Some thinking first.
ORIGINAL: We tracks
SUGGESTION: We tracked
REASON: agreement; methods in the past tense
ORIGINAL: "in order to estimate"
SUGGESTION: "to estimate"
REASON: concise
ORIGINAL: 120 particles
SUGGESTION: 100 particles
REASON: invented number
ORIGINAL: It was not trivial
SUGGESTION: It was trivial
REASON: drops the negation
ORIGINAL: words not in the text
SUGGESTION: other words
REASON: invented
ORIGINAL: the diffusion [cite]
SUGGESTION: the diffusion
REASON: drops the citation
"""


class TestPrompts(unittest.TestCase):

    def setUp(self) -> None:
        self.document = read_document(Path("draft.tex"), LATEX)
        _, self.glossary = check_document(self.document)

    def test_context_from_card_or_brief(self) -> None:
        self.assertEqual(
            document_context(CARD, "unused", self.glossary),
            "Title: Diffusion in cells\nVenue: Biophys J\n"
            "Aim: Measure diffusion with SPT.\n"
            "Acronyms defined in the document: SPT = Single-particle "
            "tracking.",
        )
        self.assertEqual(
            document_context(None, "A biophysics article.", Glossary()),
            "About the document: A biophysics article.",
        )
        self.assertEqual(document_context(None, "", Glossary()), "")

    def test_grammar_prompt(self) -> None:
        context = document_context(CARD, "", self.glossary)
        system, user = paragraph_prompt(
            LEVEL_GRAMMAR, self.document, 1, context
        )
        self.assertIn("copy editor", system)
        for part in (
            "Title: Diffusion in cells",
            "Section: 2 Methods (methods: what was done, usually in the "
            "past tense)",
            "Sentence before the paragraph: Single-particle tracking (SPT) "
            "reveals diffusion of molecules in cells.",
            '"""\nWe tracks 120 particles in order to estimate the '
            'diffusion [cite]. It was not trivial.\n"""',
            "Sentence after the paragraph: The MSD grew linearly.",
            "Find the grammar errors",
            "ORIGINAL:", "Write NONE",
        ):
            with self.subTest(part=part):
                self.assertIn(part, user)

    def test_style_prompt_and_first_paragraph(self) -> None:
        _, user = paragraph_prompt(LEVEL_STYLE, self.document, 0, "")
        self.assertIn("Suggest at most 5 changes", user)
        self.assertNotIn("Sentence before", user)
        self.assertIn("introduction: background", user)

    def test_brief(self) -> None:
        _, user = brief_prompt(self.document)
        self.assertIn("Headings: 1 Introduction; 2 Methods; 3 Results", user)
        self.assertIn("We tracks 120 particles", user)
        self.assertEqual(
            document_brief(self.document, Mock(return_value=" An  article.")),
            "An article.",
        )
        broken = Mock(side_effect=RuntimeError("down"))
        self.assertEqual(document_brief(self.document, broken), "")


class TestReplies(unittest.TestCase):

    def test_parse(self) -> None:
        self.assertEqual(parse_findings("NONE"), [])
        self.assertEqual(parse_findings(REPLY)[:2], [
            ("We tracks", "We tracked",
             "agreement; methods in the past tense"),
            ("in order to estimate", "to estimate", "concise"),
        ])
        # An unfinished block is not a finding.
        self.assertEqual(
            parse_findings("ORIGINAL: a\nSUGGESTION: b\n"), []
        )

    def test_rejection(self) -> None:
        glossary = Glossary({"SPT": "single-particle tracking"},
                            {"data set": "dataset"}, "UK")
        paragraph = ("SPT of 12 cells did not use the dataset [cite], as "
                     "shown in (ref).")
        for original, suggestion, reason in (
            ("SPT of 12 cells", "Tracking of 12 cells", "it removes “SPT”"),
            ("of 12 cells", "of 13 cells", "it changes numbers"),
            ("did not use", "used", "it changes a negation"),
            ("the dataset [cite]", "the dataset", "it changes citations"),
            ("use the dataset", "use the data set", "it removes “dataset”"),
            ("did not use", "did not employ the data set",
             "it writes “data set”, which the document does not use"),
            ("other words", "new words", "the words are not in the "
             "paragraph"),
            ("did not use", "did  not use", "nothing changes"),
            ("as shown in (ref)", "as seen in (ref)", ""),
        ):
            with self.subTest(original=original):
                self.assertEqual(
                    rejection(original, suggestion, paragraph, glossary),
                    reason,
                )


class TestLocate(unittest.TestCase):

    def test_quotes_dashes_case_and_ends(self) -> None:
        paragraph = ("We used “single-particle” tracking – a method. It is "
                     "fast.")
        for quote, words in (
            ('we used "single-particle" tracking',
             "We used “single-particle” tracking"),
            ("tracking - a method.", "tracking – a method."),
            ("...It is fast.", "It is fast"),
            ("It is fast", "It is fast"),
        ):
            with self.subTest(quote=quote):
                found = locate(quote, paragraph)
                self.assertIsNotNone(found)
                assert found is not None
                self.assertEqual(found[0], words)
                self.assertEqual(paragraph[found[1]:].startswith(words),
                                 True)
        self.assertIsNone(locate("not here", paragraph))

    def test_kept_finding_quotes_the_paragraph(self) -> None:
        document = read_document(
            Path("a.txt"), "We used “raw” data in the the analysis."
        )
        reply = ('ORIGINAL: we used "raw" data\nSUGGESTION: We used the '
                 'raw data\nREASON: article\n')
        kept, dropped, _ = review_paragraph(
            LEVEL_GRAMMAR, document, 0, "", Glossary(),
            Mock(return_value=reply),
        )
        self.assertEqual(dropped, [])
        self.assertEqual(kept[0].original, "We used “raw” data")


class TestReviewParagraph(unittest.TestCase):

    def setUp(self) -> None:
        self.document = read_document(Path("draft.tex"), LATEX)
        _, self.glossary = check_document(self.document)

    def test_kept_and_dropped(self) -> None:
        kept, dropped, failure = review_paragraph(
            LEVEL_GRAMMAR, self.document, 1, "", self.glossary,
            Mock(return_value=REPLY),
        )
        self.assertEqual(
            [(f.line, f.level, f.check, f.original, f.suggestion)
             for f in kept],
            [(5, LEVEL_GRAMMAR, "model", "We tracks", "We tracked"),
             (5, LEVEL_GRAMMAR, "model", "in order to estimate",
              "to estimate")],
        )
        self.assertEqual(failure, "")
        self.assertEqual(
            [item.why for item in dropped],
            ["it changes numbers", "it changes a negation",
             "the words are not in the paragraph", "it changes citations"],
        )

    def test_failed_calls(self) -> None:
        for error, start in (
            (RuntimeError("down"), "Model error: down"),
            (IncompleteModelResponse("cut", reason="token_limit"),
             "Not completed (token_limit)"),
        ):
            with self.subTest(error=error):
                kept, dropped, failure = review_paragraph(
                    LEVEL_STYLE, self.document, 1, "", self.glossary,
                    Mock(side_effect=error),
                )
                self.assertEqual((kept, dropped), ([], []))
                self.assertTrue(failure.startswith(start))

    def test_style_keeps_at_most_five(self) -> None:
        block = ("ORIGINAL: It was not trivial\nSUGGESTION: It was not "
                 "easy\nREASON: plainer\n")
        kept, _, _ = review_paragraph(
            LEVEL_STYLE, self.document, 1, "", self.glossary,
            Mock(return_value=block * 7),
        )
        self.assertEqual(len(kept), 5)


if __name__ == "__main__":
    unittest.main()
