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
    CONTEXT_QUOTED,
    brief_prompt,
    document_brief,
    document_context,
    locate,
    paragraph_prompt,
    parse_findings,
    rejection,
    review_paragraph,
    trim_change,
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
            "Context, for reference only (do not check or quote it):\n"
            "Sentence before the paragraph: Single-particle tracking (SPT) "
            "reveals diffusion of molecules in cells.",
            'Paragraph to check:\n"""\nWe tracks 120 particles in order '
            'to estimate the diffusion [cite]. It was not trivial.\n"""',
            "Check only the paragraph between the triple quotes.",
            "Sentence after the paragraph: The MSD grew linearly.",
            "Find the grammar errors",
            "ORIGINAL:", "Write NONE",
        ):
            with self.subTest(part=part):
                self.assertIn(part, user)

    def test_long_neighbour_is_abbreviated_not_the_paragraph(self) -> None:
        """Bound optional context without shortening the checked text.

        """

        content = ("First sentence. " + "Context " * 400
                   + "\n\nTarget paragraph.\n")
        document = read_document(Path("long.txt"), content)
        system, prompt = paragraph_prompt(
            LEVEL_GRAMMAR, document, 1, ""
        )
        self.assertIn("[context abbreviated]", prompt)
        self.assertIn('Paragraph to check:\n"""\nTarget paragraph.',
                      prompt)
        self.assertLess(len(system) + len(prompt), 3000)

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
        # The reason is one line; a NONE after the last block is not in it.
        self.assertEqual(
            parse_findings("ORIGINAL: one\nSUGGESTION: a single\nREASON: "
                           "precise NONE\nmore\n\nNONE"),
            [("one", "a single", "precise")],
        )
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
            ("did not use the", "would not use the",
             "it changes a hedge or modal verb"),
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


class TestTrim(unittest.TestCase):

    def test_cut_to_the_changed_words(self) -> None:
        for words, suggestion, expected in (
            ("the simulation and any network form one differentiable "
             "graph, so gradients reach",
             "the simulation and any network form a single differentiable "
             "graph, so gradients reach",
             ("network form one differentiable graph,",
              "network form a single differentiable graph,", 23)),
            ("We tracks", "We tracked", ("We tracks", "We tracked", 0)),
            ("the experiment itself the central object",
             "the experiment itself as the central object",
             ("experiment itself the central",
              "experiment itself as the central", 4)),
        ):
            with self.subTest(words=words):
                self.assertEqual(trim_change(words, suggestion), expected)


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

    def test_quotes_from_the_context_are_dropped_as_such(self) -> None:
        reply = ("ORIGINAL: The MSD grew linearly\nSUGGESTION: The MSD "
                 "increased linearly\nREASON: clearer\n")
        kept, dropped, _ = review_paragraph(
            LEVEL_STYLE, self.document, 1, "", self.glossary,
            Mock(return_value=reply),
        )
        self.assertEqual(kept, [])
        self.assertEqual([item.why for item in dropped], [CONTEXT_QUOTED])

    def test_kept_findings_are_trimmed(self) -> None:
        reply = ("ORIGINAL: We tracks 120 particles in order to estimate\n"
                 "SUGGESTION: We tracks 120 particles to estimate\n"
                 "REASON: concise\n")
        kept, _, _ = review_paragraph(
            LEVEL_STYLE, self.document, 1, "", self.glossary,
            Mock(return_value=reply),
        )
        self.assertEqual((kept[0].original, kept[0].suggestion),
                         ("120 particles in order to estimate",
                          "120 particles to estimate"))

    def test_style_keeps_at_most_five(self) -> None:
        block = ("ORIGINAL: It was not trivial\nSUGGESTION: It was not "
                 "easy\nREASON: plainer\n")
        kept, _, _ = review_paragraph(
            LEVEL_STYLE, self.document, 1, "", self.glossary,
            Mock(return_value=block * 7),
        )
        self.assertEqual(len(kept), 5)


class TestScientificClaimSafety(unittest.TestCase):
    """Reject visible claim reversals without claiming semantic proof.

    """

    def test_model_findings_reject_claim_reversals(self) -> None:
        """Guard meaning-bearing scientific polarity and direction words.

        """

        from kilin.tools.writing_check._checks import Glossary
        from kilin.tools.writing_check._review import meaning_change

        for before, after in (
            ("improved survival", "reduced survival"),
            ("increased from baseline", "decreased from baseline"),
            ("correlation was positive", "correlation was negative"),
            ("treatment was effective", "treatment was ineffective"),
            ("samples were included", "samples were excluded"),
            ("significant results", "insignificant results"),
            ("not measured", "never measured"),
        ):
            with self.subTest(before=before, after=after):
                self.assertTrue(meaning_change(before, after, Glossary()))

    def test_rewrite_options_share_the_same_safeguards(self) -> None:
        """Reject unsafe options in the paragraph rewrite path.

        """

        from kilin.tools.writing_check._checks import Glossary
        from kilin.tools.writing_check._followup import option_rejection

        before = "The treatment improved survival."
        after = "The treatment reduced survival."
        self.assertTrue(
            option_rejection(before, after, before, Glossary())
        )

    def test_relocated_claims_and_subjects_are_rejected(self) -> None:
        """Preserve not only protected words, but also their placement.

        """

        from kilin.tools.writing_check._checks import Glossary
        from kilin.tools.writing_check._review import meaning_change

        for before, after in (
            (
                "Treatment A was effective; treatment B was ineffective.",
                "Treatment B was effective; treatment A was ineffective.",
            ),
            (
                "Drug alpha improved, but drug beta declined.",
                "Drug beta improved, but drug alpha declined.",
            ),
            (
                "Treatment A was not effective; B was effective.",
                "Treatment A was effective; B was not effective.",
            ),
            (
                "Treatment A had positive results; B had negative ones.",
                "Treatment A had negative results; B had positive ones.",
            ),
            (
                "Treatment A was effective.",
                "Treatment B was effective.",
            ),
        ):
            with self.subTest(before=before):
                why = meaning_change(before, after, Glossary())
                self.assertTrue(why, (before, after))

    def test_benign_grammar_edits_remain_possible(self) -> None:
        """Changing a function word need not move a scientific claim.

        """

        from kilin.tools.writing_check._checks import Glossary
        from kilin.tools.writing_check._review import meaning_change

        self.assertEqual(
            meaning_change(
                "Treatment A were effective.",
                "Treatment A was effective.", Glossary(),
            ),
            "",
        )

    def test_rewrite_rejects_subject_swap_with_unchanged_word_counts(
        self,
    ) -> None:
        """Rewrites use the same scientific claim-order safeguards.

        """

        from kilin.tools.writing_check._checks import Glossary
        from kilin.tools.writing_check._followup import option_rejection

        original = (
            "Treatment A was effective; treatment B was ineffective."
        )
        changed = (
            "Treatment B was effective; treatment A was ineffective."
        )
        self.assertTrue(
            option_rejection(original, changed, original, Glossary())
        )

if __name__ == "__main__":
    unittest.main()
