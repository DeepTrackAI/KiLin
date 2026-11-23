"""Unit tests for rewriting a sentence a writing check flagged.

Check how a rewrite request and the earlier answer's findings are read,
which text is rewritten, the prompt, and which rewrites are dropped.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools.writing_check._checks import Glossary
from kilin.tools.writing_check._document import read_document
from kilin.tools.writing_check._followup import (
    option_rejection,
    parse_options,
    rewrite_prompt,
    rewrite_request,
    shown_findings,
    target_text,
)

ANSWER = """**Writing check of ms.tex** (the whole document)

1 grammar, 1 style findings.

- line 3 · grammar: “We tracks” → “We tracked” (agreement) [1]
- line 5 · style: “DeepTrack2 complements platforms like BiaPy, CellProfiler, \
and” (47 words; consider splitting it) [2]
"""
EXTRACTED_ANSWER = (
    "**Writing check of paper.pdf** (the whole document)\n\n"
    "- extracted-text line 5 · style: “DeepTrack2 complements "
    "platforms like BiaPy, CellProfiler, and” "
    "(47 words; consider splitting it) [1]\n"
)
LONG = (
    "DeepTrack2 complements platforms like BiaPy, CellProfiler, and "
    "ilastik as their upstream counterpart, supplying the physically "
    "grounded, exactly annotated data and differentiable forward models "
    "that such pipelines depend on but rarely generate themselves; "
    "resulting models are natural candidates for community repositories "
    "such as the BioImage Model Zoo [cite]."
)
TEXT = (
    "# Results\n\nWe tracks particles. It works.\n\nIt is short. "
    + LONG.replace("[cite]", "") + " More text follows here.\n"
)


class TestRequests(unittest.TestCase):

    def test_rewrite_request(self) -> None:
        for question, line in (
            ("suggest a better phrasing for line 75", 75),
            ("split the sentence at line 12", 12),
            ("line 75 · style: “DeepTrack2 …” (47 words)\n\nrephrase it", 75),
            ("rephrase extracted-text line 75 of paper.pdf", 75),
            ("extracted-text line 75 · style: text\n\nsplit it", 75),
            ("what does line 3 say?", None),
            ("rephrase the previous answer", None),
        ):
            with self.subTest(question=question):
                self.assertEqual(rewrite_request(question), line)

    def test_shown_findings(self) -> None:
        found = shown_findings(ANSWER)
        self.assertEqual(
            [(f.line, f.level, f.original, f.suggestion, f.long_sentence)
             for f in found],
            [(3, "grammar", "We tracks", "We tracked", False),
             (5, "style", "DeepTrack2 complements platforms like BiaPy, "
              "CellProfiler, and", "", True)],
        )
        self.assertEqual(shown_findings("An ordinary answer."), [])
        extracted = shown_findings(EXTRACTED_ANSWER)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].line, 5)
        self.assertTrue(extracted[0].long_sentence)


class TestTarget(unittest.TestCase):

    def setUp(self) -> None:
        self.document = read_document(Path("ms.md"), TEXT)
        self.findings = shown_findings(ANSWER)

    def test_whole_sentence_of_the_finding(self) -> None:
        index, text, finding = target_text(
            self.document, 5, self.findings, "split line 5"
        )
        self.assertEqual(index, 1)
        self.assertTrue(text.startswith("DeepTrack2 complements"))
        self.assertTrue(text.endswith("BioImage Model Zoo ."))
        self.assertTrue(finding.long_sentence)

    def test_paragraph_without_a_finding(self) -> None:
        index, text, finding = target_text(
            self.document, 3, [], "rephrase line 3"
        )
        self.assertEqual((index, text, finding),
                         (0, "We tracks particles. It works.", None))
        self.assertIsNone(target_text(self.document, 99, [], "line 99"))

    def test_extracted_text_finding_targets_sentence_not_paragraph(
        self,
    ) -> None:
        """PDF/Word findings retain their sentence when pasted back.

        """

        index, text, finding = target_text(
            self.document, 5, shown_findings(EXTRACTED_ANSWER),
            "split extracted-text line 5",
        )
        self.assertEqual(index, 1)
        self.assertTrue(text.startswith("DeepTrack2 complements"))
        self.assertFalse(text.startswith("It is short"))
        self.assertTrue(finding.long_sentence)


class TestOptions(unittest.TestCase):

    def test_prompt(self) -> None:
        finding = shown_findings(ANSWER)[1]
        system, user = rewrite_prompt(LONG, "Para " + LONG, "Title: T",
                                      finding)
        self.assertIn("rewrite", system)
        for part in ("split it into two or three sentences",
                     "Paragraph, for reference only", '"""\n' + LONG,
                     "never shortened", "OPTION:"):
            with self.subTest(part=part):
                self.assertIn(part, user)

    def test_parse(self) -> None:
        self.assertEqual(
            parse_options('Here:\nOPTION: "First."\n2. Second.\nOPTION: '
                          "First.\nOPTION: Third.\nOPTION: Fourth."),
            ["First.", "Second.", "Third."],
        )

    def test_rejections(self) -> None:
        text = "It is used by 12 groups [cite], and could help many labs."
        paragraph = text + " Labs share data."
        for option, why in (
            ("It is used by 12 groups [cite] and could help many labs.",
             ""),
            ("Twelve groups use it [cite]; it could help labs.",
             "it changes numbers"),
            ("It is used by 12 groups [cite], and...", "it is cut short "
             "with an ellipsis"),
            ("It is used by 12 groups [cite], and helps many labs.",
             "it changes a hedge or modal verb"),
            ("It is used by 12 groups [cite], offering additional "
             "functionality that could help many labs.",
             "it adds words the paragraph does not have: offering, "
             "additional, functionality"),
            (text, "nothing changes"),
        ):
            with self.subTest(option=option):
                self.assertEqual(
                    option_rejection(text, option, paragraph, Glossary()),
                    why,
                )


if __name__ == "__main__":
    unittest.main()
