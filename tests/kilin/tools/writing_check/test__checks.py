"""Unit tests for the checks writing_check does in code.

Check each finding kind, the glossary given to the model, the abstract
read on its own for acronyms, and the source line of each finding.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools.writing_check import _configuration
from kilin.tools.writing_check._checks import (
    LEVEL_CONSISTENCY,
    LEVEL_GRAMMAR,
    LEVEL_STYLE,
    Glossary,
    _initials_match,
    check_document,
)
from kilin.tools.writing_check._document import read_document

LATEX = r"""\begin{document}
\begin{abstract}
Single-particle tracking (SPT) reveals diffusion. A novel SPT toolbox.
\end{abstract}
\section{Introduction}
SPT is widely used to study the the motion of molecules \cite{a}.
Single-particle tracking (SPT) is a technique.
Again, single-particle tracking (SPT) is defined.
We use graph theory (GT) to analyse the data set and the DSA method.
\section{Methods}
We analyzed each dataset; the dataset was normalised in colour.
\end{document}
"""


def _summary(findings) -> list[tuple[int, str, str, str, str]]:
    """Return the line, level, check, words and suggestion of findings."""
    return [
        (f.line, f.level, f.check, f.original, f.suggestion)
        for f in findings
    ]


class TestChecks(unittest.TestCase):

    def setUp(self) -> None:
        self.findings, self.glossary = check_document(
            read_document(Path("draft.tex"), LATEX)
        )

    def test_findings(self) -> None:
        self.assertEqual(_summary(self.findings), [
            (3, LEVEL_STYLE, "promotional word", "novel", ""),
            (6, LEVEL_CONSISTENCY, "acronym", "SPT", ""),
            (6, LEVEL_GRAMMAR, "repeated word", "the the", "the"),
            (8, LEVEL_CONSISTENCY, "acronym",
             "single-particle tracking (SPT)", "SPT"),
            (9, LEVEL_CONSISTENCY, "term", "data set", "dataset"),
            (9, LEVEL_CONSISTENCY, "acronym", "DSA", ""),
            (11, LEVEL_CONSISTENCY, "spelling", "analyzed", "analysed"),
        ])
        reasons = {f.original: f.reason for f in self.findings}
        self.assertEqual(
            reasons["SPT"],
            "SPT is used before its definition (line 7); define it here.",
        )
        self.assertIn("already defined (line 7)",
                      reasons["single-particle tracking (SPT)"])
        self.assertIn("never defined", reasons["DSA"])

    def test_glossary(self) -> None:
        self.assertEqual(self.glossary.acronyms, {
            "SPT": "Single-particle tracking", "GT": "graph theory",
        })
        self.assertEqual(self.glossary.terms, {"data set": "dataset"})
        self.assertEqual(self.glossary.spelling, "UK")
        self.assertEqual(self.glossary.describe(), (
            "Acronyms defined in the document: SPT = Single-particle "
            "tracking; GT = graph theory.\n"
            "Terms the document writes this way: “dataset”.\n"
            "Spelling: UK English."
        ))
        self.assertEqual(Glossary().describe(), "")

    def test_abstract_is_read_on_its_own(self) -> None:
        # The abstract defines SPT and uses it: nothing to report there,
        # and its definition does not count for the body.
        lines = [f.line for f in self.findings if f.check == "acronym"]
        self.assertNotIn(3, lines)
        self.assertIn(6, lines)

    def test_initials(self) -> None:
        for words, short, expected in (
            ("rate of change", "ROC", True),
            ("signal-to-noise ratio", "SNR", True),
            ("the magnetic resonance", "MR", True),
            ("graph theory", "GTX", False),
            ("magnetic resonance imaging", "fMRI", False),
        ):
            with self.subTest(words=words):
                self.assertEqual(_initials_match(words, short), expected)

    def test_known_acronyms_and_reversed_definitions(self) -> None:
        findings, glossary = check_document(read_document(
            Path("a.txt"),
            "We used MRI and DNA. MSD (mean squared displacement) was low."
            " The MSD grew.",
        ))
        self.assertEqual(findings, [])
        self.assertEqual(glossary.acronyms,
                         {"MSD": "mean squared displacement"})

    def test_ties_and_allowed_repeats_are_not_reported(self) -> None:
        findings, glossary = check_document(read_document(
            Path("a.txt"),
            "The data set and the dataset. We said that that was fine. "
            "The color and the colour.",
        ))
        self.assertEqual(findings, [])
        self.assertEqual((glossary.terms, glossary.spelling), ({}, ""))

    def test_long_sentences(self) -> None:
        limit = _configuration.LONG_SENTENCE_WORDS
        words = [("alpha", "beta", "gamma")[i % 3] for i in range(limit + 1)]
        words[0] = "Alpha"  # A sentence starts with a capital.
        text = "Short one. " + " ".join(words) + "."
        findings, _ = check_document(read_document(Path("a.txt"), text))
        self.assertEqual(
            _summary(findings),
            [(1, LEVEL_STYLE, "long sentence", " ".join(words[:8]), "")],
        )
        self.assertEqual(findings[0].reason,
                         f"{limit + 1} words; consider splitting it.")


class TestConfiguration(unittest.TestCase):

    def test_spelling_pairs_are_us_then_uk(self) -> None:
        pairs = dict(_configuration.SPELLING_PAIRS)
        self.assertEqual(pairs["color"], "colour")
        self.assertEqual(pairs["analyze"], "analyse")
        self.assertIn("mri", _configuration.KNOWN_ACRONYMS)

    def test_bad_pairs_are_rejected(self) -> None:
        for entry in ("color", "color:color", "a:b:c", ":colour"):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    _configuration._pairs((entry,))


if __name__ == "__main__":
    unittest.main()
