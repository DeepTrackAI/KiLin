"""Unit tests for choosing the part of a document a question asks for.

Check the whole document, quoted text, line ranges, paragraph positions,
section numbers and names (loosely matched), and the problems returned
when the question names a part the document does not have.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools.writing_check._document import read_document
from kilin.tools.writing_check._scope import (
    Scope,
    ScopeProblem,
    choose_scope,
)

LATEX = r"""\begin{document}
\begin{abstract}
We present a toolbox. It is fast and robust for analysis.
\end{abstract}
\section{Introduction}
First intro paragraph about graphs.

Second intro paragraph about brains.

Third intro paragraph.
\section{Materials and methods}
Methods paragraph one.
\subsection{Data analysis}
Data paragraph.
\section{Results}
Results paragraph.
\end{document}
"""
HEADINGS = (
    "Abstract", "1 Introduction", "2 Materials and methods",
    "2.1 Data analysis", "3 Results",
)


class TestChooseScope(unittest.TestCase):

    def setUp(self) -> None:
        self.document = read_document(Path("draft.tex"), LATEX)

    def _scope(self, question: str) -> Scope:
        scope = choose_scope(question, self.document)
        self.assertIsInstance(scope, Scope)
        assert isinstance(scope, Scope)
        return scope

    def test_whole_document(self) -> None:
        for question in ("check draft.tex",
                         "check the grammar of the whole manuscript",
                         "check grammar and style of methods.tex"):
            with self.subTest(question=question):
                scope = self._scope(question)
                self.assertEqual(scope.paragraphs, tuple(range(7)))
                self.assertEqual(scope.label, "the whole document")

    def test_sections(self) -> None:
        for question, paragraphs, label in (
            ("check section 2.1", (5,), "section 2.1 Data analysis"),
            ("check §2", (4, 5), "section 2 Materials and methods"),
            ("check the methods of draft.tex", (4, 5),
             "section 2 Materials and methods"),
            ("proofread the results", (6,), "section 3 Results"),
            ("check the abstract", (0,), 'section "Abstract"'),
            ("check the data section", (5,), "section 2.1 Data analysis"),
            ("check the section on data analysis", (5,),
             "section 2.1 Data analysis"),
            ("check the style of the data", (5,),
             "section 2.1 Data analysis"),
        ):
            with self.subTest(question=question):
                scope = self._scope(question)
                self.assertEqual(
                    (scope.paragraphs, scope.label), (paragraphs, label)
                )

    def test_paragraph_positions(self) -> None:
        for question, paragraphs, label in (
            ("check the second paragraph of the introduction", (2,),
             "the second paragraph of section 1 Introduction"),
            ("check paragraph 2 of section 1", (2,),
             "paragraph 2 of section 1 Introduction"),
            ("check the last paragraph", (6,),
             "the last paragraph of the document"),
            ("check the 2nd paragraph of the methods", (5,),
             "the 2nd paragraph of section 2 Materials and methods"),
        ):
            with self.subTest(question=question):
                scope = self._scope(question)
                self.assertEqual(
                    (scope.paragraphs, scope.label), (paragraphs, label)
                )

    def test_lines(self) -> None:
        self.assertEqual(self._scope("check lines 6-8").paragraphs, (1, 2))
        self.assertEqual(self._scope("check lines 8 to 6").label,
                         "lines 6–8")
        self.assertEqual(self._scope("check line 12").paragraphs, (4,))
        self.assertEqual(self._scope("check line 2").paragraphs, ())

    def test_quoted_text(self) -> None:
        scope = self._scope('check this: "Second intro paragraph about '
                            'brains"')
        self.assertEqual((scope.paragraphs, scope.text), ((2,), ""))
        self.assertEqual(scope.label, "the quoted text (line 8)")
        scope = self._scope(
            "check this sentence: the results is very clear and robust"
        )
        self.assertEqual(scope.paragraphs, ())
        self.assertEqual(scope.text, "the results is very clear and robust")
        # Too short to be a passage: the colon is ignored.
        self.assertEqual(self._scope("check: grammar only").label,
                         "the whole document")

    def test_problems(self) -> None:
        for question, message in (
            ("check section 4", "There is no section 4."),
            ("check the discussion section",
             "There is no section called “discussion”."),
            ("check the tenth paragraph of the introduction",
             "Section 1 Introduction has 3 paragraphs."),
        ):
            with self.subTest(question=question):
                problem = choose_scope(question, self.document)
                self.assertIsInstance(problem, ScopeProblem)
                assert isinstance(problem, ScopeProblem)
                self.assertEqual(problem.message, message)
                self.assertEqual(problem.headings, HEADINGS)

    def test_ambiguous_name(self) -> None:
        document = read_document(
            Path("notes.md"),
            "## Data sources\n\nA.\n\n## Data cleaning\n\nB.\n",
        )
        problem = choose_scope("check the data section", document)
        self.assertIsInstance(problem, ScopeProblem)
        assert isinstance(problem, ScopeProblem)
        self.assertEqual(problem.headings, ("Data sources", "Data cleaning"))


if __name__ == "__main__":
    unittest.main()
