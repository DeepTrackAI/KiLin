"""Unit tests for reading a document into sections and paragraphs.

Check LaTeX, Markdown and plain text: section numbers and ranges, the
lines left out as not prose, paragraph boundaries and positions, and the
readable text given to the model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools.writing_check._document import read_document

LATEX = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}
\maketitle
\begin{abstract}
We present a toolbox \cite{a}. It is fast.
\end{abstract}
\section{Introduction}\label{sec:i}
Graph theory describes brains \citep{b,c}.
% a comment inside the paragraph
It has many uses, as \citeauthor{b} showed.

A second paragraph with math $x^2$ and a
reference to Fig.~\ref{f}.
\begin{figure}
\caption{Not prose here.}
\end{figure}
\section{Methods}
\subsection{Data}
\begin{itemize}
\item First item text.
\item Second item
continues here.
\end{itemize}
\begin{equation}
E = mc^2
\end{equation}
After the equation.
\[
a = b
\]
Final words.
\end{document}
Notes after the end.
"""


class TestLatex(unittest.TestCase):

    def setUp(self) -> None:
        self.document = read_document(Path("draft.tex"), LATEX)

    def test_sections(self) -> None:
        self.assertEqual(
            [(s.title, s.number, s.level, s.start_line, s.end_line)
             for s in self.document.sections],
            [("Abstract", "", 1, 5, 7),
             ("Introduction", "1", 1, 8, 17),
             ("Methods", "2", 1, 18, 32),
             ("Data", "2.1", 2, 19, 32)],
        )

    def test_paragraphs(self) -> None:
        self.assertEqual(
            [(p.section, p.position, p.start_line, p.end_line, p.text)
             for p in self.document.paragraphs],
            [(0, 1, 6, 6, "We present a toolbox [cite]. It is fast."),
             (1, 1, 9, 11, "Graph theory describes brains [cite]. It has "
              "many uses, as [author] showed."),
             (1, 2, 13, 14, "A second paragraph with math x^2 and a "
              "reference to Fig. (ref)."),
             (3, 1, 21, 21, "First item text."),
             (3, 2, 22, 23, "Second item continues here."),
             (3, 3, 28, 28, "After the equation."),
             (3, 4, 32, 32, "Final words.")],
        )

    def test_source_lines_are_kept(self) -> None:
        paragraph = self.document.paragraphs[1]
        self.assertEqual(
            paragraph.source,
            "\n".join(LATEX.splitlines()[8:11]),
        )

    def test_what_is_not_prose(self) -> None:
        text = " ".join(p.text for p in self.document.paragraphs)
        for left_out in ("usepackage", "Not prose here", "mc^2",
                         "a = b", "Notes after the end"):
            with self.subTest(left_out=left_out):
                self.assertNotIn(left_out, text)

    def test_chapter_file_without_document_environment(self) -> None:
        document = read_document(
            Path("chapter.tex"),
            "\\chapter{Results}\nWe found it.\n\\section*{Extra}\nMore.\n",
        )
        self.assertEqual(
            [(s.title, s.number) for s in document.sections],
            [("Results", "1"), ("Extra", "")],
        )
        self.assertEqual(
            [p.text for p in document.paragraphs],
            ["We found it.", "More."],
        )


class TestMarkdownAndText(unittest.TestCase):

    def test_markdown(self) -> None:
        document = read_document(Path("notes.md"), (
            "# Title\n\nIntro with a [link](http://x) and *emphasis*.\n\n"
            "## 2.1 Data\n\n- item one\n- item two\n\n```\n# code\n```\n"
            "| a | b |\n\nLast paragraph.\n"
        ))
        self.assertEqual(
            [(s.title, s.number, s.start_line, s.end_line)
             for s in document.sections],
            [("Title", "", 1, 15), ("Data", "2.1", 5, 15)],
        )
        self.assertEqual(
            [(p.section, p.position, p.text) for p in document.paragraphs],
            [(0, 1, "Intro with a link and emphasis."),
             (1, 1, "item one"), (1, 2, "item two"),
             (1, 3, "Last paragraph.")],
        )

    def test_plain_text(self) -> None:
        document = read_document(Path("a.txt"), "One.\nTwo.\n\nThree.\n")
        self.assertEqual(document.sections, ())
        self.assertEqual(
            [(p.section, p.position, p.start_line, p.end_line, p.text)
             for p in document.paragraphs],
            [(None, 1, 1, 2, "One. Two."), (None, 2, 4, 4, "Three.")],
        )


if __name__ == "__main__":
    unittest.main()
