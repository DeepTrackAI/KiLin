"""Unit tests for LaTeX section extraction.

The extractor splits the document text at sectioning commands, leaves out
the setup before ``\\begin{document}``, and ignores commands in comments
and verbatim environments. Tests preserve this lightweight contract.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.sources.extensions import sources_tex
from kilin.sources.sources import SourceFile
from tests.kilin.sources._support import fixture_text


def _units(text: str, name: str = "draft.tex") -> list:
    """Return the units of LaTeX text."""
    return sources_tex.extract_source_units(SourceFile(Path(name), text))


class TestLatexExtraction(unittest.TestCase):
    """Verify section boundaries, names, and provenance."""

    def test_fixture_front_matter_and_sections(self) -> None:
        text = fixture_text("sample.tex")
        units = _units(text, "sample.tex")
        self.assertEqual(
            [unit.kind for unit in units],
            ["latex_front"] + ["latex_section"] * 3,
        )
        self.assertEqual(
            [unit.section for unit in units],
            [None, "Introduction", "Graph theory measures",
             "Methods and materials"],
        )
        self.assertEqual(
            [(unit.start_line, unit.end_line) for unit in units],
            [(7, 11), (12, 15), (16, 21), (22, 24)],
        )
        # Units keep the source text as written.
        lines = text.splitlines(keepends=True)
        for unit in units:
            self.assertEqual(
                unit.text,
                "".join(lines[unit.start_line - 1:unit.end_line]),
            )

    def test_setup_and_text_after_the_document_are_left_out(self) -> None:
        text = fixture_text("sample.tex")
        joined = "".join(unit.text for unit in _units(text, "sample.tex"))
        self.assertNotIn("\\usepackage", joined)
        self.assertNotIn("\\end{document}", joined)
        self.assertNotIn("Notes after the end", joined)
        self.assertIn("We present", joined)

    def test_file_without_document_environment_is_all_text(self) -> None:
        text = "Opening text.\n\\chapter{Results}\nWe found it.\n"
        units = _units(text, "chapter2.tex")
        self.assertEqual(
            [(unit.kind, unit.section) for unit in units],
            [("latex_front", None), ("latex_section", "Results")],
        )
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_commented_and_verbatim_commands_are_not_sections(self) -> None:
        text = (
            "% \\section{Commented}\n"
            "Text.\n"
            "\\begin{lstlisting}\n\\section{Code}\n\\end{lstlisting}\n"
            "  \\section*{Real}\n"
            "Body.\n"
        )
        units = _units(text)
        self.assertEqual(
            [unit.section for unit in units], [None, "Real"]
        )
        self.assertEqual(units[1].start_line, 6)

    def test_section_names_are_plain_words(self) -> None:
        for source, name in (
            ("\\section{The \\emph{BRAPH} tool}", "The BRAPH tool"),
            ("\\subsection{Results~and \\LaTeX{} notes}",
             "Results and notes"),
            ("\\section{Data\\label{sec:data}}", "Data"),
            ("\\subsubsection[Short]{Long {nested} title}",
             "Long nested title"),
        ):
            with self.subTest(source=source):
                units = _units(f"{source}\nText.\n")
                self.assertEqual(units[0].section, name)

    def test_unclosed_title_does_not_fail(self) -> None:
        units = _units("\\section{Broken\nText.\n")
        self.assertEqual(units[0].kind, "latex_section")
        self.assertEqual(units[0].section, "Broken")

    def test_empty_file(self) -> None:
        self.assertEqual(_units(""), [])

    def test_title(self) -> None:
        self.assertEqual(
            sources_tex.latex_title(fixture_text("sample.tex")),
            "The BRAPH 2 toolbox",
        )
        self.assertIsNone(sources_tex.latex_title("% \\title{Old}\nText"))
        self.assertIsNone(sources_tex.latex_title("\\title{}"))
        self.assertEqual(
            sources_tex.latex_title(
                "Text % \\title{Wrong}\n\\title{Actual title}\n"
            ),
            "Actual title",
        )


if __name__ == "__main__":
    unittest.main()
