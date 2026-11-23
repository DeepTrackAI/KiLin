"""Unit tests for reducing LaTeX lines to readable text.

Check comments, citations through the caller's function, math brackets,
cross-references, accents, formatting commands and section titles.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools._shared.latex_text import (
    LATEX_CITE,
    NOT_CITING,
    readable_latex,
)


def _keys(match) -> str:
    """Write a citation as its keys, to see what the caller receives."""
    return f"<{match['name']}:{match['keys']}>"


class TestReadableLatex(unittest.TestCase):

    def test_lines(self) -> None:
        for line, readable in (
            ("As \\citep[see][p.~2]{a, b}.", "As <citep:a, b>."),
            ("Rate 5\\% % \\cite{x} commented", "Rate 5% "),
            ("Range $[0,1]$ here.", "Range (0,1) here."),
            ("Na\\\"{i}ve \\emph{bold} Fig.~\\ref{f}",
             "Naive bold Fig. (ref)"),
            ("\\section{The \\emph{X} tool}\\label{s}", "The X tool."),
            ("\\textbf{Bold} \\v{c}a \\bibitem", "Bold ca "),
            ("vs.\\ 81 in ($15\\times15\\,\\mu$m) at 37$^\\circ$C",
             "vs. 81 in (15×15 µm) at 37°C"),
            ("one stage --- a change --- in Fig.~\\ref{f}b--g, $x^2$",
             "one stage — a change — in Fig. (ref)b–g, x^2"),
            ("``quoted''", "“quoted”"),
        ):
            with self.subTest(line=line):
                self.assertEqual(readable_latex(line, _keys), readable)

    def test_citation_commands(self) -> None:
        names = [
            match["name"] for match in LATEX_CITE.finditer(
                "\\cite{a} \\parencite*{b} \\textcite[p.~1]{c} \\nocite{d}"
            )
        ]
        self.assertEqual(names, ["cite", "parencite", "textcite", "nocite"])
        self.assertIn("nocite", NOT_CITING)
        self.assertNotIn("cite", NOT_CITING)


if __name__ == "__main__":
    unittest.main()
