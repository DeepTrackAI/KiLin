"""Verify LaTeX bibliography access uses selected workspace sources only.

Real reference-check runs exercise manifest authorization, local-first
resolution, ambiguous names, parent traversal, symlinks and compiled
bibliographies. No model or network service is invoked.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.tools._contract import ToolRequest
from kilin.tools.reference_check import tool
from tests.kilin.tools._shared._support import chunk


class TestSelectedBibliographies(unittest.TestCase):
    """Verify bibliography resolution in the complete tool workflow."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.root = self.folder / "workspace"
        self.sources = self.root / "sources"
        self.sources.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _file(self, path: str, text: str) -> None:
        file = self.sources / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")

    def _check(
        self, paper: str,
        selected: set[str],
        declaration: str = r"\bibliography{refs}",
        extra_chunks: list[object] | None = None,
        select_paper: bool = True,
    ) -> object:
        source = (
            r"\begin{document}" + "\n"
            r"Our observations match a method \cite{smith}." + "\n"
            + declaration + "\n"
            + r"\end{document}" + "\n"
        )
        self._file(paper, source)
        passages = [chunk("c1", source, path=paper)]
        passages.extend(extra_chunks or [])
        context = SimpleNamespace(
            project_root=self.root,
            source_root=self.sources,
            source_files=lambda: {
                Path(name): 100
                for name in selected | ({paper} if select_paper else set())
            },
            chunks=passages,
            rank=Mock(return_value=[]),
            progress=Mock(), done=Mock(),
            trace=SimpleNamespace(emit=Mock()), trace_turn_id=None,
            choose=None,
        )
        request = ToolRequest(
            f"check the references of {paper}",
            f"check the references of {paper}",
        )
        with patch(
            "kilin.tools.reference_check.tool.choose_document",
            return_value=(Path(paper), "named", False),
        ):
            return tool.run(
                tool.ReferenceCheckSettings(verify=False),
                request, context,
            )

    def test_two_papers_prefer_same_directory_bibliographies(self) -> None:
        self._file("one/refs.bib", "@article{smith, title={One}}\n")
        self._file("two/refs.bib", "@article{smith, title={Two}}\n")
        chosen = {"one/refs.bib", "two/refs.bib"}
        first = self._check("one/paper.tex", chosen)
        second = self._check("two/paper.tex", chosen)
        self.assertEqual(first.summary["references"], 1)
        self.assertEqual(second.summary["references"], 1)
        self.assertIn("one/refs.bib", first.summary["source_note"])
        self.assertIn("two/refs.bib", second.summary["source_note"])
        self.assertNotIn("two/refs.bib", first.summary["source_note"])

    def test_unselected_bibliography_is_not_read(self) -> None:
        self._file("paper/refs.bib", "@article{smith, title={Private}}")
        result = self._check("paper/paper.tex", set())
        self.assertEqual(result.summary["references"], 0)
        self.assertIn("unavailable", result.summary["source_note"])

    def test_unique_selected_basename_fallback(self) -> None:
        self._file("shared/refs.bib", "@article{smith, title={Shared}}")
        result = self._check(
            "one/paper.tex", {"shared/refs.bib"}
        )
        self.assertEqual(result.summary["references"], 1)
        self.assertIn("shared/refs.bib", result.summary["source_note"])

    def test_ambiguous_basename_does_not_guess(self) -> None:
        self._file("one/refs.bib", "@article{smith, title={One}}")
        self._file("two/refs.bib", "@article{smith, title={Two}}")
        result = self._check(
            "other/paper.tex", {"one/refs.bib", "two/refs.bib"}
        )
        self.assertEqual(result.summary["references"], 0)
        self.assertIn("ambiguous", result.summary["source_note"])

    def test_declared_parent_path_within_sources(self) -> None:
        self._file("shared/refs.bib", "@article{smith, title={Shared}}")
        result = self._check(
            "paper/main.tex", {"shared/refs.bib"},
            r"\bibliography{../shared/refs}",
        )
        self.assertEqual(result.summary["references"], 1)

    def test_traversal_does_not_read_external_file(self) -> None:
        (self.folder / "secret.bib").write_text(
            "@article{smith, title={Private}}", encoding="utf-8"
        )
        result = self._check(
            "paper.tex", set(), r"\bibliography{../../secret}"
        )
        self.assertEqual(result.summary["references"], 0)
        self.assertIn("outside sources", result.summary["source_note"])

    def test_symlinked_selected_bibliography_is_rejected(self) -> None:
        external = self.folder / "private.bib"
        external.write_text(
            "@article{smith, title={Private}}", encoding="utf-8"
        )
        (self.sources / "refs.bib").symlink_to(external)
        result = self._check("paper.tex", {"refs.bib"})
        self.assertEqual(result.summary["references"], 0)

    def test_workspace_root_cannot_shadow_selected_paper(self) -> None:
        self._file("refs.bib", "@article{smith, title={Selected}}")
        (self.root / "paper.tex").write_text(
            "Other content with no citations", encoding="utf-8"
        )
        result = self._check("paper.tex", {"refs.bib"})
        self.assertEqual(result.summary["references"], 1)

    def test_stale_indexed_paper_is_not_selected(self) -> None:
        result = self._check(
            "paper.tex", set(), select_paper=False,
        )
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertIn("no longer", result.answer)
        self.assertEqual(result.evidence, ())

    def test_stale_unselected_cited_paper_is_not_considered(self) -> None:
        selected = {"refs.bib"}
        self._file(
            "refs.bib", "@article{smith, title={A study}}\n",
        )
        extra = [chunk(
            "other", "A study by Smith with matching reference text.",
            path="unselected.pdf",
        )]
        with patch(
            "kilin.tools.reference_check.tool.paper_identity"
        ) as identity:
            self._check(
                "paper.tex", selected, extra_chunks=extra,
            )
        identity.assert_not_called()

    def test_indexed_bib_is_not_treated_as_the_cited_paper(self) -> None:
        metadata = (
            "@article{smith, author={Smith, A.}, "
            "title={A study}, year={2025}, "
            "doi={10.1111/example}}\n"
        )
        self._file("refs.bib", metadata)
        result = self._check(
            "paper.tex", {"refs.bib"},
            extra_chunks=[chunk("b1", metadata, path="refs.bib")],
        )
        self.assertEqual(result.summary["references"], 1)
        self.assertEqual(result.summary["found"], 0)
        self.assertEqual(result.summary["probable"], 0)

    def test_compiled_bibliography_requires_selection(self) -> None:
        self._file(
            "paper/paper.bbl",
            "\\begin{thebibliography}{9}\n"
            "\\bibitem{smith} A. Smith, Important study (2025).\n"
            "\\end{thebibliography}\n",
        )
        absent = self._check("paper/paper.tex", set())
        self.assertEqual(absent.summary["references"], 0)
        found = self._check("paper/paper.tex", {"paper/paper.bbl"})
        self.assertEqual(found.summary["references"], 1)
        self.assertIn("paper/paper.bbl", found.summary["source_note"])


if __name__ == "__main__":
    unittest.main()
