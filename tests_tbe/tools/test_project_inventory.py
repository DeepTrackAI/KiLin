"""Check project_inventory: exact file counts, filters, and coverage."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.tools import ToolRequest
from qachat.tools.project_inventory import (
    InventorySettings,
    file_types_in,
    run,
)
from tests.qachat.tools.helpers import chunk, context


class TestProjectInventory(unittest.TestCase):
    """Counts come from the manifest and index, never from a model."""

    def setUp(self) -> None:
        names = [f"repo/papers/p{n}.pdf" for n in range(7)]
        names += ["repo/notes/a.md", "repo/notes/b.md", "repo/scan.pdf"]
        self.sizes = {Path(name): 2048 for name in names}
        # Every file but the scanned PDF has text; p0 has two parts.
        self.chunks = [chunk(name, ["text"]) for name in names[:-1]]
        self.chunks.append(chunk("repo/papers/p0.pdf", ["more"], start=2))

    def ask(self, question: str):
        built, events = context(self.chunks, sizes=self.sizes)
        request = ToolRequest(question, question)
        result = run(InventorySettings(), request, built)
        return result, events

    def test_known_files_are_counted_exactly(self) -> None:
        result, events = self.ask("how many papers are in this project?")
        self.assertIn(
            "**10 source files**; 9 have searchable text", result.answer
        )
        self.assertIn("- PDF: 8\n- Markdown: 2", result.answer)
        self.assertIn("- papers: 7 (7 PDF)", result.answer)
        self.assertIn("**No searchable text**\n\n- scan.pdf", result.answer)
        self.assertEqual(result.summary["files"], 10)
        self.assertEqual(
            result.summary["unsearchable_files"], ["repo/scan.pdf"]
        )
        self.assertEqual(result.evidence, ())
        self.assertEqual(events[-1][:2], ("done", "Counted 10 files"))

    def test_list_narrowed_by_type_and_folder(self) -> None:
        result, _ = self.ask("list the PDFs in papers")
        self.assertIn("**7 PDF files in papers**", result.answer)
        self.assertIn("- p0.pdf (2 parts, 2.0 kB)", result.answer)
        self.assertNotIn("- a.md", result.answer)
        self.assertEqual(result.summary["selected"], 7)

    def test_file_type_words(self) -> None:
        self.assertEqual(file_types_in("list the PDFs"), (".pdf",))
        self.assertEqual(file_types_in("python scripts?"), (".py", ".m"))
        self.assertEqual(file_types_in("show .md files"), (".md",))
        self.assertEqual(file_types_in("how many papers?"), ())
        self.assertEqual(file_types_in("which have no searchable text?"), ())
        self.assertEqual(file_types_in("list the text files"), (".txt",))

    def test_unreadable_question_lists_only_those_files(self) -> None:
        result, _ = self.ask("which files have no searchable text?")
        self.assertIn("**No searchable text**\n\n- scan.pdf", result.answer)
        self.assertNotIn("p3.pdf", result.answer)
        self.assertTrue(result.summary["asked_about_unreadable"])


if __name__ == "__main__":
    unittest.main()
