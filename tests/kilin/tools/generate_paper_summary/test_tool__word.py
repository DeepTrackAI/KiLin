"""Check the paper-summary tool's Word selection and citation behavior.

Use the stable Word document fixture in ``outputs/``. A real metadata
property, rather than the file name, must supply the citation; missing
properties must preserve the explicit filename fallback.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from docx import Document

from kilin.tools._contract import ToolRequest
from kilin.tools.generate_paper_summary.tool import (
    choose_paper,
    paper_citation,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "sources/extensions/fixtures/sample.docx"
)


class TestWordPaperSummary(unittest.TestCase):
    """Verify named Word sources and metadata-based paper citations."""

    def setUp(self) -> None:
        """Prepare a project holding an actual Word document."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        outputs = self.root / "outputs"
        outputs.mkdir()
        self.file = outputs / "sample.docx"
        shutil.copyfile(FIXTURE, self.file)
        self.context = SimpleNamespace(project_root=self.root, chunks=())

    def test_choose_word_output_and_cite_its_properties(self) -> None:
        """Select a named Word document without model inference."""

        chosen = choose_paper(
            ToolRequest("Summarize sample.docx", "Summarize sample.docx"),
            self.context,
        )
        self.assertNotIsInstance(chosen, str)
        path, passages, reason, guessed = chosen
        self.assertEqual(path, Path("outputs/sample.docx"))
        self.assertEqual((reason, guessed), ("named in the question", False))
        self.assertTrue(passages)
        citation = paper_citation(self.root / path, passages)
        self.assertEqual(
            citation,
            "Lovelace, A. A Sample Study of Word Documents.",
        )

    def test_docx_without_title_uses_file_name(self) -> None:
        """Never invent bibliographic metadata for a Word document."""

        missing = self.root / "outputs" / "untitled.docx"
        Document().save(missing)
        self.assertEqual(
            paper_citation(missing, ()),
            "untitled.docx (the file has no title metadata).",
        )

    @unittest.skipIf(find_spec("olefile") is None, "olefile unavailable")
    def test_named_legacy_doc_uses_the_real_reader(self) -> None:
        """Select a binary Word paper through the normal output path."""

        legacy = self.file.with_suffix(".doc")
        shutil.copyfile(FIXTURE.with_suffix(".doc"), legacy)
        chosen = choose_paper(
            ToolRequest("Summarize sample.doc", "Summarize sample.doc"),
            self.context,
        )
        self.assertNotIsInstance(chosen, str)
        path, passages, reason, guessed = chosen
        self.assertEqual(path, Path("outputs/sample.doc"))
        self.assertTrue(passages)
        self.assertEqual((reason, guessed), ("named in the question", False))

    def test_doc_uses_ole_metadata_when_available(self) -> None:
        """Accept legacy Word metadata through the shared reader."""

        doc = self.root / "outputs" / "old.doc"
        doc.write_bytes(b"old document")
        with patch(
            "kilin.sources._word_metadata.read_word_metadata",
            return_value=("A Historical Document Study", "Grace Hopper"),
        ):
            self.assertEqual(
                paper_citation(doc, ()),
                "Hopper, G. A Historical Document Study.",
            )


if __name__ == "__main__":
    unittest.main()
