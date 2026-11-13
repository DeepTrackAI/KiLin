"""Verify that saved Word documents retain citable output passages.

Output files are separate from indexed sources but share their readers,
unit extraction and chunk coordinates. The stable Word fixture exercises
that contract using a document converted independently of python-docx.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from kilin.tools._shared.outputs import load_output

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "sources/extensions/fixtures/sample.docx"
)


class TestWordOutput(unittest.TestCase):
    """Check output file reading and provenance for Word documents."""

    def test_docx_output_keeps_text_and_relative_provenance(self) -> None:
        """Load an actual docx into passages suitable for tool citations."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            shutil.copyfile(FIXTURE, outputs / "sample.docx")
            document = load_output(root, "sample.docx")

        self.assertEqual(document.path, Path("outputs/sample.docx"))
        self.assertGreater(document.estimated_tokens, 0)
        self.assertTrue(document.passages)
        self.assertTrue(all(
            chunk.path == document.path
            and chunk.identifier.startswith("outputs:")
            and chunk.start_line >= 1
            and chunk.end_line >= chunk.start_line
            for chunk in document.passages
        ))
        contents = "\n".join(chunk.text for chunk in document.passages)
        self.assertIn("café and naïve", contents)
        self.assertIn("Name | Value", contents)

    @unittest.skipIf(find_spec("olefile") is None, "olefile unavailable")
    def test_doc_output_uses_the_same_provenance_contract(self) -> None:
        """Read the legacy binary fixture through the output pipeline."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            legacy = FIXTURE.with_suffix(".doc")
            shutil.copyfile(legacy, outputs / "sample.doc")
            document = load_output(root, "sample.doc")

        self.assertEqual(document.path, Path("outputs/sample.doc"))
        self.assertGreater(len(document.passages), 0)
        self.assertIn(
            "café and naïve",
            "\n".join(chunk.text for chunk in document.passages),
        )
        self.assertTrue(all(
            chunk.path == document.path and chunk.start_line >= 1
            for chunk in document.passages
        ))



if __name__ == "__main__":
    unittest.main()
