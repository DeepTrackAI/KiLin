"""Validate workflow notebook inputs and interpreter metadata.

A literature evaluation notebook must use the checked-in PDF fixtures,
not the parent fixture folder. Check this without executing model cells.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = (
    _ROOT / "tests/workflows/KiLin_Literature_Project_Evaluation.ipynb"
)


class TestLiteratureWorkflowFixture(unittest.TestCase):
    """Keep the standalone evaluation notebook reproducible."""

    def test_pdf_fixture_directory_and_names_exist(self):
        """The imports must point at the actual checked-in PDF folder."""

        notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(
            "".join(item["source"])
            for item in notebook["cells"]
            if "FIXTURE_NAMES = (" in "".join(item["source"])
        )
        self.assertIn(
            'fixture_dir = repo_root / "tests" / "fixtures" / "pdfs"',
            cell,
        )
        fixtures = _ROOT / "tests/fixtures/pdfs"
        for name in (
            "mijalkov17braph.pdf",
            "mijalkov25computational.pdf",
            "patil25deep.pdf",
            "tiunn26braph2.pdf",
            "yao25connecting.pdf",
        ):
            with self.subTest(name=name):
                self.assertTrue((fixtures / name).is_file())
                self.assertIn(name, cell)

    def test_python_version_is_well_formed(self):
        """Notebook metadata must not contain an invalid patch version."""

        notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
        version = notebook["metadata"]["language_info"]["version"]
        self.assertRegex(version, re.compile(r"^3\.12\.\d+$"))


if __name__ == "__main__":
    unittest.main()
