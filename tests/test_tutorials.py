"""Tests that tutorial notebooks remain clean and usable."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = (
    "10_setup.ipynb",
    "20_huggingface_models.ipynb",
    "30_model_chat.ipynb",
    "40_repository_files.ipynb",
    "50_source_structure.ipynb",
    "60_embeddings.ipynb",
)


class TestTutorials(unittest.TestCase):

    def test_notebooks_are_clean_and_valid(self) -> None:
        for name in TUTORIALS:
            with self.subTest(name=name):
                path = ROOT / "tutorials" / name
                notebook = json.loads(path.read_text(encoding="utf-8"))

                self.assertEqual(notebook["nbformat"], 4)

                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
                self.assertTrue(code_cells)

                for cell in code_cells:
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

                    source = "".join(cell["source"])
                    compile(source, f"{name}:cell", "exec")

                    for line in source.splitlines():
                        self.assertLessEqual(len(line), 79)

    def test_every_tutorial_is_checked(self) -> None:
        tutorial_directory = ROOT / "tutorials"
        paths = tutorial_directory.glob("*.ipynb")
        names = tuple(
            path.name
            for path in sorted(
                paths,
                key=lambda path: int(path.name.split("_", 1)[0]),
            )
        )

        self.assertEqual(names, TUTORIALS)

    def test_notebooks_use_application_functions(self) -> None:
        expected = {
            "10_setup.ipynb": "from setup.check_setup import check_setup",
            "20_huggingface_models.ipynb": (
                "from qachat.huggingface import download_model"
            ),
            "30_model_chat.ipynb": "from qachat.chat import ChatSession",
            "40_repository_files.ipynb": (
                "from qachat.repository import scan_repository"
            ),
            "50_source_structure.ipynb": (
                "from qachat.sources.sources import extract_source_units"
            ),
            "60_embeddings.ipynb": (
                "from qachat.embeddings import EmbeddingSession"
            ),
        }

        for name, required in expected.items():
            with self.subTest(name=name):
                path = ROOT / "tutorials" / name
                notebook = json.loads(path.read_text(encoding="utf-8"))

                source = "\n".join(
                    "".join(cell["source"]) for cell in notebook["cells"]
                )

                self.assertIn(
                    "from setup.notebook import prepare_notebook",
                    source,
                )
                self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
