"""Tests for clean, standalone tutorial notebooks."""

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

    def test_notebooks_use_shared_code(self) -> None:
        expected = {
            "10_setup.ipynb": "from setup.check_setup import check_setup",
            "20_huggingface_models.ipynb": (
                "from qachat.huggingface import download_model"
            ),
            "30_model_chat.ipynb": "from qachat.chat import ChatSession",
            "40_repository_files.ipynb": (
                "from qachat.repository import scan_repository"
            ),
        }
        forbidden = (
            "app_core",
            "scripts/",
            "QACHAT_MODEL_HOME",
            "MODEL_DIRECTORY",
        )
        for name, required in expected.items():
            with self.subTest(name=name):
                path = ROOT / "tutorials" / name
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source = "\n".join(
                    "".join(cell["source"]) for cell in notebook["cells"]
                )
                self.assertIn(required, source)
                self.assertIn(".venv", source)
                for old_name in forbidden:
                    self.assertNotIn(old_name, source)


if __name__ == "__main__":
    unittest.main()
