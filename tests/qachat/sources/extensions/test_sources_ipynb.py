"""Tests for Jupyter notebook natural-unit extraction."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.sources.sources import (
    extract_source_units,
    read_repository_files,
)


class TestSourcesIpynb(unittest.TestCase):

    def test_notebook_file_becomes_visible_cells(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Purpose\n", "Explain the project."],
                },
                {
                    "cell_type": "code",
                    "source": ["value = 1\n"],
                    "outputs": [{"text": ["DO NOT INDEX"]}],
                },
                {
                    "cell_type": "raw",
                    "source": ["ignored"],
                },
            ],
            "metadata": {"secret": "DO NOT INDEX"},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.ipynb").write_text(
                json.dumps(notebook),
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".ipynb"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(
            [unit.kind for unit in units],
            ["notebook_markdown", "notebook_code"],
        )
        self.assertEqual([unit.cell for unit in units], [1, 2])
        self.assertEqual(units[0].section, "Purpose")
        self.assertNotIn(
            "DO NOT INDEX",
            "".join(unit.text for unit in units),
        )

    def test_invalid_notebook_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.ipynb").write_text(
                "{",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".ipynb"])
            files = read_repository_files(report)

            with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
                extract_source_units(files)


if __name__ == "__main__":
    unittest.main()
