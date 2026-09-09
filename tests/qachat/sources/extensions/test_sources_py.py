"""Tests for Python natural-unit extraction."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.sources.sources import (
    extract_source_units,
    read_repository_files,
)


class TestSourcesPy(unittest.TestCase):

    def test_python_file_becomes_natural_units(self) -> None:
        source = (
            '"""Example."""\n'
            "\n"
            "def greet():\n"
            "    return 'hello'\n"
            "\n"
            "class Greeter:\n"
            "    name = 'Example'\n"
            "\n"
            "    def reply(self):\n"
            "        return self.name\n"
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "example.py").write_text(source, encoding="utf-8")
            report = scan_repository(root, extensions=[".py"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        names = [(unit.kind, unit.symbol) for unit in units]
        self.assertIn(("python_function", "greet"), names)
        self.assertIn(("python_class", "Greeter"), names)

        class_unit = next(
            unit for unit in units if unit.kind == "python_class"
        )
        self.assertEqual(class_unit.start_line, 6)
        self.assertEqual(class_unit.end_line, 10)
        self.assertEqual(
            "".join(child.text for child in class_unit.children),
            class_unit.text,
        )
        self.assertEqual(
            class_unit.children[-1].symbol,
            "Greeter.reply",
        )

    def test_invalid_python_stays_one_safe_unit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.py").write_text(
                "def broken(:\n",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".py"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "python_file")
        self.assertEqual(units[0].path, Path("broken.py"))


if __name__ == "__main__":
    unittest.main()
