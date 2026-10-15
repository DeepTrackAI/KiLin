"""Unit tests for Python syntax-tree source extraction.

Synthetic Python source is parsed, never imported or executed. Assertions
cover declaration provenance, structural partitioning, and syntax fallback.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.sources.extensions import sources_py
from qachat.sources.sources import SourceFile
from tests.qachat.sources._support import fixture_text


class TestPythonExtraction(unittest.TestCase):
    """Verify Python AST extraction and class-child boundaries."""

    def test_fixture_preserves_declarations_and_context(self) -> None:
        """Retain functions, classes, and intervening source text."""

        text = fixture_text("sample.py")
        units = sources_py.extract_source_units(
            SourceFile(Path("sample.py"), text)
        )
        self.assertEqual(
            [line for unit in units for line in unit.text.splitlines()
             if line.strip()],
            [line for line in text.splitlines() if line.strip()],
        )
        self.assertIn(
            ("python_function", "calculate"),
            {(unit.kind, unit.symbol) for unit in units},
        )
        self.assertIn(
            ("python_function", "fetch"),
            {(unit.kind, unit.symbol) for unit in units},
        )
        self.assertIn(
            ("python_class", "Example"),
            {(unit.kind, unit.symbol) for unit in units},
        )
        self.assertTrue(any(unit.kind == "python_module_context"
                            for unit in units))

    def test_decorated_function_starts_at_decorator(self) -> None:
        """Include decorator lines in top-level function provenance."""

        units = sources_py.extract_source_units(SourceFile(
            Path("decorated.py"),
            "@decorator\nasync def compute():\n    return 1\n",
        ))
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].symbol, "compute")
        self.assertEqual((units[0].start_line, units[0].end_line), (1, 3))
        self.assertTrue(units[0].text.startswith("@decorator\n"))

    def test_class_children_preserve_nonblank_parent_lines(self) -> None:
        """Preserve visible class lines, context, and method order."""

        units = sources_py.extract_source_units(SourceFile(
            Path("sample.py"), fixture_text("sample.py")
        ))
        parent = next(unit for unit in units if unit.kind == "python_class")
        self.assertEqual(
            [line for child in parent.children
             for line in child.text.splitlines() if line.strip()],
            [line for line in parent.text.splitlines() if line.strip()],
        )
        self.assertEqual(
            [child.symbol for child in parent.children
             if child.kind == "python_method"],
            ["Example.describe", "Example.compute"],
        )
        self.assertTrue(all(child.path == parent.path
                            for child in parent.children))
        self.assertEqual(
            [child.start_line for child in parent.children],
            sorted(child.start_line for child in parent.children),
        )

    def test_class_without_methods_has_context_child(self) -> None:
        """Preserve class attributes when there are no methods."""

        text = "class Data:\n    value = 1\n"
        units = sources_py.extract_source_units(
            SourceFile(Path("data.py"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "python_class")
        self.assertEqual(len(units[0].children), 1)
        self.assertEqual(units[0].children[0].text, text)

    def test_module_without_declarations_is_context(self) -> None:
        """Return one context unit for a declaration-free module."""

        text = "VALUE = 1\nprint(VALUE)\n"
        units = sources_py.extract_source_units(
            SourceFile(Path("constants.py"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "python_module_context")
        self.assertEqual(units[0].symbol, "constants")
        self.assertEqual(units[0].text, text)

    def test_invalid_python_falls_back_to_one_file_unit(self) -> None:
        """Preserve syntax errors as source text rather than executing it."""

        text = "def invalid(:\n"
        units = sources_py.extract_source_units(
            SourceFile(Path("broken.py"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "python_file")
        self.assertEqual(units[0].text, text)

    def test_empty_and_blank_python_sources(self) -> None:
        """Avoid units without visible content."""

        for text in ("", "  \n\n"):
            with self.subTest(text=text):
                self.assertEqual(
                    sources_py.extract_source_units(
                        SourceFile(Path("empty.py"), text)
                    ),
                    [],
                )

    def test_function_coordinates_match_original_source(self) -> None:
        """Keep inclusive one-based lines for extracted declarations."""

        text = fixture_text("sample.py")
        lines = text.splitlines(keepends=True)
        units = sources_py.extract_source_units(
            SourceFile(Path("sample.py"), text)
        )
        for unit in units:
            with self.subTest(kind=unit.kind, symbol=unit.symbol):
                self.assertEqual(
                    unit.text,
                    "".join(lines[unit.start_line - 1:unit.end_line]),
                )


if __name__ == "__main__":
    unittest.main()
