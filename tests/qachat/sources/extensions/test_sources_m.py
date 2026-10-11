"""Unit tests for MATLAB source-structure heuristics.

Tests verify recognized class and function declarations and retained source
context. MATLAB is never executed and no MATLAB installation is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.sources.extensions import sources_m
from qachat.sources.sources import SourceFile
from tests.qachat.sources._support import fixture_text


class TestMatlabExtraction(unittest.TestCase):
    """Verify MATLAB source partitions and declaration provenance."""

    def test_fixture_recognizes_class_methods_and_blocks(self) -> None:
        """Identify class, constructor, method, and class sections."""

        text = fixture_text("sample.m")
        units = sources_m.extract_source_units(
            SourceFile(Path("sample.m"), text)
        )
        parent = next(unit for unit in units if unit.kind == "matlab_class")
        self.assertEqual(parent.symbol, "Sample")
        self.assertEqual(parent.start_line, 2)
        self.assertEqual("".join(child.text for child in parent.children),
                         parent.text)
        kinds = {child.kind for child in parent.children}
        self.assertIn("matlab_properties_block", kinds)
        self.assertIn("matlab_methods_block", kinds)
        self.assertIn("matlab_constructor", kinds)
        self.assertIn("matlab_method", kinds)
        self.assertIn(
            "Sample.compute",
            {child.symbol for child in parent.children},
        )

    def test_script_and_multiple_functions_preserve_order(self) -> None:
        """Keep script preamble and each recognizable function."""

        text = (
            "value = 1;\n"
            "function output = addOne(input)\n"
            "output = input + 1;\n"
            "end\n"
            "function show(value)\n"
            "disp(value);\n"
            "end\n"
        )
        units = sources_m.extract_source_units(
            SourceFile(Path("calculate.m"), text)
        )
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [
                ("matlab_script", "calculate"),
                ("matlab_function", "addOne"),
                ("matlab_function", "show"),
            ],
        )
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_class_without_methods_keeps_context(self) -> None:
        """Retain a class even if no methods are recognized."""

        text = "classdef Data\n    properties\n        Value\n    end\nend\n"
        units = sources_m.extract_source_units(
            SourceFile(Path("Data.m"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "matlab_class")
        self.assertEqual("".join(child.text for child in units[0].children),
                         text)

    def test_comment_is_not_a_function_declaration(self) -> None:
        """Ignore function-shaped comments when finding markers."""

        text = "% function fake()\nvalue = 1;\n"
        units = sources_m.extract_source_units(
            SourceFile(Path("script.m"), text)
        )
        self.assertEqual([unit.kind for unit in units], ["matlab_script"])
        self.assertEqual(units[0].text, text)

    def test_incomplete_function_declaration_keeps_source(self) -> None:
        """Retain a visible function marker without a parsed name."""

        text = "function\n    disp('incomplete');\nend\n"
        units = sources_m.extract_source_units(
            SourceFile(Path("broken.m"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "matlab_function")
        self.assertIsNone(units[0].symbol)
        self.assertEqual(units[0].text, text)

    def test_empty_and_blank_matlab_sources(self) -> None:
        """Return no units when source contains no visible text."""

        for text in ("", " \n\n"):
            with self.subTest(text=text):
                self.assertEqual(
                    sources_m.extract_source_units(
                        SourceFile(Path("blank.m"), text)
                    ),
                    [],
                )

    def test_class_child_ranges_match_original_lines(self) -> None:
        """Preserve exact text within each MATLAB class child."""

        text = fixture_text("sample.m")
        lines = text.splitlines(keepends=True)
        units = sources_m.extract_source_units(
            SourceFile(Path("sample.m"), text)
        )
        parent = next(unit for unit in units if unit.kind == "matlab_class")
        for child in parent.children:
            with self.subTest(kind=child.kind, symbol=child.symbol):
                self.assertEqual(
                    child.text,
                    "".join(lines[child.start_line - 1:child.end_line]),
                )


if __name__ == "__main__":
    unittest.main()
