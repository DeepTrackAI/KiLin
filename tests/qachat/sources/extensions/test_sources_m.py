"""Tests for MATLAB natural-unit extraction."""

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


class TestSourcesM(unittest.TestCase):

    def test_matlab_files_become_natural_units(self) -> None:
        matlab_class = (
            "% function fake()\n"
            "classdef (Abstract) Child < Base\n"
            "    properties\n"
            "        Value\n"
            "    end\n"
            "    methods (Static)\n"
            "        function obj = Child(value)\n"
            "            obj.Value = value;\n"
            "        end\n"
            "        function [first, second] = compute(obj)\n"
            "            first = obj.Value;\n"
            "            second = first + 1;\n"
            "        end\n"
            "    end\n"
            "end\n"
        )
        matlab_script = (
            "value = 1;\n"
            "function output = addOne(input)\n"
            "output = input + 1;\n"
            "end\n"
            "function show(value)\n"
            "disp(value);\n"
            "end\n"
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Child.m").write_text(
                matlab_class,
                encoding="utf-8",
            )
            (root / "calculate.m").write_text(
                matlab_script,
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".m"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        matlab_parent = next(
            unit for unit in units if unit.kind == "matlab_class"
        )
        child_kinds = {child.kind for child in matlab_parent.children}
        child_symbols = {
            child.symbol
            for child in matlab_parent.children
            if child.symbol is not None
        }
        self.assertEqual(matlab_parent.symbol, "Child")
        self.assertEqual(matlab_parent.start_line, 2)
        self.assertIn("matlab_properties_block", child_kinds)
        self.assertIn("matlab_methods_block", child_kinds)
        self.assertIn("matlab_constructor", child_kinds)
        self.assertIn("matlab_method", child_kinds)
        self.assertIn("Child.Child", child_symbols)
        self.assertIn("Child.compute", child_symbols)
        self.assertNotIn("fake", child_symbols)

        kinds = [unit.kind for unit in units]
        symbols = [unit.symbol for unit in units]
        self.assertIn("matlab_script", kinds)
        self.assertIn("addOne", symbols)
        self.assertIn("show", symbols)


if __name__ == "__main__":
    unittest.main()
