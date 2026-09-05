"""Tests for safe source reading and natural source units."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.source import (
    SourceFile,
    extract_source_units,
    read_repository_files,
)


class TestSource(unittest.TestCase):

    def test_read_repository_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            (root / "module.py").write_text(
                "value = 1\n",
                encoding="utf-8",
            )
            (root / "binary.py").write_bytes(b"\xff\xfe")

            report = scan_repository(root)
            with self.assertRaisesRegex(ValueError, "not UTF-8"):
                read_repository_files(report)

            report = scan_repository(
                root,
                include_paths=["module.py"],
            )
            files = read_repository_files(report)
            self.assertEqual(
                files,
                [SourceFile(Path("module.py"), "value = 1\n")],
            )

            outside_report = {
                "repository": str(root),
                "selected_files": ["../outside.py"],
            }
            with self.assertRaisesRegex(ValueError, "inside"):
                read_repository_files(outside_report)

    def test_extract_source_units(self) -> None:
        python = SourceFile(
            Path("example.py"),
            '"""Example."""\n'
            "\n"
            "def greet():\n"
            "    return 'hello'\n"
            "\n"
            "class Greeter:\n"
            "    name = 'QAChat'\n"
            "\n"
            "    def reply(self):\n"
            "        return self.name\n",
        )
        markdown = SourceFile(
            Path("guide.md"),
            "Introduction.\n\n# Setup\nRun setup.\n",
        )
        invalid_python = SourceFile(Path("broken.py"), "def broken(:\n")
        plain_text = SourceFile(Path("notes.txt"), "Keep this whole.\n")

        units = extract_source_units(
            [python, markdown, invalid_python, plain_text]
        )
        names = [(unit.kind, unit.symbol) for unit in units]
        self.assertIn(("python_function", "greet"), names)
        self.assertIn(("python_class", "Greeter"), names)
        self.assertIn(("markdown_section", None), names)
        self.assertIn(("python_file", None), names)
        self.assertIn(("file", None), names)

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

        section = next(
            unit for unit in units if unit.kind == "markdown_section"
        )
        self.assertEqual(section.section, "Setup")
        self.assertIsNone(section.symbol)

        matlab_class = SourceFile(
            Path("Child.m"),
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
            "end\n",
        )
        matlab_script = SourceFile(
            Path("calculate.m"),
            "value = 1;\n"
            "function output = addOne(input)\n"
            "output = input + 1;\n"
            "end\n"
            "function show(value)\n"
            "disp(value);\n"
            "end\n",
        )

        matlab_units = extract_source_units([matlab_class, matlab_script])
        matlab_parent = next(
            unit for unit in matlab_units if unit.kind == "matlab_class"
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

        script_kinds = [unit.kind for unit in matlab_units]
        script_symbols = [unit.symbol for unit in matlab_units]
        self.assertIn("matlab_script", script_kinds)
        self.assertIn("addOne", script_symbols)
        self.assertIn("show", script_symbols)

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
        notebook_units = extract_source_units(
            [
                SourceFile(
                    Path("guide.ipynb"),
                    json.dumps(notebook),
                )
            ]
        )
        self.assertEqual(
            [unit.kind for unit in notebook_units],
            ["notebook_markdown", "notebook_code"],
        )
        self.assertEqual(
            [unit.cell for unit in notebook_units],
            [1, 2],
        )
        self.assertEqual(notebook_units[0].section, "Purpose")
        self.assertNotIn(
            "DO NOT INDEX",
            "".join(unit.text for unit in notebook_units),
        )

        invalid_notebook = SourceFile(Path("broken.ipynb"), "{")
        with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
            extract_source_units([invalid_notebook])


if __name__ == "__main__":
    unittest.main()
