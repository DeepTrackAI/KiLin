"""Regression tests for KíLín documentation and formatting conventions.

These checks apply the Python parts of ``DTDV411_style.ipynb`` to the
library, setup helpers, and test suite. Source-extraction fixtures are not
required to define documented classes or functions because the extractor
must also accept incomplete external code.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (ROOT / "kilin", ROOT / "setup")
PYTHON_DIRECTORIES = (*SOURCE_DIRECTORIES, ROOT / "tests")
DOCSTRING_SECTIONS = ("Key Features", "Module Structure", "Examples")
DOCSTRING_PARAMETER = re.compile(r"^\**[a-zA-Z_]\w*\s*:")
JAVASCRIPT_FUNCTION = re.compile(r"^function (\w+)\((.*?)\)\s*\{")


def source_files(*directories: Path) -> list[Path]:
    """Return Python files in stable order, ignoring generated caches.

    Parameters
    ----------
    *directories: pathlib.Path
        Directory roots to search recursively for Python source files.

    Returns
    -------
    list[pathlib.Path]
        Source paths in deterministic relative-path order.

    """

    return sorted(
        file
        for directory in directories
        for file in directory.rglob("*.py")
        if "__pycache__" not in file.parts
    )


def definitions(tree: ast.AST) -> list[ast.AST]:
    """Collect all nested Python definitions and the module itself.

    Parameters
    ----------
    tree: ast.AST
        Parsed Python module.

    Returns
    -------
    list[ast.AST]
        The module and its function, asynchronous function, and class nodes.

    """

    return [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef,
             ast.AsyncFunctionDef),
        )
    ]


def documented_parameters(docstring: str) -> set[str]:
    """Read parameter names from a NumPy-style Parameters section.

    Parameters
    ----------
    docstring: str
        Cleaned function docstring to inspect.

    Returns
    -------
    set[str]
        Documented parameter names, excluding unpacking prefixes.

    """

    lines = inspect.cleandoc(docstring).splitlines()
    try:
        start = lines.index("Parameters")
    except ValueError:
        return set()
    if start + 1 >= len(lines) or lines[start + 1] != "----------":
        return set()

    names: set[str] = set()
    for line in lines[start + 2:]:
        if line and not line.startswith(" "):
            if DOCSTRING_PARAMETER.match(line):
                names.add(line.split(":", 1)[0].strip().lstrip("*"))
            else:
                break
    return names


class TestDocumentationStyle(unittest.TestCase):
    """Keep source documentation aligned with the DeepTrack conventions."""

    def test_python_line_length(self) -> None:
        """Enforce 79 characters in library, setup, and test Python files."""

        failures = []
        for file in source_files(*PYTHON_DIRECTORIES):
            for number, line in enumerate(file.read_text().splitlines(), 1):
                if len(line) > 79:
                    failures.append(
                        f"{file.relative_to(ROOT)}:{number}: {len(line)}"
                    )
        self.assertEqual(failures, [])

    def test_library_definition_docstrings(self) -> None:
        """Document every library definition with proper closing lines."""

        failures = []
        for file in source_files(*SOURCE_DIRECTORIES):
            lines = file.read_text().splitlines()
            for node in definitions(ast.parse("\n".join(lines))):
                name = getattr(node, "name", "<module>")
                docstring = ast.get_docstring(node)
                if docstring is None:
                    failures.append(f"{file.relative_to(ROOT)}: {name}")
                    continue
                first = node.body[0]
                if first.lineno != first.end_lineno:
                    if lines[first.end_lineno - 2].strip():
                        failures.append(
                            f"{file.relative_to(ROOT)}: {name}: closing"
                        )
        self.assertEqual(failures, [])

    def test_numpy_parameter_sections(self) -> None:
        """Keep documented parameters synchronized with Python signatures."""

        failures = []
        for file in source_files(*SOURCE_DIRECTORIES):
            tree = ast.parse(file.read_text())
            for node in definitions(tree):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                arguments = node.args
                names = {
                    item.arg
                    for item in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    )
                    if item.arg not in ("self", "cls")
                }
                if arguments.vararg is not None:
                    names.add(arguments.vararg.arg)
                if arguments.kwarg is not None:
                    names.add(arguments.kwarg.arg)
                actual = documented_parameters(ast.get_docstring(node) or "")
                if names != actual:
                    failures.append(
                        f"{file.relative_to(ROOT)}: {node.name}: "
                        f"{names!r} != {actual!r}"
                    )
        self.assertEqual(failures, [])

    def test_module_structure_documentation(self) -> None:
        """Require feature, structure, and example sections in modules."""

        failures = []
        for file in source_files(ROOT / "kilin"):
            docstring = ast.get_docstring(ast.parse(file.read_text())) or ""
            for title in DOCSTRING_SECTIONS:
                heading = f"\n{title}\n{'-' * len(title)}\n"
                if heading not in f"\n{docstring}":
                    failures.append(f"{file.relative_to(ROOT)}: {title}")
        self.assertEqual(failures, [])

    def test_test_modules_use_common_lint_policy(self) -> None:
        """Retain the three standard Pylint suppressions in test modules."""

        directives = (
            "C0115:missing-class-docstring",
            "C0116:missing-function-docstring",
            "C0103:invalid-name",
        )
        failures = []
        for file in sorted((ROOT / "tests").rglob("test_*.py")):
            first_lines = "\n".join(file.read_text().splitlines()[:24])
            if any(item not in first_lines for item in directives):
                failures.append(str(file.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_notebook_code_line_length(self) -> None:
        """Apply Python line length to notebook code cells, not output data."""

        failures = []
        for file in sorted(ROOT.rglob("*.ipynb")):
            notebook = json.loads(file.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", []), 1):
                if cell.get("cell_type") != "code":
                    continue
                for number, line in enumerate(
                    "".join(cell.get("source", [])).splitlines(), 1
                ):
                    if len(line) > 79:
                        failures.append(
                            f"{file.relative_to(ROOT)}:cell{index}:"
                            f"{number}"
                        )
        self.assertEqual(failures, [])

    def test_frontend_named_functions_have_jsdoc(self) -> None:
        """Describe named JavaScript functions and their parameters."""

        file = ROOT / "kilin" / "gui" / "web" / "app.js"
        lines = file.read_text().splitlines()
        failures = []
        for index, line in enumerate(lines):
            match = JAVASCRIPT_FUNCTION.match(line)
            if match is None:
                continue
            if index == 0 or lines[index - 1].strip() != "*/":
                failures.append(f"{match[1]}: missing adjacent JSDoc")
                continue
            start = index - 2
            while start >= 0 and lines[start].strip() != "/**":
                start -= 1
            if start < 0:
                failures.append(f"{match[1]}: missing JSDoc opening")
                continue
            text = "\n".join(lines[start:index])
            parameters = {
                item.split("=", 1)[0].strip()
                for item in match[2].split(",")
                if item.strip()
            }
            documented = set(re.findall(r"@param\s+(\w+)", text))
            if parameters - documented:
                failures.append(f"{match[1]}: missing parameter docs")
        self.assertEqual(failures, [])
