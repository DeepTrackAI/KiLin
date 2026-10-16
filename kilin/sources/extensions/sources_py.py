"""Partition Python source using syntax-tree declaration boundaries.

The extractor parses text without importing or executing the analyzed file.
Top-level functions, asynchronous functions, and classes retain decorator and
line-range provenance. Class units contain recognized method and intervening
context units so the chunker can split oversized classes structurally.

A syntax error falls back to a whole-file Python unit. The extractor describes
source structure, not runtime objects, imports, or executed control flow.

Key Features
------------

- **AST-based boundaries**

    Use declaration locations, including decorators, to preserve source ranges.

- **Hierarchical classes**

    Expose child methods and context while retaining the complete class text.

- **Text fallback**

    Keep syntactically invalid Python available for retrieval as source text.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one Python source file.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".py"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

"""

from __future__ import annotations

import ast
from pathlib import Path

from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "extract_source_units",
]


def _node_start(node: ast.AST) -> int:
    """Return the first line of a Python definition and its decorators.

    Parameters
    ----------
    node: ast.AST
        Function or class definition.

    Returns
    -------
    int
        First one-based source line.

    """

    decorators = getattr(node, "decorator_list", ())
    lines = [node.lineno]
    lines.extend(decorator.lineno for decorator in decorators)
    return min(lines)


def _node_end(node: ast.AST) -> int:
    """Return the final line of a Python definition.

    Parameters
    ----------
    node: ast.AST
        Function or class definition.

    Returns
    -------
    int
        Last one-based source line.

    Raises
    ------
    ValueError
        If Python does not report an ending line.

    """

    end_line = getattr(node, "end_lineno", None)
    if not isinstance(end_line, int):
        raise ValueError("Python did not report the definition's final line.")

    return end_line


def _python_class_unit(
    path: Path,
    lines: list[str],
    node: ast.ClassDef,
) -> SourceUnit:
    """Build one class unit with complete, non-overlapping child units.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.
    node: ast.ClassDef
        Parsed class definition.

    Returns
    -------
    SourceUnit
        Complete class unit.

    Raises
    ------
    ValueError
        If the class source slice is unexpectedly empty.

    """

    start_line = _node_start(node)
    end_line = _node_end(node)
    methods = sorted(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=_node_start,
    )

    children: list[SourceUnit] = []
    cursor = start_line
    for method in methods:
        method_start = _node_start(method)
        context = _source_unit(
            path,
            lines,
            cursor,
            method_start - 1,
            kind="python_class_context",
            symbol=node.name,
        )
        if context is not None:
            children.append(context)

        method_unit = _source_unit(
            path,
            lines,
            method_start,
            _node_end(method),
            kind="python_method",
            symbol=f"{node.name}.{method.name}",
        )
        if method_unit is not None:
            children.append(method_unit)

        cursor = _node_end(method) + 1

    tail = _source_unit(
        path,
        lines,
        cursor,
        end_line,
        kind="python_class_context",
        symbol=node.name,
    )
    if tail is not None:
        children.append(tail)

    unit = _source_unit(
        path,
        lines,
        start_line,
        end_line,
        kind="python_class",
        symbol=node.name,
        children=tuple(children),
    )
    if unit is None:
        raise ValueError("A parsed Python class unexpectedly had no text.")

    return unit


def _python_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify top-level Python definitions and surrounding context.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        Python units. Invalid Python remains one safe file unit.

    """

    source = "".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        fallback = _source_unit(
            path,
            lines,
            1,
            len(lines),
            kind="python_file",
        )
        return [fallback] if fallback is not None else []

    definitions = sorted(
        (
            node
            for node in tree.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ),
        key=_node_start,
    )

    units: list[SourceUnit] = []
    cursor = 1
    for node in definitions:
        start_line = _node_start(node)
        context = _source_unit(
            path,
            lines,
            cursor,
            start_line - 1,
            kind="python_module_context",
            symbol=path.stem,
        )
        if context is not None:
            units.append(context)

        if isinstance(node, ast.ClassDef):
            units.append(_python_class_unit(path, lines, node))
        else:
            function = _source_unit(
                path,
                lines,
                start_line,
                _node_end(node),
                kind="python_function",
                symbol=node.name,
            )
            if function is not None:
                units.append(function)

        cursor = _node_end(node) + 1

    tail = _source_unit(
        path,
        lines,
        cursor,
        len(lines),
        kind="python_module_context",
        symbol=path.stem,
    )
    if tail is not None:
        units.append(tail)

    return units


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one Python source file.

    Parameters
    ----------
    source_file: SourceFile
        Source file converted to text.

    Returns
    -------
    list[SourceUnit]
        Natural units in source order.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []

    return _python_units(source_file.path, lines)
