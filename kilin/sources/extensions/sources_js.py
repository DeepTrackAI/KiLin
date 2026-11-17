"""Partition JavaScript text at top-level declarations.

The extractor recognizes functions, classes and functions assigned to
``const``, ``let`` or ``var`` that start a line, with or without
``export`` or ``async``, and the methods of a class. A comment block right
above a declaration stays with it. Text between declarations is kept as
context. Like the MATLAB reader, this is a line-pattern reader, not a
JavaScript parser: a declaration that does not start a line, or code that
builds functions at run time, is kept as context. No code is executed.

Key Features
------------

- **Declarations**

    Label top-level functions and classes, and the methods of a class.

- **Comments stay attached**

    Keep the comment block above a declaration in the same unit.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one JavaScript source file.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".js"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "extract_source_units",
]

_NAME: Final[str] = r"([A-Za-z_$][\w$]*)"
_EXPORT: Final[str] = r"^(?:export\s+(?:default\s+)?)?"
JS_FUNCTION: Final[re.Pattern[str]] = re.compile(
    _EXPORT + r"(?:async\s+)?function\s*\*?\s*" + _NAME + r"\s*\("
)
JS_CLASS: Final[re.Pattern[str]] = re.compile(_EXPORT + r"class\s+" + _NAME)
# A function stored in a variable: "const f = function", "= (a) =>",
# "= a =>", or "= (" whose parameters continue on the next lines.
JS_ASSIGNED_FUNCTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+" + _NAME + r"\s*=\s*(?:async\s+)?"
    r"(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>|\([^)]*$)"
)
# A method line inside a class body, such as "  async save(name) {".
JS_METHOD: Final[re.Pattern[str]] = re.compile(
    r"^\s+(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?\*?"
    r"(#?[A-Za-z_$][\w$]*)\s*\([^;]*\)\s*\{"
)
# Statements that look like a method line.
_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"if", "for", "while", "switch", "catch", "function", "return", "with"}
)
_COMMENT: Final[re.Pattern[str]] = re.compile(r"^\s*(?://|/\*|\*)")
# The closing brace of a top-level block starts its line.
_TOP_LEVEL_END: Final[re.Pattern[str]] = re.compile(r"^[}\]]")


def _declaration(line: str) -> tuple[str, str] | None:
    """Return the kind and name of a top-level declaration line.

    Parameters
    ----------
    line: str
        One source line.

    Returns
    -------
    tuple[str, str] | None
        ``("js_function", name)`` or ``("js_class", name)``; None for other
        lines.

    """

    for pattern, kind in (
        (JS_FUNCTION, "js_function"),
        (JS_ASSIGNED_FUNCTION, "js_function"),
        (JS_CLASS, "js_class"),
    ):
        match = pattern.match(line)
        if match is not None:
            return kind, match.group(1)
    return None


def _with_comment(lines: list[str], line_number: int, first: int) -> int:
    """Move a declaration's start up over the comment block right above it.

    Parameters
    ----------
    lines: list[str]
        Source lines.
    line_number: int
        One-based line of the declaration.
    first: int
        One-based line the comment may not start before.

    Returns
    -------
    int
        One-based first line of the comment, or ``line_number`` when there
        is none.

    """

    start = line_number
    while start - 1 >= first and _COMMENT.match(lines[start - 2]):
        start -= 1
    return start


def _block_end(lines: list[str], line_number: int, last: int) -> int:
    """Return the last line of a top-level declaration.

    Parameters
    ----------
    lines: list[str]
        Source lines.
    line_number: int
        One-based line of the declaration.
    last: int
        Last one-based line it may reach.

    Returns
    -------
    int
        The line of its closing brace at the start of a line, the
        declaration line itself when it ends with ``;``, or ``last``.

    """

    if lines[line_number - 1].rstrip().endswith(";"):
        return line_number
    for number in range(line_number + 1, last + 1):
        if _TOP_LEVEL_END.match(lines[number - 1]):
            return number
    return last


def _partition(
    path: Path,
    lines: list[str],
    starts: list[tuple[int, str, str]],
    first: int,
    last: int,
    context_kind: str,
    *,
    top_level: bool = False,
) -> list[tuple[int, int, str | None, str | None]]:
    """Split a line range at declarations, keeping the text between them.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path; not used, kept for symmetry with callers.
    lines: list[str]
        Source lines.
    starts: list[tuple[int, str, str]]
        Declaration lines with their kind and name, in order.
    first: int
        First one-based line of the range.
    last: int
        Last one-based line of the range.
    context_kind: str
        Kind of the text between declarations.
    top_level: bool, optional
        End each declaration at its closing brace, so code after it is
        context. Defaults to ``False`` (a method runs to the next one).

    Returns
    -------
    list[tuple[int, int, str | None, str | None]]
        ``(start, end, kind, name)`` per part, covering the whole range.

    """

    del path
    parts: list[tuple[int, int, str | None, str | None]] = []
    position = first
    for index, (line_number, kind, name) in enumerate(starts):
        start = _with_comment(lines, line_number, position)
        if start > position:
            parts.append((position, start - 1, context_kind, None))
        end = (
            _with_comment(lines, starts[index + 1][0], line_number + 1) - 1
            if index + 1 < len(starts)
            else last
        )
        closed = _block_end(lines, line_number, end) if top_level else end
        parts.append((start, closed, kind, name))
        if closed < end:
            parts.append((closed + 1, end, context_kind, None))
        position = end + 1
    if position <= last:
        parts.append((position, last, context_kind, None))
    return parts


def _class_unit(
    path: Path, lines: list[str], start: int, end: int, name: str
) -> SourceUnit | None:
    """Return a class unit whose children are its methods.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines.
    start: int
        First one-based line of the class, comment included.
    end: int
        Last one-based line of the class.
    name: str
        Class name.

    Returns
    -------
    SourceUnit | None
        The class with one child per method and its other lines as context;
        None when the range is blank.

    """

    methods = [
        (number, "js_method", match.group(1))
        for number in range(start + 1, end + 1)
        if (match := JS_METHOD.match(lines[number - 1])) is not None
        and match.group(1) not in _KEYWORDS
    ]
    children = tuple(
        unit
        for first, last, kind, method in _partition(
            path, lines, methods, start, end, "js_class_context"
        )
        if (
            unit := _source_unit(
                path,
                lines,
                first,
                last,
                kind=str(kind),
                symbol=None if method is None else f"{name}.{method}",
            )
        )
        is not None
    )
    return _source_unit(
        path,
        lines,
        start,
        end,
        kind="js_class",
        symbol=name,
        children=children if len(children) > 1 else (),
    )


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one JavaScript source file.

    Parameters
    ----------
    source_file: SourceFile
        Source file read as text.

    Returns
    -------
    list[SourceUnit]
        Top-level functions, classes with their methods, and the context
        between them, in source order.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []
    path = source_file.path
    starts = [
        (number, *found)
        for number, line in enumerate(lines, start=1)
        if (found := _declaration(line)) is not None
    ]
    units: list[SourceUnit] = []
    for first, last, kind, name in _partition(
        path,
        lines,
        starts,
        1,
        len(lines),
        "js_module_context",
        top_level=True,
    ):
        if kind == "js_class" and name is not None:
            unit = _class_unit(path, lines, first, last, name)
        else:
            unit = _source_unit(
                path, lines, first, last, kind=str(kind), symbol=name
            )
        if unit is not None:
            units.append(unit)
    return units
