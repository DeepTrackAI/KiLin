"""Partition R text at its top-level functions and classes.

An R file is split where a top-level function, class or S4 method is
declared, such as ``smooth <- function(x) {``, ``Store <- setRefClass(``,
``setClass("Base")`` or ``setMethod("area", "Circle", ...)``. A
declaration ends where its brackets close, so code after it is context of
the file. Comment lines right above a declaration, roxygen ``#'`` blocks
included, stay with it. Comments between a function signature and its
opening brace also remain inside that function. Methods in a reference or
R6 class body, such as ``save = function(x) {``, are parts of their class.
Only declarations starting their line are found; functions defined inside
other code stay
part of the code around them. The source is never run.

Key Features
------------

- **Functions and classes**

    Label each top-level function, class and S4 method with its name.

- **Comments**

    Keep the comment block above a declaration with it.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one R source file.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".R"])
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

# R names may contain dots, as in "read.data"; backticks allow any name.
_NAME: Final[str] = r"([A-Za-z.][\w.]*|`[^`]+`)"
_ASSIGN: Final[str] = r"^" + _NAME + r"\s*(?:<<?-|=)\s*"
R_FUNCTION: Final[re.Pattern[str]] = re.compile(_ASSIGN + r"function\b")
R_CLASS: Final[re.Pattern[str]] = re.compile(
    _ASSIGN + r"(?:[\w.]+::)?(?:setRefClass|setClass|R6Class)\s*\("
)
R_SET_CLASS: Final[re.Pattern[str]] = re.compile(
    r"^(?:methods::)?set(?:Ref)?Class\s*\(\s*[\"']([^\"']+)[\"']"
)
R_METHOD: Final[re.Pattern[str]] = re.compile(
    r"^(?:methods::)?setMethod\s*\(\s*[\"']([^\"']+)[\"']"
    r"(?:\s*,\s*(?:signature\s*\(\s*)?[\"']([^\"']+)[\"'])?"
)
# A method in a reference or R6 class body: "  save = function(x) {".
R_CLASS_METHOD: Final[re.Pattern[str]] = re.compile(
    r"^\s+" + _NAME + r"\s*=\s*function\b"
)
_COMMENT: Final[re.Pattern[str]] = re.compile(r"^\s*#")


def _declaration(line: str) -> tuple[str, str] | None:
    """Return the kind and name of a top-level declaration line.

    Parameters
    ----------
    line: str
        One source line.

    Returns
    -------
    tuple[str, str] | None
        ``("r_function", name)``, ``("r_class", name)`` or
        ``("r_method", "Class.name")``; None for other lines.

    """

    for pattern, kind in (
        (R_FUNCTION, "r_function"),
        (R_CLASS, "r_class"),
        (R_SET_CLASS, "r_class"),
    ):
        match = pattern.match(line)
        if match is not None:
            return kind, match.group(1).strip("`")
    method = R_METHOD.match(line)
    if method is None:
        return None
    name, signature = method.groups()
    return "r_method", name if signature is None else f"{signature}.{name}"


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
    """Return the line where a declaration's brackets close.

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
        The first line on which every bracket opened since the declaration
        is closed again, or ``last`` when they never are. Brackets inside
        comments and strings are not counted.

    """

    depth = 0
    opened = False
    quote: str | None = None
    for number in range(line_number, last + 1):
        line = lines[number - 1]
        index = 0
        while index < len(line):
            character = line[index]
            if quote is not None:
                if character == "\\":
                    index += 1
                elif character == quote:
                    quote = None
            elif character == "#":
                break
            elif character in "\"'`":
                quote = character
            elif character in "({[":
                depth += 1
                opened = True
            elif character in ")}]":
                depth -= 1
            index += 1
        # A one-line function without brackets, such as
        # "square <- function(x) x^2", has closed its parameters. A body
        # whose "{" starts the next line continues there.
        if opened and depth <= 0 and quote is None:
            following = next(
                (
                    text.strip()
                    for text in lines[number:last]
                    if text.strip() and not _COMMENT.match(text)
                ),
                "",
            )
            if not following.startswith("{"):
                return number
    return last


def _class_unit(
    path: Path, lines: list[str], start: int, end: int, name: str
) -> SourceUnit | None:
    """Return a class unit whose children are the methods in its body.

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
        The class with one child per method, such as ``Store.save``, and
        its other lines as context; None when the range is blank.

    """

    children: list[SourceUnit] = []
    position = start
    number = start + 1
    while number <= end:
        match = R_CLASS_METHOD.match(lines[number - 1])
        if match is None:
            number += 1
            continue
        first = _with_comment(lines, number, position)
        last = _block_end(lines, number, end)
        for low, high, kind, symbol in (
            (position, first - 1, "r_class_context", None),
            (first, last, "r_method", f"{name}.{match.group(1).strip('`')}"),
        ):
            unit = _source_unit(
                path, lines, low, high, kind=kind, symbol=symbol
            )
            if unit is not None:
                children.append(unit)
        position = number = last + 1
    if children:
        rest = _source_unit(path, lines, position, end, kind="r_class_context")
        if rest is not None:
            children.append(rest)
    return _source_unit(
        path,
        lines,
        start,
        end,
        kind="r_class",
        symbol=name,
        children=tuple(children) if len(children) > 1 else (),
    )


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one R source file.

    Parameters
    ----------
    source_file: SourceFile
        Source file read as text.

    Returns
    -------
    list[SourceUnit]
        Top-level functions, classes and S4 methods, each with the comment
        block above it, and the code between them, in source order.

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
    parts: list[tuple[int, int, str, str | None]] = []
    position = 1
    for line_number, kind, name in starts:
        if line_number < position:
            # Declared inside the brackets of the previous declaration.
            continue
        start = _with_comment(lines, line_number, position)
        if start > position:
            parts.append((position, start - 1, "r_script", None))
        end = _block_end(lines, line_number, len(lines))
        parts.append((start, end, kind, name))
        position = end + 1
    if position <= len(lines):
        parts.append((position, len(lines), "r_script", None))
    units: list[SourceUnit] = []
    for first, last, kind, name in parts:
        if kind == "r_class" and name is not None:
            unit = _class_unit(path, lines, first, last, name)
        else:
            unit = _source_unit(
                path, lines, first, last, kind=kind, symbol=name
            )
        if unit is not None:
            units.append(unit)
    return units
