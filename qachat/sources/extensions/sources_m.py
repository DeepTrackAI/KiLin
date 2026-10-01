"""Partition MATLAB text using recognizable declaration boundaries.

The extractor recognizes class and function declarations and selected class
blocks using line-oriented patterns. It preserves intervening source context
and marks class children such as methods or constructors where recognized.
It is not a full MATLAB parser: multiline declarations, block comments, and
nested constructs can require interpretation beyond these heuristics.

Key Features
------------

- **Structural hints**

    Label recognizable MATLAB classes, functions, and class sections.

- **Context retention**

    Preserve text between declarations for later chunking.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one MATLAB source file.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository("/path/to/repository", extensions=[".m"])
>>> units = qachat.extract_source_units(qachat.read_repository_files(report))

"""

from __future__ import annotations

from pathlib import Path
import re

from qachat.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)


__all__ = [
    "extract_source_units",
]

MATLAB_BLOCK = re.compile(
    r"^\s*(properties|methods|events|enumeration)\b",
    re.IGNORECASE,
)
MATLAB_CLASS = re.compile(
    r"^\s*classdef(?:\s*\([^)]*\))?\s+([A-Za-z]\w*)",
    re.IGNORECASE,
)
MATLAB_FUNCTION = re.compile(r"^\s*function\b", re.IGNORECASE)


def _matlab_function_name(line: str) -> str | None:
    """Return the declared name from one MATLAB function line.

    Parameters
    ----------
    line: str
        MATLAB source line beginning with function.

    Returns
    -------
    str | None
        Declared function name, or None for an incomplete declaration.

    """

    signature = line.split("%", 1)[0].strip()
    signature = re.sub(
        r"^function\b",
        "",
        signature,
        flags=re.IGNORECASE,
    ).strip()
    if "=" in signature:
        signature = signature.split("=", 1)[1].strip()

    match = re.match(r"([A-Za-z]\w*)", signature)
    return match.group(1) if match is not None else None


def _matlab_markers(
    lines: list[str],
    *,
    class_name: str | None,
    first_line: int,
) -> list[tuple[int, str, str | None]]:
    """Find visible MATLAB blocks and function declarations.

    Parameters
    ----------
    lines: list[str]
        Source lines with their endings.
    class_name: str | None
        Enclosing class name, when the file defines a class.
    first_line: int
        First one-based line to inspect.

    Returns
    -------
    list[tuple[int, str, str | None]]
        Line, source-unit kind, and optional symbol for each marker.

    """

    markers: list[tuple[int, str, str | None]] = []
    for line_number in range(first_line, len(lines) + 1):
        line = lines[line_number - 1]
        if line.lstrip().startswith("%"):
            continue

        if MATLAB_FUNCTION.match(line) is not None:
            name = _matlab_function_name(line)
            if class_name is None:
                kind = "matlab_function"
                symbol = name
            else:
                kind = (
                    "matlab_constructor"
                    if name == class_name
                    else "matlab_method"
                )
                symbol = (
                    f"{class_name}.{name}" if name is not None else class_name
                )
            markers.append((line_number, kind, symbol))
            continue

        block = MATLAB_BLOCK.match(line)
        if block is not None:
            name = block.group(1).lower()
            markers.append(
                (
                    line_number,
                    f"matlab_{name}_block",
                    class_name,
                )
            )

    return markers


def _partition_matlab(
    path: Path,
    lines: list[str],
    *,
    start_line: int,
    end_line: int,
    markers: list[tuple[int, str, str | None]],
    context_kind: str,
    context_symbol: str | None,
) -> tuple[SourceUnit, ...]:
    """Partition a MATLAB range at visible structural markers.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.
    start_line: int
        First one-based line in the range.
    end_line: int
        Last one-based line in the range.
    markers: list[tuple[int, str, str | None]]
        Structural markers inside the range.
    context_kind: str
        Kind used for text outside a marked declaration.
    context_symbol: str | None
        Symbol used for surrounding context.

    Returns
    -------
    tuple[SourceUnit, ...]
        Complete, non-overlapping source units.

    """

    if not markers:
        context = _source_unit(
            path,
            lines,
            start_line,
            end_line,
            kind=context_kind,
            symbol=context_symbol,
        )
        return (context,) if context is not None else ()

    units: list[SourceUnit] = []
    cursor = start_line
    for index, (marker_line, kind, symbol) in enumerate(markers):
        context = _source_unit(
            path,
            lines,
            cursor,
            marker_line - 1,
            kind=context_kind,
            symbol=context_symbol,
        )
        if context is not None:
            units.append(context)

        next_line = (
            markers[index + 1][0] - 1 if index + 1 < len(markers) else end_line
        )
        marked = _source_unit(
            path,
            lines,
            marker_line,
            next_line,
            kind=kind,
            symbol=symbol,
        )
        if marked is not None:
            units.append(marked)
        cursor = next_line + 1

    return tuple(units)


def _matlab_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify MATLAB classes, blocks, methods, functions, and scripts.

    Parameters
    ----------
    path: pathlib.Path
        Relative .m source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        MATLAB units. The source is inspected as text and never executed.

    """

    class_line: int | None = None
    class_name: str | None = None
    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("%"):
            continue
        match = MATLAB_CLASS.match(line)
        if match is not None:
            class_line = line_number
            class_name = match.group(1)
            break

    if class_line is None or class_name is None:
        markers = _matlab_markers(
            lines,
            class_name=None,
            first_line=1,
        )
        return list(
            _partition_matlab(
                path,
                lines,
                start_line=1,
                end_line=len(lines),
                markers=markers,
                context_kind="matlab_script",
                context_symbol=path.stem,
            )
        )

    units: list[SourceUnit] = []
    preamble = _source_unit(
        path,
        lines,
        1,
        class_line - 1,
        kind="matlab_file_context",
        symbol=path.stem,
    )
    if preamble is not None:
        units.append(preamble)

    markers = _matlab_markers(
        lines,
        class_name=class_name,
        first_line=class_line + 1,
    )
    children = _partition_matlab(
        path,
        lines,
        start_line=class_line,
        end_line=len(lines),
        markers=markers,
        context_kind="matlab_class_context",
        context_symbol=class_name,
    )
    class_unit = _source_unit(
        path,
        lines,
        class_line,
        len(lines),
        kind="matlab_class",
        symbol=class_name,
        children=children,
    )
    if class_unit is not None:
        units.append(class_unit)

    return units


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one MATLAB source file.

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

    return _matlab_units(source_file.path, lines)
