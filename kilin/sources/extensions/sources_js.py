"""Partition JavaScript text into declarations and surrounding context.

The extractor recognizes line-leading top-level functions, classes, and
functions assigned to variables, including class methods. A lightweight
lexical scan distinguishes code from comments, strings, templates, and
regular-expression literals, and matches balanced braces to avoid assigning
unrelated statements to declarations. Comments above declarations remain
attached. This is not a JavaScript parser: dynamic constructs, declarations
embedded in expressions, and methods on the declaration line of a class
may remain context. No source code is executed.

Key Features
------------

- **Declarations**

    Label functions, classes and class methods at recognizable boundaries.

- **Conservative boundaries**

    Preserve source order and text without inventing code inside literals.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one JavaScript source file.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".js"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import re
from typing import Final

from kilin.sources.extensions._js_lex import mask_non_code
from kilin.sources.sources import SourceFile, SourceUnit, _source_unit

__all__ = ["extract_source_units"]

_NAME: Final[str] = r"([A-Za-z_$][\w$]*)"
_EXPORT: Final[str] = r"^(?:export\s+(?:default\s+)?)?"
JS_FUNCTION: Final[re.Pattern[str]] = re.compile(
    _EXPORT + r"(?:async\s+)?function\s*\*?\s*" + _NAME + r"\s*\("
)
JS_CLASS: Final[re.Pattern[str]] = re.compile(_EXPORT + r"class\s+" + _NAME)
JS_ASSIGNED_FUNCTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+" + _NAME + r"\s*=\s*(?:async\s+)?"
    r"(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>|\([^)]*$)"
)
JS_METHOD: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?\*?"
    r"(#?[A-Za-z_$][\w$]*)\s*\([^;]*\)\s*\{"
)
_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"if", "for", "while", "switch", "catch", "function", "return", "with"}
)
_COMMENT: Final[re.Pattern[str]] = re.compile(r"^\s*(?://|/\*|\*)")


def _declaration(line: str) -> tuple[str, str] | None:
    """Recognize a top-level declaration in a masked source line.

    Parameters
    ----------
    line: str
        JavaScript line outside comments and literal text.

    Returns
    -------
    tuple[str, str] | None
        Source-unit kind and symbol, or None for unrecognized syntax.

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


def _line_offsets(lines: list[str]) -> list[int]:
    """Return each source line's starting character offset.

    Parameters
    ----------
    lines: list[str]
        Source lines retaining their endings.

    Returns
    -------
    list[int]
        Zero-based offsets aligned with line numbers.

    """

    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    return starts


def _line_depths(lines: list[str]) -> list[tuple[int, int]]:
    """Return brace and parenthesis/bracket depths before each line.

    Parameters
    ----------
    lines: list[str]
        Lexically masked source lines.

    Returns
    -------
    list[tuple[int, int]]
        Brace depth and nonbrace nesting depth for each line.

    """

    depths: list[tuple[int, int]] = []
    braces = 0
    other = 0
    for line in lines:
        depths.append((braces, other))
        for char in line:
            if char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
            elif char in "([":
                other += 1
            elif char in ")]":
                other -= 1
    return depths


def _matching_brace(text: str, opening: int) -> int | None:
    """Find the matching closing brace in masked JavaScript code.

    Parameters
    ----------
    text: str
        Masked JavaScript source.
    opening: int
        Offset of an opening brace.

    Returns
    -------
    int | None
        Matching closing-brace offset or None for incomplete code.

    """

    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def _declaration_end(
    text: str, offsets: list[int], line: int, assigned: bool
) -> int | None:
    """Find the line closing a declaration without crossing its scope.

    Parameters
    ----------
    text: str
        Offset-preserving masked source.
    offsets: list[int]
        Start of each line in the source.
    line: int
        One-based declaration line.
    assigned: bool
        Whether the declaration begins with a variable assignment.

    Returns
    -------
    int | None
        One-based closing line, or None if the construct is incomplete.

    """

    position = offsets[line - 1]
    parens = 0
    brackets = 0
    arrow = False
    expression = False
    while position < len(text):
        char = text[position]
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif parens == 0 and brackets == 0:
            if char == "=" and text[position:position + 2] == "=>":
                arrow = True
                position += 2
                continue
            if arrow and not char.isspace():
                expression = char != "{"
                arrow = False
            if char == "{" and not expression:
                # Parenthesized arrow expressions can contain object
                # literals; they are handled by the semicolon boundary.
                closing = _matching_brace(text, position)
                if closing is None:
                    return None
                return bisect_right(offsets, closing)
            if char == ";" and assigned:
                return bisect_right(offsets, position)
            if char == "\n" and assigned and expression:
                # Without a block or semicolon, let the next declaration
                # bound the expression instead of guessing its grammar.
                return None
        position += 1
    return None


def _with_comment(lines: list[str], line: int, first: int) -> int:
    """Include immediately preceding line comments or JSDoc lines.

    Parameters
    ----------
    lines: list[str]
        Original source lines.
    line: int
        One-based declaration line.
    first: int
        Earliest line allowed for this partition.

    Returns
    -------
    int
        First line of attached comments, or the declaration line.

    """

    start = line
    while start - 1 >= first:
        previous = lines[start - 2].lstrip()
        if previous.startswith("//") or previous.startswith("/*"):
            start -= 1
            continue
        if previous.startswith("*/"):
            start -= 1
            while start - 1 >= first:
                start -= 1
                if lines[start - 1].lstrip().startswith("/*"):
                    break
            continue
        if _COMMENT.match(lines[start - 2]):
            start -= 1
            continue
        break
    return start


def _partition(
    lines: list[str],
    starts: list[tuple[int, str, str, int | None]],
    first: int,
    last: int,
    context_kind: str,
) -> list[tuple[int, int, str, str | None]]:
    """Partition whole source lines into declaration and context units.

    Parameters
    ----------
    lines: list[str]
        Original source lines.
    starts: list[tuple[int, str, str, int | None]]
        Declaration lines, kinds, names and structural closing lines.
    first: int
        First source line in this scope.
    last: int
        Last source line in this scope.
    context_kind: str
        Kind assigned to intervening code.

    Returns
    -------
    list[tuple[int, int, str, str | None]]
        Ordered line ranges covering the scope without overlaps.

    """

    parts: list[tuple[int, int, str, str | None]] = []
    position = first
    for index, (line, kind, name, closing) in enumerate(starts):
        start = _with_comment(lines, line, position)
        if start > position:
            parts.append((position, start - 1, context_kind, None))
        next_line = (
            _with_comment(lines, starts[index + 1][0], line + 1)
            if index + 1 < len(starts)
            else last + 1
        )
        boundary = next_line - 1
        end = min(closing if closing is not None else boundary, boundary)
        parts.append((start, end, kind, name))
        if end < boundary:
            parts.append((end + 1, boundary, context_kind, None))
        position = boundary + 1
    if position <= last:
        parts.append((position, last, context_kind, None))
    return parts


def _class_unit(
    path: Path,
    lines: list[str],
    safe_lines: list[str],
    depths: list[tuple[int, int]],
    masked: str,
    offsets: list[int],
    first: int,
    last: int,
    name: str,
) -> SourceUnit | None:
    """Create a class unit whose direct methods are children.

    Parameters
    ----------
    path: pathlib.Path
        Source path.
    lines: list[str]
        Original text lines.
    safe_lines: list[str]
        Masked lines with code delimiters preserved.
    depths: list[tuple[int, int]]
        Brace and parenthesis/bracket depth at each line start.
    masked: str
        Full masked source for matching method braces.
    offsets: list[int]
        Source line offsets.
    first: int
        First one-based line of this class's unit.
    last: int
        Last one-based line of the class unit.
    name: str
        Declared class name.

    Returns
    -------
    SourceUnit | None
        Class unit with immediate methods and intervening context.

    """

    methods = []
    for number in range(first + 1, last + 1):
        if depths[number - 1] != (1, 0):
            continue
        match = JS_METHOD.match(safe_lines[number - 1])
        if match is None or match.group(1) in _KEYWORDS:
            continue
        end = _declaration_end(masked, offsets, number, False)
        methods.append((number, "js_method", match.group(1), end))

    if methods:
        number, kind, method, end = methods[-1]
        if end is not None and all(
            line.strip() in {"", "}", "};"}
            for line in safe_lines[end:last]
        ):
            methods[-1] = (number, kind, method, last)

    children = tuple(
        unit
        for start, end, kind, method in _partition(
            lines, methods, first, last, "js_class_context"
        )
        if (
            unit := _source_unit(
                path, lines, start, end, kind=kind,
                symbol=None if method is None else f"{name}.{method}",
            )
        ) is not None
    )
    return _source_unit(
        path, lines, first, last, kind="js_class", symbol=name,
        children=children if methods else (),
    )


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one JavaScript source file.

    Parameters
    ----------
    source_file: SourceFile
        JavaScript source file read as text.

    Returns
    -------
    list[SourceUnit]
        Declarations and context, covering source lines in order.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []
    masked = mask_non_code(source_file.text)
    safe_lines = masked.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    depths = _line_depths(safe_lines)
    declarations: list[tuple[int, str, str, int | None]] = []
    for number, line in enumerate(safe_lines, start=1):
        if depths[number - 1] != (0, 0):
            continue
        found = _declaration(line)
        if found is None:
            continue
        kind, name = found
        assigned = JS_ASSIGNED_FUNCTION.match(line) is not None
        end = _declaration_end(masked, offsets, number, assigned)
        declarations.append((number, kind, name, end))

    units: list[SourceUnit] = []
    for first, last, kind, name in _partition(
        lines, declarations, 1, len(lines), "js_module_context"
    ):
        if kind == "js_class" and name is not None:
            unit = _class_unit(
                source_file.path, lines, safe_lines, depths, masked,
                offsets, first, last, name,
            )
        else:
            unit = _source_unit(
                source_file.path, lines, first, last,
                kind=kind, symbol=name,
            )
        if unit is not None:
            units.append(unit)
    return units
