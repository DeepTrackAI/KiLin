"""Partition JSON text at the keys of its top-level object.

A JSON file is one unit. When it is too large for one passage, it is split
where each top-level key starts, and each part is labelled with its key,
such as ``"DEFAULT_MODELS"``. Line numbers refer to the file as written; a
file written on one line, or whose top level is not an object, stays one
unit and is split along its text. The text is scanned, not loaded, so a
file with an error is still read.

Key Features
------------

- **Top-level keys**

    Label the parts of a large object with the key they hold.

- **Original lines**

    Keep the file as written; nothing is reformatted.

Module Structure
----------------

- ``extract_source_units``

    Return one JSON file, split at its top-level keys.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".json"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

"""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from typing import Final

from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "extract_source_units",
]


# Changing this version reindexes profiles that select JSON files.
JSON_SOURCE_UNITS_FORMAT: Final[str] = "top-level-keys-v2"


def _top_level_keys(text: str, lines: list[str]) -> list[tuple[int, str]]:
    """Find where each key of the top-level object starts.

    Parameters
    ----------
    text: str
        JSON text.
    lines: list[str]
        Source lines, using the same line boundaries as source units.

    Returns
    -------
    list[tuple[int, str]]
        One-based line and name of each top-level key, in order; empty when
        the top level is not an object.

    """

    keys: list[tuple[int, str]] = []
    depth = 0
    expecting_key = False
    index = 0
    started = False
    while index < len(text):
        character = text[index]
        if character == '"':
            end = index + 1
            while end < len(text) and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            if depth == 1 and expecting_key:
                keys.append((index, text[index + 1 : end]))
                expecting_key = False
            index = end
        elif character in "{[":
            if not started and character == "[":
                return []
            started = True
            depth += 1
            expecting_key = depth == 1 and character == "{"
        elif character in "}]":
            depth -= 1
        elif character == "," and depth == 1:
            expecting_key = True
        index += 1
    # Convert key offsets using the exact splitlines boundaries that the
    # source-unit builder uses, including Unicode and mixed line endings.
    line_ends = list(accumulate(len(part) for part in lines))
    return [
        (bisect_right(line_ends, position) + 1, key)
        for position, key in keys
    ]


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return one JSON file, split at its top-level keys.

    Parameters
    ----------
    source_file: SourceFile
        JSON text read as written.

    Returns
    -------
    list[SourceUnit]
        One unit for the whole file. Its children start at each top-level
        key on its own line, with the text before the first key as context,
        and are used only when the file is too large for one passage.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []
    path = source_file.path
    # Keys sharing a line stay in one part, named by the first of them.
    starts: list[tuple[int, str]] = []
    for line, key in _top_level_keys(source_file.text, lines):
        if not starts or starts[-1][0] != line:
            starts.append((line, key))
    children: list[SourceUnit] = []
    first = starts[0][0] if starts else len(lines) + 1
    context = _source_unit(path, lines, 1, first - 1, kind="json_context")
    if context is not None:
        children.append(context)
    for index, (start, key) in enumerate(starts):
        end = (
            starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        )
        part = _source_unit(
            path, lines, start, end, kind="json_key", section=f'"{key}"'
        )
        if part is not None:
            children.append(part)
    whole = _source_unit(
        path,
        lines,
        1,
        len(lines),
        kind="json_file",
        children=tuple(children) if len(children) > 1 else (),
    )
    return [] if whole is None else [whole]
