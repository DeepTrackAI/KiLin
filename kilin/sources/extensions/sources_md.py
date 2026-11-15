"""Partition Markdown text at ATX-style heading lines.

Heading lines beginning with one to six hash characters define successive
flat sections. The text before the first heading is preserved separately.
Headings inside backtick or tilde code fences are ignored. This lightweight
implementation does not parse nested Markdown structure or setext headings.

Key Features
------------

- **Heading sections**

    Attach recognized heading text to section provenance.

- **Original coordinates**

    Keep inclusive source-file line ranges and original section text.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one Markdown source file.

- ``markdown_headings``

    The ATX headings outside code fences, with their level.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".md"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

"""

from __future__ import annotations

from pathlib import Path
import re

from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "extract_source_units",
    "markdown_headings",
]

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")


def _markdown_headings(lines: list[str]) -> list[tuple[int, str]]:
    """Find ATX headings outside backtick and tilde code fences.

    Parameters
    ----------
    lines: list[str]
        Markdown source lines, retaining line endings where available.

    Returns
    -------
    list[tuple[int, str]]
        One-based line numbers and normalized heading titles.

    Notes
    -----
    Fences may be indented by up to three spaces. A closing fence must use
    the opening character at least as many times, with only trailing spaces.
    Backtick fence info strings cannot contain backticks. Heading detection
    otherwise keeps the existing lightweight ATX syntax.

    """

    headings: list[tuple[int, str]] = []
    fence: str | None = None
    for line_number, source_line in enumerate(lines, start=1):
        line = source_line.rstrip("\r\n")
        if fence is not None:
            closing = _FENCE_CLOSE.match(line)
            if (
                closing is not None
                and closing.group(1)[0] == fence[0]
                and len(closing.group(1)) >= len(fence)
            ):
                fence = None
            continue

        opening = _FENCE_OPEN.match(line)
        if opening is not None:
            marker = opening.group(1)
            if marker[0] != "`" or "`" not in opening.group(2):
                fence = marker
                continue

        match = MARKDOWN_HEADING.match(line)
        if match is not None:
            title = match.group(2).strip().rstrip("#").rstrip()
            headings.append((line_number, title))

    return headings


def markdown_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return the ATX headings of Markdown text, with their level.

    Parameters
    ----------
    lines: list[str]
        Markdown source lines.

    Returns
    -------
    list[tuple[int, int, str]]
        One-based line, level (the number of ``#``, 1 to 6) and title of
        each heading outside code fences, in order.

    """

    return [
        (number, len(match.group(1)), title)
        for number, title in _markdown_headings(lines)
        if (match := MARKDOWN_HEADING.match(lines[number - 1].rstrip("\r\n")))
    ]


def _markdown_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify Markdown preamble and heading sections.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        Markdown units.

    """

    headings = _markdown_headings(lines)

    units: list[SourceUnit] = []
    first_heading = headings[0][0] if headings else len(lines) + 1
    preamble = _source_unit(
        path,
        lines,
        1,
        first_heading - 1,
        kind="markdown_preamble",
    )
    if preamble is not None:
        units.append(preamble)

    for index, (start_line, title) in enumerate(headings):
        end_line = (
            headings[index + 1][0] - 1
            if index + 1 < len(headings)
            else len(lines)
        )
        section = _source_unit(
            path,
            lines,
            start_line,
            end_line,
            kind="markdown_section",
            section=title,
        )
        if section is not None:
            units.append(section)

    return units


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one Markdown source file.

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

    return _markdown_units(source_file.path, lines)
