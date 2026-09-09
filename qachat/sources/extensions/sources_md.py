"""Extract heading sections from Markdown source.

Functions
---------
extract_source_units
    Keep the preamble and make one natural unit for each heading section.

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

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


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

    headings: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, 1):
        match = MARKDOWN_HEADING.match(line.rstrip("\r\n"))
        if match is not None:
            title = match.group(2).strip().rstrip("#").rstrip()
            headings.append((line_number, title))

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
