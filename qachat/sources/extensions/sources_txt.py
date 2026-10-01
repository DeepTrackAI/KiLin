"""Split plain-text sources into nonblank paragraph units.

Blank lines delimit paragraphs and are not included in the resulting units.
Each unit retains its inclusive one-based location in the original file.
The extractor does not infer headings, sentence boundaries, encoding, or
semantic structure; decoding has already occurred in the source reader.

Key Features
------------

- **Paragraph boundaries**

    Group consecutive nonblank lines without rewriting their text.

- **File provenance**

    Keep source paths and original line ranges for each paragraph.

Module Structure
----------------

- ``extract_source_units``

    Return natural units from one plain-text source file.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository("/path/to/repository", extensions=[".txt"])
>>> units = qachat.extract_source_units(qachat.read_repository_files(report))

"""

from __future__ import annotations

from pathlib import Path

from qachat.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)


__all__ = [
    "extract_source_units",
]


def _is_blank(line: str) -> bool:
    """Return whether a source line has no visible content.

    Parameters
    ----------
    line: str
        One source line, with or without a line ending.

    Returns
    -------
    bool
        True if the line contains no non-whitespace characters.

    """

    return not line.strip()


def _paragraph_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify plain-text paragraphs, separated by blank lines.

    A paragraph is a maximal run of non-blank lines. Blank lines between
    paragraphs are not included in either neighbor's span, matching how
    Markdown headings mark section boundaries without belonging to either
    section.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        Paragraph units in source order.

    """

    units: list[SourceUnit] = []
    paragraph_start: int | None = None

    for line_number, line in enumerate(lines, 1):
        if _is_blank(line):
            if paragraph_start is not None:
                unit = _source_unit(
                    path,
                    lines,
                    paragraph_start,
                    line_number - 1,
                    kind="text_paragraph",
                )
                if unit is not None:
                    units.append(unit)
                paragraph_start = None
        elif paragraph_start is None:
            paragraph_start = line_number

    if paragraph_start is not None:
        unit = _source_unit(
            path,
            lines,
            paragraph_start,
            len(lines),
            kind="text_paragraph",
        )
        if unit is not None:
            units.append(unit)

    return units


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return natural units from one plain-text source file.

    Parameters
    ----------
    source_file: SourceFile
        Source file converted to text.

    Returns
    -------
    list[SourceUnit]
        Natural units in source order. A file with no blank lines at all
        becomes a single paragraph unit covering the whole file -- the same
        result the generic whole-file fallback would have produced, so short
        documents are unaffected by this extension existing.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []

    return _paragraph_units(source_file.path, lines)
