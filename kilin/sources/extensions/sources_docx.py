"""Read Word ``.docx`` files as Markdown-style text with heading sections.

The specialized reader opens the document with ``python-docx`` and walks the
body in reading order. Headings become Markdown ``#`` lines, list items
become ``-`` lines, and each table row becomes one line with its cells
separated by `` | ``. Blocks are separated by blank lines. Because the text
is Markdown-style, extraction reuses the Markdown heading sections, so a
Word passage is cited by its line range and its heading like a Markdown
passage. Line numbers refer to the extracted text, not to Word pages.

Headers, footers, footnotes, comments, text boxes, images, and equations
are not read. Macros and embedded objects are never executed.

A file is recognized by its first bytes, not by its name: a ``.docx`` file
that is really an old ``.doc`` file is read by the ``.doc`` reader.

Key Features
------------

- **Reading order**

    Keep paragraphs, headings, list items, and tables in document order.

- **Heading sections**

    Turn Word heading styles into Markdown headings, so each passage keeps
    the heading it belongs to.

- **Clear failures**

    Report damaged or password-protected files, and a missing reader
    package, with a readable message.

Module Structure
----------------

- ``read_source_file``

    Extract the text of one Word file without executing embedded content.

- ``extract_source_units``

    Return Markdown-style heading sections of the extracted text.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".docx"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

The Word hook requires the ``python-docx`` package when imported.

"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
from typing import Final
from zipfile import BadZipFile

try:
    from docx import Document as _open_document
    from docx.opc.exceptions import PackageNotFoundError
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    from lxml.etree import XMLSyntaxError
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Reading Word (.docx) files needs the python-docx package. "
        "Reinstall KíLín, or run: pip install python-docx",
        name=error.name,
    ) from error

from kilin.sources.extensions import sources_md
from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
)

__all__ = [
    "read_source_file",
    "extract_source_units",
]

# Word style names such as "Heading 1"; Markdown has six heading levels.
_HEADING_STYLE: Final[re.Pattern[str]] = re.compile(r"^Heading ([1-9])$")
_TITLE_STYLE: Final[str] = "Title"
_MAX_HEADING_LEVEL: Final[int] = 6
# Body lines that Markdown would read as a heading or a code fence.
_MARKDOWN_MARKUP: Final[re.Pattern[str]] = re.compile(r"^(?:#|```|~~~)")
_CELL_SEPARATOR: Final[str] = " | "
_OLE_SIGNATURE: Final[bytes] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_BUILTIN_LIST_STYLE: Final[re.Pattern[str]] = re.compile(
    r"^List (?:Bullet|Number)(?: ([2-9]))?$"
)
_BLOCK_SEPARATOR: Final[str] = "\n\n"
# Errors raised by python-docx and its zip and XML layers for bad files.
_READ_ERRORS: Final[tuple[type[Exception], ...]] = (
    BadZipFile,
    KeyError,
    OSError,
    PackageNotFoundError,
    ValueError,
    XMLSyntaxError,
)


def read_source_file(target: Path, relative_path: Path) -> SourceFile:
    """Extract the text of one Word file without executing embedded content.

    Parameters
    ----------
    target: pathlib.Path
        Absolute ``.docx`` path that has already passed the safety checks.
    relative_path: pathlib.Path
        File path relative to the repository root.

    Returns
    -------
    SourceFile
        Markdown-style text of the document body; empty when the document
        has no text.

    Raises
    ------
    ValueError
        If the file is not a readable Word document, for example because it
        is damaged or password-protected.

    """

    with target.open("rb") as handle:
        start = handle.read(len(_OLE_SIGNATURE))
    if start == _OLE_SIGNATURE:
        # Only an actual OLE file needs the legacy reader and olefile.
        from kilin.sources.extensions import sources_doc

        return sources_doc.read_source_file(target, relative_path)

    try:
        document = _open_document(str(target))
        blocks = _block_texts(document.iter_inner_content())
    except _READ_ERRORS as error:
        raise ValueError(
            f"Cannot read Word file; it may be damaged or "
            f"password-protected: {relative_path.as_posix()}"
        ) from error

    text = _BLOCK_SEPARATOR.join(blocks)
    return SourceFile(relative_path, f"{text}\n" if text else "")


def _block_texts(blocks: Iterable[Paragraph | Table]) -> list[str]:
    """Convert body paragraphs and tables into nonblank text blocks.

    Parameters
    ----------
    blocks: Iterable[Paragraph | Table]
        Paragraphs and tables in reading order, as returned by
        ``iter_inner_content``.

    Returns
    -------
    list[str]
        One text block per nonblank paragraph or table, in reading order.

    """

    texts: list[str] = []
    for block in blocks:
        if isinstance(block, Table):
            text = "\n".join(_plain_line(row) for row in _table_rows(block))
        else:
            text = _paragraph_text(block)
        if text:
            texts.append(text)
    return texts


def _paragraph_text(paragraph: Paragraph) -> str:
    """Return one paragraph as a Markdown heading, list item, or text.

    Parameters
    ----------
    paragraph: docx.text.paragraph.Paragraph
        Body paragraph.

    Returns
    -------
    str
        Markdown-style text, or an empty string for a blank paragraph.

    """

    lines = [line.strip() for line in paragraph.text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    level = _heading_level(paragraph)
    if level is not None:
        return f"{'#' * level} {' '.join(lines)}"

    lines = [_plain_line(line) for line in lines]
    depth = _list_depth(paragraph)
    if depth is not None:
        indent = "  " * depth
        lines[0] = f"{indent}- {lines[0]}"
        lines[1:] = [f"{indent}  {line}" for line in lines[1:]]
    return "\n".join(lines)


def _heading_level(paragraph: Paragraph) -> int | None:
    """Return the Markdown heading level of a Word heading paragraph.

    Parameters
    ----------
    paragraph: docx.text.paragraph.Paragraph
        Body paragraph.

    Returns
    -------
    int | None
        Level from 1 to 6 for the "Title" and "Heading N" styles; None for
        other paragraphs.

    Notes
    -----
    Word stores the names of its built-in styles in English, so documents
    written in other languages are recognized too.

    """

    style = paragraph.style
    name = style.name if style is not None else None
    if name == _TITLE_STYLE:
        return 1
    match = _HEADING_STYLE.match(name or "")
    if match is None:
        return None
    return min(int(match.group(1)), _MAX_HEADING_LEVEL)


def _list_depth(paragraph: Paragraph) -> int | None:
    """Return the nesting depth of a bulleted or numbered list item.

    Parameters
    ----------
    paragraph: docx.text.paragraph.Paragraph
        Body paragraph.

    Returns
    -------
    int | None
        Zero for a top-level item, larger for nested items; None when the
        paragraph is not a list item.

    Notes
    -----
    ``python-docx`` has no public list API. Numbering comes from paragraph
    or inherited style XML; indented but unnumbered styles stay plain text.
    Built-in nested list styles specify indentation without an ``ilvl``.

    """

    properties = paragraph._p.pPr
    direct = properties.numPr if properties is not None else None

    # Numbering can be on the paragraph or inherited from a paragraph
    # style. "List Paragraph" has indentation but no numbering and must
    # not be treated as a bullet merely because its name starts "List ".
    style = paragraph.style
    inherited = None
    seen: set[str] = set()
    while style is not None and style.style_id not in seen:
        seen.add(style.style_id)
        properties = style._element.pPr
        candidate = properties.numPr if properties is not None else None
        if candidate is not None:
            inherited = candidate
            break
        style = style.base_style

    if direct is not None and direct.numId is not None:
        if direct.numId.val == 0:
            # Word uses numId=0 to suppress numbering from the style.
            return None
    elif inherited is None or inherited.numId is None:
        return None

    if direct is not None and direct.ilvl is not None:
        return int(direct.ilvl.val)
    if inherited is not None and inherited.ilvl is not None:
        return int(inherited.ilvl.val)

    # Built-in List Bullet/Number 2 and 3 use distinct numbering IDs with
    # a level-zero definition but are visibly indented. Their style name
    # supplies the logical depth when no explicit level is recorded.
    style = paragraph.style
    while style is not None:
        match = _BUILTIN_LIST_STYLE.match(style.name or "")
        if match is not None:
            return int(match.group(1) or 1) - 1
        style = style.base_style
    return 0


def _table_rows(table: Table) -> list[str]:
    """Return each nonblank table row as one line of cell texts.

    Parameters
    ----------
    table: docx.table.Table
        Word table, possibly with merged or nested cells.

    Returns
    -------
    list[str]
        Row lines with cells separated by ``" | "``.

    Notes
    -----
    A merged cell appears once per row, not once per grid column it spans.

    """

    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen: set[int] = set()
        for cell in row.cells:
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(_cell_text(cell))
        if any(cells):
            rows.append(_CELL_SEPARATOR.join(cells))
    return rows


def _cell_text(cell: _Cell) -> str:
    """Return a table cell's text, including nested tables, on one line.

    Parameters
    ----------
    cell: docx.table._Cell
        Table cell.

    Returns
    -------
    str
        Cell text with all whitespace runs collapsed to single spaces.

    """

    parts: list[str] = []
    for block in cell.iter_inner_content():
        if isinstance(block, Table):
            parts.extend(_table_rows(block))
        else:
            parts.append(block.text)
    return " ".join(" ".join(parts).split())


def _plain_line(line: str) -> str:
    """Escape a body line that Markdown would read as markup.

    Parameters
    ----------
    line: str
        One line of paragraph or table text.

    Returns
    -------
    str
        The line, with a backslash before a leading ``#`` or code fence so
        it cannot start a false section.

    """

    if _MARKDOWN_MARKUP.match(line):
        return f"\\{line}"
    return line


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return Markdown-style heading sections of the extracted text.

    Parameters
    ----------
    source_file: SourceFile
        Text returned by ``read_source_file``.

    Returns
    -------
    list[SourceUnit]
        ``markdown_preamble`` and ``markdown_section`` units in reading
        order; empty for a document without text.

    """

    return sources_md.extract_source_units(source_file)
