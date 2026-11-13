"""Read the text of old binary Word ``.doc`` files.

The old Word format (Word 97-2003) stores a document as streams inside one
container file. The text sits in the ``WordDocument`` stream, and a small
index there, the piece table, says which byte ranges hold text and whether
each range uses one byte (Windows-1252) or two bytes (UTF-16) per letter.
The reader opens the container with ``olefile``, follows the piece table,
and keeps the main body text only. Headers, footers, footnotes, comments,
text boxes, images, and formatting are not read, and headings are not
recognized, so a ``.doc`` passage is cited by line range only. Macros and
embedded objects are never executed.

Paragraphs become blocks separated by blank lines, and each table row
becomes one line with its cells separated by `` | ``, as for ``.docx``.
Field codes, such as the target of a link, are dropped and the shown text
kept. Line numbers refer to the extracted text, not to Word pages.

A file is recognized by its first bytes, not by its name: a ``.doc`` file
that is really a ``.docx`` file is read by the ``.docx`` reader.

Key Features
------------

- **Content only**

    Read the body text in reading order without Word or LibreOffice.

- **Same layout as .docx**

    Use the same blocks and table rows as the ``.docx`` reader.

- **Clear failures**

    Report damaged, password-protected, and pre-1997 Word files with a
    readable message.

Module Structure
----------------

- ``read_source_file``

    Extract the body text of one old Word file.

- ``extract_source_units``

    Return Markdown-style sections of the extracted text.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".doc"])
>>> units = kilin.extract_source_units(kilin.read_repository_files(report))

The old Word hook requires the ``olefile`` package when imported.

"""

from __future__ import annotations

from pathlib import Path
import re
import struct
from typing import Final

try:
    import olefile
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Reading old Word (.doc) files needs the olefile package. "
        "Reinstall KíLín, or run: pip install olefile",
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

# First bytes of an old Office container and of a zip (.docx) file.
OLE_SIGNATURE: Final[bytes] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_SIGNATURE: Final[bytes] = b"PK\x03\x04"

# File information block (FIB) at the start of the WordDocument stream.
_WORD_IDENT: Final[int] = 0xA5EC
_WORD_97_VERSION: Final[int] = 0x00C1
_FLAGS_OFFSET: Final[int] = 0x000A
_FLAG_TABLE_1: Final[int] = 0x0200
_FLAG_ENCRYPTED: Final[int] = 0x0100
_FLAG_OBFUSCATED: Final[int] = 0x8000
_FIB_BASE_SIZE: Final[int] = 32
# Position of the main text length and of the piece table in the FIB.
_TEXT_LENGTH_INDEX: Final[int] = 3
_PIECE_TABLE_INDEX: Final[int] = 33

# Piece table records and their flags.
_CLX_PROPERTIES: Final[int] = 0x01
_CLX_PIECES: Final[int] = 0x02
_PIECE_SIZE: Final[int] = 8
_PIECE_COMPRESSED: Final[int] = 0x40000000
_PIECE_OFFSET: Final[int] = 0x3FFFFFFF

# Word control characters in the text.
_FIELD_BEGIN: Final[str] = "\x13"
_FIELD_SEPARATOR: Final[str] = "\x14"
_FIELD_END: Final[str] = "\x15"
_ROW_END: Final[str] = "\x07\x07"
_CELL_END: Final[str] = "\x07"
_PARAGRAPH_END: Final[str] = "\r"
# Line, page, section, and column breaks start a new block.
_BREAKS: Final[re.Pattern[str]] = re.compile(r"[\x0b\x0c\x0e]")
# Other control characters (pictures, notes, optional hyphens) carry no text.
_CONTROL: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0f-\x1f]")
_NON_BREAKING_HYPHEN: Final[str] = "\x1e"

# Body lines that Markdown would read as a heading or a code fence.
_MARKDOWN_MARKUP: Final[re.Pattern[str]] = re.compile(r"^(?:#|```|~~~)")
_CELL_SEPARATOR: Final[str] = " | "
_BLOCK_SEPARATOR: Final[str] = "\n\n"
# Errors raised by olefile, struct, and decoding for bad files.
_READ_ERRORS: Final[tuple[type[Exception], ...]] = (
    IndexError,
    OSError,
    struct.error,
    ValueError,
)


def read_source_file(target: Path, relative_path: Path) -> SourceFile:
    """Extract the body text of one old Word file.

    Parameters
    ----------
    target: pathlib.Path
        Absolute ``.doc`` path that has already passed the safety checks.
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
        is damaged, password-protected, or made by Word 95 or older.

    """

    with target.open("rb") as handle:
        start = handle.read(len(OLE_SIGNATURE))
    if start.startswith(ZIP_SIGNATURE):
        # A .docx file with a .doc name; imported here to avoid a cycle.
        from kilin.sources.extensions import sources_docx

        return sources_docx.read_source_file(target, relative_path)

    name = relative_path.as_posix()
    try:
        raw = _raw_text(target)
    except _DocumentProblem as problem:
        raise ValueError(f"{problem}: {name}") from None
    except _READ_ERRORS as error:
        raise ValueError(
            f"Cannot read Word file; it may be damaged or "
            f"password-protected: {name}"
        ) from error

    text = _BLOCK_SEPARATOR.join(_blocks(raw))
    return SourceFile(relative_path, f"{text}\n" if text else "")


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return Markdown-style sections of the extracted text.

    Parameters
    ----------
    source_file: SourceFile
        Text returned by ``read_source_file``.

    Returns
    -------
    list[SourceUnit]
        ``markdown_preamble`` and ``markdown_section`` units in reading
        order; empty for a document without text. The text has no headings
        unless it came from a ``.docx`` file with a ``.doc`` name.

    """

    return sources_md.extract_source_units(source_file)


class _DocumentProblem(Exception):
    """A readable Word file that KíLín cannot extract, with the reason."""


def _raw_text(target: Path) -> str:
    """Return the main body text of a Word 97-2003 file as stored.

    Parameters
    ----------
    target: pathlib.Path
        Old Word file to read.

    Returns
    -------
    str
        Body text with Word's control characters still in place.

    Raises
    ------
    _DocumentProblem
        If the file is protected or older than Word 97.
    OSError
        If the file is not an old Office container or lacks a stream.
    struct.error
        If a stream is shorter than its records say.

    """

    with olefile.OleFileIO(str(target)) as container:
        word = container.openstream("WordDocument").read()
        ident, version = struct.unpack_from("<HH", word, 0)
        if ident != _WORD_IDENT:
            raise OSError("Not a Word document.")
        if version < _WORD_97_VERSION:
            raise _DocumentProblem(
                "Word file made by Word 95 or older; open it in Word or "
                "LibreOffice and save it as .docx"
            )
        (flags,) = struct.unpack_from("<H", word, _FLAGS_OFFSET)
        if flags & (_FLAG_ENCRYPTED | _FLAG_OBFUSCATED):
            raise _DocumentProblem("Word file is password-protected")
        table_name = "1Table" if flags & _FLAG_TABLE_1 else "0Table"
        table = container.openstream(table_name).read()

    text_length, clx_offset, clx_size = _fib_fields(word)
    clx = table[clx_offset : clx_offset + clx_size]
    return _piece_text(word, clx)[:text_length]


def _fib_fields(word: bytes) -> tuple[int, int, int]:
    """Return the body text length and piece table location from the FIB.

    Parameters
    ----------
    word: bytes
        Content of the ``WordDocument`` stream.

    Returns
    -------
    tuple[int, int, int]
        Number of body characters, and the offset and size of the piece
        table records in the table stream.

    """

    # The FIB has a fixed base, then three arrays, each after its length.
    position = _FIB_BASE_SIZE
    (short_count,) = struct.unpack_from("<H", word, position)
    position += 2 + 2 * short_count
    (long_count,) = struct.unpack_from("<H", word, position)
    longs = position + 2
    position = longs + 4 * long_count
    pairs = position + 2

    (text_length,) = struct.unpack_from(
        "<i", word, longs + 4 * _TEXT_LENGTH_INDEX
    )
    clx_offset, clx_size = struct.unpack_from(
        "<II", word, pairs + 8 * _PIECE_TABLE_INDEX
    )
    return max(text_length, 0), clx_offset, clx_size


def _piece_text(word: bytes, clx: bytes) -> str:
    """Join the text pieces listed in the piece table, in order.

    Parameters
    ----------
    word: bytes
        Content of the ``WordDocument`` stream, which holds the text.
    clx: bytes
        Piece table records from the table stream.

    Returns
    -------
    str
        Document text, including notes and headers after the body.

    Raises
    ------
    ValueError
        If the records have no piece table.

    """

    # Formatting records come first; each gives its own size.
    position = 0
    while position < len(clx) and clx[position] == _CLX_PROPERTIES:
        (size,) = struct.unpack_from("<H", clx, position + 1)
        position += 3 + size
    if position >= len(clx) or clx[position] != _CLX_PIECES:
        raise ValueError("No piece table.")

    (size,) = struct.unpack_from("<I", clx, position + 1)
    pieces = clx[position + 5 : position + 5 + size]
    # n + 1 character positions, then n piece descriptors.
    count = (len(pieces) - 4) // (4 + _PIECE_SIZE)
    positions = struct.unpack_from(f"<{count + 1}I", pieces, 0)
    descriptors = 4 * (count + 1)

    parts = []
    for index in range(count):
        length = positions[index + 1] - positions[index]
        (location,) = struct.unpack_from(
            "<I", pieces, descriptors + _PIECE_SIZE * index + 2
        )
        if location & _PIECE_COMPRESSED:
            start = (location & _PIECE_OFFSET) // 2
            parts.append(word[start : start + length].decode("cp1252"))
        else:
            start = location & _PIECE_OFFSET
            data = word[start : start + 2 * length]
            parts.append(data.decode("utf-16-le"))
    return "".join(parts)


def _blocks(raw: str) -> list[str]:
    """Turn stored Word text into paragraph blocks and table rows.

    Parameters
    ----------
    raw: str
        Body text with Word's control characters.

    Returns
    -------
    list[str]
        Nonempty blocks in reading order, with spaces normalized and
        Markdown markup escaped.

    Notes
    -----
    A table cell ends with one cell mark and a row with two. An empty last
    cell therefore also ends the row, so such a row may split in two.

    """

    text = _shown_text(raw)
    text = text.replace(_ROW_END, _PARAGRAPH_END)
    text = text.replace(_CELL_END, _CELL_SEPARATOR)
    text = text.replace(_NON_BREAKING_HYPHEN, "-")
    text = _BREAKS.sub(_PARAGRAPH_END, text)
    text = _CONTROL.sub("", text)

    blocks = []
    for paragraph in text.split(_PARAGRAPH_END):
        line = " ".join(paragraph.split())
        if line:
            blocks.append(_plain_line(line))
    return blocks


def _shown_text(raw: str) -> str:
    """Drop field codes and keep the text each field shows.

    Parameters
    ----------
    raw: str
        Body text with Word's field marks.

    Returns
    -------
    str
        The text without field codes, such as ``HYPERLINK "..."``.

    """

    kept = []
    # One entry per open field: True while in its code part.
    fields: list[bool] = []
    for character in raw:
        if character == _FIELD_BEGIN:
            fields.append(True)
        elif character == _FIELD_SEPARATOR:
            if fields:
                fields[-1] = False
        elif character == _FIELD_END:
            if fields:
                fields.pop()
        elif not any(fields):
            kept.append(character)
    return "".join(kept)


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
