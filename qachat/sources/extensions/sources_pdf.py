"""Extract selectable PDF text and expose page-scoped source units.

The specialized reader extracts page text with ``pdfminer.six`` layout
analysis, which builds words from glyph boxes and real font metrics, so word
spacing follows the printed page (``pypdf``'s plain mode split some
letter-spaced words, such as acronyms, in two). ``pypdf`` still checks
encryption and
counts pages, and supplies the text when ``pdfminer.six`` cannot parse a
file. The reader cleans problematic characters and joins pages with
form-feed separators. Extraction
turns nonblank pages into units labeled ``Page N``. Line numbers refer to the
extracted text within a page, not to a stable visual layout in the PDF.

No optical character recognition is performed. Encrypted files and documents
without extractable text are rejected. Image-only content, equations, columns,
and reading order are limited by the underlying text extractor.

Key Features
------------

- **Specialized reading**

    Convert supported PDF page text into the shared ``SourceFile``
    representation.

- **Page provenance**

    Preserve original page numbers even when blank pages yield no unit.

- **Explicit limitations**

    Report unsupported text extraction rather than silently returning an empty
    document.

Module Structure
----------------

- ``read_source_file``

    Extract text from one PDF without executing embedded content.

- ``extract_source_units``

    Return one natural unit for each non-empty PDF page.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository("/path/to/repository", extensions=[".pdf"])
>>> units = qachat.extract_source_units(qachat.read_repository_files(report))

The PDF hook requires its optional reader dependency when imported.

"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from pdfminer.high_level import extract_text as _pdfminer_text
from pdfminer.layout import LAParams
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from qachat.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "read_source_file",
    "extract_source_units",
]

PAGE_BREAK = "\f"
# Identifies the text extractor in index metadata; changing it re-indexes
# projects that read PDFs, because their extracted text changes.
PDF_TEXT_EXTRACTOR: Final[str] = "pdfminer.six layout v1"
# Glyphs without a Unicode mapping, such as some bullets and math symbols.
_UNMAPPED_GLYPH = re.compile(r"\(cid:\d+\)")
_BLANK_LINES = re.compile(r"\n[ \t]*(?:\n[ \t]*){2,}")


def read_source_file(target: Path, relative_path: Path) -> SourceFile:
    """Extract text from one PDF without executing embedded content.

    Parameters
    ----------
    target: pathlib.Path
        Absolute PDF path that has already passed the safety checks.
    relative_path: pathlib.Path
        File path relative to the repository root.

    Returns
    -------
    SourceFile
        Extracted pages separated by form-feed characters.

    Raises
    ------
    ValueError
        If the PDF is invalid, encrypted, or contains no extractable text.

    """

    try:
        reader = PdfReader(target)
    except (OSError, PdfReadError) as error:
        raise ValueError(
            f"Cannot read PDF: {relative_path.as_posix()}"
        ) from error

    if reader.is_encrypted:
        raise ValueError(
            f"Encrypted PDF is not supported: {relative_path.as_posix()}"
        )

    pages = _layout_pages(target, len(reader.pages))
    try:
        if pages is None:
            pages = [page.extract_text() or "" for page in reader.pages]
        pages = [_clean_extracted_text(page) for page in pages]
    except (KeyError, PdfReadError) as error:
        raise ValueError(
            f"Cannot extract PDF text: {relative_path.as_posix()}"
        ) from error

    if not any(page.strip() for page in pages):
        raise ValueError(
            f"PDF has no extractable text; scanned PDFs need OCR: "
            f"{relative_path.as_posix()}"
        )

    return SourceFile(relative_path, PAGE_BREAK.join(pages))


def _layout_pages(target: Path, page_count: int) -> list[str] | None:
    """Extract each page's text with ``pdfminer.six`` layout analysis.

    Parameters
    ----------
    target: pathlib.Path
        PDF to read.
    page_count: int
        Number of pages ``pypdf`` found; the result must match it so page
        numbers stay exact.

    Returns
    -------
    list[str] | None
        One text per page, or None when ``pdfminer.six`` cannot parse the
        file or finds a different number of pages; the caller then uses
        ``pypdf``'s plain extraction.

    """

    try:
        text = _pdfminer_text(str(target), laparams=LAParams())
    except Exception:  # Any parser failure falls back to pypdf.
        return None
    pages = text.split(PAGE_BREAK)
    # pdfminer ends every page, including the last, with a form feed.
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    if len(pages) != page_count:
        return None
    return pages


def _clean_extracted_text(text: str) -> str:
    """Replace malformed PDF Unicode while preserving valid characters.

    PDF font maps can produce isolated UTF-16 surrogate characters. They are
    not valid Unicode text and cannot be passed to a model tokenizer. This
    round trip combines valid surrogate pairs and replaces only isolated ones.

    Parameters
    ----------
    text: str
        Text returned by the PDF reader, potentially containing surrogate code
        points.

    Returns
    -------
    str
        Text with surrogate handling normalized by an intermediate UTF-16
        encoding/decoding step.

    """

    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    text = _UNMAPPED_GLYPH.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text)


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return one natural unit for each non-empty PDF page.

    Parameters
    ----------
    source_file: SourceFile
        PDF text with form-feed page separators.

    Returns
    -------
    list[SourceUnit]
        Non-empty page units in document order.

    """

    units: list[SourceUnit] = []
    pages = source_file.text.split(PAGE_BREAK)
    for page_number, page in enumerate(pages, start=1):
        lines = page.splitlines(keepends=True)
        if page and not lines:
            lines = [page]

        unit = _source_unit(
            source_file.path,
            lines,
            1,
            len(lines),
            kind="pdf_page",
            section=f"Page {page_number}",
        )
        if unit is not None:
            units.append(unit)

    return units
