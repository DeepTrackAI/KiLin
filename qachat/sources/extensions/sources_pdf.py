"""Extract selectable PDF text and expose page-scoped source units.

The specialized reader uses ``pypdf`` to extract page text, cleans problematic
surrogate characters, and joins pages with form-feed separators. Extraction
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

    try:
        pages = [
            _clean_extracted_text(page.extract_text() or "")
            for page in reader.pages
        ]
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

    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


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
