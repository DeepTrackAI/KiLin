"""Read PDF text and make one unit per non-empty page.

Functions
---------
read_source_file
    Extract text from a binary PDF file.
extract_source_units
    Make page-based units from the extracted text.

Scanned image PDFs require OCR and are not supported here.

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
        pages = [page.extract_text() or "" for page in reader.pages]
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
