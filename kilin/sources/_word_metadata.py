"""Read bibliographic metadata from Word documents without a model.

The metadata helper supports both modern OOXML (``.docx``) and OLE
Word 97-2003 (``.doc``) documents. It recognizes the actual container
signature, including when a document has the other Word suffix. Reader
imports are local so that reading a ``.docx`` never requires ``olefile``.

Unreadable or missing metadata is not an extraction error. Callers that
need a citation or a search header can fall back to document text or to
its filename without changing the document's source-reading policy.

Key Features
------------

- **Optional metadata**

    Read stored bibliographic properties without parsing the document body.

- **Format-independent dispatch**

    Use the container signature, not the filename, to select the reader.

Module Structure
----------------

- ``read_word_metadata``

    Return the document title and author when the stored properties exist.

Examples
--------
>>> from pathlib import Path
>>> from kilin.sources._word_metadata import read_word_metadata
>>> title, author = read_word_metadata(Path("paper.docx"))

"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "read_word_metadata",
]

# Word OLE and OOXML containers are identified independently of suffixes.
_OLE_SIGNATURE: Final[bytes] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_WORD_SUFFIXES: Final[frozenset[str]] = frozenset({".doc", ".docx"})


def read_word_metadata(file: Path) -> tuple[str | None, str | None]:
    """Return a Word file's stored title and author, when readable.

    Parameters
    ----------
    file: pathlib.Path
        Word file. The suffix must be ``.doc`` or ``.docx``; the content
        signature determines which metadata reader is used.

    Returns
    -------
    tuple[str | None, str | None]
        Title and full author string, or ``None`` for missing or unreadable
        properties. No title or author is inferred from the body text.

    """

    if file.suffix.lower() not in _WORD_SUFFIXES:
        return None, None

    try:
        with file.open("rb") as handle:
            signature = handle.read(len(_OLE_SIGNATURE))

        if signature == _OLE_SIGNATURE:
            import olefile

            with olefile.OleFileIO(str(file)) as document:
                metadata = document.get_metadata()
                return (
                    _property(metadata.title),
                    _property(metadata.author),
                )

        from docx import Document

        properties = Document(str(file)).core_properties
        return _property(properties.title), _property(properties.author)
    except Exception:  # pylint: disable=broad-exception-caught
        # Metadata is optional; a reader error cannot abort indexing.
        return None, None


def _property(value: object) -> str | None:
    """Normalize a stored Word property without inventing information.

    Parameters
    ----------
    value: object
        A core-property string or an OLE metadata value.

    Returns
    -------
    str | None
        A nonempty, whitespace-normalized string, when available.

    """

    if isinstance(value, bytes):
        # OLE metadata is usually decoded already; retain a readable
        # fallback for binary strings returned by older file producers.
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None
