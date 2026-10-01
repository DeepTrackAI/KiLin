"""Expose source-file records and structured source extraction.

This package separates selected file reading from text-unit extraction.
``SourceFile`` stores repository-relative text, and ``SourceUnit`` adds
structural provenance for downstream chunking. Suffix-specific hooks provide
Python, MATLAB, Markdown, notebook, PDF, and plain-text handling.

Key Features
------------

- **Shared source records**

    Keep provenance consistent across format-specific readers and extractors.

- **Explicit I/O**

    Read only the selected files when the caller invokes the source-reading
    API.

Module Structure
----------------

- ``SourceFile``

    One repository file converted to text.

- ``SourceUnit``

    One natural section of a source file.

- ``extract_source_units``

    Identify natural source units without executing source code.

- ``read_repository_files``

    Read files selected by qachat.repository.scan_repository.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository("/path/to/repository")
>>> files = qachat.read_repository_files(report)
>>> units = qachat.extract_source_units(files)

"""

from .sources import (
    SourceFile,
    SourceUnit,
    extract_source_units,
    read_repository_files,
)


__all__ = [
    "SourceFile",
    "SourceUnit",
    "extract_source_units",
    "read_repository_files",
]
