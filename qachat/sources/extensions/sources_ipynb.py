"""Extract code and Markdown cell text from notebook JSON.

This hook reads notebook cells from the already-loaded JSON text. Only code
and Markdown cells become source units; outputs, execution state, attachments,
and raw cells are not indexed here. Cell numbers are one-based positions in
the original cell sequence, including cells skipped by the extractor.
Line ranges are local to each cell, not to the notebook JSON document.

Key Features
------------

- **Cell provenance**

    Preserve the original notebook path and cell position.

- **Text-only extraction**

    Use cell source text without executing cells or indexing output payloads.

Module Structure
----------------

- ``extract_source_units``

    Return non-empty code and Markdown cells from one notebook.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository(
...     "/path/to/repository", extensions=[".ipynb"],
... )
>>> units = qachat.extract_source_units(qachat.read_repository_files(report))

"""

from __future__ import annotations

import json
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


def _notebook_units(source_file: SourceFile) -> list[SourceUnit]:
    """Extract code and Markdown cells without stored outputs or metadata.

    Parameters
    ----------
    source_file: SourceFile
        UTF-8 notebook JSON.

    Returns
    -------
    list[SourceUnit]
        Non-empty code and Markdown cells with one-based cell provenance.

    Raises
    ------
    ValueError
        If the notebook is malformed.

    """

    try:
        notebook = json.loads(source_file.text)
        cells = notebook["cells"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
        ) from error

    if not isinstance(cells, list):
        raise ValueError(
            f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
        )

    units: list[SourceUnit] = []
    for cell_number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise ValueError(
                f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
            )

        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown"}:
            continue

        raw_source = cell.get("source", "")
        if isinstance(raw_source, list):
            if not all(isinstance(line, str) for line in raw_source):
                raise ValueError(
                    f"Invalid Jupyter notebook: "
                    f"{source_file.path.as_posix()}"
                )
            cell_text = "".join(raw_source)
        elif isinstance(raw_source, str):
            cell_text = raw_source
        else:
            raise ValueError(
                f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
            )

        if not cell_text.strip():
            continue

        lines = cell_text.splitlines(keepends=True)
        if not lines:
            lines = [cell_text]

        section: str | None = None
        if cell_type == "markdown":
            for line in lines:
                heading = MARKDOWN_HEADING.match(line.rstrip("\r\n"))
                if heading is not None:
                    section = heading.group(2).strip().rstrip("#").rstrip()
                    break

        unit = _source_unit(
            source_file.path,
            lines,
            1,
            len(lines),
            kind=f"notebook_{cell_type}",
            section=section,
            cell=cell_number,
        )
        if unit is not None:
            units.append(unit)

    return units


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return non-empty code and Markdown cells from one notebook.

    Parameters
    ----------
    source_file: SourceFile
        Notebook JSON converted to text.

    Returns
    -------
    list[SourceUnit]
        Code and Markdown cell units in notebook order.

    """

    return _notebook_units(source_file)
