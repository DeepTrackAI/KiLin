r"""Index BibTeX entries as individually searchable source units.

A selected ``.bib`` database is normal source material, not an implicit
file dependency. Entry and macro boundaries preserve the exact UTF-8
source text, citation keys, and one-based physical line coordinates.
No TeX expansion or BibTeX execution is performed.

Key Features
------------

- **Per-entry retrieval**

    Each top-level ``@type{key, ...}`` begins an indexed source unit.

- **Exact source provenance**

    Original lines, including blank separators, remain unchanged.

Module Structure
----------------

- ``extract_source_units``

    Partition a selected BibTeX source into searchable units.

Examples
--------
>>> from pathlib import Path
>>> from kilin.sources.sources import SourceFile
>>> from kilin.sources.extensions.sources_bib import extract_source_units
>>> units = extract_source_units(SourceFile(
...     Path("refs.bib"), "@article{key, title={Paper}}\n"))
>>> (units[0].kind, units[0].symbol)
('bib_entry', 'key')

"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from kilin.sources.sources import SourceFile, SourceUnit, _source_unit

__all__ = ["extract_source_units"]

_ENTRY: Final[re.Pattern[str]] = re.compile(
    r"^\s*@(?P<kind>[A-Za-z]+)\s*[{(]\s*(?P<key>[^,\s{}()]*)"
)
_NONREFERENCES: Final[frozenset[str]] = frozenset({
    "string", "comment", "preamble",
})


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Split a BibTeX database into provenance-bearing entry units.

    Parameters
    ----------
    source_file: SourceFile
        Selected BibTeX file and decoded text.

    Returns
    -------
    list[SourceUnit]
        One unit per recognized top-level entry, and a leading unit for
        any preceding text. Unrecognized content is retained rather than
        silently dropped. Malformed entries remain searchable as text.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []
    headings = [
        (index, match.group("kind").lower(), match.group("key").strip())
        for index, line in enumerate(lines, 1)
        if (match := _ENTRY.match(line)) is not None
    ]
    if not headings:
        unit = _source_unit(
            source_file.path, lines, 1, len(lines), kind="bib_text"
        )
        return [unit] if unit is not None else []
    units = []
    if headings[0][0] > 1:
        initial = _source_unit(
            source_file.path, lines, 1, headings[0][0] - 1,
            kind="bib_preamble",
        )
        if initial is not None:
            units.append(initial)
    for index, (start, kind, key) in enumerate(headings):
        end = (
            headings[index + 1][0] - 1
            if index + 1 < len(headings) else len(lines)
        )
        unit = _source_unit(
            source_file.path, lines, start, end,
            kind="bib_definition" if kind in _NONREFERENCES else "bib_entry",
            symbol=key or None, section=kind,
        )
        if unit is not None:
            units.append(unit)
    return units
