r"""Index compiled LaTeX bibliography entries as searchable source units.

A user-selected ``.bbl`` file is indexed independently of any ``.tex``
document that cites it. Entries retain their original content and
one-based source lines; LaTeX compilation is never performed.

Key Features
------------

- **Reference boundaries**

    ``\\bibitem`` starts an individual searchable bibliography entry.

- **Exact provenance**

    File text and coordinates are kept unchanged for citation previews.

Module Structure
----------------

- ``extract_source_units``

    Split a compiled bibliography into entry and surrounding text units.

Examples
--------
>>> from pathlib import Path
>>> from kilin.sources.sources import SourceFile
>>> from kilin.sources.extensions.sources_bbl import extract_source_units
>>> units = extract_source_units(SourceFile(
...     Path("paper.bbl"), "\\\\bibitem{key} A paper.\n"))
>>> (units[0].kind, units[0].symbol)
('bbl_entry', 'key')

"""

from __future__ import annotations

import re
from typing import Final

from kilin.sources.sources import SourceFile, SourceUnit, _source_unit

__all__ = ["extract_source_units"]

_ITEM: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}"
)


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Split a compiled bibliography into exact-text reference units.

    Parameters
    ----------
    source_file: SourceFile
        Selected ``.bbl`` file and decoded source text.

    Returns
    -------
    list[SourceUnit]
        Each ``\\bibitem`` and its following lines form one unit.
        Leading content, or a file without items, remains searchable.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []
    starts = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines, 1)
        if (match := _ITEM.match(line)) is not None
    ]
    if not starts:
        unit = _source_unit(
            source_file.path, lines, 1, len(lines), kind="bbl_text"
        )
        return [unit] if unit is not None else []
    units = []
    if starts[0][0] > 1:
        initial = _source_unit(
            source_file.path, lines, 1, starts[0][0] - 1,
            kind="bbl_preamble",
        )
        if initial is not None:
            units.append(initial)
    for index, (start, key) in enumerate(starts):
        end = (
            starts[index + 1][0] - 1
            if index + 1 < len(starts) else len(lines)
        )
        unit = _source_unit(
            source_file.path, lines, start, end, kind="bbl_entry",
            symbol=key or None, section="Bibliography",
        )
        if unit is not None:
            units.append(unit)
    return units
