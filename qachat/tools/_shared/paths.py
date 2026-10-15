"""Name sources by the shortest unique ending of each path.

Algorithms shared by registered tools belong here rather than in
another executable tool. The module has no GUI dependency.

Key Features
------------

- **Unique source names**

    Deterministic operations reusable by multiple tool packages.

Module Structure
----------------

- ``short_names``

    Public operations shared within the tools package.

Examples
--------
>>> from qachat.tools._shared.paths import short_names
>>> short_names([])
{}

"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "short_names",
]


def short_names(paths: Sequence[Path]) -> dict[Path, str]:
    """Name each file by the shortest path ending that is unique.

    Parameters
    ----------
    paths: Sequence[pathlib.Path]
        Files to name.

    Returns
    -------
    dict[pathlib.Path, str]
        For example ``dev_intro/readme.md`` when several ``readme.md``
        files exist, and ``smith2020.pdf`` otherwise.

    """

    names: dict[Path, str] = {}
    for path in paths:
        parts = path.parts
        for size in range(1, len(parts) + 1):
            ending = parts[-size:]
            if not any(
                other != path and other.parts[-size:] == ending
                for other in paths
            ):
                break
        names[path] = "/".join(ending)

    return names

