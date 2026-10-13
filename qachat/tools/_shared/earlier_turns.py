"""Select an earlier turn with listed supporting evidence.

Algorithms shared by registered tools belong here rather than in
another executable tool. The module has no GUI dependency.

Key Features
------------

- **Evidence-aware turn selection**

    Deterministic operations reusable by multiple tool packages.

Module Structure
----------------

- ``newest_listed``

    Public operations shared within the tools package.

Examples
--------
>>> from qachat.tools._shared.earlier_turns import newest_listed
>>> newest_listed([]) is None
True

"""

from __future__ import annotations

from collections.abc import Sequence
from qachat.tools._contract import EarlierTurn

__all__ = [
    "newest_listed",
]


def newest_listed(earlier: Sequence[EarlierTurn]) -> EarlierTurn | None:
    """Return the newest related turn with evidence, else the newest one.

    Parameters
    ----------
    earlier: Sequence[EarlierTurn]
        Earlier turns, newest first.

    Returns
    -------
    EarlierTurn | None
        The turn whose evidence a follow-up is about, or None when no
        earlier turn has evidence.

    """

    listed = [turn for turn in earlier if turn.texts]
    linked = next((turn for turn in listed if turn.linked), None)
    return linked or next(iter(listed), None)

