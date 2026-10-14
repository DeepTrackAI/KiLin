"""Small source objects for model-free tool tests.

Construct immutable in-memory source fragments with realistic provenance.

"""

from __future__ import annotations

from pathlib import Path

from qachat.chunking import Chunk

__all__ = ["chunk"]


def chunk(
    identifier: str, text: str, *, path: str = "paper.md",
    line: int = 1,
) -> Chunk:
    """Construct a minimal indexed source fragment.

    Parameters
    ----------
    identifier: str
        Identifier unique within the test.
    text: str
        Exact indexed text.
    path: str, optional
        Relative file path. Defaults to ``"paper.md"``.
    line: int, optional
        First source line. Defaults to ``1``.

    Returns
    -------
    Chunk
        Immutable source fragment.

    """

    return Chunk(
        identifier=identifier,
        path=Path(path),
        text=text,
        start_line=line,
        end_line=line + text.count("\n"),
        kind="markdown",
        symbol=None,
        section=None,
    )
