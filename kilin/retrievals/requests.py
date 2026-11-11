"""Define bounded retrieval requests and results for core and tools.

Tool-owned retrieval requests share the answering pipeline's source-first
search and evidence budgeting without inheriting an answering intent's
limits. The result supplies original indexed chunks, not card summaries;
source cards affect discovery but never replace citeable source passages.

Key Features
------------

- **Explicit retrieval policy**

    A validated strategy, source scope, card-kind filter, and hard budget.

- **Original evidence**

    Return indexed chunks, selected source paths, and coverage diagnostics.

Module Structure
----------------

- ``RetrievalRequest``

    A bounded, model-free retrieval request.

- ``RetrievalResult``

    Selected original chunks and source-representation report.

Examples
--------
>>> from kilin.retrievals.requests import RetrievalRequest
>>> from kilin.retrievals.evidence_budget import EvidenceLimits
>>> limits = EvidenceLimits(3, 1, 8, 5000)
>>> request = RetrievalRequest("protein folding", limits)
>>> request.strategy
'source_first'

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Final, Literal

from kilin.retrievals.evidence_budget import EvidenceLimits

if TYPE_CHECKING:
    from kilin.chunking import Chunk

__all__ = [
    "RetrievalRequest",
    "RetrievalResult",
]

RetrievalStrategy = Literal["hybrid", "source_first"]
_STRATEGIES: Final[tuple[RetrievalStrategy, ...]] = (
    "hybrid", "source_first",
)


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Request bounded evidence through the core retrieval coordinator.

    Parameters
    ----------
    query: str
        Nonempty query, independent of an answering intent.
    limits: EvidenceLimits
        Tool- or caller-owned hard evidence limits.
    strategy: "hybrid" | "source_first", optional
        Retrieval strategy, defaulting to ``source_first``.
    scope: tuple[pathlib.Path, ...], optional
        Explicit source-relative paths. Multiple paths restrict the search.
    compare: bool, optional
        Allow other sources alongside a single named source. Defaults to
        ``False``.
    card_kinds: tuple[str, ...], optional
        Restrict card discovery to these registered kind identifiers.
        Empty means all kinds. Valid only for ``source_first``.

    Raises
    ------
    ValueError
        If the query, strategy, source paths, or kind filter is invalid.
    TypeError
        If limits are not an ``EvidenceLimits`` instance.

    """

    query: str
    limits: EvidenceLimits
    strategy: RetrievalStrategy = "source_first"
    scope: tuple[Path, ...] = ()
    compare: bool = False
    card_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous requests before indexing or model operations.

        Raises
        ------
        ValueError
            If settings or paths do not satisfy the retrieval contract.
        TypeError
            If the hard evidence limits have the wrong type.

        """

        if not isinstance(self.limits, EvidenceLimits):
            raise TypeError("limits must be EvidenceLimits.")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Retrieval query must be nonempty text.")
        if self.strategy not in _STRATEGIES:
            raise ValueError("Unknown retrieval strategy.")
        if type(self.compare) is not bool:
            raise ValueError("compare must be boolean.")
        if type(self.scope) is not tuple:
            raise ValueError("scope must be a tuple of relative paths.")
        for path in self.scope:
            if not isinstance(path, Path) or not _relative_path(path):
                raise ValueError("scope requires valid relative paths.")
        if type(self.card_kinds) is not tuple or any(
            not isinstance(kind, str) or not kind.strip()
            for kind in self.card_kinds
        ):
            raise ValueError("card_kinds must contain nonempty names.")
        if self.card_kinds and self.strategy != "source_first":
            raise ValueError("Card filters require source_first retrieval.")


def _relative_path(path: Path) -> bool:
    """Check portable source-relative paths without filesystem access.

    Parameters
    ----------
    path: pathlib.Path
        Candidate source path.

    Returns
    -------
    bool
        True for nonempty relative paths without parent traversal.

    """

    windows = PureWindowsPath(str(path))
    return (
        not path.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and bool(path.parts)
        and all(part not in ("", ".", "..") for part in path.parts)
        and all(part not in ("", ".", "..") for part in windows.parts)
    )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Original chunks selected through one bounded retrieval request.

    Parameters
    ----------
    passages: tuple[Chunk, ...]
        Original indexed chunks in final citation order; citation numbers
        are one-based positions. Card content is never returned as evidence.
    sources: tuple[pathlib.Path, ...]
        Sources chosen by the strategy, ordered by source relevance. A
        selected source can have no fitting evidence under the token budget.
    coverage: dict[str, object]
        JSON-compatible source-representation report. Coverage does not
        establish that the complete content of a source was examined.

    """

    passages: tuple[Chunk, ...]
    sources: tuple[Path, ...]
    coverage: dict[str, object]
