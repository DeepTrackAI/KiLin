"""Apply deterministic passage, source, and evidence-token budgets.

Evidence limits are shared retrieval primitives, not model settings. They
preserve original passages and citation provenance while reserving source
coverage before distributing additional passages. Tools may use the same
functions with their own validated settings; they do not inherit the limits
of an unrelated answering intent.

Key Features
------------

- **Hard evidence limits**

    Bound the number of sources, original passages, and estimated evidence
    tokens, including headers and source locations.

- **Source coverage**

    Prefer one original passage per selected source before allocating
    additional passages up to the configured per-source target.

- **Deterministic reporting**

    Record omitted passages and the limits that prevented their inclusion.

Module Structure
----------------

- ``EvidenceLimits``

    Validated per-intent or per-tool retrieval limits.

- ``EvidenceSelection``

    The selected original evidence and its estimated resource usage.

- ``allocate_evidence``

    Reserve source coverage and select whole evidence items within limits.

- ``evidence_block``

    Render one numbered evidence block, also used in answer prompts.

- ``evidence_tokens``

    Estimate all displayed evidence text and headers.

Examples
--------
>>> from kilin.retrievals.evidence_budget import EvidenceLimits
>>> EvidenceLimits(3, 1, 8, 5000).max_passages
8

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final, Generic, Literal, Protocol, TypeVar

from kilin.retrievals.document_reading import estimate_tokens

__all__ = [
    "COVERAGE_BOUNDED",
    "COVERAGE_EXHAUSTIVE",
    "COVERAGE_MODES",
    "EvidenceLimits",
    "EvidenceSelection",
    "allocate_evidence",
    "coverage_report",
    "evidence_block",
    "evidence_tokens",
]

CoverageMode = Literal["bounded", "exhaustive"]
COVERAGE_BOUNDED: Final[CoverageMode] = "bounded"
COVERAGE_EXHAUSTIVE: Final[CoverageMode] = "exhaustive"
COVERAGE_MODES: Final[tuple[CoverageMode, ...]] = (
    COVERAGE_BOUNDED, COVERAGE_EXHAUSTIVE,
)


class _EvidenceItem(Protocol):
    """Fields required to budget a provenance-bearing evidence item.

    Attributes
    ----------
    path: pathlib.Path
        Relative source path.
    start_line: int
        Source start line.
    end_line: int
        Source end line.
    symbol: str | None
        Optional section or code symbol.
    context_role: str | None
        Role shown in the prompt when present.
    text: str
        Original passage text.

    """

    path: Path
    start_line: int
    end_line: int
    symbol: str | None
    context_role: str | None
    text: str


_EvidenceT = TypeVar("_EvidenceT", bound=_EvidenceItem)


@dataclass(frozen=True, slots=True)
class EvidenceLimits:
    """Limits for source-first evidence, independent of model calls.

    Parameters
    ----------
    max_sources: int
        Maximum number of represented sources.
    passages_per_source: int
        Desired passages per selected source; a target, not a guarantee.
    max_passages: int
        Hard number of evidence passages.
    max_tokens: int
        Hard estimated evidence tokens, including source headers.
    coverage: "bounded" | "exhaustive", optional
        Requested coverage mode. Defaults to ``"bounded"``.

    Raises
    ------
    ValueError
        If limits are invalid or mutually inconsistent.

    """

    max_sources: int
    passages_per_source: int
    max_passages: int
    max_tokens: int
    coverage: CoverageMode = COVERAGE_BOUNDED

    def __post_init__(self) -> None:
        """Reject invalid limits before retrieval or model inference.

        Raises
        ------
        ValueError
            If a value is not a positive integer, or coverage is unknown.

        """

        values = (
            self.max_sources, self.passages_per_source,
            self.max_passages, self.max_tokens,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Evidence limits must be positive integers.")
        if self.coverage not in COVERAGE_MODES:
            raise ValueError("Unknown evidence coverage mode.")


@dataclass(frozen=True, slots=True)
class EvidenceSelection(Generic[_EvidenceT]):
    """Whole evidence items retained under a validated budget.

    Parameters
    ----------
    items: tuple[_EvidenceT, ...]
        Retained original evidence, in answer-citation order.
    estimated_tokens: int
        Estimated tokens of rendered evidence, including headers.
    omitted: int
        Number of considered evidence items not selected.
    limits_hit: tuple[str, ...]
        Applicable ``sources``, ``passages``, or ``tokens`` limits.

    """

    items: tuple[_EvidenceT, ...]
    estimated_tokens: int
    omitted: int
    limits_hit: tuple[str, ...]


def evidence_block(item: _EvidenceItem, citation: int) -> str:
    """Render exactly the evidence block used by the answer prompt.

    Parameters
    ----------
    item: _EvidenceItem
        One original evidence item.
    citation: int
        One-based citation number used in the prompt.

    Returns
    -------
    str
        Citation header, source location, role, and original text.

    """

    location = (
        f"{PureWindowsPath(str(item.path)).as_posix()}:"
        f"{item.start_line}-{item.end_line}"
    )
    symbol = f" ({item.symbol})" if item.symbol else ""
    role = (
        f" [context: {item.context_role}]"
        if item.context_role else ""
    )
    return f"[{citation}] {location}{symbol}{role}\n{item.text}"


def evidence_tokens(items: Sequence[_EvidenceItem]) -> int:
    """Estimate all evidence blocks as represented in the answer prompt.

    Parameters
    ----------
    items: Sequence[_EvidenceItem]
        Items in their final citation order.

    Returns
    -------
    int
        Upper bound under the established four-characters/token estimate.

    Notes
    -----
    Each block includes its separator. Separately rounded blocks never
    underestimate the token estimator applied to their concatenation.
    This is not an exact model-tokenizer count.

    """

    return sum(
        estimate_tokens("\n\n" + evidence_block(item, citation))
        for citation, item in enumerate(items, start=1)
    )


def allocate_evidence(
    items: Sequence[_EvidenceT],
    limits: EvidenceLimits,
    *,
    priority_sources: Sequence[Path] = (),
) -> EvidenceSelection[_EvidenceT]:
    """Reserve source coverage, then fill remaining evidence capacity.

    Parameters
    ----------
    items: Sequence[_EvidenceT]
        Original evidence items, in retrieval priority order.
    limits: EvidenceLimits
        Hard source, passage, and estimated-token limits.
    priority_sources: Sequence[pathlib.Path], optional
        Sources ranked by the source-first strategy. Sources not in this
        list can fill unused capacity, but cannot displace a priority
        source's first passage. Defaults to ``()``.

    Returns
    -------
    EvidenceSelection[_EvidenceT]
        Selected whole passages and applicable budget limits.

    Notes
    -----
    First reserve one fitting passage per priority source. Then approach
    the per-source target, and finally consider all remaining passages in
    retrieval order. Oversized passages are skipped without truncation.
    Citations can be numbered sequentially after selection. Every candidate
    is costed with the *maximum* possible citation-number width, so final
    renumbering cannot exceed the configured evidence-token limit.

    """

    ordered = tuple(items)
    by_path: dict[Path, list[int]] = {}
    for index, item in enumerate(ordered):
        by_path.setdefault(item.path, []).append(index)
    preferred = tuple(dict.fromkeys(priority_sources))
    priority = preferred or tuple(by_path)
    chosen: list[int] = []
    used: set[int] = set()
    sources: set[Path] = set()
    hit: set[str] = set()
    token_total = 0
    # Using the largest permitted citation width avoids underestimating
    # a later, sequentially renumbered reference.
    citation_width = len(str(limits.max_passages))

    def add(index: int) -> bool:
        """Accept a whole passage if all three limits permit it.

        Parameters
        ----------
        index: int
            Original evidence index to consider.

        Returns
        -------
        bool
            Whether this passage was newly accepted.

        """

        nonlocal token_total
        if index in used:
            return False
        item = ordered[index]
        if len(chosen) >= limits.max_passages:
            hit.add("passages")
            return False
        if item.path not in sources and len(sources) >= limits.max_sources:
            hit.add("sources")
            return False
        number = "9" * citation_width
        rendered = evidence_block(item, int(number))
        cost = estimate_tokens("\n\n" + rendered)
        if token_total + cost > limits.max_tokens:
            hit.add("tokens")
            return False
        chosen.append(index)
        used.add(index)
        sources.add(item.path)
        token_total += cost
        return True

    # First guarantee the best *fitting* passage of each chosen source.
    for path in priority:
        for index in by_path.get(path, ()):
            if add(index):
                break
    # Next attempt per-source quotas, without overriding hard limits.
    for path in priority:
        present = sum(ordered[index].path == path for index in chosen)
        for index in by_path.get(path, ()):
            if present >= limits.passages_per_source:
                break
            if add(index):
                present += 1
    # Related expansion and related-turn evidence compete for any space
    # left after sources and quotas have been represented.
    for index in range(len(ordered)):
        add(index)
    retained = tuple(ordered[index] for index in chosen)
    return EvidenceSelection(
        items=retained,
        estimated_tokens=evidence_tokens(retained),
        omitted=len(ordered) - len(retained),
        limits_hit=tuple(sorted(hit)) if len(used) < len(ordered) else (),
    )


def coverage_report(
    requested: Sequence[Path],
    items: Sequence[_EvidenceItem],
    *,
    mode: CoverageMode = COVERAGE_BOUNDED,
    unavailable: Sequence[Path] = (),
    limits_hit: Sequence[str] = (),
) -> dict[str, object]:
    """Describe source representation without claiming content coverage.

    Parameters
    ----------
    requested: Sequence[pathlib.Path]
        Sources in the explicitly requested scope or selected source set.
    items: Sequence[_EvidenceItem]
        Final evidence passages actually shown to the answering model.
    mode: "bounded" | "exhaustive", optional
        Whether coverage of every requested source was attempted.
    unavailable: Sequence[pathlib.Path], optional
        Requested source files with no indexed passages.
    limits_hit: Sequence[str], optional
        Limits that excluded candidate evidence, including earlier stages.

    Returns
    -------
    dict[str, object]
        JSON-safe source coverage, missing paths and limiting reasons.

    Notes
    -----
    A represented source has at least one included original passage. This
    makes no claim that the whole source was reviewed or every finding was
    examined. Missing passages or unreadable files are not silently called
    verified absence of information.

    """

    expected = tuple(dict.fromkeys(requested))
    represented = {item.path for item in items}
    missing = tuple(path for path in expected if path not in represented)
    unreadable = set(unavailable)
    reasons = set(limits_hit)
    if missing and len(expected) > len(represented):
        reasons.add("source_coverage")
    if any(path in unreadable for path in missing):
        reasons.add("unindexed")
    return {
        "mode": mode,
        "requested_sources": len(expected),
        "represented_sources": len(represented.intersection(expected)),
        "missing_sources": [path.as_posix() for path in missing],
        "unindexed_sources": [
            path.as_posix() for path in missing if path in unreadable
        ],
        "complete": not missing,
        "passage_count": len(items),
        "estimated_evidence_tokens": evidence_tokens(items),
        "limits_hit": sorted(reasons),
        "meaning": "source representation, not full content review",
    }
