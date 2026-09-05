"""Choose repository retrieval methods for one question.

The planner uses only visible words, paths, and symbols. It does not call a
language model. The resulting plan is therefore fast, repeatable, and easy to
show in tutorials or diagnostics.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from qachat.chunking import Chunk

QuestionKind = Literal[
    "general",
    "file",
    "symbol",
    "relationship",
    "architecture",
    "exact",
]
RetrievalMethod = Literal[
    "semantic",
    "lexical",
    "metadata",
    "graph",
]

ARCHITECTURE_CUES = (
    "architecture",
    "data flow",
    "execution flow",
    "entry point",
    "whole codebase",
    "whole repository",
)
EXACT_CUES = (
    "how many",
    "count the",
    "number of",
    "list all",
    "list every",
)
RELATIONSHIP_CUES = (
    "called by",
    "calls ",
    "depends on",
    "inherits",
    "imports",
    "relationship",
    "subclass",
    "used by",
)


@dataclass(frozen=True, slots=True)
class QuestionPlan:
    """A transparent retrieval plan for one repository question.

    Attributes
    ----------
    kind: "general" | "file" | "symbol" | "relationship" | \
            "architecture" | "exact"
        Detected question kind.
    methods: tuple[RetrievalMethod, ...]
        Retrieval methods selected for the question.
    matched_paths: tuple[pathlib.Path, ...]
        Repository paths written explicitly in the question.
    matched_symbols: tuple[str, ...]
        Source symbols written explicitly in the question.
    reason: str
        Short, human-readable explanation of the decision.

    """

    kind: QuestionKind
    methods: tuple[RetrievalMethod, ...]
    matched_paths: tuple[Path, ...] = ()
    matched_symbols: tuple[str, ...] = ()
    reason: str = ""


def plan_question(
    question: str,
    chunks: list[Chunk] | tuple[Chunk, ...],
) -> QuestionPlan:
    """Choose retrieval methods from visible question cues and metadata.

    Parameters
    ----------
    question: str
        Repository question to classify.
    chunks: list[Chunk] | tuple[Chunk, ...]
        Searchable chunks whose paths and symbols may appear in the question.

    Returns
    -------
    QuestionPlan
        Deterministic question kind, retrieval methods, and matched metadata.

    Raises
    ------
    ValueError
        If the question is empty or chunks are not a list or tuple.

    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a non-empty repository question.")

    if not isinstance(chunks, (list, tuple)):
        raise ValueError("chunks must be a list or tuple.")

    normalized = " ".join(question.casefold().split())
    paths = _matched_paths(question, chunks)
    symbols = _matched_symbols(question, chunks)

    if any(cue in normalized for cue in EXACT_CUES):
        return QuestionPlan(
            kind="exact",
            methods=("graph", "metadata"),
            matched_paths=paths,
            matched_symbols=symbols,
            reason="The question asks for an exact list or count.",
        )

    if any(cue in normalized for cue in ARCHITECTURE_CUES):
        return QuestionPlan(
            kind="architecture",
            methods=("semantic", "lexical", "metadata", "graph"),
            matched_paths=paths,
            matched_symbols=symbols,
            reason="The question asks how repository parts fit together.",
        )

    if any(cue in normalized for cue in RELATIONSHIP_CUES):
        return QuestionPlan(
            kind="relationship",
            methods=("graph", "lexical", "metadata"),
            matched_paths=paths,
            matched_symbols=symbols,
            reason="The question asks about a source relationship.",
        )

    if paths:
        return QuestionPlan(
            kind="file",
            methods=("metadata", "lexical", "semantic", "graph"),
            matched_paths=paths,
            matched_symbols=symbols,
            reason="The question names a repository file.",
        )

    if symbols:
        return QuestionPlan(
            kind="symbol",
            methods=("metadata", "lexical", "semantic", "graph"),
            matched_symbols=symbols,
            reason="The question names a source symbol.",
        )

    return QuestionPlan(
        kind="general",
        methods=("semantic", "lexical"),
        reason="The question asks for meaning rather than an exact name.",
    )


def _contains_name(question: str, name: str) -> bool:
    """Return whether a path or symbol appears as a complete name.

    Parameters
    ----------
    question: str
        Original question.
    name: str
        Candidate path or symbol.

    Returns
    -------
    bool
        Whether the candidate appears without joining another identifier.

    """

    pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    return re.search(pattern, question, flags=re.IGNORECASE) is not None


def _matched_paths(
    question: str,
    chunks: list[Chunk] | tuple[Chunk, ...],
) -> tuple[Path, ...]:
    """Return repository paths named in a question.

    Parameters
    ----------
    question: str
        Original question.
    chunks: list[Chunk] | tuple[Chunk, ...]
        Chunks containing known repository paths.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Stable, unique path matches.

    """

    paths = sorted(
        {chunk.path for chunk in chunks},
        key=lambda path: path.as_posix(),
    )
    full = [
        path
        for path in paths
        if "/" in path.as_posix() and _contains_name(question, path.as_posix())
    ]
    if full:
        return tuple(full)

    names = {
        path.name
        for path in paths
        if len(path.name) >= 3 and _contains_name(question, path.name)
    }
    return tuple(path for path in paths if path.name in names)


def _matched_symbols(
    question: str,
    chunks: list[Chunk] | tuple[Chunk, ...],
) -> tuple[str, ...]:
    """Return source symbols named in a question.

    Parameters
    ----------
    question: str
        Original question.
    chunks: list[Chunk] | tuple[Chunk, ...]
        Chunks containing known source symbols.

    Returns
    -------
    tuple[str, ...]
        Stable, unique symbol matches.

    """

    symbols = sorted(
        {
            chunk.symbol
            for chunk in chunks
            if chunk.symbol is not None and len(chunk.symbol) >= 3
        }
    )
    return tuple(
        symbol for symbol in symbols if _contains_name(question, symbol)
    )
