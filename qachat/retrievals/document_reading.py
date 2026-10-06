"""Read one whole document or function as evidence.

Some questions need a complete source rather than its best-matching chunks:
"read the whole Smith 2020 paper", "read the whole train.py", "go through the
fit function". Document reading chooses one target and supplies all of its
chunks in source order, so the answer can draw on parts that ranked
retrieval would miss.

The target is chosen deterministically. A function named in the question
(known to the code graph) is read alone. Otherwise a file named in the
question, by path, file name, or file stem, is read. Otherwise the file
that dominates the question's fused search results is read. When the
target is longer than the token budget, its chunks most relevant to the
question are kept, still in source order, and the selection is marked
incomplete.

Key Features
------------

- **Target choice**

    Named function, named file, or the file dominating the search results.

- **Token budget**

    Token counts are estimated from text length (about four characters per
    token for English text with Qwen3 tokenizers), so no tokenizer is
    loaded.

- **Source order**

    Selected chunks keep the order they have in the source, so the model
    reads the document as written.

Module Structure
----------------

- ``DocumentReason``
- ``DocumentSelection``
- ``dominant_path``
- ``estimate_tokens``
- ``named_paths``
- ``read_document``

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias

from qachat.chunking import Chunk
from qachat.retrievals.code_graph import CodeGraph
from qachat.retrievals.routing import (
    RetrievalPlan,
    expand_retrieval_context,
    plan_retrieval,
)
from qachat.retrievals.strategies.retrieval_hybrid import HybridResult

__all__ = [
    "DOCUMENT_REASON_CHOICES",
    "DOCUMENT_REASON_NAMED_FILE",
    "DOCUMENT_REASON_NAMED_FUNCTION",
    "DOCUMENT_REASON_SEARCH",
    "DocumentReason",
    "DocumentSelection",
    "dominant_path",
    "estimate_tokens",
    "named_paths",
    "read_document",
]

# Why a document was chosen for reading.
DocumentReason: TypeAlias = Literal[
    "named_function", "named_file", "search"
]
DOCUMENT_REASON_NAMED_FUNCTION: Final[DocumentReason] = "named_function"
DOCUMENT_REASON_NAMED_FILE: Final[DocumentReason] = "named_file"
DOCUMENT_REASON_SEARCH: Final[DocumentReason] = "search"
DOCUMENT_REASON_CHOICES: Final[tuple[DocumentReason, ...]] = (
    DOCUMENT_REASON_NAMED_FUNCTION,
    DOCUMENT_REASON_NAMED_FILE,
    DOCUMENT_REASON_SEARCH,
)
# Characters per token for the estimate; measured on a 31-page paper with
# the Qwen3 tokenizer (107,912 characters, 26,986 tokens).
_CHARACTERS_PER_TOKEN: Final[int] = 4
# File stems shorter than this are too generic to count as a name.
_MIN_STEM_LENGTH: Final[int] = 4


@dataclass(frozen=True, slots=True)
class DocumentSelection:
    """The document chosen for reading and the chunks supplied from it.

    Attributes
    ----------
    path: pathlib.Path
        The file read (for a function, the file containing it).
    functions: tuple[str, ...]
        Functions read, when a function was named; otherwise empty.
    reason: DocumentReason
        Why this target was chosen.
    chunks: tuple[Chunk, ...]
        Selected chunks in source order.
    total_chunks: int
        Number of chunks in the whole target.
    estimated_tokens: int
        Estimated tokens of the selected chunks.
    complete: bool
        True when every chunk of the target fits the budget.

    """

    path: Path
    functions: tuple[str, ...]
    reason: DocumentReason
    chunks: tuple[Chunk, ...]
    total_chunks: int
    estimated_tokens: int
    complete: bool


def estimate_tokens(text: str) -> int:
    """Estimate the token count of text without a tokenizer.

    Parameters
    ----------
    text: str
        Text to measure.

    Returns
    -------
    int
        About one token per four characters, rounded up.

    """

    return math.ceil(len(text) / _CHARACTERS_PER_TOKEN)


def read_document(
    question: str,
    chunks: Sequence[Chunk],
    graph: CodeGraph,
    ranked: Sequence[HybridResult],
    *,
    max_tokens: int,
    top: int = 20,
) -> DocumentSelection | None:
    """Choose one document for a question and fit it to a token budget.

    Parameters
    ----------
    question: str
        The question (with any earlier questions it refers to).
    chunks: Sequence[Chunk]
        All indexed chunks, in index order.
    graph: CodeGraph
        Code graph used to recognize named functions; may be empty.
    ranked: Sequence[HybridResult]
        Fused search results for the question, best first. They choose the
        target when none is named, and order chunks by relevance when the
        target exceeds the budget.
    max_tokens: int
        Evidence budget in estimated tokens.
    top: int, optional
        How many leading results decide the dominant file. Defaults to
        ``20``.

    Returns
    -------
    DocumentSelection | None
        The selection, or None when no chunks or results identify a
        target.

    """

    if not chunks:
        return None
    plan = plan_retrieval(question, graph)
    if plan.functions:
        target = expand_retrieval_context(
            (),
            chunks,
            graph,
            RetrievalPlan(
                paths=(),
                functions=plan.functions,
                relationship=False,
                related_functions=(),
                signals=("exact function",),
            ),
        )
        if target:
            return _fit(
                target,
                ranked,
                max_tokens,
                path=target[0].path,
                functions=plan.functions,
                reason=DOCUMENT_REASON_NAMED_FUNCTION,
            )
    paths = list(dict.fromkeys(chunk.path for chunk in chunks))
    named = named_paths(question, paths)
    path = dominant_path(ranked[:top], named or None)
    reason = DOCUMENT_REASON_NAMED_FILE if named else DOCUMENT_REASON_SEARCH
    if path is None and named:
        path = named[0]
    if path is None:
        return None
    target = [chunk for chunk in chunks if chunk.path == path]
    return _fit(
        target, ranked, max_tokens, path=path, functions=(), reason=reason
    )


def named_paths(question: str, paths: Sequence[Path]) -> list[Path]:
    """Return the files a question names.

    Parameters
    ----------
    question: str
        The question.
    paths: Sequence[pathlib.Path]
        Candidate files.

    Returns
    -------
    list[pathlib.Path]
        Files named by full path; otherwise by file name; otherwise by
        file stem of at least four characters ("smith2020"). Matching
        ignores case and needs a boundary on both sides.

    """

    text = question.replace("\\", "/").casefold()
    full = [path for path in paths if path.as_posix().casefold() in text]
    if full:
        return full
    names = [path for path in paths if _mentions(text, path.name.casefold())]
    if names:
        return names
    return [
        path
        for path in paths
        if len(path.stem) >= _MIN_STEM_LENGTH
        and _mentions(text, path.stem.casefold())
    ]


def _mentions(text: str, name: str) -> bool:
    return bool(
        re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", text)
    )


def dominant_path(
    ranked: Sequence[HybridResult], candidates: list[Path] | None = None
) -> Path | None:
    """Return the file that dominates a ranking.

    Parameters
    ----------
    ranked: Sequence[HybridResult]
        Fused search results to weigh, best first.
    candidates: list[pathlib.Path] | None, optional
        Only these files may be chosen; None allows any. Defaults to
        ``None``.

    Returns
    -------
    pathlib.Path | None
        The file with the highest summed fusion score, or None when no
        result belongs to an allowed file.

    """

    scores: dict[Path, float] = {}
    for result in ranked:
        path = result.chunk.path
        if candidates is None or path in candidates:
            scores[path] = scores.get(path, 0.0) + result.score
    if not scores:
        return None
    return max(scores, key=lambda path: scores[path])


def _fit(
    target: Sequence[Chunk],
    ranked: Sequence[HybridResult],
    max_tokens: int,
    *,
    path: Path,
    functions: tuple[str, ...],
    reason: DocumentReason,
) -> DocumentSelection:
    """Keep the whole target, or its most relevant chunks within budget."""

    sizes = {chunk.identifier: estimate_tokens(chunk.text) for chunk in target}
    chosen: set[str] = set(sizes)
    if sum(sizes.values()) > max_tokens:
        # Most relevant first; chunks the ranking missed come last.
        order = list(
            dict.fromkeys(
                [
                    result.chunk.identifier
                    for result in ranked
                    if result.chunk.identifier in sizes
                ]
                + [chunk.identifier for chunk in target]
            )
        )
        chosen = set()
        used = 0
        for identifier in order:
            if used + sizes[identifier] <= max_tokens:
                chosen.add(identifier)
                used += sizes[identifier]
    selected = tuple(chunk for chunk in target if chunk.identifier in chosen)
    return DocumentSelection(
        path=path,
        functions=functions,
        reason=reason,
        chunks=selected,
        total_chunks=len(target),
        estimated_tokens=sum(sizes[c.identifier] for c in selected),
        complete=len(selected) == len(target),
    )
