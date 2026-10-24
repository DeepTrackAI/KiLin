"""Read one whole document or function as evidence.

Some questions need a complete source rather than its best-matching chunks:
"read the whole Smith 2020 paper", "read the whole train.py", "go through the
fit function". Document reading chooses one target and supplies all of its
chunks in source order, so the answer can draw on parts that ranked
retrieval would miss.

The target is chosen deterministically. An explicitly supplied path takes
precedence; otherwise named functions are selected from one source file.
When functions in several files are named, ranking selects one file, with a
stable path-order fallback. Otherwise a named file or the file dominating
the fused search results is read. Over-budget selections keep the most
relevant chunks in source order and are marked incomplete.

Only text already in the index is read; nothing is loaded from disk.
Token counts are estimates, so a selection can be slightly larger or
smaller than the model's real count. The budget is not model-specific.

Key Features
------------

- **Target choice**

    Named function, named file, or the file dominating the search results.

- **Token budget**

    Token counts are estimated using the configured characters-per-token
    heuristic (initially four for English text with Qwen3 tokenizers), so
    no tokenizer is loaded.

- **Source order**

    Selected chunks keep the order they have in the source, so the model
    reads the document as written.

Module Structure
----------------

- ``DOCUMENT_REASON_NAMED_FUNCTION``, ``DOCUMENT_REASON_NAMED_FILE``, and
  ``DOCUMENT_REASON_SEARCH``

    Canonical string values for why a document was chosen.

- ``DOCUMENT_REASON_CHOICES``

    Ordered collection of the document reasons.

- ``DocumentReason``

    Type alias for ``"named_function"``, ``"named_file"``, or ``"search"``.

- ``DocumentSelection``

    The document chosen for reading and the chunks supplied from it.

- ``estimate_tokens``

    Estimate the token count of text without a tokenizer.

- ``read_document``

    Choose one document for a question and fit it to a token budget.

- ``named_paths``

    Return the files a question names.

- ``dominant_path``

    Return the file that dominates a ranking.

Examples
--------
>>> from pathlib import Path
>>> from kilin.retrievals import estimate_tokens, named_paths
>>> estimate_tokens("one two three")
4
>>> named_paths("summarize smith2020", [Path("papers/smith2020.pdf")])
[PosixPath('papers/smith2020.pdf')]

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias

from kilin.chunking import Chunk
from kilin.defaults import load_defaults_section, require_default_int
from kilin.retrievals.code_graph import CodeGraph
from kilin.retrievals.context_expansion import (
    ContextExpansionPlan,
    _chunk_matches_function,
    expand_retrieval_context,
    plan_context_expansion,
)
from kilin.retrievals.strategies.retrieval_hybrid import HybridResult

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

# Token estimation and filename heuristics; the estimate was originally
# calibrated on a paper with Qwen3 (107,912 chars and 26,986 tokens).
_RETRIEVAL_DEFAULTS = load_defaults_section("RETRIEVAL")
_CHARACTERS_PER_TOKEN: Final[int] = require_default_int(
    _RETRIEVAL_DEFAULTS, "DOCUMENT_CHARACTERS_PER_TOKEN", "RETRIEVAL"
)
# File stems shorter than this are too generic to count as a name.
_MIN_STEM_LENGTH: Final[int] = require_default_int(
    _RETRIEVAL_DEFAULTS, "MIN_DOCUMENT_STEM_LENGTH", "RETRIEVAL"
)
# Nouns that mark the word before them as an author ("smith paper").
_DOCUMENT_NOUNS: Final[str] = (
    r"(?:paper|article|study|review|preprint|book|thesis|report|chapter|"
    r"manuscript|work)s?(?![a-z])"
)


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
    path: Path | None = None,
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
    path: pathlib.Path | None, optional
        The file to read, when the caller already chose it (for example
        the user did). Takes precedence over named functions and search
        ranking. Defaults to ``None``.

    Returns
    -------
    DocumentSelection | None
        The selection, or None when no indexed chunks belong to the
        identified target. Selected chunks always come from one file.

    """

    if not chunks:
        return None

    # A caller-specified path takes precedence over named functions.
    plan = plan_context_expansion(question, graph)
    if path is None and plan.functions:
        target = expand_retrieval_context(
            (),
            chunks,
            graph,
            ContextExpansionPlan(
                paths=(),
                functions=plan.functions,
                relationship=False,
                related_functions=(),
                signals=("exact function",),
            ),
        )
        if target:
            # A DocumentSelection describes exactly one source file.
            candidates = sorted({chunk.path for chunk in target})
            target_path = (
                dominant_path(ranked[:top], candidates) or candidates[0]
            )
            target = [
                chunk for chunk in target if chunk.path == target_path
            ]
            matched_functions = {
                entity.name
                for entity in graph.entities
                if entity.name in plan.functions
                and any(
                    _chunk_matches_function(chunk, entity)
                    for chunk in target
                )
            }
            functions = tuple(
                name for name in plan.functions if name in matched_functions
            )
            return _fit(
                target,
                ranked,
                max_tokens,
                path=target_path,
                functions=functions,
                reason=DOCUMENT_REASON_NAMED_FUNCTION,
            )

    # Otherwise a named file, or the file dominating the search results;
    # among several named files the search results decide.
    paths = list(dict.fromkeys(chunk.path for chunk in chunks))
    named = [path] if path is not None else named_paths(question, paths)
    target_path = dominant_path(ranked[:top], named or None)
    reason = DOCUMENT_REASON_NAMED_FILE if named else DOCUMENT_REASON_SEARCH
    if target_path is None and named:
        target_path = named[0]
    if target_path is None:
        return None

    target = [chunk for chunk in chunks if chunk.path == target_path]
    if not target:
        return None
    return _fit(
        target,
        ranked,
        max_tokens,
        path=target_path,
        functions=(),
        reason=reason,
    )


def named_paths(
    question: str, paths: Sequence[Path], *, authors: bool = False
) -> list[Path]:
    """Return the files a question names.

    Parameters
    ----------
    question: str
        The question.
    paths: Sequence[pathlib.Path]
        Candidate files.
    authors: bool, optional
        Also match author names that start file stems, for requests about
        one document ("summarize Smith's paper"). Defaults to ``False``.

    Returns
    -------
    list[pathlib.Path]
        Files named by full path; otherwise by file name; otherwise by
        file stem of at least four characters ("smith2020"; a plain-word
        stem such as ``project`` only when marked as a file name, see
        ``_names_stem``); otherwise,
        with ``authors``, by a name that starts an author-year file stem
        ("smith", "Smith's paper", or "smith20" for ``smith2020.pdf``;
        "lee25" for ``lee25networks.pdf``). A stem without digits
        (``smith_review.pdf``) is matched by an author name only when the
        question uses the word as one: possessive, or followed by a
        document noun ("Smith's", "smith paper"). Matching ignores case
        and needs a boundary on both sides.

    """

    text = question.replace("\\", "/").casefold()
    full = [path for path in paths if path.as_posix().casefold() in text]
    if full:
        return full

    names = [path for path in paths if _mentions(text, path.name.casefold())]
    if names:
        return names

    stems = [
        path
        for path in paths
        if len(path.stem) >= _MIN_STEM_LENGTH
        and _names_stem(text, path.stem.casefold())
    ]
    if stems or not authors:
        return stems

    # Candidate author names: words of three or more letters, optionally
    # followed by digits ("lee25").
    words = set(re.findall(r"(?<![a-z0-9])[a-z]{3,}\d*(?![a-z])", text))
    # A stem without a year ("source_pipeline") is matched only by a word
    # used as an author name, so ordinary words ("the source folder") do
    # not select a file.
    cited = {word for word in words if _cites_author(text, word)}
    return [
        path
        for path in paths
        if _starts_stem(
            path.stem,
            words if any(c.isdigit() for c in path.stem) else cited,
        )
    ]


def _names_stem(text: str, stem: str) -> bool:
    """Whether a question names a file by its stem.

    Parameters
    ----------
    text: str
        Case-folded question.
    stem: str
        Case-folded file stem.

    Returns
    -------
    bool
        True when the stem appears as a whole word and either looks like a
        file name (it contains a digit, ``_``, ``-``, or ``.``, as in
        ``smith2020`` or ``data_loader``), or is marked as one: in
        backticks, or followed by a word such as "file", "module",
        "script", "notebook", or a document noun ("the project module",
        "the hinton paper").

    Notes
    -----
    A plain-word stem alone ("the project type", "the chat model") does
    not name ``project.py`` or ``chat.py``, so ordinary words do not limit
    the search to one file.

    """

    if not _mentions(text, stem):
        return False
    if any(character.isdigit() or character in "_-." for character in stem):
        return True
    name = rf"(?<![\w.]){re.escape(stem)}(?![\w])"
    if re.search(rf"`{re.escape(stem)}`", text):
        return True
    return bool(
        re.search(
            rf"{name}\s+(?:(?:file|module|script|notebook)s?(?![a-z])|"
            rf"{_DOCUMENT_NOUNS})",
            text,
        )
    )


def _cites_author(text: str, word: str) -> bool:
    """Whether a word is used as an author name in a question.

    Parameters
    ----------
    text: str
        Case-folded question.
    word: str
        Candidate author word, possibly with year digits ("lee25").

    Returns
    -------
    bool
        True when the word has digits, is possessive ("smith's"), or is
        followed, possibly after "et al", by a document noun such as
        "paper", "article", or "study" ("smith paper", "lee et al.
        review").

    """

    if any(character.isdigit() for character in word):
        return True
    name = rf"(?<![a-z0-9]){re.escape(word)}"
    if re.search(rf"{name}(?:'s|\u2019s)(?![a-z])", text):
        return True
    return bool(
        re.search(
            rf"{name}(?:\s+et\s+al\.?)?\s+{_DOCUMENT_NOUNS}",
            text,
        )
    )


def _starts_stem(stem: str, words: set[str]) -> bool:
    """Whether a word starts an author-year stem.

    A name ends where the year starts ("smith" in smith2020) or at ``_``
    or ``-``; a name with its year ("lee25" in lee25networks) may be
    followed by anything.

    Parameters
    ----------
    stem: str
        Normalized filename stem to compare.
    words: set[str]
        Normalized words to compare.

    Returns
    -------
    bool
        True when a word starts an author-year stem.

    """

    stem = stem.casefold()
    for word in words:
        if not stem.startswith(word):
            continue
        rest = stem[len(word) :]
        if (
            word[-1].isdigit()
            or not rest
            or rest[0].isdigit()
            or rest[0] in "_-"
        ):
            return True

    return False


def _mentions(text: str, name: str) -> bool:
    """Whether ``text`` contains ``name`` as a whole word or file name.

    Parameters
    ----------
    text: str
        Text to analyze or transform.
    name: str
        Filename or project name to use.

    Returns
    -------
    bool
        True when ``text`` contains ``name`` as a whole word or file name.

    """

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
    """Keep the whole target, or its most relevant chunks within budget.

    Parameters
    ----------
    target: Sequence[Chunk]
        Every chunk of the chosen document or function, in source order.
    ranked: Sequence[HybridResult]
        Fused search results, best first; they decide which chunks to keep
        when the target exceeds the budget.
    max_tokens: int
        Evidence budget in estimated tokens.
    path: pathlib.Path
        File that holds the target.
    functions: tuple[str, ...]
        Functions read, or empty for a whole file.
    reason: "named_function" | "named_file" | "search"
        Why the target was chosen.

    Returns
    -------
    DocumentSelection
        Selected chunks in source order. Over budget, chunks are taken in
        ranking order (unranked chunks last), and a chunk that does not fit
        is skipped while smaller later ones may still fit.

    Raises
    ------
    ValueError
        If chunks refer to a different file or an identifier refers to
        conflicting chunk records.

    Notes
    -----
    Repeated identical chunk identifiers are counted only once. Valid
    persistent indexes already require unique identifiers.

    """

    unique: dict[str, Chunk] = {}
    for chunk in target:
        if chunk.path != path:
            raise ValueError("Document chunks must belong to one file.")
        previous = unique.get(chunk.identifier)
        if previous is not None and previous != chunk:
            raise ValueError("Conflicting chunks share an identifier.")
        unique.setdefault(chunk.identifier, chunk)

    target = sorted(
        unique.values(),
        key=lambda chunk: (chunk.start_line, chunk.end_line),
    )
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
        estimated_tokens=sum(sizes[chunk.identifier] for chunk in selected),
        complete=len(selected) == len(target),
    )
