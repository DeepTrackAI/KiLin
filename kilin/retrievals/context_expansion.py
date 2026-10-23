"""Expand retrieval context from explicit source references in a question.

Context expansion recognizes file paths and function names, and uses a
configurable vocabulary of relationship requests. It produces a
``ContextExpansionPlan`` and adds matching file or function chunks and
eligible graph neighbors to an
existing ranking.
This is deterministic text matching over the supplied source metadata, not a
language-model router or a complete interpretation of the question.

Expansion deduplicates chunk identifiers while preserving the initial ranked
results. The expansion itself has no token or result-count ceiling, so callers
must account for the full expanded context before generating an answer.

Key Features
------------

- **Explicit references**

    Match known paths and function symbols mentioned in the question.

- **Graph neighbors**

    Recognize selected relationship phrases and include matching neighboring
    functions.

- **Ordered expansion**

    Keep ranked evidence first and append source-ordered matches without
    duplicate chunk identifiers.

Module Structure
----------------

- ``ContextExpansionPlan``

    Deterministic context expansion requested by one question.

- ``expand_retrieval_context``

    Add complete requested context after normal RRF retrieval.

- ``plan_context_expansion``

    Detect exact repository context requested by one question.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> graph = kilin.CodeGraph(entities=(), relations=())
>>> plan = kilin.plan_context_expansion("Explain this project.", graph)

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

from kilin.chunking import Chunk
from kilin.defaults import (
    load_defaults_section,
    require_default_nonnegative_int,
    require_default_strings,
)
from kilin.retrievals.code_graph import CodeEntity, CodeGraph
from kilin.retrievals.strategies.retrieval_hybrid import HybridResult

__all__ = [
    "ContextExpansionPlan",
    "expand_retrieval_context",
    "plan_context_expansion",
]


_FUNCTION_KINDS = {
    "matlab_function",
    "python_function",
    "python_method",
}

# Relationship phrases influence deterministic graph-context expansion.
_RETRIEVAL_DEFAULTS = load_defaults_section("RETRIEVAL")
_RELATION_PHRASES = require_default_strings(
    _RETRIEVAL_DEFAULTS, "RELATION_PHRASES", "RETRIEVAL"
)
if any(
    phrase != phrase.strip() or phrase != phrase.lower()
    for phrase in _RELATION_PHRASES
):
    raise ValueError(
        "RETRIEVAL.RELATION_PHRASES must contain lowercase, "
        "trimmed phrases."
    )

# Most direct callees added for a named function when the question does not
# ask about relationships; zero disables callee expansion.
_CALLEE_EXPANSION_MAX = require_default_nonnegative_int(
    _RETRIEVAL_DEFAULTS, "CALLEE_EXPANSION_MAX", "RETRIEVAL"
)
_WORD = re.compile(r"[A-Za-z0-9]+")
# Words used in plain prose that introduce a code name ("the reply method").
_CODE_NOUNS = ("function", "method")


@dataclass(frozen=True, slots=True)
class ContextExpansionPlan:
    """Deterministic context expansion requested by one question.

    Parameters
    ----------
    paths: tuple[pathlib.Path, ...]
        Exact repository paths or file names found in the question.
    functions: tuple[str, ...]
        Exact repository functions found in the question.
    relationship: bool
        Whether the question asks about relationships around a function.
    related_functions: tuple[str, ...]
        Direct function neighbors when relationship expansion is requested;
        otherwise the named functions' most relevant direct callees.
    signals: tuple[str, ...]
        Human-readable signals that explain the expansion.

    Attributes
    ----------
    paths: tuple[pathlib.Path, ...]
        Exact repository paths or file names found in the question.
    functions: tuple[str, ...]
        Exact repository functions found in the question.
    relationship: bool
        Whether the question asks about relationships around a function.
    related_functions: tuple[str, ...]
        Direct function neighbors when relationship expansion is requested;
        otherwise the named functions' most relevant direct callees.
    signals: tuple[str, ...]
        Human-readable signals that explain the expansion.

    """

    paths: tuple[Path, ...]
    functions: tuple[str, ...]
    relationship: bool
    related_functions: tuple[str, ...]
    signals: tuple[str, ...]


def plan_context_expansion(
    query_text: str,
    graph: CodeGraph,
) -> ContextExpansionPlan:
    """Detect exact repository context requested by one question.

    The plan does not change Reciprocal Rank Fusion. It only records context
    that should be guaranteed in addition to the normal RRF top results.

    Parameters
    ----------
    query_text: str
        User question.
    graph: CodeGraph
        Current repository code graph.

    Returns
    -------
    ContextExpansionPlan
        Exact files, functions, and direct related functions to include.

    Raises
    ------
    ValueError
        If the question is empty.

    Notes
    -----
    Recognizes only kinds in _FUNCTION_KINDS and phrases in _RELATION_PHRASES.
    A relationship request requires at least one explicit function match. It
    does not infer arbitrary semantic references.

    A function is recognized by its qualified name, by a dotted suffix of it
    (``Class.method``), or by its leaf name. A leaf name that reads as a
    plain word (no underscore, digit, or internal capital) matches only when
    it is unambiguous (one top-level function, not a method, longer than two
    letters) or is marked as code in the question (backticks, a following
    ``(``, or the word "function" or "method" next to it). Ordinary words
    such as "to" or "reply" therefore do not match methods of those names.

    Without relationship language, the named functions' direct callees are
    added, at most ``CALLEE_EXPANSION_MAX``, preferring callees whose names
    share words with the question: explaining a function usually requires
    the functions it calls. Related functions in test code (a ``tests`` or
    ``test`` folder, or a ``test_*`` or ``*_test`` file) are left out unless
    the question mentions tests.

    """

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be non-empty text.")

    paths = _explicit_paths(query_text, graph)
    functions = _explicit_functions(query_text, graph, paths)
    relationship = bool(functions) and _has_relationship(query_text)
    if relationship:
        related_functions = _direct_function_neighbors(functions, graph)
    elif functions:
        related_functions = _ranked_callees(
            functions, graph, query_text, _CALLEE_EXPANSION_MAX
        )
    else:
        related_functions = ()
    if related_functions and not _mentions_tests(query_text):
        related_functions = _without_test_code(related_functions, graph)

    signals: list[str] = []
    if paths:
        signals.append("exact file")
    if functions:
        signals.append("exact function")
    if relationship:
        signals.append("relationship")
    elif related_functions:
        signals.append("callees")
    if not signals:
        signals.append("none")

    return ContextExpansionPlan(
        paths=paths,
        functions=functions,
        relationship=relationship,
        related_functions=related_functions,
        signals=tuple(signals),
    )


def expand_retrieval_context(
    rrf_results: Sequence[HybridResult],
    chunks: Sequence[Chunk],
    graph: CodeGraph,
    plan: ContextExpansionPlan,
) -> list[Chunk]:
    """Add complete requested context after normal RRF retrieval.

    Normal RRF chunks stay first and keep their original order. Exact file
    chunks, exact function chunks, and direct related-function chunks are
    appended when requested. Duplicate chunks are removed.

    Parameters
    ----------
    rrf_results: Sequence[HybridResult]
        Normal RRF results in best-first order.
    chunks: Sequence[Chunk]
        All searchable repository chunks.
    graph: CodeGraph
        Current repository code graph.
    plan: ContextExpansionPlan
        Deterministic expansion plan for the question.

    Returns
    -------
    list[Chunk]
        RRF chunks followed by any guaranteed context expansion.

    Notes
    -----
    Initial fused results stay first. Explicit-path chunks, named-function
    chunks, and the planned related functions (direct neighbors or callees)
    follow. Deduplication is by chunk identifier. No output-count or
    token-budget limit is applied.

    """

    expanded: list[Chunk] = []
    seen: set[str] = set()

    _append_unique(
        expanded,
        seen,
        (result.chunk for result in rrf_results),
    )

    file_chunks = (
        chunk for chunk in _source_order(chunks) if chunk.path in plan.paths
    )
    _append_unique(expanded, seen, file_chunks)

    function_chunks = _chunks_for_functions(
        chunks,
        graph,
        plan.functions,
    )
    _append_unique(expanded, seen, function_chunks)

    if plan.related_functions:
        related_chunks = _chunks_for_functions(
            chunks,
            graph,
            plan.related_functions,
        )
        _append_unique(expanded, seen, related_chunks)

    return expanded


def _explicit_paths(
    query_text: str,
    graph: CodeGraph,
) -> tuple[Path, ...]:
    """Return exact repository paths or file names in the question.

    Parameters
    ----------
    query_text: str
        Question text to inspect for explicit file/function mentions and
        recognized relationship phrases.
    graph: CodeGraph
        Graph supplying known paths, function entities, and relationships.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Exact repository paths or file names in the question.

    Notes
    -----
    Case-folded full path substrings take precedence over basename matches.
    Basenames use identifier-boundary matching; multiple matching files can be
    returned.

    """

    normalized = query_text.replace("\\", "/").casefold()
    paths = {entity.path for entity in graph.entities}

    full_matches = {
        path for path in paths if path.as_posix().casefold() in normalized
    }
    if full_matches:
        return tuple(sorted(full_matches, key=lambda path: path.as_posix()))

    name_matches = {
        path
        for path in paths
        if _contains_identifier(normalized, path.name.casefold())
    }
    return tuple(sorted(name_matches, key=lambda path: path.as_posix()))


def _explicit_functions(
    query_text: str,
    graph: CodeGraph,
    paths: tuple[Path, ...],
) -> tuple[str, ...]:
    """Return exact repository function names in the question.

    Parameters
    ----------
    query_text: str
        Question text to inspect for explicit file/function mentions and
        recognized relationship phrases.
    graph: CodeGraph
        Graph supplying known paths, function entities, and relationships.
    paths: tuple[pathlib.Path, ...]
        Repository-relative paths already recognized in the question.

    Returns
    -------
    tuple[str, ...]
        Exact repository function names in the question.

    """

    normalized = query_text.replace("\\", "/").casefold()
    normalized = _remove_path_mentions(normalized, paths)

    functions = [
        entity for entity in graph.entities if entity.kind in _FUNCTION_KINDS
    ]
    leaf_counts = Counter(
        entity.name.rsplit(".", 1)[-1].casefold() for entity in functions
    )
    matches = set()
    for entity in functions:
        leaf = entity.name.rsplit(".", 1)[-1].casefold()
        # A plain word names a function only when it is unambiguous: one
        # top-level function, not a method, and longer than two letters.
        plain_word_ok = (
            entity.kind != "python_method"
            and leaf_counts[leaf] == 1
            and len(leaf) > 2
        )
        if _function_is_named(normalized, entity.name, plain_word_ok):
            matches.add(entity.name)
    return tuple(sorted(matches))


def _has_relationship(query_text: str) -> bool:
    """Return whether wording asks about code relationships.

    Parameters
    ----------
    query_text: str
        Question text to inspect for explicit file/function mentions and
        recognized relationship phrases.

    Returns
    -------
    bool
        Whether wording asks about code relationships.

    """

    normalized = " ".join(_WORD.findall(query_text.casefold()))
    padded = f" {normalized} "
    return any(f" {phrase} " in padded for phrase in _RELATION_PHRASES)


def _direct_function_neighbors(
    functions: tuple[str, ...],
    graph: CodeGraph,
) -> tuple[str, ...]:
    """Return function entities one graph edge from the target functions.

    Parameters
    ----------
    functions: tuple[str, ...]
        Qualified names of recognized function entities.
    graph: CodeGraph
        Graph supplying known paths, function entities, and relationships.

    Returns
    -------
    tuple[str, ...]
        Function entities one graph edge from the target functions.

    Notes
    -----
    Relations are treated as undirected and are not filtered to calls alone.
    Only neighbor entities of recognized function kinds are retained; matched
    seed functions are excluded.

    """

    function_names = {
        entity.name
        for entity in graph.entities
        if entity.kind in _FUNCTION_KINDS
    }
    targets = set(functions)
    neighbors: set[str] = set()

    for relation in graph.relations:
        if relation.source in targets:
            neighbor = relation.target
        elif relation.target in targets:
            neighbor = relation.source
        else:
            continue

        if neighbor in function_names and neighbor not in targets:
            neighbors.add(neighbor)

    return tuple(sorted(neighbors))


def _mentions_tests(query_text: str) -> bool:
    """Return whether a question is about tests.

    Parameters
    ----------
    query_text: str
        Question text.

    Returns
    -------
    bool
        True when the question contains "test", "tests", "tested", or
        "testing" as a word or inside an identifier.

    """

    return any(
        word in {"test", "tests", "tested", "testing"}
        for word in _identifier_words(query_text)
    )


def _without_test_code(
    functions: tuple[str, ...], graph: CodeGraph
) -> tuple[str, ...]:
    """Drop functions defined in test code.

    Parameters
    ----------
    functions: tuple[str, ...]
        Qualified names of function entities.
    graph: CodeGraph
        Graph supplying the path of each entity.

    Returns
    -------
    tuple[str, ...]
        Functions whose file is not in a ``tests`` or ``test`` folder and
        is not named ``test_*`` or ``*_test``, in their original order.

    """

    paths = {entity.name: entity.path for entity in graph.entities}
    return tuple(
        name for name in functions if not _is_test_path(paths.get(name))
    )


def _is_test_path(path: Path | None) -> bool:
    """Return whether a source path belongs to test code.

    Parameters
    ----------
    path: pathlib.Path | None
        Repository-relative source path, or None when unknown.

    Returns
    -------
    bool
        True for files in a ``tests`` or ``test`` folder and for files named
        ``test_*`` or ``*_test``.

    """

    if path is None:
        return False
    return (
        any(part in {"tests", "test"} for part in path.parts[:-1])
        or path.stem.startswith("test_")
        or path.stem.endswith("_test")
    )


def _chunks_for_functions(
    chunks: Sequence[Chunk],
    graph: CodeGraph,
    functions: tuple[str, ...],
) -> list[Chunk]:
    """Return all chunks belonging to the requested functions.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Source chunks available for context expansion.
    graph: CodeGraph
        Graph supplying known paths, function entities, and relationships.
    functions: tuple[str, ...]
        Qualified names of recognized function entities.

    Returns
    -------
    list[Chunk]
        All chunks belonging to the requested functions.

    """

    entities = {
        entity.name: entity
        for entity in graph.entities
        if entity.kind in _FUNCTION_KINDS
    }
    selected: list[Chunk] = []

    for function in functions:
        entity = entities.get(function)
        if entity is None:
            continue

        selected.extend(
            chunk
            for chunk in _source_order(chunks)
            if _chunk_matches_function(chunk, entity)
        )

    return selected


def _chunk_matches_function(
    chunk: Chunk,
    entity: CodeEntity,
) -> bool:
    """Return whether a chunk belongs to one function entity.

    Parameters
    ----------
    chunk: Chunk
        One candidate source chunk.
    entity: CodeEntity
        Function entity whose path and symbol should match the candidate chunk.

    Returns
    -------
    bool
        Whether a chunk belongs to one function entity.

    """

    if chunk.path != entity.path or chunk.symbol is None:
        return False

    symbol = chunk.symbol.casefold()
    name = entity.name.casefold()
    leaf = name.rsplit(".", 1)[-1]

    return symbol == name or name.endswith(f".{symbol}") or symbol == leaf


def _remove_path_mentions(
    query_text: str,
    paths: tuple[Path, ...],
) -> str:
    """Remove matched paths before looking for function names.

    Parameters
    ----------
    query_text: str
        Question text to inspect for explicit file/function mentions and
        recognized relationship phrases.
    paths: tuple[pathlib.Path, ...]
        Repository-relative paths already recognized in the question.

    Returns
    -------
    str
        Question text with recognized path forms removed before function-name
        detection.

    """

    cleaned = query_text
    for path in paths:
        cleaned = cleaned.replace(path.as_posix().casefold(), " ")
        cleaned = re.sub(
            _identifier_pattern(path.name.casefold()),
            " ",
            cleaned,
        )
    return cleaned


def _function_is_named(
    query_text: str, function: str, plain_word_ok: bool = True
) -> bool:
    """Return whether a function is named in the question.

    Parameters
    ----------
    query_text: str
        Case-folded question text, with recognized paths removed.
    function: str
        Qualified function name, in its original case.
    plain_word_ok: bool, optional
        Whether a leaf name that reads as a plain word may match without a
        code marker. Defaults to ``True``.

    Returns
    -------
    bool
        True for the qualified name or a dotted suffix of it
        (``Class.method``), or for the leaf name when it looks like code, is
        marked as code in the question, or is an accepted plain word.

    """

    parts = function.casefold().split(".")
    for start in range(len(parts) - 1):
        if _contains_identifier(query_text, ".".join(parts[start:])):
            return True
    leaf = parts[-1]
    if not _contains_identifier(query_text, leaf):
        return False
    original_leaf = function.rsplit(".", 1)[-1]
    return (
        plain_word_ok
        or _looks_like_code(original_leaf)
        or _marked_as_code(query_text, leaf)
    )


def _looks_like_code(name: str) -> bool:
    """Return whether an identifier cannot be an ordinary word.

    Parameters
    ----------
    name: str
        Identifier in its original case.

    Returns
    -------
    bool
        True when the name contains an underscore or a digit, or a capital
        letter after its first character (``camelCase``, ``CamelCase``).

    """

    return (
        "_" in name
        or any(character.isdigit() for character in name)
        or any(character.isupper() for character in name[1:])
    )


def _marked_as_code(query_text: str, leaf: str) -> bool:
    """Return whether the question marks a plain name as code.

    Parameters
    ----------
    query_text: str
        Case-folded question text.
    leaf: str
        Case-folded leaf function name.

    Returns
    -------
    bool
        True when the name is in backticks, is followed by ``(``, or stands
        next to the word "function" or "method".

    """

    name = re.escape(leaf)
    nouns = "|".join(_CODE_NOUNS)
    patterns = (
        rf"`{name}(\(\))?`",
        rf"(?<![A-Za-z0-9_.]){name}\s*\(",
        rf"\b(?:{nouns})\s+`?{name}(?![A-Za-z0-9_])",
        rf"(?<![A-Za-z0-9_.]){name}`?\s+(?:{nouns})\b",
    )
    return any(re.search(pattern, query_text) for pattern in patterns)


def _ranked_callees(
    functions: tuple[str, ...],
    graph: CodeGraph,
    query_text: str,
    limit: int,
) -> tuple[str, ...]:
    """Return the most relevant functions called by the named functions.

    Parameters
    ----------
    functions: tuple[str, ...]
        Qualified names of recognized function entities.
    graph: CodeGraph
        Graph supplying function entities and ``calls`` relations.
    query_text: str
        Question text, used to prefer callees whose names share words with
        it.
    limit: int
        Most callees returned; zero returns none.

    Returns
    -------
    tuple[str, ...]
        Callee names, most shared words first, then in name order.

    """

    if limit <= 0:
        return ()
    function_names = {
        entity.name
        for entity in graph.entities
        if entity.kind in _FUNCTION_KINDS
    }
    targets = set(functions)
    callees = {
        relation.target
        for relation in graph.relations
        if relation.relation == "calls"
        and relation.source in targets
        and relation.target in function_names
        and relation.target not in targets
    }
    question_words = set(_identifier_words(query_text))
    shared = {
        name: len(
            question_words & set(_identifier_words(name.rsplit(".", 1)[-1]))
        )
        for name in callees
    }
    ranked = sorted(callees, key=lambda name: (-shared[name], name))
    return tuple(ranked[:limit])


def _identifier_words(text: str) -> list[str]:
    """Split text and identifiers into lowercase words.

    Parameters
    ----------
    text: str
        Prose or an identifier such as ``_parse_route`` or ``sideReply``.

    Returns
    -------
    list[str]
        Lowercase words; ``route_question`` gives ``route`` and
        ``question``.

    """

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).replace("_", " ")
    return [word.casefold() for word in _WORD.findall(spaced)]


def _contains_identifier(text: str, value: str) -> bool:
    """Return whether one identifier appears with safe boundaries.

    Parameters
    ----------
    text: str
        Text in which an identifier is sought.
    value: str
        Literal identifier whose boundaries or pattern are checked.

    Returns
    -------
    bool
        Whether one identifier appears with safe boundaries.

    """

    return re.search(_identifier_pattern(value), text) is not None


def _identifier_pattern(value: str) -> str:
    """Return a regex pattern for one exact code identifier.

    Parameters
    ----------
    value: str
        Literal identifier whose boundaries or pattern are checked.

    Returns
    -------
    str
        A regex pattern for one exact code identifier.

    """

    return rf"(?<![A-Za-z0-9_]){re.escape(value)}" r"(?![A-Za-z0-9_])"


def _source_order(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Return chunks in deterministic source order.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Source chunks available for context expansion.

    Returns
    -------
    list[Chunk]
        Chunks in deterministic source order.

    """

    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.path.as_posix(),
            chunk.start_line,
            chunk.end_line,
            chunk.identifier,
        ),
    )


def _append_unique(
    destination: list[Chunk],
    seen: set[str],
    candidates,
) -> None:
    """Append chunks once while preserving candidate order.

    Parameters
    ----------
    destination: list[Chunk]
        Mutable output list to extend with previously unseen chunks.
    seen: set[str]
        Mutable set of chunk identifiers already included in the output.
    candidates: object
        Iterable of candidate chunks appended in iteration order when not
        already seen.

    Notes
    -----
    Mutates both destination and seen. Candidate objects are retained by
    reference; their contents are not copied or merged.

    """

    for chunk in candidates:
        if chunk.identifier in seen:
            continue
        seen.add(chunk.identifier)
        destination.append(chunk)
