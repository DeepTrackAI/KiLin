"""Expand retrieval context from explicit source references in a question.

Routing recognizes file paths, known function names, and a configurable
vocabulary of relationship requests. It produces a ``RetrievalPlan``
and adds matching file or function chunks and eligible graph neighbors to an
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

- ``RetrievalPlan``

    Deterministic context expansion requested by one question.

- ``demote_test_results``

    Rank results from test code after the others.

- ``expand_retrieval_context``

    Add complete requested context after normal RRF retrieval.

- ``plan_retrieval``

    Detect exact repository context requested by one question.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> graph = kilin.CodeGraph(entities=(), relations=())
>>> plan = kilin.plan_retrieval("Explain this project.", graph)

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import os
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
    "RetrievalPlan",
    "demote_test_results",
    "expand_retrieval_context",
    "plan_retrieval",
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
# Call steps followed from a named function when collecting callees.
_CALLEE_EXPANSION_DEPTH = require_default_nonnegative_int(
    _RETRIEVAL_DEFAULTS, "CALLEE_EXPANSION_DEPTH", "RETRIEVAL"
)
# Most methods added for a named class; smaller classes are added whole.
_CLASS_METHOD_EXPANSION_MAX = require_default_nonnegative_int(
    _RETRIEVAL_DEFAULTS, "CLASS_METHOD_EXPANSION_MAX", "RETRIEVAL"
)
# Most chunks kept from one long related function; zero keeps all.
_RELATED_CHUNKS_MAX = require_default_nonnegative_int(
    _RETRIEVAL_DEFAULTS, "RELATED_CHUNKS_MAX", "RETRIEVAL"
)
# Most functions added because every word of their name occurs in the
# question ("which PyTorch device" adds ``torch_device``); zero disables it.
_NAME_WORD_EXPANSION_MAX = require_default_nonnegative_int(
    _RETRIEVAL_DEFAULTS, "NAME_WORD_EXPANSION_MAX", "RETRIEVAL"
)
_CLASS_KINDS = {"matlab_class", "python_class"}
_WORD = re.compile(r"[A-Za-z0-9]+")
# Words used in plain prose that introduce a code name ("the reply method").
_CODE_NOUNS = ("function", "method", "class")


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
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
        otherwise the named functions' most relevant callees. Methods of
        named classes and functions described by the question's words are
        appended.
    signals: tuple[str, ...]
        Human-readable signals that explain the expansion.
    focus_terms: tuple[str, ...], optional
        Words of the question and leaf names of the named functions, used to
        choose the most relevant chunks of a long related function.
        Defaults to ``()``.

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
        otherwise the named functions' most relevant callees. Methods of
        named classes and functions described by the question's words are
        appended.
    signals: tuple[str, ...]
        Human-readable signals that explain the expansion.
    focus_terms: tuple[str, ...]
        Words of the question and leaf names of the named functions, used to
        choose the most relevant chunks of a long related function.

    """

    paths: tuple[Path, ...]
    functions: tuple[str, ...]
    relationship: bool
    related_functions: tuple[str, ...]
    signals: tuple[str, ...]
    focus_terms: tuple[str, ...] = ()


def plan_retrieval(
    query_text: str,
    graph: CodeGraph,
) -> RetrievalPlan:
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
    RetrievalPlan
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

    Without relationship language, the named functions' callees are added,
    up to ``CALLEE_EXPANSION_DEPTH`` call steps away and at most
    ``CALLEE_EXPANSION_MAX``, preferring callees whose names share words with
    the question, then nearer ones: explaining a function usually requires
    the functions it calls. A named class (recognized like a function, and
    by the word "class" next to a plain name) adds its methods: all of them
    when there are at most ``CLASS_METHOD_EXPANSION_MAX``, otherwise those
    whose names share most words with the question. When ranking callees
    and methods, the words of the named functions and classes themselves
    are ignored, since every candidate tends to share them. Question words
    match identifier words in simple variants ("written" matches
    ``write``; "PyTorch" gives "py" and "torch"). Finally, up to
    ``NAME_WORD_EXPANSION_MAX`` functions are added whose names have at
    least two words of four or more letters, all occurring in the question
    (``torch_device`` for "which PyTorch device"). Related functions in
    test code (a ``tests`` or ``test`` folder, or a ``test_*`` or ``*_test``
    file) are left out unless the question mentions tests.

    """

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be non-empty text.")

    paths = _explicit_paths(query_text, graph)
    functions = _explicit_functions(query_text, graph, paths)
    relationship = bool(functions) and _has_relationship(query_text)
    classes = _explicit_classes(query_text, graph, paths)
    # Candidates would all share the words of the names that led to them,
    # so those words do not rank callees, methods, or described functions.
    ranking_text = _without_names(query_text, (*functions, *classes))
    if relationship:
        related_functions = _direct_function_neighbors(functions, graph)
    elif functions:
        related_functions = _ranked_callees(
            functions,
            graph,
            ranking_text,
            _CALLEE_EXPANSION_MAX,
            _CALLEE_EXPANSION_DEPTH,
        )
    else:
        related_functions = ()
    class_methods = _class_methods(
        classes, graph, ranking_text, _CLASS_METHOD_EXPANSION_MAX
    )
    tests_allowed = _mentions_tests(query_text)
    described = _described_functions(
        ranking_text,
        graph,
        set(functions) | set(related_functions) | set(class_methods),
        _NAME_WORD_EXPANSION_MAX,
        tests_allowed,
    )
    related_functions = tuple(
        dict.fromkeys(
            name
            for name in (*related_functions, *class_methods, *described)
            if name not in functions
        )
    )
    if related_functions and not tests_allowed:
        related_functions = _without_test_code(related_functions, graph)

    signals: list[str] = []
    if paths:
        signals.append("exact file")
    if functions:
        signals.append("exact function")
    if classes:
        signals.append("exact class")
    if relationship:
        signals.append("relationship")
    elif functions and set(related_functions) - set(class_methods) - set(
        described
    ):
        signals.append("callees")
    if described:
        signals.append("name words")
    if not signals:
        signals.append("none")

    focus_terms = tuple(
        dict.fromkeys(
            [
                *_identifier_words(query_text),
                *(name.rsplit(".", 1)[-1].casefold() for name in functions),
            ]
        )
    )
    return RetrievalPlan(
        paths=paths,
        functions=functions,
        relationship=relationship,
        related_functions=related_functions,
        signals=tuple(signals),
        focus_terms=focus_terms,
    )


def expand_retrieval_context(
    rrf_results: Sequence[HybridResult],
    chunks: Sequence[Chunk],
    graph: CodeGraph,
    plan: RetrievalPlan,
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
    plan: RetrievalPlan
        Deterministic expansion plan for the question.

    Returns
    -------
    list[Chunk]
        RRF chunks followed by any guaranteed context expansion.

    Notes
    -----
    Initial fused results stay first. Explicit-path chunks, named-function
    chunks, and the planned related functions (direct neighbors, callees, or
    methods of named classes) follow. A related function longer than
    ``RELATED_CHUNKS_MAX`` chunks contributes only its most relevant chunks:
    those that mention a named function (such as a call site), then those
    sharing most words with the question. Deduplication is by chunk
    identifier. No overall output-count or token-budget limit is applied.

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
        named_leaves = {
            name.rsplit(".", 1)[-1].casefold() for name in plan.functions
        }
        for function in plan.related_functions:
            related_chunks = _chunks_for_functions(
                chunks, graph, (function,)
            )
            _append_unique(
                expanded,
                seen,
                _focused_chunks(
                    related_chunks,
                    named_leaves,
                    plan.focus_terms,
                    _RELATED_CHUNKS_MAX,
                ),
            )

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


def demote_test_results(
    results: Sequence[HybridResult], query_text: str
) -> list[HybridResult]:
    """Rank results from test code after the others.

    Parameters
    ----------
    results: Sequence[HybridResult]
        Fused results, best first.
    query_text: str
        Question text; results are left unchanged when it mentions tests.

    Returns
    -------
    list[HybridResult]
        The same results with those from test code (a ``tests`` or ``test``
        folder, or a ``test_*`` or ``*_test`` file) moved after the others,
        each group keeping its order.

    Notes
    -----
    Test code often repeats the names and wording of the function it tests,
    so it can outrank the function itself; for a question about how the code
    works, the source code is the better evidence.

    """

    if _mentions_tests(query_text):
        return list(results)
    source = [r for r in results if not _is_test_path(r.chunk.path)]
    tests = [r for r in results if _is_test_path(r.chunk.path)]
    return source + tests


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
    depth: int = 1,
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
    depth: int, optional
        Call steps followed from the named functions. Defaults to ``1``.

    Returns
    -------
    tuple[str, ...]
        Callee names, most shared words first, then fewer call steps, then
        those calling more repository functions themselves (they do more of
        the work than small utilities), then in name order.

    """

    if limit <= 0 or depth <= 0:
        return ()
    function_names = {
        entity.name
        for entity in graph.entities
        if entity.kind in _FUNCTION_KINDS
    }
    calls: dict[str, set[str]] = {}
    for relation in graph.relations:
        if relation.relation == "calls" and relation.target in function_names:
            calls.setdefault(relation.source, set()).add(relation.target)
    targets = set(functions)
    steps: dict[str, int] = {}
    frontier = set(functions)
    for step in range(1, depth + 1):
        reached = set()
        for caller in frontier:
            for callee in calls.get(caller, ()):
                if callee not in targets and callee not in steps:
                    steps[callee] = step
                    reached.add(callee)
        frontier = reached
    question_words = _identifier_words(query_text)
    ranked = sorted(
        steps,
        key=lambda name: (
            -_shared_words(question_words, name.rsplit(".", 1)[-1]),
            steps[name],
            -len(calls.get(name, ())),
            name,
        ),
    )
    return tuple(ranked[:limit])


def _explicit_classes(
    query_text: str,
    graph: CodeGraph,
    paths: tuple[Path, ...],
) -> tuple[str, ...]:
    """Return repository classes named in the question.

    Parameters
    ----------
    query_text: str
        Question text.
    graph: CodeGraph
        Graph supplying class entities.
    paths: tuple[pathlib.Path, ...]
        Repository-relative paths already recognized in the question.

    Returns
    -------
    tuple[str, ...]
        Qualified class names, in name order.

    Notes
    -----
    A class is named by its qualified name, a dotted suffix of it, or its
    leaf name when that name looks like code (``ConversationFileLock``) or
    is marked as code (backticks or the word "class" next to it). A plain
    capitalized word such as "Chunk" alone does not name a class.

    """

    normalized = query_text.replace("\\", "/").casefold()
    normalized = _remove_path_mentions(normalized, paths)
    return tuple(
        sorted(
            entity.name
            for entity in graph.entities
            if entity.kind in _CLASS_KINDS
            and _function_is_named(normalized, entity.name, False)
        )
    )


def _class_methods(
    classes: tuple[str, ...],
    graph: CodeGraph,
    query_text: str,
    limit: int,
) -> tuple[str, ...]:
    """Return the methods of named classes most relevant to the question.

    Parameters
    ----------
    classes: tuple[str, ...]
        Qualified names of the named classes.
    graph: CodeGraph
        Graph supplying method entities.
    query_text: str
        Question text, used to rank the methods of large classes.
    limit: int
        Classes with at most this many methods contribute all of them;
        larger classes contribute at most this many, each sharing at least
        one word with the question. Special methods such as ``__init__``
        are not counted or added. Zero contributes none.

    Returns
    -------
    tuple[str, ...]
        Method names, class by class.

    """

    if limit <= 0:
        return ()
    question_words = _identifier_words(query_text)
    selected: list[str] = []
    for name in classes:
        prefix = f"{name}."
        # Special methods (``__init__``, ``__enter__``...) rarely explain
        # what a class does, so they are not candidates.
        methods = sorted(
            entity.name
            for entity in graph.entities
            if entity.kind in _FUNCTION_KINDS
            and entity.name.startswith(prefix)
            and "." not in entity.name[len(prefix):]
            and not (
                entity.name[len(prefix):].startswith("__")
                and entity.name.endswith("__")
            )
        )
        if len(methods) <= limit:
            selected.extend(methods)
            continue
        scored = [
            (_shared_words(question_words, method.rsplit(".", 1)[-1]), method)
            for method in methods
        ]
        ranked = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: (-item[0], item[1]),
        )
        selected.extend(method for _, method in ranked[:limit])
    return tuple(selected)


def _described_functions(
    query_text: str,
    graph: CodeGraph,
    exclude: set[str],
    limit: int,
    tests_allowed: bool,
) -> tuple[str, ...]:
    """Return functions whose name words all occur in the question.

    Parameters
    ----------
    query_text: str
        Question text, without the names it states explicitly.
    graph: CodeGraph
        Graph supplying function entities and their paths.
    exclude: set[str]
        Qualified names already in the plan.
    limit: int
        Most functions returned; zero returns none.
    tests_allowed: bool
        Whether functions in test code may be returned.

    Returns
    -------
    tuple[str, ...]
        Qualified names, those with more name words first, then those whose
        words occur unchanged in the question, then in name order.

    Notes
    -----
    A function qualifies only when its leaf name has at least two words of
    four or more letters and every word of it matches a question word
    (``_words_match``). Special methods never qualify. This describes a
    function in prose ("which PyTorch device") without naming it.

    """

    if limit <= 0:
        return ()
    question_words = set(_identifier_words(query_text))
    scored: list[tuple[int, int, str]] = []
    for entity in graph.entities:
        if entity.kind not in _FUNCTION_KINDS or entity.name in exclude:
            continue
        leaf = entity.name.rsplit(".", 1)[-1]
        if leaf.startswith("__") and leaf.endswith("__"):
            continue
        words = set(_identifier_words(leaf))
        if sum(len(word) >= 4 for word in words) < 2:
            continue
        if not all(
            any(_words_match(word, other) for other in question_words)
            for word in words
        ):
            continue
        if not tests_allowed and _is_test_path(entity.path):
            continue
        exact = len(words & question_words)
        scored.append((-len(words), -exact, entity.name))
    return tuple(name for _, _, name in sorted(scored)[:limit])


def _without_names(query_text: str, names: Sequence[str]) -> str:
    """Remove mentions of named code entities from a question.

    Parameters
    ----------
    query_text: str
        Question text.
    names: Sequence[str]
        Qualified names of the functions and classes the question names.

    Returns
    -------
    str
        The question with each name, dotted suffix, or leaf name replaced
        by a space, ignoring case. Prose words equal to a part of a name
        ("the project folder" for ``prepare_project``) are kept.

    """

    text = query_text
    spellings = set()
    for name in names:
        parts = name.split(".")
        spellings.update(
            ".".join(parts[index:]) for index in range(len(parts))
        )
    for spelling in sorted(spellings, key=len, reverse=True):
        text = re.sub(
            _identifier_pattern(spelling), " ", text, flags=re.IGNORECASE
        )
    return text


def _focused_chunks(
    chunks: list[Chunk],
    named_leaves: set[str],
    focus_terms: tuple[str, ...],
    limit: int,
) -> list[Chunk]:
    """Keep the most relevant chunks of one long related function.

    Parameters
    ----------
    chunks: list[Chunk]
        Chunks of one function, in source order.
    named_leaves: set[str]
        Case-folded leaf names of the functions named in the question.
    focus_terms: tuple[str, ...]
        Words of the question.
    limit: int
        Most chunks kept; zero or a function with no more chunks keeps all.

    Returns
    -------
    list[Chunk]
        The selected chunks, in source order.

    """

    if limit <= 0 or len(chunks) <= limit:
        return chunks

    def relevance(chunk: Chunk) -> tuple[int, int]:
        """Rank a chunk by mentions of named functions, then shared words.

        Parameters
        ----------
        chunk: Chunk
            One chunk of the related function.

        Returns
        -------
        tuple[int, int]
            Named functions mentioned, and question words shared.

        """

        text = chunk.text.casefold()
        mentions = sum(
            1 for leaf in named_leaves if _contains_identifier(text, leaf)
        )
        words = set(_identifier_words(chunk.text))
        shared = sum(
            1
            for term in set(focus_terms)
            if any(_words_match(term, word) for word in words)
        )
        return mentions, shared

    order = {id(chunk): index for index, chunk in enumerate(chunks)}
    best = sorted(
        chunks,
        key=lambda chunk: (
            -relevance(chunk)[0],
            -relevance(chunk)[1],
            order[id(chunk)],
        ),
    )[:limit]
    return sorted(best, key=lambda chunk: order[id(chunk)])


def _shared_words(question_words: list[str], identifier: str) -> int:
    """Count identifier words that also occur in the question.

    Parameters
    ----------
    question_words: list[str]
        Words of the question, from ``_identifier_words``.
    identifier: str
        A function or method leaf name.

    Returns
    -------
    int
        Identifier words matching a question word, as decided by
        ``_words_match``.

    """

    candidates = set(question_words)
    return sum(
        1
        for word in set(_identifier_words(identifier))
        if any(_words_match(word, other) for other in candidates)
    )


def _words_match(first: str, second: str) -> bool:
    """Return whether two lowercase words are variants of one word.

    Parameters
    ----------
    first: str
        Lowercase word.
    second: str
        Lowercase word.

    Returns
    -------
    bool
        Whether the words are equal up to a simple plural ending, or the
        shorter one has at least five letters, the longer one at most three
        more, and they agree from the start up to the shorter one's
        next-to-last letter ("write" and "written", "store" and "storage",
        "selects" and "selected").

    Notes
    -----
    The length limits keep short or distant words apart: "read" does not
    match "readable" or "thread", and "event" does not match "prevents".

    """

    if _word_forms(first) & _word_forms(second):
        return True
    shorter, longer = sorted((first, second), key=len)
    if len(shorter) < 5 or len(longer) - len(shorter) > 3:
        return False
    return len(os.path.commonprefix((shorter, longer))) >= len(shorter) - 1


def _word_forms(word: str) -> set[str]:
    """Return a word with its simple singular forms.

    Parameters
    ----------
    word: str
        Lowercase word.

    Returns
    -------
    set[str]
        The word and, for plural-looking words, the forms without "s",
        "es", or with "ies" changed to "y".

    """

    forms = {word}
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        forms.add(word[:-1])
        if word.endswith("es"):
            forms.add(word[:-2])
        if word.endswith("ies"):
            forms.add(word[:-3] + "y")
    return forms


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
