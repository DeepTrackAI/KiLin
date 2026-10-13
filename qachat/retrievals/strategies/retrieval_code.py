"""Retrieve chunks near code entities recognized in a question.

Name matching normalizes identifiers and detects a configured set of operator
symbols. Resolved entities seed an undirected traversal of the supplied graph;
matching chunks are ranked by graph distance and deterministic tie-breakers.
The graph and name resolver are heuristics, so a match is not proof of a
runtime call, dependency, or intended meaning of a word in the question.

Key Features
------------

- **Entity seeds**

    Resolve known leaf names, qualified names, and configured operator-related
    method names.

- **Bounded graph traversal**

    Search neighboring entities up to the requested hop count.

- **Source-backed results**

    Return only chunks whose source path and symbol match a candidate entity.

Module Structure
----------------

- ``CodeResult``

    One chunk found through code-aware retrieval.

- ``resolve_code_entities``

    Return entities explicitly named by a query.

- ``code_search``

    Find chunks for explicitly named and related code entities.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> graph = qachat.CodeGraph(entities=(), relations=())
>>> names = qachat.resolve_code_entities(graph, "How does loading work?")

"""

from __future__ import annotations

from dataclasses import dataclass
import re

from qachat.chunking import Chunk
from qachat.defaults import (
    load_defaults_section,
    require_default_nonnegative_int,
)
from qachat.retrievals.code_graph.code_graph import CodeEntity, CodeGraph

__all__ = [
    "CodeResult",
    "resolve_code_entities",
    "code_search",
]

# Default number of graph edges traversed from explicitly named entities.
_RETRIEVAL_DEFAULTS = load_defaults_section("RETRIEVAL")
_DEFAULT_CODE_MAX_DISTANCE = require_default_nonnegative_int(
    _RETRIEVAL_DEFAULTS, "CODE_MAX_DISTANCE", "RETRIEVAL"
)

_WORD = re.compile(r"[A-Za-z0-9]+")

# Python operator symbols mapped to the dunder method(s) they invoke. A query
# like "what does & do" names no identifier at all -- _name_words() strips
# punctuation entirely, so the ordinary whole-word matching below never sees
# it. Some dunder names normalize to common English words once underscores
# become spaces (__and__ -> "and"), which would false-match any unrelated
# query merely containing that word -- so this table is matched against the
# raw query text directly, on the literal symbol, never through the
# word-normalization path used for identifier names.
_OPERATOR_DUNDERS: dict[str, tuple[str, ...]] = {
    ">>": ("__rshift__", "__rrshift__"),
    "<<": ("__lshift__", "__rlshift__"),
    "**": ("__pow__", "__rpow__"),
    "//": ("__floordiv__", "__rfloordiv__"),
    "==": ("__eq__",),
    "!=": ("__ne__",),
    "<=": ("__le__", "__rle__"),
    ">=": ("__ge__", "__rge__"),
    "&": ("__and__", "__rand__"),
    "|": ("__or__", "__ror__"),
    "^": ("__xor__", "__rxor__"),
    "+": ("__add__", "__radd__"),
    "-": ("__sub__", "__rsub__"),
    "*": ("__mul__", "__rmul__"),
    "/": ("__truediv__", "__rtruediv__"),
    "%": ("__mod__", "__rmod__"),
    "<": ("__lt__", "__rlt__"),
    ">": ("__gt__", "__rgt__"),
    "~": ("__invert__",),
}
# Reflected dunders (the second name for any operator with two): only
# invoked as Python's fallback when the left operand does not define the
# forward method, so a query naming the symbol almost always means the
# forward method. Used to break ties between same-depth siblings below.
_REFLECTED_DUNDERS: frozenset[str] = frozenset(
    names[1] for names in _OPERATOR_DUNDERS.values() if len(names) > 1
)

# Longest symbols first, so ">>" matches before the ">" it contains.
_OPERATOR_PATTERN = re.compile(
    "|".join(
        rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])"
        for symbol in sorted(_OPERATOR_DUNDERS, key=len, reverse=True)
    )
)


@dataclass(frozen=True, slots=True)
class CodeResult:
    """One chunk found through code-aware retrieval.

    Parameters
    ----------
    score: float
        Deterministic score based on graph distance.
    chunk: Chunk
        Chunk belonging to the matched or related entity.
    entity: str
        Qualified code entity that led to the chunk.
    distance: int
        Number of relationships from the query match.

    Attributes
    ----------
    score: float
        Deterministic score based on graph distance.
    chunk: Chunk
        Chunk belonging to the matched or related entity.
    entity: str
        Qualified code entity that led to the chunk.
    distance: int
        Number of relationships from the query match.

    """

    score: float
    chunk: Chunk
    entity: str
    distance: int


def resolve_code_entities(
    graph: CodeGraph,
    query_text: str,
) -> list[CodeEntity]:
    """Return entities explicitly named by a query.

    Parameters
    ----------
    graph: CodeGraph
        Source-derived graph used to recognize entities and traverse
        relationships.
    query_text: str
        Nonblank question text used for identifier and configured operator
        matching.

    Returns
    -------
    list[CodeEntity]
        Entities explicitly named by a query.

    Raises
    ------
    ValueError
        If query_text is not nonblank text.

    Notes
    -----
    Name and operator matching are heuristic. Normalized leaf-name matches
    favor the longest recognized leaf names; this does not prove the question
    refers to a particular runtime object.

    """

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be non-empty text.")

    query_words = _name_words(query_text)
    word_matches = [
        entity
        for entity in graph.entities
        if _contains_whole_words(
            query_words,
            _name_words(entity.name.split(".")[-1]),
        )
    ]
    if word_matches:
        longest = max(
            len(_name_words(entity.name.split(".")[-1]))
            for entity in word_matches
        )
        word_matches = [
            entity
            for entity in word_matches
            if len(_name_words(entity.name.split(".")[-1])) == longest
        ]

    operator_matches = _operator_seed_entities(graph, query_text)
    if not word_matches and not operator_matches:
        return []

    seen: set[str] = set()
    matches: list[CodeEntity] = []
    for entity in [*word_matches, *operator_matches]:
        if entity.name in seen:
            continue
        seen.add(entity.name)
        matches.append(entity)
    return matches


def code_search(
    graph: CodeGraph,
    chunks: list[Chunk],
    query_text: str,
    *,
    count: int = 5,
    max_distance: int = _DEFAULT_CODE_MAX_DISTANCE,
) -> list[CodeResult]:
    """Find chunks for explicitly named and related code entities.

    Parameters
    ----------
    graph: CodeGraph
        Source-derived graph used to recognize entities and traverse
        relationships.
    chunks: list[Chunk]
        Available retrieval chunks whose paths and symbols can match graph
        entities.
    query_text: str
        Nonblank question text used for identifier and configured operator
        matching.
    count: int, optional
        Positive exact integer limiting returned code results. Defaults to
        ``5``.
    max_distance: int, optional
        Nonnegative exact integer bounding undirected graph hops from matched
        seeds. Defaults to ``RETRIEVAL.CODE_MAX_DISTANCE`` (currently 1).

    Returns
    -------
    list[CodeResult]
        CodeResult list limited by count, ordered by graph distance and the
        implemented specificity/identifier tie-breakers.

    Raises
    ------
    ValueError
        If count or max_distance is invalid, or query_text is blank/non-string.

    Notes
    -----
    No model is queried. Traversal treats all relation types as undirected
    links. Only chunks matching a candidate entity path and symbol become
    results. No matched entity yields an empty result list.

    """

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")
    if type(max_distance) is not int or max_distance < 0:
        raise ValueError("max_distance must be a non-negative whole number.")

    seeds = resolve_code_entities(graph, query_text)
    distances = _graph_distances(
        graph,
        {entity.name for entity in seeds},
        max_distance,
    )
    entity_paths = {entity.name: entity.path for entity in graph.entities}
    results: list[CodeResult] = []
    for chunk in chunks:
        if chunk.symbol is None:
            continue
        candidates = [
            (entity, distance)
            for entity, distance in distances.items()
            if entity_paths.get(entity) == chunk.path
            and _symbols_match(entity, chunk.symbol)
        ]
        if not candidates:
            continue
        entity, distance = min(candidates, key=lambda item: item[1])
        results.append(
            CodeResult(
                score=1.0 / (distance + 1),
                chunk=chunk,
                entity=entity,
                distance=distance,
            )
        )

    # Tie-break: prefer the more specific matched entity before falling
    # back to the chunk identifier. Without this, a broad seed (a class
    # named in the query) and a narrow seed (a specific method within it,
    # e.g. from an operator symbol) tie at the same distance, and a large
    # class chunked into many pieces can crowd out the one chunk for the
    # narrow match on nothing more than hash order.
    #
    # A second tie-break handles same-depth siblings: an operator symbol
    # resolves to both its forward dunder (__and__) and its reflected
    # counterpart (__rand__, only invoked when the left operand does not
    # define the forward method), and both sit at the same distance and
    # depth. The forward method is what runs for the plain "a & b" case
    # a query about the symbol is almost always asking about, so it wins
    # the tie; only when the forward chunk is unavailable does the
    # reflected one surface instead.
    results.sort(
        key=lambda result: (
            result.distance,
            -result.entity.count("."),
            result.entity.split(".")[-1] in _REFLECTED_DUNDERS,
            result.chunk.identifier,
        )
    )
    return results[: min(count, len(results))]


def _name_words(text: str) -> str:
    """Return compact lowercase words for identifier matching.

    Parameters
    ----------
    text: str
        Identifier or question text normalized into lowercase word components.

    Returns
    -------
    str
        Compact lowercase words for identifier matching.

    """

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(_WORD.findall(expanded.replace("_", " ").lower()))


def _operator_seed_entities(
    graph: CodeGraph,
    query_text: str,
) -> list[CodeEntity]:
    """Return entities for an operator symbol written literally in the query.

    A symbol only counts when it is not adjacent to a letter or digit on either
    side. This keeps ordinary punctuation and numbers from triggering a match
    -- a hyphen inside "well-known" or a slash inside "9/30" never reaches this
    far -- while still recognizing a symbol set off by whitespace, quotes,
    backticks, or parentheses, the way "the & operator", "using `>>`", and "(a
    & b)" all are.

    Parameters
    ----------
    graph: CodeGraph
        Source-derived graph used to recognize entities and traverse
        relationships.
    query_text: str
        Nonblank question text used for identifier and configured operator
        matching.

    Returns
    -------
    list[CodeEntity]
        Entities for an operator symbol written literally in the query.

    """

    dunder_names: set[str] = set()
    for match in _OPERATOR_PATTERN.finditer(query_text):
        dunder_names.update(_OPERATOR_DUNDERS[match.group(0)])
    if not dunder_names:
        return []
    return [
        entity
        for entity in graph.entities
        if entity.name.split(".")[-1] in dunder_names
    ]


def _contains_whole_words(query_words: str, name_words: str) -> bool:
    """Return whether name_words appears in query_words as whole words.

    Both arguments are already space-joined, lowercase word sequences from
    :func:`_name_words`. A plain substring check would let a short entity name
    match inside an unrelated longer word that merely starts the same way --
    for example the module entity ``project`` would match the query word
    ``projects`` (an ordinary English plural, not a reference to
    ``qachat/project.py``), or ``act`` would match inside ``react``. This
    instead requires name_words to appear at a word boundary on both sides, so
    only an actual whole-word (or whole-phrase, for multi-word names)
    occurrence counts as an explicit reference.

    Parameters
    ----------
    query_words: str
        Normalized words from the query text.
    name_words: str
        Normalized words from one code entity's short name.

    Returns
    -------
    bool
        Whether name_words occurs in query_words as whole words.

    """

    if not name_words:
        return False
    return (
        re.search(
            rf"(?<!\w){re.escape(name_words)}(?!\w)",
            query_words,
        )
        is not None
    )


def _graph_distances(
    graph: CodeGraph,
    seeds: set[str],
    max_distance: int,
) -> dict[str, int]:
    """Return shortest undirected distances from seed entities.

    Parameters
    ----------
    graph: CodeGraph
        Source-derived graph used to recognize entities and traverse
        relationships.
    seeds: set[str]
        Entity names used as distance-zero graph traversal seeds.
    max_distance: int
        Nonnegative exact integer bounding undirected graph hops from matched
        seeds.

    Returns
    -------
    dict[str, int]
        Shortest undirected distances from seed entities.

    Notes
    -----
    Computes minimum hop distances in an undirected adjacency map. Relations
    need not have explicit entity records at both endpoints.

    """

    distances = {seed: 0 for seed in seeds}
    frontier = set(seeds)
    for distance in range(1, max_distance + 1):
        following: set[str] = set()
        for relation in graph.relations:
            if relation.source in frontier:
                following.add(relation.target)
            if relation.target in frontier:
                following.add(relation.source)
        following.difference_update(distances)
        distances.update({name: distance for name in following})
        frontier = following
    return distances


def _symbols_match(entity: str, symbol: str) -> bool:
    """Return whether a graph entity corresponds to a chunk symbol.

    Parameters
    ----------
    entity: str
        Qualified graph entity name to compare with a source symbol.
    symbol: str
        Optional source-unit symbol name to compare with a graph entity.

    Returns
    -------
    bool
        Whether a graph entity corresponds to a chunk symbol.

    """

    return entity == symbol or entity.endswith(f".{symbol}")
