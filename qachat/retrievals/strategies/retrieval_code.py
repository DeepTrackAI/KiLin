"""Retrieve chunks through deterministic code relationships.

Classes
-------
CodeResult
    One chunk found through a code relationship.

Functions
---------
resolve_code_entities
    Match query wording to named code entities.
code_search
    Rank chunks belonging to matched and related entities.

"""

from __future__ import annotations

from dataclasses import dataclass
import re

from qachat.chunking import Chunk
from qachat.retrievals.code_graph.code_graph import CodeEntity, CodeGraph


__all__ = [
    "CodeResult",
    "resolve_code_entities",
    "code_search",
]

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
    """Return entities explicitly named by a query."""

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
    max_distance: int = 1,
) -> list[CodeResult]:
    """Find chunks for explicitly named and related code entities."""

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
    """Return compact lowercase words for identifier matching."""

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(_WORD.findall(expanded.replace("_", " ").lower()))


def _operator_seed_entities(
    graph: CodeGraph,
    query_text: str,
) -> list[CodeEntity]:
    """Return entities for an operator symbol written literally in the query.

    A symbol only counts when it is not adjacent to a letter or digit on
    either side. This keeps ordinary punctuation and numbers from triggering
    a match -- a hyphen inside "well-known" or a slash inside "9/30" never
    reaches this far -- while still recognizing a symbol set off by
    whitespace, quotes, backticks, or parentheses, the way "the & operator",
    "using `>>`", and "(a & b)" all are.
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
    :func:`_name_words`. A plain substring check would let a short entity
    name match inside an unrelated longer word that merely starts the same
    way -- for example the module entity ``project`` would match the query
    word ``projects`` (an ordinary English plural, not a reference to
    ``qachat/project.py``), or ``act`` would match inside ``react``. This
    instead requires name_words to appear at a word boundary on both sides,
    so only an actual whole-word (or whole-phrase, for multi-word names)
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
    """Return shortest undirected distances from seed entities."""

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
    """Return whether a graph entity corresponds to a chunk symbol."""

    return entity == symbol or entity.endswith(f".{symbol}")
