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

_WORD = re.compile(r"[A-Za-z0-9]+")


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
    matches = [
        entity
        for entity in graph.entities
        if _name_words(entity.name.split(".")[-1]) in query_words
    ]
    if not matches:
        return []

    longest = max(
        len(_name_words(entity.name.split(".")[-1])) for entity in matches
    )
    return [
        entity
        for entity in matches
        if len(_name_words(entity.name.split(".")[-1])) == longest
    ]


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

    results.sort(
        key=lambda result: (
            result.distance,
            result.chunk.identifier,
        )
    )
    return results[: min(count, len(results))]


def _name_words(text: str) -> str:
    """Return compact lowercase words for identifier matching."""

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(_WORD.findall(expanded.replace("_", " ").lower()))


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
