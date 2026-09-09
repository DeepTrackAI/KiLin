"""Save and load reusable retrieval data.

Classes
-------
PersistentIndexes
    Semantic, lexical, and code-graph data with a source fingerprint.

Functions
---------
prepare_persistent_indexes
    Validate and combine in-memory retrieval data.
save_indexes
    Save combined retrieval data as readable JSON.
load_indexes
    Restore combined retrieval data from JSON.
indexes_are_current
    Compare saved content with current chunks and code graph.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from qachat.chunking import Chunk
from qachat.retrievals.code_graph.code_graph import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
)
from qachat.retrievals.strategies.retrieval_lexical import LexicalIndex
from qachat.retrievals.strategies.retrieval_semantic import SemanticIndex


__all__ = [
    "PersistentIndexes",
    "prepare_persistent_indexes",
    "save_indexes",
    "load_indexes",
    "indexes_are_current",
]

_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PersistentIndexes:
    """Retrieval data saved for the same repository state.

    Attributes
    ----------
    fingerprint: str
        Stable digest of the indexed chunks and code graph.
    semantic: SemanticIndex
        Saved semantic index.
    lexical: LexicalIndex
        Saved lexical index.
    code_graph: CodeGraph
        Saved deterministic code graph.

    """

    fingerprint: str
    semantic: SemanticIndex
    lexical: LexicalIndex
    code_graph: CodeGraph


def prepare_persistent_indexes(
    semantic: SemanticIndex,
    lexical: LexicalIndex,
    code_graph: CodeGraph,
) -> PersistentIndexes:
    """Validate and combine retrieval data from one repository state.

    Parameters
    ----------
    semantic: SemanticIndex
        In-memory semantic index.
    lexical: LexicalIndex
        In-memory lexical index.
    code_graph: CodeGraph
        In-memory deterministic code graph.

    Returns
    -------
    PersistentIndexes
        Combined retrieval data and source fingerprint.

    Raises
    ------
    ValueError
        If semantic and lexical indexes contain different chunks.

    """

    if semantic.chunks != lexical.chunks:
        raise ValueError("Semantic and lexical indexes must share chunks.")

    chunks = list(semantic.chunks)
    return PersistentIndexes(
        fingerprint=_repository_fingerprint(chunks, code_graph),
        semantic=semantic,
        lexical=lexical,
        code_graph=code_graph,
    )


def save_indexes(indexes: PersistentIndexes, path: str | Path) -> Path:
    """Save persistent retrieval data as JSON and return its path."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _FORMAT_VERSION,
        "fingerprint": indexes.fingerprint,
        "chunks": [_chunk_to_data(chunk) for chunk in indexes.semantic.chunks],
        "semantic_embeddings": [
            list(vector) for vector in indexes.semantic.embeddings
        ],
        "lexical_vectors": [
            dict(vector) for vector in indexes.lexical.vectors
        ],
        "inverse_document_frequency": dict(
            indexes.lexical.inverse_document_frequency
        ),
        "code_graph": _code_graph_to_data(indexes.code_graph),
    }
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_indexes(path: str | Path) -> PersistentIndexes:
    """Load semantic, lexical, and code-graph data from JSON."""

    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload["format_version"] != _FORMAT_VERSION:
            raise ValueError("Unsupported index format version.")

        chunks = tuple(_chunk_from_data(item) for item in payload["chunks"])
        semantic = SemanticIndex(
            chunks=chunks,
            embeddings=tuple(
                tuple(float(value) for value in vector)
                for vector in payload["semantic_embeddings"]
            ),
        )
        lexical = LexicalIndex(
            chunks=chunks,
            vectors=tuple(
                {str(term): float(value) for term, value in vector.items()}
                for vector in payload["lexical_vectors"]
            ),
            inverse_document_frequency={
                str(term): float(value)
                for term, value in payload[
                    "inverse_document_frequency"
                ].items()
            },
        )
        code_graph = _code_graph_from_data(payload["code_graph"])
        indexes = PersistentIndexes(
            fingerprint=str(payload["fingerprint"]),
            semantic=semantic,
            lexical=lexical,
            code_graph=code_graph,
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("The saved index file is invalid.") from error

    expected = prepare_persistent_indexes(
        semantic,
        lexical,
        code_graph,
    )
    if indexes.fingerprint != expected.fingerprint:
        raise ValueError("The saved index fingerprint is invalid.")
    return indexes


def indexes_are_current(
    indexes: PersistentIndexes,
    chunks: list[Chunk],
    code_graph: CodeGraph,
) -> bool:
    """Return whether saved retrieval data matches the current repository."""

    current = _repository_fingerprint(chunks, code_graph)
    return indexes.fingerprint == current


def _repository_fingerprint(
    chunks: list[Chunk],
    code_graph: CodeGraph,
) -> str:
    """Return a stable digest for chunks and deterministic code structure."""

    data = {
        "chunks": [_chunk_to_data(chunk) for chunk in chunks],
        "code_graph": _code_graph_to_data(code_graph),
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _chunk_to_data(chunk: Chunk) -> dict[str, object]:
    """Convert one chunk to JSON-safe data."""

    data = asdict(chunk)
    data["path"] = chunk.path.as_posix()
    return data


def _chunk_from_data(data: dict[str, object]) -> Chunk:
    """Restore one chunk from checked JSON data."""

    return Chunk(
        identifier=str(data["identifier"]),
        path=Path(str(data["path"])),
        text=str(data["text"]),
        start_line=int(data["start_line"]),
        end_line=int(data["end_line"]),
        kind=str(data["kind"]),
        symbol=(str(data["symbol"]) if data["symbol"] is not None else None),
        section=(
            str(data["section"]) if data["section"] is not None else None
        ),
        cell=(int(data["cell"]) if data["cell"] is not None else None),
    )


def _code_graph_to_data(code_graph: CodeGraph) -> dict[str, object]:
    """Convert one code graph to JSON-safe data."""

    return {
        "entities": [
            {
                "name": entity.name,
                "kind": entity.kind,
                "path": entity.path.as_posix(),
            }
            for entity in code_graph.entities
        ],
        "relations": [asdict(relation) for relation in code_graph.relations],
    }


def _code_graph_from_data(data: dict[str, object]) -> CodeGraph:
    """Restore one code graph from checked JSON data."""

    entities_data = data["entities"]
    relations_data = data["relations"]
    if not isinstance(entities_data, list):
        raise TypeError("Code-graph entities must be a list.")
    if not isinstance(relations_data, list):
        raise TypeError("Code-graph relations must be a list.")

    entities = tuple(
        CodeEntity(
            name=str(item["name"]),
            kind=str(item["kind"]),
            path=Path(str(item["path"])),
        )
        for item in entities_data
    )
    relations = tuple(
        CodeRelation(
            source=str(item["source"]),
            relation=str(item["relation"]),
            target=str(item["target"]),
        )
        for item in relations_data
    )
    return CodeGraph(
        entities=entities,
        relations=relations,
    )
