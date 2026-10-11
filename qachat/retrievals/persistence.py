"""Serialize retrieval indexes with chunk-and-graph provenance fingerprints.

``PersistentIndexes`` groups semantic and lexical indexes with a code graph.
JSON storage includes chunks, vectors, lexical statistics, and relationships.
The stored fingerprint identifies ordered chunk content and the graph; it
does not cover the numeric embedding vectors, lexical weights, embedding
model identity, or model revision.

Loading reconstructs the records and validates vector cardinality, finite
numeric values, lexical vocabulary consistency, unique chunk identifiers,
and the chunk/graph fingerprint. These checks are not a complete schema
validator or authenticity guarantee. Files are replaced through a temporary
path, with no inter-process lock or transaction spanning metadata files.

Key Features
------------

- **Combined index records**

    Store the three retrieval inputs together while requiring semantic and
    lexical chunk alignment.

- **JSON persistence**

    Encode paths and immutable record fields into a reconstructable JSON
    representation.

- **Content identity**

    Compare supplied chunks and graph against the stored fingerprint without
    scanning a repository.

Module Structure
----------------

- ``PersistentIndexes``

    Retrieval data saved for the same repository state.

- ``prepare_persistent_indexes``

    Validate and combine retrieval data from one repository state.

- ``save_indexes``

    Save persistent retrieval data as JSON and return its path.

- ``load_indexes``

    Load semantic, lexical, and code-graph data from JSON.

- ``indexes_are_current``

    Return whether saved retrieval data matches the current repository.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> indexes = qachat.load_indexes("/path/to/workspace/indexes/retrieval.json")
>>> matches = qachat.indexes_are_current(
...     indexes, list(indexes.semantic.chunks), indexes.code_graph,
... )

The file must have been written in the supported index format.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from qachat.chunking import Chunk
from qachat.retrievals.code_graph.code_graph import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
)
from qachat.retrievals.strategies.retrieval_lexical import LexicalIndex
from qachat.retrievals.strategies.retrieval_semantic import (
    SemanticIndex,
    _validate_semantic_embeddings,
)

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

    Parameters
    ----------
    fingerprint: str
        Stable digest of the indexed chunks and code graph.
    semantic: SemanticIndex
        Saved semantic index.
    lexical: LexicalIndex
        Saved lexical index.
    code_graph: CodeGraph
        Saved deterministic code graph.

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

    Notes
    -----
    The fingerprint covers serialized ordered chunks and the code graph. It
    excludes numeric vectors, lexical statistics, embedding model identity, and
    backend/library versions. Frozen fields do not make nested lexical
    dictionaries immutable.

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
        If chunks are not aligned, chunk identifiers are duplicated, or
        semantic/lexical vectors are malformed.

    Notes
    -----
    Validation checks vector structure and finite numeric values but does
    not establish embedding-model identity or require unit-normalized vectors.

    """

    _validate_index_records(semantic, lexical)

    chunks = list(semantic.chunks)
    return PersistentIndexes(
        fingerprint=_repository_fingerprint(chunks, code_graph),
        semantic=semantic,
        lexical=lexical,
        code_graph=code_graph,
    )


def save_indexes(indexes: PersistentIndexes, path: str | Path) -> Path:
    """Save persistent retrieval data as JSON and return its path.

    Parameters
    ----------
    indexes: PersistentIndexes
        Combined semantic, lexical, and code-graph indexes with their stored
        chunk/graph fingerprint.
    path: str | pathlib.Path
        Index JSON file path. Relative paths use the working directory; the
        path is expanded and resolved.

    Returns
    -------
    pathlib.Path
        Resolved path of the written JSON bundle.

    Raises
    ------
    OSError
        If directories, the temporary file, or the replacement cannot be
        written.
    TypeError
        If a stored field cannot be serialized to JSON.
    ValueError
        If the stored chunks or retrieval vectors fail validation.

    Notes
    -----
    Validates index structure and serializes embedding scalars as Python
    floats before replacing a temporary sibling file. The supplied fingerprint
    is stored as given, not recalculated here. Concurrent writes are not
    coordinated.

    """

    _validate_index_records(indexes.semantic, indexes.lexical)

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _FORMAT_VERSION,
        "fingerprint": indexes.fingerprint,
        "chunks": [_chunk_to_data(chunk) for chunk in indexes.semantic.chunks],
        "semantic_embeddings": [
            [float(value) for value in vector]
            for vector in indexes.semantic.embeddings
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
    """Load semantic, lexical, and code-graph data from JSON.

    Parameters
    ----------
    path: str | pathlib.Path
        Index JSON file path. Relative paths use the working directory; the
        path is expanded and resolved.

    Returns
    -------
    PersistentIndexes
        Reconstructed PersistentIndexes with a verified chunk/graph
        fingerprint.

    Raises
    ------
    ValueError
        If the format version, structural fields, vector contents, chunk
        identifiers, or source fingerprint are invalid.
    OSError
        If the file cannot be read.

    Notes
    -----
    Vector counts, dimensions, types, and finiteness are checked before
    accepting saved data. The source fingerprint still excludes numeric
    embeddings, lexical weights, and model identity; payload checksums and
    model provenance are reserved for a later format revision.

    """

    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload["format_version"] != _FORMAT_VERSION:
            raise ValueError("Unsupported index format version.")

        chunks = tuple(_chunk_from_data(item) for item in payload["chunks"])
        semantic = SemanticIndex(
            chunks=chunks,
            embeddings=tuple(
                tuple(_stored_number(value) for value in vector)
                for vector in payload["semantic_embeddings"]
            ),
        )
        lexical = LexicalIndex(
            chunks=chunks,
            vectors=tuple(
                _stored_weight_mapping(vector)
                for vector in payload["lexical_vectors"]
            ),
            inverse_document_frequency=_stored_weight_mapping(
                payload["inverse_document_frequency"]
            ),
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
    """Return whether saved retrieval data matches the current repository.

    Parameters
    ----------
    indexes: PersistentIndexes
        Combined semantic, lexical, and code-graph indexes with their stored
        chunk/graph fingerprint.
    chunks: list[Chunk]
        Ordered source chunks used to compute the content/provenance
        fingerprint.
    code_graph: CodeGraph
        Code entities and relations included in the fingerprint or serialized
        bundle.

    Returns
    -------
    bool
        Whether saved retrieval data matches the current repository.

    Notes
    -----
    Only the supplied chunk/graph fingerprint is compared. No files are read
    and model identity, model revisions, vector data, and current source
    selection are not checked.

    """

    current = _repository_fingerprint(chunks, code_graph)
    return indexes.fingerprint == current


def _repository_fingerprint(
    chunks: list[Chunk],
    code_graph: CodeGraph,
) -> str:
    """Return a stable digest for chunks and deterministic code structure.

    Parameters
    ----------
    chunks: list[Chunk]
        Ordered source chunks used to compute the content/provenance
        fingerprint.
    code_graph: CodeGraph
        Code entities and relations included in the fingerprint or serialized
        bundle.

    Returns
    -------
    str
        A stable digest for chunks and deterministic code structure.

    Notes
    -----
    Array/list ordering participates in the hash. The hash is a content
    identifier, not an authenticity signature, and does not include embeddings
    or lexical weights.

    """

    data = {
        "chunks": [_chunk_to_data(chunk) for chunk in chunks],
        "code_graph": _code_graph_to_data(code_graph),
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _chunk_to_data(chunk: Chunk) -> dict[str, object]:
    """Convert one chunk to JSON-safe data.

    Parameters
    ----------
    chunk: Chunk
        One source chunk to encode as JSON-compatible data.

    Returns
    -------
    dict[str, object]
        Dataclass-field dictionary with path converted to its POSIX string
        form.

    """

    data = asdict(chunk)
    data["path"] = chunk.path.as_posix()
    return data


def _chunk_from_data(data: dict[str, object]) -> Chunk:
    """Restore one chunk from checked JSON data.

    Parameters
    ----------
    data: dict[str, object]
        Decoded dictionary representing one serialized record.

    Returns
    -------
    Chunk
        Chunk reconstructed using the saved fields and implemented scalar
        conversions.

    Notes
    -----
    Required fields are indexed directly and paths are not checked for
    containment. Missing keys and failed type conversions propagate to the
    caller.

    """

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
    """Convert one code graph to JSON-safe data.

    Parameters
    ----------
    code_graph: CodeGraph
        Code entities and relations included in the fingerprint or serialized
        bundle.

    Returns
    -------
    dict[str, object]
        Dictionary with entity and relation lists, converting entity paths to
        POSIX strings.

    """

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
    """Restore one code graph from checked JSON data.

    Parameters
    ----------
    data: dict[str, object]
        Decoded dictionary representing one serialized record.

    Returns
    -------
    CodeGraph
        CodeGraph reconstructed from entity and relation lists.

    Raises
    ------
    TypeError
        If the entities or relations container is not a list.
    KeyError
        If a required graph or record field is absent.

    Notes
    -----
    Names and relationship labels are converted to strings; graph connectivity
    and path containment are not validated here.

    """

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


def _stored_number(value: object) -> float:
    """Convert a JSON number without accepting Boolean or text coercion.

    Parameters
    ----------
    value: object
        Raw JSON value expected to represent a finite real number.

    Returns
    -------
    float
        Finite Python floating-point value.

    Raises
    ------
    ValueError
        If the saved value is not a finite JSON number.

    """
    if type(value) not in (int, float):
        raise ValueError("The saved index contains an invalid numeric value.")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(
            "The saved index numeric value is too large."
        ) from error
    if not math.isfinite(converted):
        raise ValueError("The saved index contains a non-finite value.")
    return converted


def _stored_weight_mapping(data: object) -> dict[str, float]:
    """Decode a JSON sparse vector or inverse-frequency dictionary.

    Parameters
    ----------
    data: object
        Saved lexical term-to-weight mapping.

    Returns
    -------
    dict[str, float]
        Term weights validated as finite real values.

    Raises
    ------
    ValueError
        If the mapping or any numeric component is malformed.

    """
    if not isinstance(data, dict):
        raise ValueError("The saved lexical weights must be a dictionary.")
    return {
        str(term): _stored_number(weight)
        for term, weight in data.items()
    }


def _validate_index_records(
    semantic: SemanticIndex,
    lexical: LexicalIndex,
) -> None:
    """Validate aligned retrieval records before saving or accepting data.

    Parameters
    ----------
    semantic: SemanticIndex
        Ordered semantic retrieval data and chunk provenance.
    lexical: LexicalIndex
        Ordered lexical retrieval data and term statistics.

    Raises
    ------
    ValueError
        If chunk identities, embedding vectors, or lexical data are invalid.

    Notes
    -----
    Semantic vector norms need not equal one. Empty lexical vectors are
    permitted, but their keys must be present in the IDF vocabulary.

    """
    if semantic.chunks != lexical.chunks:
        raise ValueError("Semantic and lexical indexes must share chunks.")

    chunks = semantic.chunks
    if not chunks:
        raise ValueError("Persistent indexes require at least one chunk.")
    identifiers = [chunk.identifier for chunk in chunks]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("Chunk identifiers must be non-empty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Chunk identifiers must be unique.")

    _validate_semantic_embeddings(semantic.embeddings, len(chunks))

    if not isinstance(lexical.vectors, (tuple, list)) or (
        len(lexical.vectors) != len(chunks)
    ):
        raise ValueError("Lexical vector count does not match chunks.")
    vocabulary = lexical.inverse_document_frequency
    if not isinstance(vocabulary, dict):
        raise ValueError("Lexical IDF must be a term-weight dictionary.")

    for term, weight in vocabulary.items():
        if not isinstance(term, str) or not term:
            raise ValueError("Lexical IDF terms must be non-empty strings.")
        _validate_lexical_weight(weight, label="Lexical IDF")

    for vector in lexical.vectors:
        if not isinstance(vector, dict):
            raise ValueError("Lexical vectors must be dictionaries.")
        for term, weight in vector.items():
            if not isinstance(term, str) or not term:
                raise ValueError("Lexical terms must be non-empty strings.")
            if term not in vocabulary:
                raise ValueError("Lexical term is absent from the IDF index.")
            _validate_lexical_weight(weight, label="Lexical vector")


def _validate_lexical_weight(value: float, *, label: str) -> None:
    """Reject non-finite, negative, or nonnumeric lexical weights.

    Parameters
    ----------
    value: float
        A sparse TF-IDF component or inverse document frequency.
    label: str
        Location of the weight for validation diagnostics.

    Raises
    ------
    ValueError
        If a value cannot represent a TF-IDF weight.

    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must contain finite nonnegative weights.")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite or value < 0.0:
        raise ValueError(f"{label} must contain finite nonnegative weights.")
