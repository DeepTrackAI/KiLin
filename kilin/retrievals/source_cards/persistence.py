"""Persist source-card content and embeddings in independent files.

Cards and eligibility outcomes are stored separately from the vectors used
to search complete cards. Each file has a format version and a canonical
payload checksum; each vector group also records the digest of its own card
and the exact search texts embedded. This permits incremental reuse after
partial updates without trusting stale vectors from a previous card version.

Files are atomically replaced one at a time using the existing core writer.
There is no cross-file transaction or inter-process writer coordination.
If indexing stops between writes, card content remains readable and vector
groups inconsistent with that content are discarded on load. Missing files
are empty indexes, not an invitation to read obsolete paper-card formats.

Key Features
------------

- **Independent storage**

    Save card content/status and embedding vectors in separate JSON files.

- **Integrity and recovery**

    Reject malformed or incorrectly checksummed payloads, and ignore stale
    vector groups.

- **No migration**

    Read only the new source-card formats; older paper caches are unaffected.

Module Structure
----------------

- ``save_source_cards`` and ``load_source_cards``

    Persist and recover complete card outcomes.

- ``save_source_card_vectors`` and ``load_source_card_vectors``

    Persist and recover vectors bound to cards and search-text projections.

Examples
--------
>>> from pathlib import Path
>>> from kilin.retrievals.source_cards import SourceCardIndex
>>> from kilin.retrievals.source_cards import load_source_cards
>>> load_source_cards(Path("nonexistent_cards.json"))
SourceCardIndex(records=())

"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from kilin.atomic_writes import atomic_write_text
from kilin.retrievals.source_cards.index import (
    IndexedSourceCard,
    SourceCardIndex,
    SourceCardVector,
    SourceCardVectorIndex,
    card_digest,
    prepare_source_card_vectors,
)
from kilin.retrievals.source_cards.models import (
    CardStatement,
    SourceCard,
    SourceCardResult,
)

__all__ = [
    "load_source_card_vectors",
    "load_source_cards",
    "save_source_card_vectors",
    "save_source_cards",
]

_CARD_FORMAT = "source-cards-v1"
_VECTOR_FORMAT = "source-card-vectors-v1"


def save_source_cards(index: SourceCardIndex, path: str | Path) -> Path:
    """Atomically save validated card outcomes to one JSON file.

    Parameters
    ----------
    index: SourceCardIndex
        Previously validated card and eligibility records.
    path: str | pathlib.Path
        Destination JSON file; its parent is created if needed.

    Returns
    -------
    pathlib.Path
        Resolved destination path.

    Raises
    ------
    ValueError
        If card records cannot be serialized consistently.
    OSError
        If the destination cannot be written atomically.

    Notes
    -----
    Source-kind and chunk-provenance checks occur on index admission;
    serialization checks structural consistency and payload integrity.

    """

    if not isinstance(index, SourceCardIndex):
        raise ValueError("Expected a SourceCardIndex.")
    # Revalidate even if a caller constructed a modified frozen record.
    SourceCardIndex(index.records)
    payload = {
        "format": _CARD_FORMAT,
        "records": [
            _record_to_data(entry)
            for entry in sorted(index.records, key=lambda item: (
                item.result.path.as_posix(), item.result.kind
            ))
        ],
    }
    return _write(path, payload)


def load_source_cards(path: str | Path) -> SourceCardIndex:
    """Read source-card outcomes, including cached ineligible sources.

    Parameters
    ----------
    path: str | pathlib.Path
        Card-content file, normally ``indexes/source_cards.json``.

    Returns
    -------
    SourceCardIndex
        Checksummed records, or an empty index if the file is absent.

    Raises
    ------
    ValueError
        If the saved format, checksum, or records are invalid.
    OSError
        If an existing file cannot be read.

    Notes
    -----
    The caller must validate source hashes, active kind versions, and
    provenance against current extraction before reusing the returned cards.
    Checksums detect accidental corruption, not malicious modification.

    """

    payload = _read(path, _CARD_FORMAT)
    if payload is None:
        return SourceCardIndex(())
    _fields(payload, {"format", "records", "payload_sha256"})
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("Stored source-card records must be an array.")
    try:
        return SourceCardIndex(tuple(
            _record_from_data(item) for item in records
        ))
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Stored source-card content is invalid.") from error


def save_source_card_vectors(
    vectors: SourceCardVectorIndex,
    cards: SourceCardIndex,
    path: str | Path,
) -> Path:
    """Atomically persist vectors without rewriting source-card content.

    Parameters
    ----------
    vectors: SourceCardVectorIndex
        Vectors for one resolved embedding model.
    cards: SourceCardIndex
        Current validated cards; every vector group must match its card.
    path: str | pathlib.Path
        Destination, normally ``indexes/source_card_vectors.json``.

    Returns
    -------
    pathlib.Path
        Resolved destination path.

    Raises
    ------
    ValueError
        If vectors are inconsistent, stale, or non-finite.
    OSError
        If the destination cannot be written atomically.

    """

    if not isinstance(vectors, SourceCardVectorIndex):
        raise ValueError("Expected a SourceCardVectorIndex.")
    prepare_source_card_vectors(
        vectors.records, cards,
        embedding_model_identity=vectors.embedding_model_identity,
    )
    payload = {
        "format": _VECTOR_FORMAT,
        "embedding_model_identity": vectors.embedding_model_identity,
        "records": [
            {
                "path": entry.path.as_posix(),
                "kind": entry.kind,
                "card_sha256": entry.card_sha256,
                "text_sha256": entry.text_sha256,
                "embeddings": [
                    [float(value) for value in vector]
                    for vector in entry.embeddings
                ],
            }
            for entry in sorted(vectors.records, key=lambda item: (
                item.path.as_posix(), item.kind
            ))
        ],
    }
    return _write(path, payload)


def load_source_card_vectors(
    path: str | Path,
    cards: SourceCardIndex,
) -> SourceCardVectorIndex | None:
    """Load vectors, discarding groups whose cards have been replaced.

    Parameters
    ----------
    path: str | pathlib.Path
        Persistent vector file.
    cards: SourceCardIndex
        Current card-content index to check per-card fingerprints.

    Returns
    -------
    SourceCardVectorIndex | None
        Saved model identity with still-matching vector groups, or ``None``
        when the file does not exist. Stale groups are dropped individually.

    Raises
    ------
    ValueError
        If the checksum, stored format, or numeric vectors are corrupt.
    OSError
        If an existing vector file is unreadable.

    Notes
    -----
    The file's own checksum is validated before stale groups are filtered.
    A crash between two atomic writes leaves either the old or new vector
    file, and only card-matching entries are returned for incremental reuse.

    """

    if not isinstance(cards, SourceCardIndex):
        raise ValueError("Expected a SourceCardIndex.")
    payload = _read(path, _VECTOR_FORMAT)
    if payload is None:
        return None
    _fields(payload, {
        "format", "embedding_model_identity", "records", "payload_sha256",
    })
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("Stored source-card vectors must be an array.")
    try:
        index = SourceCardVectorIndex(
            payload["embedding_model_identity"],
            tuple(_vector_from_data(item) for item in records),
        )
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Stored source-card vectors are invalid.") from error
    owners = {record.key: record for record in cards.records}
    kept = []
    for entry in index.records:
        owner = owners.get(entry.key)
        if (
            owner is not None
            and owner.result.card is not None
            and entry.card_sha256 == card_digest(owner.result.card)
        ):
            kept.append(entry)
    return SourceCardVectorIndex(
        index.embedding_model_identity, tuple(kept)
    )


def _record_to_data(entry: IndexedSourceCard) -> dict[str, object]:
    """Serialize one indexed result without storing redundant vector data.

    Parameters
    ----------
    entry: IndexedSourceCard
        Source-card attempt and cache metadata.

    Returns
    -------
    dict[str, object]
        JSON-compatible record.

    """

    result = entry.result
    return {
        "path": result.path.as_posix(),
        "kind": result.kind,
        "status": result.status,
        "reason": result.reason,
        "source_sha256": entry.source_sha256,
        "schema_version": entry.schema_version,
        "validation_version": entry.validation_version,
        "generation_model_id": entry.generation_model_id,
        "generation_settings": entry.generation_settings,
        "card": _card_to_data(result.card) if result.card else None,
    }


def _card_to_data(card: SourceCard) -> dict[str, object]:
    """Serialize one card, preserving statement-level provenance.

    Parameters
    ----------
    card: SourceCard
        Validated card.

    Returns
    -------
    dict[str, object]
        Complete source-card content with no embedding vectors.

    """

    return {
        "path": card.path.as_posix(),
        "kind": card.kind,
        "schema_version": card.schema_version,
        "validation_version": card.validation_version,
        "source_sha256": card.source_sha256,
        "metadata": [list(pair) for pair in card.metadata],
        "statements": [
            {"field": stmt.field, "text": stmt.text,
             "chunk_ids": list(stmt.chunk_ids)}
            for stmt in card.statements
        ],
        "model_id": card.model_id,
        "generation_settings": card.generation_settings,
    }


def _record_from_data(value: object) -> IndexedSourceCard:
    """Reconstruct an indexed card attempt from checked JSON fields.

    Parameters
    ----------
    value: object
        Stored record object.

    Returns
    -------
    IndexedSourceCard
        Parsed attempt ready for structural integrity checks.

    """

    _fields(value, {
        "path", "kind", "status", "reason", "source_sha256",
        "schema_version", "validation_version", "generation_model_id",
        "generation_settings", "card",
    })
    card = _card_from_data(value["card"]) if (
        value["card"] is not None
    ) else None
    return IndexedSourceCard(
        result=SourceCardResult(
            Path(value["path"]), value["kind"], value["status"],
            card=card, reason=value["reason"],
        ),
        source_sha256=value["source_sha256"],
        schema_version=value["schema_version"],
        validation_version=value["validation_version"],
        generation_model_id=value["generation_model_id"],
        generation_settings=value["generation_settings"],
    )


def _card_from_data(value: object) -> SourceCard:
    """Reconstruct a generic card without inferring its semantic kind.

    Parameters
    ----------
    value: object
        Serialized source card.

    Returns
    -------
    SourceCard
        Card content, including original chunk citations.

    """

    _fields(value, {
        "path", "kind", "schema_version", "validation_version",
        "source_sha256", "metadata", "statements", "model_id",
        "generation_settings",
    })
    metadata = value["metadata"]
    statements = value["statements"]
    if not isinstance(metadata, list) or not isinstance(statements, list):
        raise ValueError("Stored source-card fields must be arrays.")
    pairs = []
    for item in metadata:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Stored card metadata must contain pairs.")
        pairs.append(tuple(item))
    parsed = []
    for item in statements:
        _fields(item, {"field", "text", "chunk_ids"})
        if not isinstance(item["chunk_ids"], list):
            raise ValueError("Stored chunk citations must be an array.")
        parsed.append(CardStatement(
            item["field"], item["text"], tuple(item["chunk_ids"])
        ))
    return SourceCard(
        path=Path(value["path"]),
        kind=value["kind"],
        schema_version=value["schema_version"],
        validation_version=value["validation_version"],
        source_sha256=value["source_sha256"],
        metadata=tuple(pairs),
        statements=tuple(parsed),
        model_id=value["model_id"],
        generation_settings=value["generation_settings"],
    )


def _vector_from_data(value: object) -> SourceCardVector:
    """Reconstruct one vector group from strict JSON fields.

    Parameters
    ----------
    value: object
        Saved vector group.

    Returns
    -------
    SourceCardVector
        Immutable vector group for structural and numerical validation.

    """

    _fields(value, {
        "path", "kind", "card_sha256", "text_sha256", "embeddings",
    })
    vectors = value["embeddings"]
    if not isinstance(vectors, list):
        raise ValueError("Stored embeddings must be an array.")
    if any(not isinstance(row, list) for row in vectors):
        raise ValueError("Stored embedding rows must be arrays.")
    return SourceCardVector(
        path=Path(value["path"]),
        kind=value["kind"],
        card_sha256=value["card_sha256"],
        text_sha256=value["text_sha256"],
        embeddings=tuple(tuple(row) for row in vectors),
    )


def _write(path: str | Path, payload: dict[str, object]) -> Path:
    """Seal and atomically replace one independently versioned index file.

    Parameters
    ----------
    path: str | pathlib.Path
        File destination.
    payload: dict[str, object]
        Strict JSON payload without a checksum field.

    Returns
    -------
    pathlib.Path
        Resolved destination path.

    """

    target = Path(path).expanduser().resolve()
    sealed = {**payload, "payload_sha256": _checksum(payload)}
    text = json.dumps(
        sealed, sort_keys=True, ensure_ascii=False, allow_nan=False,
        indent=2,
    ) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, text)
    return target


def _read(path: str | Path, expected_format: str) -> dict | None:
    """Read an independently sealed index, or None if not created yet.

    Parameters
    ----------
    path: str | pathlib.Path
        Saved index file.
    expected_format: str
        Accepted current format identifier.

    Returns
    -------
    dict | None
        Verified payload or ``None`` when the file is missing.

    Raises
    ------
    ValueError
        For an invalid or outdated file, checksum, or duplicate JSON key.

    """

    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(
            "Saved source-card index is not valid JSON."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Saved source-card index must be an object.")
    if payload.get("format") != expected_format:
        raise ValueError("Unknown source-card index format version.")
    saved = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items()
            if key != "payload_sha256"}
    if not isinstance(saved, str) or saved != _checksum(body):
        raise ValueError("Saved source-card index checksum is invalid.")
    return payload


def _unique_object(items: list[tuple[str, object]]) -> dict:
    """Refuse ambiguous duplicate JSON object keys during loading.

    Parameters
    ----------
    items: list[tuple[str, object]]
        JSON key/value pairs in encounter order.

    Returns
    -------
    dict
        Object with unique keys.

    Raises
    ------
    ValueError
        If an object contains the same key more than once.

    """

    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate key in source-card index.")
        result[key] = value
    return result


def _fields(value: object, required: set[str]) -> None:
    """Require exact keys for a versioned persistent record.

    Parameters
    ----------
    value: object
        Candidate JSON object.
    required: set[str]
        Expected set of field names.

    Raises
    ------
    ValueError
        If the object is missing fields or contains unknown fields.

    """

    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Saved source-card fields are invalid.")


def _checksum(value: dict[str, object]) -> str:
    """Hash a canonical JSON payload without its checksum field.

    Parameters
    ----------
    value: dict[str, object]
        File payload without its integrity field.

    Returns
    -------
    str
        SHA-256 of the canonical JSON encoding.

    """

    text = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    )
    return sha256(text.encode("utf-8")).hexdigest()
