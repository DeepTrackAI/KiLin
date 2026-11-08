"""Build and search an independent, optional source-card index.

Source cards summarize entire sources and are not evidence passages. The
index retains validated outcomes separately from vectors, which are tied to
individual cards and their projected search texts. An interrupted update can
therefore reuse vectors for unchanged cards without trusting stale vectors.

No language model, embedding model, GUI, or active RAG session is loaded here.
Callers supply validated cards and precomputed embeddings. Generation and
integration with project indexing follow in later PRs.

Key Features
------------

- **Independent records**

    Store successful, ineligible, and failed results by source and card kind.

- **Incremental planning**

    Reuse unchanged cards and vectors; identify the records to rebuild.

- **Source-level search**

    Score prepared card vectors without another model-generation call.

Module Structure
----------------

- ``IndexedSourceCard`` and ``SourceCardIndex``

    Versioned, source-bound card-generation outcomes.

- ``SourceCardVector`` and ``SourceCardVectorIndex``

    Embeddings bound to one card and one search-text projection.

- ``prepare_source_card_index`` and ``plan_source_card_updates``

    Admit validated results and plan incremental content updates.

- ``prepare_source_card_vectors`` and ``plan_source_card_vectors``

    Validate vectors and plan independent embedding reuse.

- ``search_source_cards``

    Rank source/card-kind pairs using existing query embeddings.

Examples
--------
>>> from kilin.retrievals.source_cards import SourceCardIndex
>>> SourceCardIndex(()).records
()

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from numbers import Real
from pathlib import Path
from typing import TypeAlias

from kilin.chunking import Chunk
from kilin.retrievals.source_cards.models import (
    CARD_STATUS_FAILED,
    CARD_STATUS_VALID,
    SourceCard,
    SourceCardResult,
    SourceDescriptor,
    _SHA256,
    _check_descriptor,
    _check_identifier,
    _check_path,
)
from kilin.retrievals.source_cards.registry import SourceCardRegistry

__all__ = [
    "CardKey",
    "IndexedSourceCard",
    "SourceCardIndex",
    "SourceCardMatch",
    "SourceCardUpdatePlan",
    "SourceCardVector",
    "SourceCardVectorIndex",
    "SourceCardVectorPlan",
    "card_digest",
    "plan_source_card_updates",
    "plan_source_card_vectors",
    "prepare_source_card_index",
    "prepare_source_card_vectors",
    "search_source_cards",
    "search_text_digest",
]

CardKey: TypeAlias = tuple[Path, str]


@dataclass(frozen=True, slots=True)
class IndexedSourceCard:
    """One card attempt and the inputs needed to validate its cache.

    Parameters
    ----------
    result: SourceCardResult
        Valid, ineligible, or failed source-card attempt.
    source_sha256: str
        Original source-content SHA-256 for all three result statuses.
    schema_version: int
        Version of the kind schema used for this attempt.
    validation_version: int
        Version of the kind validator used for this attempt.
    generation_model_id: str | None, optional
        Model responsible for the attempt, if any. Defaults to ``None``.
    generation_settings: str | None, optional
        Generation-parameter fingerprint, if any. Defaults to ``None``.

    Notes
    -----
    For valid results, these fields must agree with the stored card. For
    excluded and failed results they preserve the inputs needed to avoid
    unnecessary retries when the same source is indexed again.

    """

    result: SourceCardResult
    source_sha256: str
    schema_version: int
    validation_version: int
    generation_model_id: str | None = None
    generation_settings: str | None = None

    @property
    def key(self) -> CardKey:
        """Return the source-path and card-kind identity.

        Returns
        -------
        CardKey
            Immutable identity for this attempt.

        """

        return (self.result.path, self.result.kind)


@dataclass(frozen=True, slots=True)
class SourceCardIndex:
    """Source-card outcomes without embedding vectors.

    Parameters
    ----------
    records: tuple[IndexedSourceCard, ...]
        Unique source/card-kind records, including exclusions and failures.

    """

    records: tuple[IndexedSourceCard, ...]

    def __post_init__(self) -> None:
        """Reject structurally inconsistent or duplicate cached records.

        Raises
        ------
        ValueError
            If entries cannot be admitted as source-card cache records.

        """

        if type(self.records) is not tuple:
            raise ValueError("Source-card records must be an immutable tuple.")
        seen: set[CardKey] = set()
        for record in self.records:
            _check_record(record)
            if record.key in seen:
                raise ValueError("Duplicate source/card-kind cache entry.")
            seen.add(record.key)

    def get(self, path: Path, kind: str) -> IndexedSourceCard | None:
        """Look up one source/card-kind record.

        Parameters
        ----------
        path: pathlib.Path
            Source-relative path.
        kind: str
            Registered card-kind identifier.

        Returns
        -------
        IndexedSourceCard | None
            Matching record, or ``None`` when not cached.

        """

        return next(
            (item for item in self.records if item.key == (path, kind)),
            None,
        )


@dataclass(frozen=True, slots=True)
class SourceCardUpdatePlan:
    """Reusable outcomes and source/card-kind pairs requiring new attempts.

    Parameters
    ----------
    reused: tuple[IndexedSourceCard, ...]
        Current records which need no new card-generation attempt.
    pending: tuple[CardKey, ...]
        Selected source/kind pairs that must be generated or classified.

    """

    reused: tuple[IndexedSourceCard, ...]
    pending: tuple[CardKey, ...]


@dataclass(frozen=True, slots=True)
class SourceCardVector:
    """Embeddings of one card, keyed by its content and projected text.

    Parameters
    ----------
    path: pathlib.Path
        Source-relative path.
    kind: str
        Card-kind identifier.
    card_sha256: str
        Fingerprint of the full card content, including provenance.
    text_sha256: str
        Fingerprint of the exact ordered texts used for embedding.
    embeddings: tuple[tuple[float, ...], ...]
        Nonempty vectors of a consistent finite, nonzero dimension.

    """

    path: Path
    kind: str
    card_sha256: str
    text_sha256: str
    embeddings: tuple[tuple[float, ...], ...]

    @property
    def key(self) -> CardKey:
        """Return the source/card-kind identity of this vector group.

        Returns
        -------
        CardKey
            Source path and card kind.

        """

        return (self.path, self.kind)


@dataclass(frozen=True, slots=True)
class SourceCardVectorIndex:
    """Search vectors sharing one embedding-model identity.

    Parameters
    ----------
    embedding_model_identity: str
        Resolved identity of the model used for every stored vector.
    records: tuple[SourceCardVector, ...]
        Valid card vectors; ineligible and failed sources have no vectors.

    """

    embedding_model_identity: str
    records: tuple[SourceCardVector, ...]

    def __post_init__(self) -> None:
        """Validate model identity, cardinality, and numerical integrity.

        Raises
        ------
        ValueError
            If a vector group is invalid or duplicates another key.

        """

        if (
            not isinstance(self.embedding_model_identity, str)
            or not self.embedding_model_identity.strip()
        ):
            raise ValueError("Embedding model identity must be nonempty.")
        if type(self.records) is not tuple:
            raise ValueError("Vector records must be an immutable tuple.")
        seen: set[CardKey] = set()
        dimension: int | None = None
        for record in self.records:
            current = _check_vector(record)
            if record.key in seen:
                raise ValueError("Duplicate source-card vector identity.")
            if dimension is not None and dimension != current:
                raise ValueError(
                "All source-card vector dimensions must match."
            )
            dimension = current
            seen.add(record.key)


@dataclass(frozen=True, slots=True)
class SourceCardVectorPlan:
    """Reusable vector groups and cards requiring new embeddings.

    Parameters
    ----------
    reused: tuple[SourceCardVector, ...]
        Valid groups for the current model, card and text projection.
    pending: tuple[CardKey, ...]
        Valid cards whose search texts require embedding.

    """

    reused: tuple[SourceCardVector, ...]
    pending: tuple[CardKey, ...]


@dataclass(frozen=True, slots=True)
class SourceCardMatch:
    """One source/card-kind pair matched by its stored vector groups.

    Parameters
    ----------
    score: float
        Highest cosine similarity to any vector for this card.
    path: pathlib.Path
        Matching source path.
    kind: str
        Matching card kind.

    """

    score: float
    path: Path
    kind: str


def prepare_source_card_index(
    records: Sequence[IndexedSourceCard],
    sources: Mapping[Path, SourceDescriptor],
    chunks: Sequence[Chunk],
    source_hashes: Mapping[Path, str],
    registry: SourceCardRegistry,
) -> SourceCardIndex:
    """Admit current records only after validating provenance and kind.

    Parameters
    ----------
    records: Sequence[IndexedSourceCard]
        Card attempts to admit to the index.
    sources: Mapping[pathlib.Path, SourceDescriptor]
        Current source descriptions.
    chunks: Sequence[Chunk]
        Available original source chunks for provenance checks.
    source_hashes: Mapping[pathlib.Path, str]
        Source-byte digests for the current source snapshot.
    registry: SourceCardRegistry
        Trusted kind validators and their current versions.

    Returns
    -------
    SourceCardIndex
        Validated records, including legitimate ineligible/failed outcomes.

    Raises
    ------
    ValueError
        If a result has stale metadata or unsupported provenance.

    """

    index = SourceCardIndex(tuple(records))
    by_path: dict[Path, list[Chunk]] = {}
    for chunk in chunks:
        by_path.setdefault(chunk.path, []).append(chunk)
    for record in index.records:
        source = sources.get(record.result.path)
        if source is None:
            raise ValueError("Card record has no selected source.")
        _check_descriptor(source)
        if record.source_sha256 != source_hashes.get(source.path):
            raise ValueError("Source-card record is stale for its source.")
        kind = registry.get(record.result.kind)
        if (
            record.schema_version != kind.schema_version
            or record.validation_version != kind.validation_version
        ):
            raise ValueError("Source-card kind version is stale.")
        if record.result.status == CARD_STATUS_VALID:
            registry.validate(
                record.result.card, source, by_path.get(source.path, ())
            )
    return index


def plan_source_card_updates(
    previous: SourceCardIndex,
    sources: Mapping[Path, SourceDescriptor],
    source_hashes: Mapping[Path, str],
    selected_kinds: Mapping[Path, str],
    chunks: Sequence[Chunk],
    registry: SourceCardRegistry,
    *,
    generation_model_id: str | None = None,
    generation_settings: str | None = None,
    retry_failed: bool = False,
) -> SourceCardUpdatePlan:
    """Identify which source-card attempts can be reused incrementally.

    Parameters
    ----------
    previous: SourceCardIndex
        Previously cached outcomes.
    sources: Mapping[pathlib.Path, SourceDescriptor]
        Current source descriptions.
    source_hashes: Mapping[pathlib.Path, str]
        Current source content hashes.
    selected_kinds: Mapping[pathlib.Path, str]
        One candidate kind per selected source, chosen by project policy.
    chunks: Sequence[Chunk]
        Current original chunks, used to check cached valid provenance.
    registry: SourceCardRegistry
        Trusted current kind implementations.
    generation_model_id: str | None, optional
        Configured generation model; only failed attempts require a model
        match before reuse. Defaults to ``None``.
    generation_settings: str | None, optional
        Generation settings fingerprint, or ``None``. Defaults to ``None``.
    retry_failed: bool, optional
        Force retry of failed attempts. Defaults to ``False``.

    Returns
    -------
    SourceCardUpdatePlan
        Reusable attempts and pending source/card-kind keys.

    Notes
    -----
    Changes to generation models do not invalidate valid cards. Failed
    attempts are retried when their model/settings change, while verified
    exclusions are reused until the source or kind validation changes.

    """

    if type(retry_failed) is not bool:
        raise ValueError("retry_failed must be a boolean.")
    _check_optional_text(generation_model_id)
    _check_optional_text(generation_settings)
    by_path: dict[Path, list[Chunk]] = {}
    for chunk in chunks:
        by_path.setdefault(chunk.path, []).append(chunk)
    existing = {record.key: record for record in previous.records}
    reused: list[IndexedSourceCard] = []
    pending: list[CardKey] = []
    for path, kind_id in selected_kinds.items():
        source = sources.get(path)
        if source is None or source.path != path:
            raise ValueError("Selected card kind has no matching source.")
        _check_descriptor(source)
        _check_hash(source_hashes.get(path), "Source SHA-256")
        kind = registry.get(kind_id)
        record = existing.get((path, kind_id))
        key = (path, kind_id)
        if record is None or (
            record.source_sha256 != source_hashes[path]
            or record.schema_version != kind.schema_version
            or record.validation_version != kind.validation_version
        ):
            pending.append(key)
            continue
        if record.result.status == CARD_STATUS_VALID:
            try:
                registry.validate(
                    record.result.card, source, by_path.get(path, ())
                )
            except ValueError:
                pending.append(key)
                continue
        elif record.result.status == CARD_STATUS_FAILED and (
            retry_failed
            or record.generation_model_id != generation_model_id
            or record.generation_settings != generation_settings
        ):
            pending.append(key)
            continue
        reused.append(record)
    return SourceCardUpdatePlan(tuple(reused), tuple(pending))


def card_digest(card: SourceCard) -> str:
    """Return a stable fingerprint of one card's content and provenance.

    Parameters
    ----------
    card: SourceCard
        Card to fingerprint independently of other sources.

    Returns
    -------
    str
        Hex SHA-256 digest of the canonical card record.

    """

    if not isinstance(card, SourceCard):
        raise ValueError("Expected a SourceCard.")
    payload = {
        "path": card.path.as_posix(),
        "kind": card.kind,
        "schema_version": card.schema_version,
        "validation_version": card.validation_version,
        "source_sha256": card.source_sha256,
        "metadata": list(map(list, card.metadata)),
        "statements": [
            {
                "field": stmt.field,
                "text": stmt.text,
                "chunk_ids": list(stmt.chunk_ids),
            }
            for stmt in card.statements
        ],
        "model_id": card.model_id,
        "generation_settings": card.generation_settings,
    }
    return _digest(payload)


def search_text_digest(texts: Sequence[str]) -> str:
    """Fingerprint the exact ordered texts embedded for one source card.

    Parameters
    ----------
    texts: Sequence[str]
        Nonempty, ordered card search texts.

    Returns
    -------
    str
        SHA-256 hash of the canonical text list.

    Raises
    ------
    ValueError
        If no usable search texts are supplied.

    """

    if (
        isinstance(texts, (str, bytes))
        or not isinstance(texts, Sequence)
        or not texts
        or any(not isinstance(item, str) or not item.strip()
               for item in texts)
    ):
        raise ValueError("Search texts must be nonempty text strings.")
    return _digest(list(texts))


def prepare_source_card_vectors(
    records: Sequence[SourceCardVector],
    cards: SourceCardIndex,
    *,
    embedding_model_identity: str,
) -> SourceCardVectorIndex:
    """Admit independent embedding groups matching validated cards.

    Parameters
    ----------
    records: Sequence[SourceCardVector]
        Prepared vector groups keyed by source and kind.
    cards: SourceCardIndex
        Card records used when the embeddings were constructed.
    embedding_model_identity: str
        Identity of the model used for all vectors.

    Returns
    -------
    SourceCardVectorIndex
        Validated searchable vectors; missing card vectors are permitted.

    Raises
    ------
    ValueError
        If a vector is invalid or its card fingerprint is stale.

    """

    index = SourceCardVectorIndex(
        embedding_model_identity, tuple(records)
    )
    owners = {entry.key: entry for entry in cards.records}
    for record in index.records:
        owner = owners.get(record.key)
        if owner is None or owner.result.status != CARD_STATUS_VALID:
            raise ValueError("Vectors require a valid source card.")
        if record.card_sha256 != card_digest(owner.result.card):
            raise ValueError("Vector content does not match the source card.")
    return index


def plan_source_card_vectors(
    cards: SourceCardIndex,
    previous: SourceCardVectorIndex | None,
    search_texts: Mapping[CardKey, Sequence[str]],
    *,
    embedding_model_identity: str,
) -> SourceCardVectorPlan:
    """Reuse vectors only for unchanged cards, texts, and embedding model.

    Parameters
    ----------
    cards: SourceCardIndex
        Current source-card outcomes.
    previous: SourceCardVectorIndex | None
        Previously loaded vectors, when available.
    search_texts: Mapping[CardKey, Sequence[str]]
        Kind-generated text projections of currently valid cards.
    embedding_model_identity: str
        Resolved current embedding-model identity.

    Returns
    -------
    SourceCardVectorPlan
        Reusable vectors and valid cards needing new embeddings.

    Notes
    -----
    A card added, changed, or removed does not force recomputation of the
    unrelated cards' vectors. Invalid and missing card states have no vectors.

    """

    if not isinstance(embedding_model_identity, str) or (
        not embedding_model_identity.strip()
    ):
        raise ValueError("Embedding model identity must be nonempty.")
    good = {
        record.key: record for record in previous.records
    } if (
        previous is not None
        and previous.embedding_model_identity == embedding_model_identity
    ) else {}
    reused: list[SourceCardVector] = []
    pending: list[CardKey] = []
    valid_keys = {
        item.key for item in cards.records
        if item.result.status == CARD_STATUS_VALID
    }
    if set(search_texts) != valid_keys:
        raise ValueError("Search texts must cover exactly the valid cards.")
    for item in cards.records:
        if item.result.status != CARD_STATUS_VALID:
            continue
        key = item.key
        card_sha = card_digest(item.result.card)
        text_sha = search_text_digest(search_texts[key])
        old = good.get(key)
        if old is not None and (
            old.card_sha256 == card_sha
            and old.text_sha256 == text_sha
            and len(old.embeddings) == len(search_texts[key])
        ):
            reused.append(old)
        else:
            pending.append(key)
    return SourceCardVectorPlan(tuple(reused), tuple(pending))


def search_source_cards(
    vectors: SourceCardVectorIndex,
    query_vector: Sequence[float],
    *,
    count: int = 5,
) -> tuple[SourceCardMatch, ...]:
    """Rank source-card groups by their best cosine similarity to a query.

    Parameters
    ----------
    vectors: SourceCardVectorIndex
        Prepared source-card embedding index.
    query_vector: Sequence[float]
        Precomputed query embedding from the same embedding model.
    count: int, optional
        Maximum cards to return. Defaults to ``5``.

    Returns
    -------
    tuple[SourceCardMatch, ...]
        Highest-scoring source/card-kind pairs, with stable tie ordering.

    Raises
    ------
    ValueError
        If the query is invalid or has incompatible dimensions.

    """

    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive whole number.")
    norm = _norm(query_vector)
    if vectors.records and len(query_vector) != len(
        vectors.records[0].embeddings[0]
    ):
        raise ValueError("Query and card vector dimensions do not match.")
    results = []
    for record in vectors.records:
        score = max(
            _cosine(query_vector, norm, vector)
            for vector in record.embeddings
        )
        results.append(SourceCardMatch(score, record.path, record.kind))
    results.sort(key=lambda item: (
        -item.score, item.path.as_posix(), item.kind
    ))
    return tuple(results[:count])


def _cosine(
    query: Sequence[float], query_norm: float, vector: Sequence[float]
) -> float:
    """Compute cosine similarity without overflowing raw dot products.

    Parameters
    ----------
    query: Sequence[float]
        Validated query embedding.
    query_norm: float
        Its positive Euclidean norm.
    vector: Sequence[float]
        One validated card embedding of the same dimension.

    Returns
    -------
    float
        Cosine similarity, using normalized components.

    """

    vector_norm = _norm(vector)
    return sum(
        (a / query_norm) * (b / vector_norm)
        for a, b in zip(query, vector)
    )


def _check_record(record: IndexedSourceCard) -> None:
    """Validate cache metadata and structural card contents.

    Parameters
    ----------
    record: IndexedSourceCard
        Cached result and its content/validator identity.

    Raises
    ------
    ValueError
        If the record is malformed or disagrees with its valid card.

    """

    if not isinstance(record, IndexedSourceCard):
        raise ValueError("Expected an IndexedSourceCard.")
    if not isinstance(record.result, SourceCardResult):
        raise ValueError("Cached result must be a SourceCardResult.")
    _check_path(record.result.path)
    _check_identifier(record.result.kind, "card kind")
    _check_hash(record.source_sha256, "Source SHA-256")
    for value in (record.schema_version, record.validation_version):
        if type(value) is not int or value < 1:
            raise ValueError("Cache versions must be positive integers.")
    _check_optional_text(record.generation_model_id)
    _check_optional_text(record.generation_settings)
    if record.result.status != CARD_STATUS_VALID:
        return
    card = record.result.card
    if (
        card.source_sha256 != record.source_sha256
        or card.schema_version != record.schema_version
        or card.validation_version != record.validation_version
        or card.model_id != record.generation_model_id
        or card.generation_settings != record.generation_settings
    ):
        raise ValueError("Card content and cache metadata disagree.")
    _check_card_shape(card)


def _check_card_shape(card: SourceCard) -> None:
    """Check saved card fields without requiring the original chunks.

    Parameters
    ----------
    card: SourceCard
        Card reconstructed from persistent storage.

    Raises
    ------
    ValueError
        If its immutable metadata or statement records are malformed.

    """

    if not isinstance(card, SourceCard):
        raise ValueError("Expected a SourceCard.")
    if type(card.metadata) is not tuple or (
        len({key for key, _ in card.metadata}) != len(card.metadata)
    ):
        raise ValueError("Malformed or duplicate source-card metadata.")
    for key, value in card.metadata:
        _check_identifier(key, "metadata key")
        if not isinstance(value, str):
            raise ValueError("Source-card metadata values must be text.")
    if type(card.statements) is not tuple or not card.statements:
        raise ValueError("Source cards must have substantive statements.")
    for statement in card.statements:
        _check_identifier(statement.field, "statement field")
        if not isinstance(statement.text, str) or not statement.text.strip():
            raise ValueError("Invalid source-card statement text.")
        if (
            type(statement.chunk_ids) is not tuple
            or not statement.chunk_ids
            or any(not isinstance(cid, str) or not cid.strip()
                   for cid in statement.chunk_ids)
            or len(set(statement.chunk_ids)) != len(statement.chunk_ids)
        ):
            raise ValueError("Invalid source-card statement provenance.")


def _check_vector(record: SourceCardVector) -> int:
    """Validate one vector group and return its embedding dimension.

    Parameters
    ----------
    record: SourceCardVector
        Source-card vectors to inspect.

    Returns
    -------
    int
        Shared positive vector dimension.

    Raises
    ------
    ValueError
        If vector cardinality, finiteness, or lengths are invalid.

    """

    if not isinstance(record, SourceCardVector):
        raise ValueError("Expected a SourceCardVector.")
    _check_path(record.path)
    _check_identifier(record.kind, "card kind")
    _check_hash(record.card_sha256, "Card SHA-256")
    _check_hash(record.text_sha256, "Search-text SHA-256")
    if type(record.embeddings) is not tuple or not record.embeddings:
        raise ValueError("Source-card embeddings must be nonempty.")
    dimension = None
    for vector in record.embeddings:
        if type(vector) is not tuple or not vector:
            raise ValueError("Vectors must be nonempty immutable tuples.")
        if any(
            not _finite_number(value)
            for value in vector
        ):
            raise ValueError("Source-card vectors must be finite numbers.")
        _norm(vector)
        if dimension is not None and len(vector) != dimension:
            raise ValueError("Source-card vectors have mismatched dimensions.")
        dimension = len(vector)
    return dimension


def _check_hash(value: object, label: str) -> None:
    """Validate a canonical SHA-256 hexadecimal digest.

    Parameters
    ----------
    value: object
        Candidate hexadecimal digest.
    label: str
        Name used for diagnostics.

    Raises
    ------
    ValueError
        If the digest is not a lowercase, 64-character SHA-256 string.

    """

    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be lowercase SHA-256 hex.")


def _check_optional_text(value: object) -> None:
    """Reject invalid optional generation metadata.

    Parameters
    ----------
    value: object
        Optional model identity or generation settings fingerprint.

    Raises
    ------
    ValueError
        If a supplied value is not nonempty text.

    """

    if value is not None and (
        not isinstance(value, str) or not value.strip()
    ):
        raise ValueError("Optional generation metadata must be text.")


def _finite_number(value: object) -> bool:
    """Accept finite real embedding components, excluding booleans.

    Parameters
    ----------
    value: object
        Candidate component from a model or a saved JSON vector.

    Returns
    -------
    bool
        Whether the component is a finite real number.

    """

    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _norm(values: Sequence[float]) -> float:
    """Validate a query/vector and return its Euclidean norm.

    Parameters
    ----------
    values: Sequence[float]
        Precomputed vector whose numerical validity is required.

    Returns
    -------
    float
        Positive Euclidean norm.

    Raises
    ------
    ValueError
        If values are empty, nonfinite, or have zero norm.

    """

    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(
            not _finite_number(value)
            for value in values
        )
    ):
        raise ValueError("Query vector must contain finite numbers.")
    scale = max(abs(value) for value in values)
    norm = scale * sqrt(sum((value / scale) ** 2 for value in values)) if (
        scale
    ) else 0.0
    if not isfinite(norm) or norm <= 0:
        raise ValueError("Query vector must have nonzero finite norm.")
    return norm


def _digest(value: object) -> str:
    """Hash a canonical JSON value without depending on storage location.

    Parameters
    ----------
    value: object
        JSON-compatible content.

    Returns
    -------
    str
        Canonical SHA-256 digest.

    """

    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    return sha256(text.encode("utf-8")).hexdigest()
