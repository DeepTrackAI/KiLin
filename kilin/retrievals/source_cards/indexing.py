"""Reconcile source cards after normal project retrieval indexing.

The coordinator selects one registered card kind per source, validates saved
outcomes and reuse plans, generates only missing cards with an isolated model,
and embeds only missing vector groups with the existing embedding session.
Source-card failures do not affect ordinary passage retrieval. No old paper
cache is imported or migrated.

Key Features
------------

- **Incremental generation**

    Keep complete cards and cached exclusions until their source or kind
    contract changes; retry failures only when their model/settings change.

- **Independent vectors**

    Bind each vector group to a card and its projected search text, allowing
    partial recovery after interrupted writes or an embedding-model change.

- **Bounded lifecycle**

    Load the dedicated model on the first missing eligible card and release
    it before loading the interactive answering model.

Module Structure
----------------

- ``reconcile_source_cards``

    Return current card records and searchable vectors for one project.

Examples
--------
>>> from kilin.retrievals.source_cards import SourceCardIndex
>>> SourceCardIndex(()).records
()

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from kilin.chunking import Chunk
from kilin.model_settings import read_source_card_generation_settings
from kilin.project import QAProject
from kilin.retrievals.model_identity import embedding_model_identity
from kilin.retrievals.source_cards.descriptors import describe_sources
from kilin.retrievals.source_cards.generation import SourceCardGenerator
from kilin.retrievals.source_cards.index import (
    IndexedSourceCard,
    SourceCardIndex,
    SourceCardVector,
    SourceCardVectorIndex,
    card_digest,
    plan_source_card_updates,
    plan_source_card_vectors,
    prepare_source_card_index,
    prepare_source_card_vectors,
    search_text_digest,
)
from kilin.retrievals.source_cards.kinds.scientific_paper import (
    card_texts,
    generate_card,
)
from kilin.retrievals.source_cards.models import (
    CARD_STATUS_FAILED,
    CARD_STATUS_VALID,
    SourceCardResult,
)
from kilin.retrievals.source_cards.persistence import (
    load_source_card_vectors,
    load_source_cards,
    save_source_card_vectors,
    save_source_cards,
)
from kilin.retrievals.source_cards.registry import SOURCE_CARD_KINDS
from kilin.source_card_policy import (
    read_source_card_overrides,
    read_source_card_policy,
    select_source_card_kinds,
)
from kilin.project_type import project_type

__all__ = ["reconcile_source_cards"]

_CARDS_FILE: Final[str] = "source_cards.json"
_VECTORS_FILE: Final[str] = "source_card_vectors.json"


def reconcile_source_cards(
    project: QAProject,
    chunks: Sequence[Chunk],
    manifest: Mapping[str, object],
    embedding_session: object,
    *,
    backend: str,
    titles: Mapping[Path, str] | None = None,
    expected_dimension: int | None = None,
    trace: object | None = None,
    progress: Callable[[int, int], None] | None = None,
    generator_factory: Callable[..., SourceCardGenerator] = (
        SourceCardGenerator
    ),
) -> tuple[SourceCardIndex, SourceCardVectorIndex | None]:
    """Reconcile saved cards and vectors with current selected sources.

    Parameters
    ----------
    project: QAProject
        Opened project with saved profile and indexing directory.
    chunks: Sequence[Chunk]
        Current source-relative chunks prepared by ordinary retrieval.
    manifest: Mapping[str, object]
        Current manifest containing source SHA-256 digests.
    embedding_session: object
        Already loaded embedding session used for normal chunk indexes.
    backend: str
        Selected model backend.
    titles: Mapping[pathlib.Path, str] | None, optional
        Titles inferred by existing document-header extraction.
    expected_dimension: int | None, optional
        Dimension of the ordinary semantic vectors. Defaults to ``None``.
    trace: object | None, optional
        Optional RAG trace sink for index failures. Defaults to ``None``.
    progress: Callable[[int, int], None] | None, optional
        Count-based callback for pending generations. Defaults to ``None``.
    generator_factory: Callable[..., SourceCardGenerator], optional
        Isolated generation-session factory for deterministic tests.

    Returns
    -------
    tuple[SourceCardIndex, SourceCardVectorIndex | None]
        Current card outcomes and searchable vectors, or empty/None when
        the project has disabled the source-card channel.

    Notes
    -----
    The source-card content index is saved before its vectors. A crash
    between writes leaves valid card text and stale vectors that can be
    re-embedded on the next indexing run. Cache corruption is treated as
    derived-data loss, not as an error in the original source files.

    """

    policy = read_source_card_policy(
        project.root, project_type(project.profile.type_id)
    )
    if not policy.enabled:
        return SourceCardIndex(()), None
    if not isinstance(manifest, Mapping):
        raise ValueError("Source-card manifest must be a mapping.")
    files = manifest.get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError("Source-card manifest files must be a mapping.")
    hashes = {}
    for name, item in files.items():
        if not isinstance(name, str) or not isinstance(item, Mapping):
            raise ValueError("Source-card manifest entry is invalid.")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Source-card manifest has no source digest.")
        hashes[Path(name)] = digest
    sources = describe_sources(tuple(sorted(hashes)), chunks, titles=titles)
    kinds = select_source_card_kinds(
        sources, policy, read_source_card_overrides(project.root)
    )
    if not kinds:
        return SourceCardIndex(()), None
    settings = read_source_card_generation_settings(project.root, backend)
    cards_file = project.indexes / _CARDS_FILE
    vectors_file = project.indexes / _VECTORS_FILE
    try:
        previous = load_source_cards(cards_file)
    except (OSError, ValueError) as error:
        _log_error(trace, "source_cards.load.failed", error)
        previous = SourceCardIndex(())
    plan = plan_source_card_updates(
        previous, sources, hashes, kinds, chunks, SOURCE_CARD_KINDS,
        generation_model_id=settings.model_id,
        generation_settings=settings.fingerprint(),
    )
    records = list(plan.reused)
    by_path: dict[Path, list[Chunk]] = {}
    for chunk in chunks:
        by_path.setdefault(chunk.path, []).append(chunk)
    pending = plan.pending if policy.automatic else ()
    if pending:
        with generator_factory(
            settings, backend=backend,
            allow_download=project.profile.allow_download, trace=trace,
        ) as generator:
            for number, (path, kind_id) in enumerate(pending, 1):
                kind = SOURCE_CARD_KINDS.get(kind_id)
                if kind_id != "scientific_paper":
                    raise ValueError("Card generation kind is unsupported.")
                try:
                    result = generate_card(
                        sources[path], by_path.get(path, ()),
                        source_sha256=hashes[path], generator=generator,
                    )
                    if result.status == CARD_STATUS_VALID:
                        SOURCE_CARD_KINDS.validate(
                            result.card, sources[path], by_path.get(path, ())
                        )
                except Exception as error:
                    _log_error(trace, "source_cards.generate.failed", error)
                    result = SourceCardResult(
                        path, kind_id, CARD_STATUS_FAILED,
                        reason=f"Generation failed: {type(error).__name__}.",
                    )
                records.append(IndexedSourceCard(
                    result, hashes[path], kind.schema_version,
                    kind.validation_version, settings.model_id,
                    settings.fingerprint(),
                ))
                if progress is not None:
                    progress(number, len(pending))
    cards = prepare_source_card_index(
        sorted(records, key=lambda record: (
            record.result.path.as_posix(), record.result.kind
        )),
        sources, chunks, hashes, SOURCE_CARD_KINDS,
    )
    if cards != previous:
        save_source_cards(cards, cards_file)
    identity = getattr(
        embedding_session, "_retrieval_model_identity", None
    )
    if not identity:
        model_id = getattr(embedding_session, "model_id", None)
        model_source = getattr(embedding_session, "model_source", None)
        if not model_id or not model_source:
            raise ValueError("Embedding model identity is unavailable.")
        identity = embedding_model_identity(model_id, model_source)
    try:
        old_vectors = load_source_card_vectors(vectors_file, cards)
    except (OSError, ValueError) as error:
        _log_error(trace, "source_card_vectors.load.failed", error)
        old_vectors = None
    if old_vectors is not None and expected_dimension is not None:
        old_vectors = SourceCardVectorIndex(
            old_vectors.embedding_model_identity,
            tuple(record for record in old_vectors.records if (
                len(record.embeddings[0]) == expected_dimension
            )),
        )
    texts = {
        entry.key: card_texts(entry.result.card)
        for entry in cards.records
        if entry.result.status == CARD_STATUS_VALID
    }
    vector_plan = plan_source_card_vectors(
        cards, old_vectors, texts, embedding_model_identity=identity
    )
    updated = list(vector_plan.reused)
    pending_vectors = vector_plan.pending
    if pending_vectors:
        flat = [text for key in pending_vectors for text in texts[key]]
        try:
            generated = embedding_session.embed_documents(flat)
            from kilin.retrievals.strategies.retrieval_semantic import (
                _validate_semantic_embeddings,
            )
            _validate_semantic_embeddings(generated, len(flat))
            if expected_dimension is not None and (
                len(generated[0]) != expected_dimension
            ):
                raise ValueError("Card and passage vector sizes differ.")
            offset = 0
            for key in pending_vectors:
                group = tuple(tuple(vector) for vector in generated[
                    offset:offset + len(texts[key])
                ])
                offset += len(texts[key])
                owner = cards.get(*key)
                updated.append(SourceCardVector(
                    key[0], key[1], card_digest(owner.result.card),
                    search_text_digest(texts[key]), group,
                ))
        except (ValueError, TypeError, RuntimeError) as error:
            _log_error(trace, "source_card_vectors.embed.failed", error)
    vectors = prepare_source_card_vectors(
        updated, cards, embedding_model_identity=identity
    )
    if old_vectors != vectors:
        save_source_card_vectors(vectors, cards, vectors_file)
    return cards, vectors


def _log_error(trace: object | None, event: str, error: Exception) -> None:
    """Record recoverable source-card errors without affecting retrieval.

    Parameters
    ----------
    trace: object | None
        Optional trace sink implementing ``exception``.
    event: str
        Stable diagnostic event name.
    error: Exception
        Original cache or indexing error.

    """

    if trace is not None:
        trace.exception(event, error, component="source_cards")
