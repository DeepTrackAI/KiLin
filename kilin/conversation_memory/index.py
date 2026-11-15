"""Synchronize independently indexed project conversation messages.

Saved conversations and user feedback are authoritative. The memory index is
rebuildable, project-local derived state. One record identifies a turn UUID
and message role; several active branches can be origins of the same record.
Distinct turns with identical text retain separate feedback but share the
same fingerprint and embedding vector.

No language-model summarization, source extraction, or source reindexing is
performed. The caller supplies an already loaded embedding model.

Key Features
------------

- **Intent-aware selection**

    Respect automatic, explicit, and hard-never policies independently for
    questions and answers, including user Dislike/Like/Save.

- **Incremental embeddings**

    Reuse vectors for unchanged normalized texts under the same model identity.

- **Active-origin reconciliation**

    Remove archived, deleted, or ineligible occurrences on synchronization.

Module Structure
----------------

- ``MemoryRecord`` and ``MemoryIndex``

    Immutable retrieval records and the corresponding vector cache.

- ``synchronize_memory``

    Refresh derived memory without modifying conversations or feedback.

Examples
--------
>>> from kilin.conversation_memory.index import memory_fingerprint
>>> (memory_fingerprint('user', 'Example') ==
...      memory_fingerprint('user', ' Example '))
True

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Final

from kilin.answering import strip_thinking
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackSnapshot, FeedbackStore
from kilin.conversation_memory.models import require_saved_turn_ids
from kilin.conversation_memory.persistence import (
    memory_index_path,
    read_memory_index,
    write_memory_index,
)
from kilin.project_type import ProjectType

__all__ = [
    "MemoryIndex",
    "MemoryRecord",
    "MemoryOrigin",
    "memory_fingerprint",
    "synchronize_memory",
]

_MAX_CONVERSATION_BYTES: Final[int] = 128 * 1024 * 1024
_MAX_MESSAGE_CHARS: Final[int] = 12_000
_BATCH_SIZE: Final[int] = 24
_WORDS: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class MemoryOrigin:
    """One active conversation containing a committed message.

    Attributes
    ----------
    conversation_id: str
        Saved conversation identifier, or filename stem for a core session.
    turn_number: int
        One-based position in the saved conversation.

    """

    conversation_id: str
    turn_number: int


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One eligible original question or assistant response.

    Attributes
    ----------
    turn_uuid: str
        Stable identity, shared by inherited branch copies.
    role: str
        User question or assistant answer.
    text: str
        Searchable text with hidden thinking removed from answers.
    timestamp: str
        Timestamp of the completed turn.
    fingerprint: str
        Content-derived embedding identity, independent of turn UUID.
    priority: int
        0 for automatic, 1 for saved, 2 for liked selections.
    origins: tuple[MemoryOrigin, ...]
        Active project conversations containing this exact turn.

    """

    turn_uuid: str
    role: str
    text: str
    timestamp: str
    fingerprint: str
    priority: int
    origins: tuple[MemoryOrigin, ...]


@dataclass(frozen=True, slots=True)
class MemoryIndex:
    """Derived records and embeddings for one project and model identity.

    Attributes
    ----------
    model_identity: str
        Identity inherited from the verified primary-source index.
    records: tuple[MemoryRecord, ...]
        Indexed message occurrences with independent feedback.
    vectors: Mapping[str, tuple[float, ...]]
        Normalized embeddings keyed by text fingerprints.
    changed: bool
        Whether this synchronization replaced the saved derived index.
    embedded_count: int
        Number of new unique content vectors computed in this call.

    """

    model_identity: str
    records: tuple[MemoryRecord, ...]
    vectors: Mapping[str, tuple[float, ...]]
    changed: bool = False
    embedded_count: int = 0


def memory_fingerprint(role: str, text: str) -> str:
    """Compute a role-separated fingerprint for normalized message text.

    Parameters
    ----------
    role: str
        ``user`` or ``assistant``.
    text: str
        Searchable message content.

    Returns
    -------
    str
        SHA-256 hex digest shared by duplicate text of the same role.

    """

    normalized = _WORDS.sub(" ", text.strip())
    return sha256(f"{role}\0{normalized}".encode("utf-8")).hexdigest()


def _manifest(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Describe active, non-symlink conversation files without reading them.

    Parameters
    ----------
    root: pathlib.Path
        Owning project directory.

    Returns
    -------
    tuple[tuple[str, int, int], ...]
        Filename, size, and nanosecond modification timestamp.

    Raises
    ------
    ValueError
        If a conversation folder or file is a symbolic link.

    """

    folder = root / "logs" / "traces"
    if folder.is_symlink():
        raise ValueError("The conversation folder must not be a link.")
    if not folder.exists():
        return ()
    if not folder.is_dir():
        raise ValueError("The conversation folder is invalid.")
    result: list[tuple[str, int, int]] = []
    for path in sorted(folder.glob("conversation_*.json")):
        if path.is_symlink():
            raise ValueError("Conversation memory cannot read linked files.")
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size > _MAX_CONVERSATION_BYTES:
            raise ValueError("The conversation file exceeds the memory limit.")
        result.append((path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(result)


def _policy_digest(config: ProjectType) -> str:
    """Identify settings which determine memory index membership.

    Parameters
    ----------
    config: ProjectType
        Validated effective project behavior.

    Returns
    -------
    str
        Hash of content and intent policies, excluding search budgets.

    """

    settings = config.conversation_memory
    content = {
        "questions": settings.questions,
        "answers": settings.answers,
        "intents": {
            key: value.memory_indexing
            for key, value in sorted(config.intents.items())
        },
    }
    encoded = json.dumps(content, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _eligible(
    role: str, intent: object, config: ProjectType,
    feedback: FeedbackSnapshot, turn_uuid: str,
) -> tuple[bool, int]:
    """Apply explicit and automatic policy to one saved message.

    Parameters
    ----------
    role: str
        User question or assistant answer.
    intent: object
        Validated stored intent, if still defined in the project.
    config: ProjectType
        Effective project content and intent settings.
    feedback: FeedbackSnapshot
        Authoritative project feedback.
    turn_uuid: str
        Identity of the stored turn.

    Returns
    -------
    tuple[bool, int]
        Eligibility and bounded feedback-priority tier.

    """

    policy = config.intents.get(intent) if isinstance(intent, str) else None
    if policy is None or policy.memory_indexing == "never":
        return False, 0
    rating = feedback.get(turn_uuid, role)
    if rating.reaction == "disliked":
        return False, 0
    selected = rating.selected
    if role == "assistant" and config.conversation_memory.answers == "none":
        return False, 0
    if policy.memory_indexing == "explicit" and not selected:
        return False, 0
    if role == "user":
        allowed = config.conversation_memory.questions == "all" or selected
    else:
        allowed = (
            config.conversation_memory.answers == "all" or selected
        )
    priority = 2 if rating.reaction == "liked" else int(rating.saved)
    return allowed, priority


def _read_records(
    root: Path, manifest: tuple[tuple[str, int, int], ...],
    config: ProjectType, feedback: FeedbackSnapshot,
) -> tuple[MemoryRecord, ...]:
    """Collect eligible occurrences from active saved conversations.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace root.
    manifest: tuple[tuple[str, int, int], ...]
        Active files already checked for ownership and size.
    config: ProjectType
        Effective intent and content policies.
    feedback: FeedbackSnapshot
        Validated user feedback snapshot.

    Returns
    -------
    tuple[MemoryRecord, ...]
        One distinct UUID/role record with all active branch origins.

    Raises
    ------
    ValueError
        If an active saved record is malformed or inherited content differs.

    """

    entries: dict[
        tuple[str, str], tuple[str, str, int, list[MemoryOrigin]]
    ] = {}
    for name, _, _ in manifest:
        path = root / "logs" / "traces" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = conversation_state(payload)
        turns = record.get("turns", [])
        ids = require_saved_turn_ids(turns)
        conversation_id = record.get("conversation_id", path.stem)
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("The conversation identifier is invalid.")
        for number, (turn, turn_uuid) in enumerate(zip(turns, ids), 1):
            if not isinstance(turn, dict):
                raise ValueError("The saved conversation turn is invalid.")
            params = turn.get("parameters", {})
            if not isinstance(params, dict):
                raise ValueError("The saved memory parameters are invalid.")
            intent = params.get("intent")
            timestamp = turn.get("timestamp")
            if not isinstance(timestamp, str) or not timestamp:
                raise ValueError("The saved memory timestamp is invalid.")
            for role, field in (("user", "question"),
                                ("assistant", "answer")):
                if role == "assistant" and params.get("completion"):
                    continue
                eligible, priority = _eligible(
                    role, intent, config, feedback, turn_uuid
                )
                if not eligible:
                    continue
                raw_text = turn.get(field)
                if not isinstance(raw_text, str):
                    raise ValueError("The saved memory text is invalid.")
                text = (
                    strip_thinking(raw_text) if role == "assistant"
                    else raw_text
                ).strip()
                if not text:
                    continue
                text = text[:_MAX_MESSAGE_CHARS]
                key = (turn_uuid, role)
                origin = MemoryOrigin(conversation_id, number)
                previous = entries.get(key)
                if previous is None:
                    entries[key] = (text, timestamp, priority, [origin])
                else:
                    old_text, old_date, old_priority, origins = previous
                    if old_text != text or old_date != timestamp:
                        raise ValueError(
                            "Branches disagree on an inherited memory turn."
                        )
                    origins.append(origin)
                    entries[key] = (
                        old_text, old_date, max(old_priority, priority),
                        origins,
                    )
    records: list[MemoryRecord] = []
    for (turn_uuid, role), values in entries.items():
        text, timestamp, priority, origins = values
        records.append(MemoryRecord(
            turn_uuid, role, text, timestamp,
            memory_fingerprint(role, text), priority,
            tuple(sorted(set(origins), key=lambda item: (
                item.conversation_id, item.turn_number
            ))),
        ))
    return tuple(sorted(records, key=lambda item: (
        item.timestamp, item.turn_uuid, item.role
    )))


def _decode(
    saved: dict[str, object], model_identity: str,
    manifest: tuple[tuple[str, int, int], ...], policy_digest: str,
    feedback_revision: str,
) -> MemoryIndex | None:
    """Validate a saved index for reuse without trusting its JSON records.

    Parameters
    ----------
    saved: dict[str, object]
        Checksum-validated derived JSON.
    model_identity: str
        Identity of the current embedding model.
    manifest: tuple[tuple[str, int, int], ...]
        Current active conversation file inventory.
    policy_digest: str
        Effective index-membership policy hash.
    feedback_revision: str
        Feedback file's SHA-256 revision.

    Returns
    -------
    MemoryIndex | None
        Valid index if its model and structural data match, else None.

    """

    try:
        if saved["model_identity"] != model_identity:
            return None
        vectors = saved["vectors"]
        records = saved["records"]
        if not isinstance(vectors, dict) or not isinstance(records, list):
            return None
        decoded_vectors: dict[str, tuple[float, ...]] = {}
        dimension = None
        for key, vector in vectors.items():
            if (
                not isinstance(key, str)
                or re.fullmatch(r"[0-9a-f]{64}", key) is None
                or not isinstance(vector, list)
                or not vector
                or any(
                    type(value) not in (float, int) or not isfinite(value)
                    for value in vector
                )
            ):
                return None
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                return None
            decoded_vectors[key] = tuple(float(v) for v in vector)
        decoded_records: list[MemoryRecord] = []
        for value in records:
            if not isinstance(value, dict) or set(value) != {
                "turn_uuid", "role", "text", "timestamp",
                "fingerprint", "priority", "origins",
            }:
                return None
            origins = tuple(MemoryOrigin(**item) for item in value["origins"])
            record = MemoryRecord(**{**value, "origins": origins})
            if (
                record.role not in ("user", "assistant")
                or not isinstance(record.text, str) or not record.text
                or record.fingerprint != memory_fingerprint(
                    record.role, record.text
                )
                or record.fingerprint not in decoded_vectors
                or record.priority not in (0, 1, 2)
            ):
                return None
            decoded_records.append(record)
        index = MemoryIndex(
            model_identity, tuple(decoded_records), decoded_vectors
        )
        # A validated stale index can supply vectors for unchanged content.
        if (
            saved["manifest"] != [list(item) for item in manifest]
            or saved["policy_digest"] != policy_digest
            or saved["feedback_revision"] != feedback_revision
        ):
            return index
        return index
    except (KeyError, TypeError, ValueError):
        return None


def synchronize_memory(
    project_directory: str | Path,
    config: ProjectType,
    embedding_session: object,
    *,
    model_identity: str,
) -> MemoryIndex:
    """Synchronize the independent memory index with active conversations.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        Root of the active KíLín project.
    config: ProjectType
        Validated effective project behavior.
    embedding_session: object
        Already loaded embedding session with ``embed_documents``.
    model_identity: str
        Verified current embedding-model identity.

    Returns
    -------
    MemoryIndex
        Eligible records and vectors; no writes when the state is current.

    Raises
    ------
    ValueError
        If project state, policy or active conversation data is invalid.

    Notes
    -----
    Disabled memory returns immediately, without accessing the filesystem or
    embedding model. The index does not automatically delete old checkpoints.

    """

    if not config.conversation_memory.enabled:
        return MemoryIndex(model_identity, (), {})
    root = Path(project_directory).resolve()
    manifest = _manifest(root)
    feedback = FeedbackStore(root).load()
    digest = _policy_digest(config)
    path = memory_index_path(root)
    saved = read_memory_index(path)
    previous = (
        _decode(saved, model_identity, manifest, digest, feedback.revision)
        if saved is not None else None
    )
    if (
        saved is not None and previous is not None
        and saved["manifest"] == [list(item) for item in manifest]
        and saved["policy_digest"] == digest
        and saved["feedback_revision"] == feedback.revision
    ):
        return previous
    records = _read_records(root, manifest, config, feedback)
    # Detailed traces share the conversation JSON. An unrelated progress
    # event may change its file stat without changing committed turns.
    # Avoid rewriting the index merely because diagnostic events changed.
    if (
        previous is not None
        and previous.records == records
        and saved is not None
        and saved["policy_digest"] == digest
        and saved["feedback_revision"] == feedback.revision
    ):
        return previous
    text_by_fingerprint = {
        item.fingerprint: item.text for item in records
    }
    old_vectors = previous.vectors if previous is not None else {}
    vectors = {
        key: old_vectors[key] for key in text_by_fingerprint
        if key in old_vectors
    }
    missing = sorted(set(text_by_fingerprint) - set(vectors))
    dimension = len(next(iter(vectors.values()))) if vectors else None
    for start in range(0, len(missing), _BATCH_SIZE):
        batch = missing[start:start + _BATCH_SIZE]
        prepared = [text_by_fingerprint[key] for key in batch]
        embedded = embedding_session.embed_documents(
            prepared, purpose="documents"
        )
        if len(embedded) != len(batch):
            raise ValueError("Memory embedding count does not match input.")
        for key, vector in zip(batch, embedded):
            if (
                not isinstance(vector, (list, tuple))
                or not vector
                or any(
                    type(value) not in (float, int) or not isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError("The computed memory embedding is invalid.")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError("Memory embedding dimensions do not match.")
            vectors[key] = tuple(float(value) for value in vector)
    content = {
        "model_identity": model_identity,
        "manifest": [list(item) for item in manifest],
        "policy_digest": digest,
        "feedback_revision": feedback.revision,
        "records": [
            {
                "turn_uuid": item.turn_uuid,
                "role": item.role,
                "text": item.text,
                "timestamp": item.timestamp,
                "fingerprint": item.fingerprint,
                "priority": item.priority,
                "origins": [
                    {
                        "conversation_id": origin.conversation_id,
                        "turn_number": origin.turn_number,
                    }
                    for origin in item.origins
                ],
            }
            for item in records
        ],
        "vectors": {key: list(value) for key, value in vectors.items()},
    }
    write_memory_index(path, content)
    return MemoryIndex(
        model_identity, records, vectors, changed=True,
        embedded_count=len(missing),
    )
