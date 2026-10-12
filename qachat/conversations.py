"""Manage named conversation records in a project's trace directory.

``ConversationStore`` maintains a registry of ``conversation_*.json`` files
under ``logs/traces``. It discovers saved metadata, chooses an active record,
creates empty conversations, and branches by copying a saved resumable state.
A branch records its parent's identifier and checkpoint reference; it does
not duplicate the parent's complete trace-event history.

Files use only the canonical grouped conversation-session record through
the shared internal conversation-format contract. Writes use unique
atomically replaced temporary files. A stable companion-file lock excludes
concurrent QAChat writers across processes. The in-memory registry and
filesystem are not updated as one transaction.
Conversation files can contain full questions, answers, and source excerpts.

Key Features
------------

- **Named conversations**

    Create, select, rename, and timestamp conversations independently of loaded
    models.

- **Branch metadata**

    Record parent identity and the saved checkpoint from which a branch was
    created.

- **Canonical storage**

    Read validated grouped conversation records without starting inference.

Module Structure
----------------

- ``ConversationInfo``

    Immutable metadata identifying one named conversation.

- ``ConversationStore``

    Manage named JSON conversation records for one project.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat.conversations
>>> store = qachat.conversations.ConversationStore("/path/to/workspace")
>>> active = store.ensure()
>>> path = store.state_path(active)

``ensure()`` creates storage and an initial record when none exists.

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

from qachat.atomic_writes import atomic_write_text
from qachat.conversation_ownership import (
    ConversationFileLock,
    conversation_write_ownership,
)
from qachat.conversation_format import (
    CONVERSATION_SESSION_KEY,
    conversation_state,
    new_conversation_session,
)

__all__ = ["ConversationInfo", "ConversationStore"]

# Accept existing opaque IDs as well as generated ``conversation_<hex>`` IDs.
# Do not constrain loaded IDs to the generated UUID format: older records and
# imported conversations may use other alphanumeric identifiers.
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# Saved metadata uses UTC ISO 8601 timestamps, with optional fractions.
# Both the current ``Z`` spelling and historical ``+00:00`` are accepted.
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|\+00:00)"
)


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    """Immutable metadata identifying one named conversation.

    The store serializes these fields alongside resumable state. Dataclass
    construction does not independently validate timestamp syntax or identifier
    uniqueness.

    Parameters
    ----------
    conversation_id: str
        Identifier of a conversation already discovered or created by this
        store.
    name: str
        Display name. Renaming collapses whitespace after string conversion and
        requires 1 to 80 characters.
    parent_conversation_id: str | None
        Parent conversation identifier for a branch, or None for a new
        independent conversation.
    origin_checkpoint_id: str | None
        Saved parent checkpoint identifier, or None when no checkpoint was
        available.
    created_at_utc: str
        Creation timestamp as an ISO 8601 UTC string.
    updated_at_utc: str
        Last metadata-update timestamp as an ISO 8601 UTC string.

    Attributes
    ----------
    conversation_id: str
        Identifier of a conversation already discovered or created by this
        store.
    name: str
        Display name. Renaming collapses whitespace after string conversion and
        requires 1 to 80 characters.
    parent_conversation_id: str | None
        Parent conversation identifier for a branch, or None for a new
        independent conversation.
    origin_checkpoint_id: str | None
        Saved parent checkpoint identifier, or None when no checkpoint was
        available.
    created_at_utc: str
        Creation timestamp as an ISO 8601 UTC string.
    updated_at_utc: str
        Last metadata-update timestamp as an ISO 8601 UTC string.

    """

    conversation_id: str
    name: str
    parent_conversation_id: str | None
    origin_checkpoint_id: str | None
    created_at_utc: str
    updated_at_utc: str


class ConversationStore:
    """Manage named JSON conversation records for one project.

    The constructor establishes paths and an empty registry. Call ensure to
    discover records and select an initial conversation, or create to write a
    new record.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        Project directory whose logs/traces folder stores conversation JSON
        files. Construction resolves the path without creating files.

    Attributes
    ----------
    project_directory: pathlib.Path
        Resolved project directory.
    root: pathlib.Path
        Conversation storage directory under logs/traces.

    Methods
    -------
    ensure()
        Scan conversation files and select the most recently updated one.
    active()
        Return the active conversation metadata without reading its file.
    conversations()
        Return the registered conversations in creation order.
    select(conversation_id)
        Select a registered conversation without loading model history.
    create(*, branch_from=None)
        Create an empty conversation or a resumable copy of a parent.
    rename(conversation_id, name)
        Normalize and persist a conversation display name.
    touch(conversation_id)
        Update and persist the last-modified timestamp of a conversation.
    is_empty(conversation_id)
        Return whether a conversation has no saved turns.
    archive(conversation_id)
        Move a conversation's file aside and select another conversation.
    state_path(conversation)
        Resolve the registered file path of a conversation.

    Notes
    -----
    Metadata and saved state are local JSON, not an encrypted store.
    File mutations borrow a caller-held ownership token or acquire a
    short-lived cross-process lock. Registry updates are not transactional.
    The store does not load language models.

    """

    def __init__(self, project_directory: str | Path) -> None:
        """Resolve storage paths and initialize an empty in-memory registry.

        Parameters
        ----------
        project_directory: str | pathlib.Path
            Project directory whose logs/traces folder stores conversation JSON
            files. Construction resolves the path without creating files.

        """

        self.project_directory = Path(project_directory).expanduser().resolve()
        self.root = self.project_directory / "logs" / "traces"
        self._items: dict[str, ConversationInfo] = {}
        self._paths: dict[str, Path] = {}
        self._active_id: str | None = None

    def ensure(self) -> ConversationInfo:
        """Scan conversation files and select the most recently updated one.

        Returns
        -------
        ConversationInfo
            Most recently updated discovered record, or a newly created record
            when storage has none.

        Raises
        ------
        ValueError
            If the traces directory itself is a symbolic link.
        OSError
            If storage cannot be created, scanned, or written.

        Notes
        -----
        Discovery skips invalid records and duplicate identifiers after their
        first discovered path. Existing records are ordered by updated
        timestamp and identifier when selecting the active one.

        """

        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("The traces folder must not be a link.")
        self._scan()
        if not self._items:
            return self.create()
        newest = max(
            self._items.values(),
            key=lambda item: (
                _utc_datetime(item.updated_at_utc), item.conversation_id
            ),
        )
        self._active_id = newest.conversation_id
        return newest

    def active(self) -> ConversationInfo:
        """Return the active conversation metadata without reading its file.

        Returns
        -------
        ConversationInfo
            The currently registered active record.

        Raises
        ------
        RuntimeError
            If no active identifier is registered.

        """

        if self._active_id is None or self._active_id not in self._items:
            raise RuntimeError("No active conversation is available.")
        return self._items[self._active_id]

    def conversations(self) -> list[ConversationInfo]:
        """Return the registered conversations in creation order.

        Returns
        -------
        list[ConversationInfo]
            A newly allocated list ordered by creation timestamp, then
            identifier.

        """

        return sorted(
            self._items.values(),
            key=lambda item: (
                _utc_datetime(item.created_at_utc), item.conversation_id
            ),
        )

    def select(self, conversation_id: str) -> ConversationInfo:
        """Select a registered conversation without loading model history.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.

        Returns
        -------
        ConversationInfo
            The selected metadata record.

        Raises
        ------
        ValueError
            If the identifier is not registered.

        """

        item = self._require(conversation_id)
        self._active_id = item.conversation_id
        return item

    def create(
        self,
        *,
        branch_from: str | None = None,
        ownership: ConversationFileLock | None = None,
    ) -> ConversationInfo:
        """Create an empty conversation or a resumable copy of a parent.

        Parameters
        ----------
        branch_from: str | None, optional
            Identifier of a registered parent to copy resumable state from, or
            None to create an empty independent conversation. Defaults to
            ``None``.
        ownership: ConversationFileLock | None, optional
            Borrowed lock for the parent conversation when branching. The
            new destination always acquires its own short-lived lock.

        Returns
        -------
        ConversationInfo
            Newly written and registered record, also made active.

        Raises
        ------
        ValueError
            If a requested parent is unknown or its saved JSON/state is
            invalid.
        OSError
            If the new conversation cannot be written.

        Notes
        -----
        A branch copies non-metadata fields from the parent resume state and
        records parent/checkpoint identity. Old trace operations are not
        copied. The new filename avoids existing paths at the moment it is
        chosen; each candidate path is locked and checked before creation.
        A collision detected after selection raises rather than overwriting.

        """

        parent = (
            self._require(branch_from) if branch_from is not None else None
        )
        name = (
            self._branch_name(parent.name)
            if parent is not None
            else f"Conversation {len(self._items) + 1}"
        )
        checkpoint = None
        now = _utc_timestamp()
        item = ConversationInfo(
            conversation_id=f"conversation_{uuid4().hex[:12]}",
            name=name,
            parent_conversation_id=(
                parent.conversation_id if parent is not None else None
            ),
            origin_checkpoint_id=checkpoint,
            created_at_utc=now,
            updated_at_utc=now,
        )
        path = self._new_path()
        copied_state: dict[str, object] = {}
        if parent is not None:
            parent_path = self.state_path(parent)
            with conversation_write_ownership(
                parent_path, ownership=ownership
            ):
                saved = conversation_state(_read_json(parent_path))
                checkpoint = saved.get("checkpoint_id")
                checkpoint = (
                    checkpoint
                    if isinstance(checkpoint, str) and checkpoint
                    else None
                )
                metadata = ConversationInfo.__dataclass_fields__
                copied_state = {
                    key: value
                    for key, value in saved.items()
                    if key not in metadata
                }
            item = replace(item, origin_checkpoint_id=checkpoint)
        with conversation_write_ownership(path):
            if path.exists():
                raise FileExistsError(
                    f"Conversation path already exists: {path}"
                )
            _write_json(
                path,
                {
                    "sessions": {
                        CONVERSATION_SESSION_KEY: new_conversation_session(
                            {**asdict(item), **copied_state}
                        )
                    }
                },
            )
        self._items[item.conversation_id] = item
        self._paths[item.conversation_id] = path
        self._active_id = item.conversation_id
        return item

    def rename(
        self,
        conversation_id: str,
        name: str,
        *,
        ownership: ConversationFileLock | None = None,
        state_writer: Callable[[Mapping[str, object]], None] | None = None,
    ) -> ConversationInfo:
        """Normalize and persist a conversation display name.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.
        name: str
            Display name. Renaming collapses whitespace after string conversion
            and requires 1 to 80 characters.
        ownership: ConversationFileLock | None, optional
            Borrowed lock held by an active session for this file. Without
            one, the store acquires and releases its own exclusive lock.
        state_writer: Callable[[Mapping[str, object]], None] | None, optional
            Writer for the same JSON file, such as an active trace logger's
            ``update_conversation``. Requires borrowed ownership and replaces
            the store's own file write. Defaults to ``None``.

        Returns
        -------
        ConversationInfo
            Updated immutable metadata record.

        Raises
        ------
        ValueError
            If the identifier is unknown, the normalized name is empty or over
            80 characters, or saved state is invalid.
        OSError
            If updated JSON cannot be written.

        """

        item = self._require(conversation_id)
        clean = " ".join(str(name).split())
        if not clean:
            raise ValueError("Conversation name cannot be empty.")
        if len(clean) > 80:
            raise ValueError(
                "Conversation name must be 80 characters or less."
            )
        renamed = replace(item, name=clean, updated_at_utc=_utc_timestamp())
        self._update_metadata(
            renamed, ownership=ownership, state_writer=state_writer
        )
        return renamed

    def touch(
        self,
        conversation_id: str,
        *,
        ownership: ConversationFileLock | None = None,
        state_writer: Callable[[Mapping[str, object]], None] | None = None,
    ) -> ConversationInfo:
        """Update and persist the last-modified timestamp of a conversation.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.
        ownership: ConversationFileLock | None, optional
            Borrowed lock held by an active session for this file. Without
            one, the store acquires and releases its own exclusive lock.
        state_writer: Callable[[Mapping[str, object]], None] | None, optional
            Writer for the same JSON file, such as an active trace logger's
            ``update_conversation``. Requires borrowed ownership and replaces
            the store's own file write. Defaults to ``None``.

        Returns
        -------
        ConversationInfo
            Updated immutable metadata record.

        Raises
        ------
        ValueError
            If the identifier or saved state is invalid.
        OSError
            If updated JSON cannot be written.

        """

        item = self._require(conversation_id)
        updated = replace(item, updated_at_utc=_utc_timestamp())
        self._update_metadata(
            updated, ownership=ownership, state_writer=state_writer
        )
        return updated

    def is_empty(self, conversation_id: str) -> bool:
        """Return whether a conversation has no saved turns.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.

        Returns
        -------
        bool
            True when the saved resume state has no turns.

        Raises
        ------
        ValueError
            If the identifier, the saved file, or its turn list is invalid.

        """

        item = self._require(conversation_id)
        record = conversation_state(_read_json(self.state_path(item)))
        turns = record.get("turns", [])
        if not isinstance(turns, list):
            raise ValueError("The saved conversation is invalid.")

        return not turns

    def archive(
        self,
        conversation_id: str,
        *,
        ownership: ConversationFileLock | None = None,
    ) -> tuple[ConversationInfo, Path]:
        """Remove a conversation from the catalog and retain its JSON.

        The record is moved into ``logs/traces/deleted`` so deletion remains
        recoverable; an existing file of the same name there is kept, and
        the moved file gets a ``_2``, ``_3``, ... suffix. When the last
        conversation is archived, a new empty conversation is created
        automatically. When the active conversation is archived, the most
        recently updated remaining one becomes active.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.
        ownership: ConversationFileLock | None, optional
            Borrowed lock for archiving the currently owned conversation.
            Other files are locked independently before moving.

        Returns
        -------
        tuple[ConversationInfo, pathlib.Path]
            The active conversation afterwards, and the archived file path.

        Raises
        ------
        ValueError
            If the identifier is invalid or the deleted-conversations folder
            is a symbolic link.
        OSError
            If the folder cannot be created or the file cannot be moved.

        Notes
        -----
        The source path remains exclusively owned through the rename; the
        registry update is not part of a filesystem transaction.

        """

        item = self._require(conversation_id)
        source = self.state_path(item)
        deleted = self.root / "deleted"
        deleted.mkdir(parents=True, exist_ok=True)
        if deleted.is_symlink():
            raise ValueError(
                "The deleted-conversations folder must not be a link."
            )

        with conversation_write_ownership(
            source, ownership=ownership
        ):
            destination = deleted / source.name
            number = 2
            while destination.exists():
                destination = (
                    deleted / f"{source.stem}_{number}{source.suffix}"
                )
                number += 1
            source.rename(destination)
        self._items.pop(conversation_id)
        self._paths.pop(conversation_id)

        if not self._items:
            return self.create(), destination

        if self._active_id == conversation_id:
            replacement = max(
                self._items.values(),
                key=lambda candidate: (
                    _utc_datetime(candidate.updated_at_utc),
                    candidate.conversation_id,
                ),
            )
            self._active_id = replacement.conversation_id

        return self.active(), destination

    def state_path(self, conversation: ConversationInfo | str) -> Path:
        """Resolve the registered file path of a conversation.

        Parameters
        ----------
        conversation: ConversationInfo | str
            A known ConversationInfo record or its registered identifier.

        Returns
        -------
        pathlib.Path
            The path remembered by this store, without rereading or verifying
            the file.

        Raises
        ------
        ValueError
            If the identifier is unknown.

        """

        identifier = (
            conversation.conversation_id
            if isinstance(conversation, ConversationInfo)
            else conversation
        )
        self._require(identifier)
        return self._paths[identifier]

    def _scan(self) -> None:
        """Rebuild the registry from recognizable conversation files.

        Symbolic links, non-files, malformed JSON/state, missing metadata, and
        duplicate identifiers after the first are skipped. Persisted metadata
        is validated before records enter the registry.

        """

        self._items.clear()
        self._paths.clear()
        for path in sorted(self.root.glob("conversation_*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                record = conversation_state(_read_json(path))
                if not isinstance(record, dict):
                    continue
                item = _validated_metadata(record)
            except (KeyError, TypeError, ValueError):
                continue
            if item.conversation_id not in self._items:
                self._items[item.conversation_id] = item
                self._paths[item.conversation_id] = path

    def _update_metadata(
        self,
        item: ConversationInfo,
        *,
        ownership: ConversationFileLock | None = None,
        state_writer: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        """Replace saved metadata, then update the in-memory record.

        Parameters
        ----------
        item: ConversationInfo
            Updated metadata record for a conversation already registered in
            the store.
        ownership: ConversationFileLock | None, optional
            Borrowed exclusive write lock for this record, when already held.
        state_writer: Callable[[Mapping[str, object]], None] | None, optional
            Coordinated writer for the same conversation file. Requires an
            active borrowed owner; the store updates its registry only after
            the writer succeeds. Defaults to ``None``.

        Raises
        ------
        ValueError
            If existing JSON/state is invalid.
        OSError
            If the updated record cannot be written.

        Notes
        -----
        A successful coordinated write updates the trace and store metadata
        together, without a second JSON replacement from the store. Callers
        must supply a writer targeting the borrowed conversation path.

        """

        path = self._paths[item.conversation_id]
        if state_writer is not None and ownership is None:
            raise ValueError(
                "A coordinated state writer requires held ownership."
            )
        with conversation_write_ownership(path, ownership=ownership):
            payload = _read_json(path)
            conversation = conversation_state(payload, create=True)
            changes = asdict(item)
            if state_writer is None:
                conversation.update(changes)
                _write_json(path, payload)
            else:
                # The active trace is the sole JSON writer. A second
                # store replacement would leave its snapshot stale.
                state_writer(changes)
            self._items[item.conversation_id] = item

    def _new_path(self) -> Path:
        """Choose a timestamped conversation filename not currently present.

        Returns
        -------
        pathlib.Path
            Candidate JSON path with a numeric suffix added on collisions. The
            file is not created here.

        """

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        candidate = self.root / f"conversation_{timestamp}.json"
        number = 2
        while candidate.exists():
            candidate = self.root / f"conversation_{timestamp}_{number}.json"
            number += 1
        return candidate

    def _branch_name(self, parent_name: str) -> str:
        """Choose a branch name unused by the current registry.

        Parameters
        ----------
        parent_name: str
            Display name of the parent used as the prefix for a unique branch
            name.

        Returns
        -------
        str
            Parent name followed by "branch" and the first available positive
            integer.

        """

        used = {item.name for item in self._items.values()}
        number = 1
        while f"{parent_name} branch {number}" in used:
            number += 1
        return f"{parent_name} branch {number}"

    def _require(self, conversation_id: str | None) -> ConversationInfo:
        """Find a registered conversation or reject the identifier.

        Parameters
        ----------
        conversation_id: str | None
            Identifier of a conversation already discovered or created by this
            store.

        Returns
        -------
        ConversationInfo
            Metadata associated with the registered string identifier.

        Raises
        ------
        ValueError
            If the identifier is non-string or absent from the registry.

        """

        if not isinstance(conversation_id, str):
            raise ValueError("Conversation not found.")
        try:
            return self._items[conversation_id]
        except KeyError as error:
            raise ValueError("Conversation not found.") from error

    @staticmethod
    def _checkpoint_id(path: Path) -> str | None:
        """Read a saved conversation checkpoint identifier.

        Parameters
        ----------
        path: pathlib.Path
            Path of the conversation JSON file to read or write.

        Returns
        -------
        str | None
            Nonempty checkpoint string, or None if the field is absent or
            invalid.

        Raises
        ------
        ValueError
            If the file or enclosing conversation state is invalid.

        """

        record = conversation_state(_read_json(path))
        value = record.get("checkpoint_id")
        return value if isinstance(value, str) and value else None


def _utc_datetime(value: object) -> datetime:
    """Parse a valid persisted UTC timestamp for chronological ordering.

    Parameters
    ----------
    value: object
        Candidate ISO 8601 timestamp from saved metadata.

    Returns
    -------
    datetime.datetime
        Timezone-aware UTC datetime for comparing saved records.

    Raises
    ------
    ValueError
        If the timestamp is not a valid UTC date and time.

    """

    if (
        not isinstance(value, str)
        or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("The saved conversation timestamp is invalid.")

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "The saved conversation timestamp is invalid."
        ) from error


def _validated_metadata(record: dict[str, object]) -> ConversationInfo:
    """Validate metadata before registering a persisted conversation.

    Parameters
    ----------
    record: dict[str, object]
        Saved resumable conversation state containing metadata fields.

    Returns
    -------
    ConversationInfo
        Metadata with validated identifiers, text, and UTC timestamps.

    Raises
    ------
    ValueError
        If any required field is missing or has an invalid value.

    Notes
    -----
    Extra fields belong to saved RAG state and are retained in the JSON.
    Validation never rewrites, repairs, or deletes an invalid file. Discovery
    skips invalid records so other conversations remain accessible.

    """

    try:
        metadata = {
            field: record[field]
            for field in ConversationInfo.__dataclass_fields__
        }
    except KeyError as error:
        raise ValueError(
            "The saved conversation metadata is incomplete."
        ) from error

    identifier = metadata["conversation_id"]
    if (
        not isinstance(identifier, str)
        or _ID_PATTERN.fullmatch(identifier) is None
    ):
        raise ValueError("The saved conversation ID is invalid.")

    name = metadata["name"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 80
    ):
        raise ValueError("The saved conversation name is invalid.")

    parent = metadata["parent_conversation_id"]
    if (
        parent is not None
        and (
            not isinstance(parent, str)
            or _ID_PATTERN.fullmatch(parent) is None
        )
    ):
        raise ValueError("The saved conversation parent ID is invalid.")

    origin = metadata["origin_checkpoint_id"]
    if origin is not None and (not isinstance(origin, str) or not origin):
        raise ValueError("The saved conversation checkpoint is invalid.")

    _utc_datetime(metadata["created_at_utc"])
    _utc_datetime(metadata["updated_at_utc"])
    return ConversationInfo(**metadata)


def _read_json(path: Path) -> dict[str, object]:
    """Read a conversation JSON object as UTF-8.

    Parameters
    ----------
    path: pathlib.Path
        Path of the conversation JSON file to read or write.

    Returns
    -------
    dict[str, object]
        Decoded dictionary.

    Raises
    ------
    ValueError
        If reading fails, JSON is invalid, or the decoded root is not a
        dictionary.

    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("The saved conversation is invalid.") from error
    if not isinstance(payload, dict):
        raise ValueError("The saved conversation is invalid.")
    return payload


def _write_json(path: Path, payload: object) -> None:
    """Write formatted UTF-8 JSON through a temporary replacement file.

    Parameters
    ----------
    path: pathlib.Path
        Path of the conversation JSON file to read or write.
    payload: object
        Decoded or serializable JSON document containing conversation data.

    Raises
    ------
    OSError
        If the parent, temporary file, or replacement cannot be written.
    TypeError
        If the payload contains unsupported JSON values.

    Notes
    -----
    Parent directories are created. An exclusive, unpredictably named temporary
    file in the destination directory prevents following a pre-existing
    temporary-file symlink. Temporary files are removed on any write or replace
    failure. Replacement is atomic for one file, not a transaction with the
    in-memory registry or other writers.

    """

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def _utc_timestamp() -> str:
    """Format the current UTC time for conversation metadata.

    Returns
    -------
    str
        ISO 8601 timestamp at millisecond precision ending in Z.

    """

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
