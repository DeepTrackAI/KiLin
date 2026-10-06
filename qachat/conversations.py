"""Manage named conversation records in a project's trace directory.

``ConversationStore`` maintains a registry of ``conversation_*.json`` files
under ``logs/traces``. It discovers saved metadata, chooses an active record,
creates empty conversations, and branches by copying a saved resumable state.
A branch records its parent's identifier and checkpoint reference; it does
not duplicate the parent's complete trace-event history.

Files support both legacy top-level conversation records and grouped trace
sessions through the shared internal conversation-format contract. Writes
replace a temporary JSON file, but this module provides no multi-process lock
or transaction across registry and filesystem updates.
Conversation files can contain full questions, answers, and source excerpts.

Key Features
------------

- **Named conversations**

    Create, select, rename, and timestamp conversations independently of loaded
    models.

- **Branch metadata**

    Record parent identity and the saved checkpoint from which a branch was
    created.

- **Compatible storage**

    Read supported legacy and grouped conversation records without starting
    model inference.

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

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from qachat.conversation_format import (
    CONVERSATION_SESSION_KEY,
    conversation_state,
    new_conversation_session,
)

__all__ = ["ConversationInfo", "ConversationStore"]


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
    state_path(conversation)
        Resolve the registered file path of a conversation.

    Notes
    -----
    Metadata and saved state are local JSON, not an encrypted store. Registry
    operations and file writes do not provide inter-process locking. The store
    does not load language models.

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
            key=lambda item: (item.updated_at_utc, item.conversation_id),
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
            key=lambda item: (item.created_at_utc, item.conversation_id),
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

    def create(self, *, branch_from: str | None = None) -> ConversationInfo:
        """Create an empty conversation or a resumable copy of a parent.

        Parameters
        ----------
        branch_from: str | None, optional
            Identifier of a registered parent to copy resumable state from, or
            None to create an empty independent conversation. Defaults to
            ``None``.

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
        chosen; no concurrent-writer lock is used.

        """

        parent = (
            self._require(branch_from) if branch_from is not None else None
        )
        name = (
            self._branch_name(parent.name)
            if parent is not None
            else f"Conversation {len(self._items) + 1}"
        )
        checkpoint = (
            self._checkpoint_id(self.state_path(parent))
            if parent is not None
            else None
        )
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
            saved = conversation_state(_read_json(self.state_path(parent)))
            metadata = ConversationInfo.__dataclass_fields__
            copied_state = {
                key: value
                for key, value in saved.items()
                if key not in metadata
            }
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

    def rename(self, conversation_id: str, name: str) -> ConversationInfo:
        """Normalize and persist a conversation display name.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.
        name: str
            Display name. Renaming collapses whitespace after string conversion
            and requires 1 to 80 characters.

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
        self._update_metadata(renamed)
        return renamed

    def touch(self, conversation_id: str) -> ConversationInfo:
        """Update and persist the last-modified timestamp of a conversation.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation already discovered or created by this
            store.

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
        self._update_metadata(updated)
        return updated

    def is_empty(self, conversation_id: str) -> bool:
        """Return whether a conversation has no saved turns."""

        item = self._require(conversation_id)
        record = conversation_state(_read_json(self.state_path(item)))
        turns = record.get("turns", [])
        if not isinstance(turns, list):
            raise ValueError("The saved conversation is invalid.")
        return not turns

    def archive(
        self, conversation_id: str
    ) -> tuple[ConversationInfo, Path]:
        """Remove a conversation from the catalog and retain its JSON.

        The record is moved into ``logs/traces/deleted`` so deletion remains
        recoverable. When the last conversation is archived, a new empty
        conversation is created automatically.
        """

        item = self._require(conversation_id)
        source = self.state_path(item)
        deleted = self.root / "deleted"
        deleted.mkdir(parents=True, exist_ok=True)
        if deleted.is_symlink():
            raise ValueError(
                "The deleted-conversations folder must not be a link."
            )
        destination = deleted / source.name
        number = 2
        while destination.exists():
            destination = deleted / f"{source.stem}_{number}{source.suffix}"
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
                    candidate.updated_at_utc,
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
        duplicate identifiers after the first are skipped. This does not
        validate every metadata value against its annotation.

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
                item = ConversationInfo(
                    **{
                        field: record[field]
                        for field in ConversationInfo.__dataclass_fields__
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
            if item.conversation_id not in self._items:
                self._items[item.conversation_id] = item
                self._paths[item.conversation_id] = path

    def _update_metadata(self, item: ConversationInfo) -> None:
        """Replace saved metadata, then update the in-memory record.

        Parameters
        ----------
        item: ConversationInfo
            Updated metadata record for a conversation already registered in
            the store.

        Raises
        ------
        ValueError
            If existing JSON/state is invalid.
        OSError
            If the updated record cannot be written.

        """

        path = self._paths[item.conversation_id]
        payload = _read_json(path)
        conversation = conversation_state(payload, create=True)
        conversation.update(asdict(item))
        _write_json(path, payload)
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
    Parent directories are created. Replacement is a single-file operation, not
    a transaction with the in-memory registry or other writers.

    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
