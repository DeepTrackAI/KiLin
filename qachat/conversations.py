"""Discover and manage one self-contained JSON file per conversation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

__all__ = ["ConversationInfo", "ConversationStore"]


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    """User-visible identity and lineage for one saved conversation."""

    conversation_id: str
    name: str
    parent_conversation_id: str | None
    origin_checkpoint_id: str | None
    created_at_utc: str
    updated_at_utc: str


class ConversationStore:
    """Use ``project/logs/traces/conversation_<timestamp>.json`` directly."""

    def __init__(self, project_directory: str | Path) -> None:
        self.project_directory = Path(project_directory).expanduser().resolve()
        self.root = self.project_directory / "logs" / "traces"
        self._items: dict[str, ConversationInfo] = {}
        self._paths: dict[str, Path] = {}
        self._active_id: str | None = None

    def ensure(self) -> ConversationInfo:
        """Scan conversation files and select the most recently updated one."""

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
        if self._active_id is None or self._active_id not in self._items:
            raise RuntimeError("No active conversation is available.")
        return self._items[self._active_id]

    def conversations(self) -> list[ConversationInfo]:
        return sorted(
            self._items.values(),
            key=lambda item: (item.created_at_utc, item.conversation_id),
        )

    def select(self, conversation_id: str) -> ConversationInfo:
        item = self._require(conversation_id)
        self._active_id = item.conversation_id
        return item

    def create(self, *, branch_from: str | None = None) -> ConversationInfo:
        """Create an empty conversation or a resumable copy of a parent."""

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
            saved = _conversation_record(_read_json(self.state_path(parent)))
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
                    "05_conversations": {
                        "title": "Conversation turns",
                        "category": "conversations",
                        "turns": {},
                        "operations": {},
                        "resume_state": {**asdict(item), **copied_state},
                    }
                }
            },
        )
        self._items[item.conversation_id] = item
        self._paths[item.conversation_id] = path
        self._active_id = item.conversation_id
        return item

    def rename(self, conversation_id: str, name: str) -> ConversationInfo:
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
        item = self._require(conversation_id)
        updated = replace(item, updated_at_utc=_utc_timestamp())
        self._update_metadata(updated)
        return updated

    def state_path(self, conversation: ConversationInfo | str) -> Path:
        identifier = (
            conversation.conversation_id
            if isinstance(conversation, ConversationInfo)
            else conversation
        )
        self._require(identifier)
        return self._paths[identifier]

    def _scan(self) -> None:
        self._items.clear()
        self._paths.clear()
        for path in sorted(self.root.glob("conversation_*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                record = _conversation_record(_read_json(path))
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
        path = self._paths[item.conversation_id]
        payload = _read_json(path)
        conversation = _conversation_record(payload, create=True)
        conversation.update(asdict(item))
        _write_json(path, payload)
        self._items[item.conversation_id] = item

    def _new_path(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        candidate = self.root / f"conversation_{timestamp}.json"
        number = 2
        while candidate.exists():
            candidate = self.root / f"conversation_{timestamp}_{number}.json"
            number += 1
        return candidate

    def _branch_name(self, parent_name: str) -> str:
        used = {item.name for item in self._items.values()}
        number = 1
        while f"{parent_name} branch {number}" in used:
            number += 1
        return f"{parent_name} branch {number}"

    def _require(self, conversation_id: str | None) -> ConversationInfo:
        if not isinstance(conversation_id, str):
            raise ValueError("Conversation not found.")
        try:
            return self._items[conversation_id]
        except KeyError as error:
            raise ValueError("Conversation not found.") from error

    @staticmethod
    def _checkpoint_id(path: Path) -> str | None:
        record = _conversation_record(_read_json(path))
        value = record.get("checkpoint_id")
        return value if isinstance(value, str) and value else None


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("The saved conversation is invalid.") from error
    if not isinstance(payload, dict):
        raise ValueError("The saved conversation is invalid.")
    return payload


def _conversation_record(
    payload: dict[str, object], *, create: bool = False
) -> dict[str, object]:
    """Return state nested after project and indexing trace sessions."""

    legacy = payload.get("conversation")
    if isinstance(legacy, dict):
        return legacy
    sessions = (
        payload.setdefault("sessions", {})
        if create
        else payload.get("sessions")
    )
    if not isinstance(sessions, dict):
        raise ValueError("The saved conversation is invalid.")
    for session in sessions.values():
        if (
            isinstance(session, dict)
            and session.get("category") == "conversations"
        ):
            record = (
                session.setdefault("resume_state", {})
                if create
                else session.get("resume_state", session.get("conversation"))
            )
            if isinstance(record, dict):
                return record
    if create:
        session = {
            "title": "Conversation turns",
            "category": "conversations",
            "turns": {},
            "operations": {},
            "resume_state": {},
        }
        sessions["05_conversations"] = session
        return session["resume_state"]
    raise ValueError("The saved conversation is invalid.")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
