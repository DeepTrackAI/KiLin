"""Persist project-local conversation feedback with independent ownership.

``memory/feedback.json`` is authoritative user data, not a rebuildable index
or part of conversation checkpoint hashes. Each entry is keyed by turn UUID
and message role, so branches share feedback on inherited turns while
independently generated duplicate text receives independent feedback.

The existing per-file advisory lock protects read/modify/write operations
across processes, and atomic replacement prevents partial JSON publication.
Optimistic revisions detect stale editors; the lock also prevents silent lost
updates by independently acting writers without revision tokens.

Key Features
------------

- **Project-local state**

    Reject updates for turns not present in an active project conversation.

- **Independent feedback**

    Preserve Save and Like independently; support Dislike and reset.

- **Safe persistence**

    Validate all loaded data and replace only under an exclusive file lease.

Module Structure
----------------

- ``FeedbackSnapshot``

    Immutable view of feedback and the optimistic revision token.

- ``FeedbackStore``

    Read and update feedback without changing conversation files.

Examples
--------
>>> from kilin.conversation_memory.feedback import FeedbackStore
>>> store = FeedbackStore("/path/to/project")
>>> store.path.name
'feedback.json'

"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from kilin.atomic_writes import atomic_write_text
from kilin.conversation_ownership import conversation_write_ownership
from kilin.conversations import ConversationStore
from kilin.conversation_memory.models import (
    FeedbackReaction,
    FeedbackRole,
    MessageFeedback,
    require_feedback_role,
    require_turn_uuid,
)

__all__ = [
    "FeedbackConflictError",
    "FeedbackSnapshot",
    "FeedbackStore",
]

_FORMAT: Final[str] = "conversation-memory-feedback"
_FILE: Final[str] = "feedback.json"
_MAX_BYTES: Final[int] = 1_048_576
_EMPTY_REVISION: Final[str] = sha256(b"").hexdigest()


class FeedbackConflictError(RuntimeError):
    """Report a stale revision when writing project feedback."""


@dataclass(frozen=True, slots=True)
class FeedbackSnapshot:
    """Validated, immutable feedback and its revision token.

    Attributes
    ----------
    revision: str
        SHA-256 of the existing file bytes or of empty bytes if absent.
    entries: Mapping[str, Mapping[str, MessageFeedback]]
        Turn UUID keys with independently rated question/answer roles.

    """

    revision: str
    entries: Mapping[str, Mapping[str, MessageFeedback]]

    def get(
        self, turn_uuid: str, role: FeedbackRole
    ) -> MessageFeedback:
        """Read one rating without changing the feedback store.

        Parameters
        ----------
        turn_uuid: str
            Identity of the completed turn.
        role: FeedbackRole
            ``user`` question or ``assistant`` answer.

        Returns
        -------
        MessageFeedback
            Persisted rating or the neutral default when absent.

        """

        key = require_turn_uuid(turn_uuid)
        kind = require_feedback_role(role)
        return self.entries.get(key, {}).get(kind, MessageFeedback())


class FeedbackStore:
    """Read and update authoritative feedback in one project workspace.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        Root of the owning KíLín workspace, not its source directory.

    Attributes
    ----------
    project_directory: pathlib.Path
        Canonical root of the requested workspace.
    path: pathlib.Path
        Project-local authoritative feedback JSON file.

    Notes
    -----
    The store never creates a missing project. An update verifies that its
    turn belongs to the named active conversation. Archived or removed
    conversations are not valid new rating targets; existing ratings are
    retained, allowing another active branch to reference the same turn.

    """

    def __init__(self, project_directory: str | Path) -> None:
        """Record the project-owned path without creating any files.

        Parameters
        ----------
        project_directory: str | pathlib.Path
            Project workspace root, resolved without creating it.

        """

        self.project_directory = Path(project_directory).expanduser().resolve()
        self.path = self.project_directory / "memory" / _FILE

    def load(self) -> FeedbackSnapshot:
        """Read a complete immutable feedback snapshot without writes.

        Returns
        -------
        FeedbackSnapshot
            Validated records and revision; empty if no file exists.

        Raises
        ------
        ValueError
            If the directory is linked or the saved format is invalid.
        OSError
            If the file cannot be read.

        """

        directory = self.path.parent
        if directory.is_symlink() or (
            directory.exists() and not directory.is_dir()
        ):
            raise ValueError("The project memory folder is invalid.")
        if self.path.is_symlink():
            raise ValueError("The project feedback file cannot be a link.")
        if not self.path.exists():
            return FeedbackSnapshot(_EMPTY_REVISION, MappingProxyType({}))
        if not self.path.is_file() or self.path.stat().st_size > _MAX_BYTES:
            raise ValueError("The saved conversation feedback is invalid.")
        content = self.path.read_bytes()
        if len(content) > _MAX_BYTES:
            raise ValueError("The saved conversation feedback is invalid.")
        revision = sha256(content).hexdigest()
        try:
            payload = json.loads(content)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "The saved conversation feedback is invalid."
            ) from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"format", "entries"}
            or payload["format"] != _FORMAT
            or not isinstance(payload["entries"], dict)
        ):
            raise ValueError("The saved conversation feedback is invalid.")
        entries: dict[str, Mapping[str, MessageFeedback]] = {}
        for turn_uuid, ratings in payload["entries"].items():
            require_turn_uuid(turn_uuid)
            if not isinstance(ratings, dict) or not ratings:
                raise ValueError("The saved conversation feedback is invalid.")
            validated: dict[str, MessageFeedback] = {}
            for role, data in ratings.items():
                require_feedback_role(role)
                if (
                    not isinstance(data, dict)
                    or not {"saved", "reaction"}.issubset(data)
                    or not set(data).issubset({
                        "saved", "reaction", "excluded"
                    })
                ):
                    raise ValueError(
                        "The saved conversation feedback is invalid."
                    )
                entry = MessageFeedback(**data)
                if entry == MessageFeedback():
                    raise ValueError(
                        "Neutral feedback must not be stored explicitly."
                    )
                validated[role] = entry
            entries[turn_uuid] = MappingProxyType(validated)
        return FeedbackSnapshot(revision, MappingProxyType(entries))

    def update(
        self,
        conversation_id: str,
        turn_uuid: str,
        role: FeedbackRole,
        *,
        saved: bool | None,
        reaction: FeedbackReaction | None,
        excluded: bool | None = None,
        expected_revision: str | None = None,
    ) -> FeedbackSnapshot:
        """Persist one rating while preserving other writers' entries.

        Parameters
        ----------
        conversation_id: str
            Active project conversation containing the referenced turn.
        turn_uuid: str
            Stable identifier of the completed turn to rate.
        role: FeedbackRole
            ``user`` question or ``assistant`` answer.
        saved: bool | None
            Explicit Save state; None preserves the saved state under lock.
        reaction: FeedbackReaction | None
            Rating; None preserves the current rating under lock.
        excluded: bool | None, optional
            Explicit memory exclusion; None preserves the current state.
        expected_revision: str | None, optional
            Revision observed by the caller; stale edits fail. None accepts
            the latest state and merges under an exclusive lock.

        Returns
        -------
        FeedbackSnapshot
            Newly persisted full state and its revision token.

        Raises
        ------
        ValueError
            If the target does not belong to the project or is invalid.
        FeedbackConflictError
            If expected_revision does not match the current saved file.
        OSError
            If atomic persistence fails.

        Notes
        -----
        An exclusion remains authoritative until explicitly restored. A
        neutral, unsaved, non-excluded rating removes the entry. The
        conversation checkpoint and all other feedback entries are untouched.

        """

        identifier = require_turn_uuid(turn_uuid)
        kind = require_feedback_role(role)
        # Reject invalid supplied values before entering the file lease.
        MessageFeedback(
            saved=False if saved is None else saved,
            reaction="neutral" if reaction is None else reaction,
            excluded=False if excluded is None else excluded,
        )
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("Feedback needs an active conversation ID.")
        if expected_revision is not None and (
            not isinstance(expected_revision, str)
            or len(expected_revision) != 64
        ):
            raise ValueError("Feedback revision must be a SHA-256 value.")
        if not self.project_directory.is_dir():
            raise ValueError("The project workspace does not exist.")
        if not ConversationStore(
            self.project_directory
        ).contains_turn(conversation_id, identifier):
            raise ValueError(
                "The feedback target is not an active project turn."
            )
        directory = self.path.parent
        if directory.is_symlink() or (
            directory.exists() and not directory.is_dir()
        ):
            raise ValueError("The project memory folder is invalid.")
        directory.mkdir(exist_ok=True)
        with conversation_write_ownership(self.path):
            current = self.load()
            if (
                expected_revision is not None
                and current.revision != expected_revision
            ):
                raise FeedbackConflictError(
                    "Conversation feedback changed since it was loaded."
                )
            if not ConversationStore(
                self.project_directory
            ).contains_turn(conversation_id, identifier):
                raise ValueError(
                    "The feedback target is not an active project turn."
                )
            entries = {
                key: dict(value) for key, value in current.entries.items()
            }
            ratings = entries.setdefault(identifier, {})
            existing = ratings.get(kind, MessageFeedback())
            rating = MessageFeedback(
                saved=existing.saved if saved is None else saved,
                reaction=(
                    existing.reaction if reaction is None else reaction
                ),
                excluded=(
                    existing.excluded if excluded is None else excluded
                ),
            )
            if rating == MessageFeedback():
                ratings.pop(kind, None)
            else:
                ratings[kind] = rating
            if not ratings:
                entries.pop(identifier, None)
            payload = {
                "format": _FORMAT,
                "entries": {
                    key: {
                        item: {
                            "saved": value.saved,
                            "reaction": value.reaction,
                            **({"excluded": True} if value.excluded else {}),
                        }
                        for item, value in sorted(ratings.items())
                    }
                    for key, ratings in sorted(entries.items())
                },
            }
            content = json.dumps(
                payload, indent=2, ensure_ascii=False, sort_keys=True
            ) + "\n"
            if len(content.encode("utf-8")) > _MAX_BYTES:
                raise ValueError("Conversation feedback exceeds its limit.")
            atomic_write_text(self.path, content)
            return self.load()

    def reset(
        self,
        conversation_id: str,
        turn_uuid: str,
        role: FeedbackRole,
        *,
        expected_revision: str | None = None,
    ) -> FeedbackSnapshot:
        """Clear Save and reaction without lifting an explicit exclusion.

        Parameters
        ----------
        conversation_id: str
            Active project conversation containing the turn.
        turn_uuid: str
            Stable identity of the turn to reset.
        role: FeedbackRole
            Question or answer role being reset.
        expected_revision: str | None, optional
            Revision read by an editor. Defaults to None.

        Returns
        -------
        FeedbackSnapshot
            New snapshot with this message reset to neutral and unsaved.

        """

        return self.update(
            conversation_id, turn_uuid, role,
            saved=False, reaction="neutral",
            expected_revision=expected_revision,
        )
