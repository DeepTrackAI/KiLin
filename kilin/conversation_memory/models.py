"""Define durable turn identities and conversation feedback values.

A turn UUID identifies one committed question/answer pair, including copies
retained in branches. A message role identifies the question or answer within
that pair. Neither a timestamp nor the text of a message is used as feedback
identity: independently generated duplicate text may be rated differently.

Key Features
------------

- **Unique completed turns**

    Fresh UUID4 identifiers and strict validation of stored identities.

- **Immutable feedback values**

    Save and Like remain distinguishable; Dislike excludes the answer later.

Module Structure
----------------

- ``new_turn_uuid`` and ``require_turn_uuid``

    Generate and validate canonical turn identifiers.

- ``require_saved_turn_ids``

    Reject unsupported saved turns before they are copied into branches.

- ``MessageFeedback``

    Represent the user's selection and reaction for one message.

Examples
--------
>>> from kilin.conversation_memory.models import MessageFeedback
>>> MessageFeedback(saved=True).selected
True

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Final, Literal, TypeAlias
from uuid import UUID, uuid4

__all__ = [
    "MessageFeedback",
    "FeedbackReaction",
    "FeedbackRole",
    "new_turn_uuid",
    "require_turn_uuid",
    "require_saved_turn_ids",
    "require_feedback_role",
]

FeedbackReaction: TypeAlias = Literal["neutral", "liked", "disliked"]
FeedbackRole: TypeAlias = Literal["user", "assistant"]
_REACTIONS: Final[frozenset[str]] = frozenset(
    ("neutral", "liked", "disliked")
)
_ROLES: Final[frozenset[str]] = frozenset(("user", "assistant"))
_UUID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{32}")


def new_turn_uuid() -> str:
    """Create a canonical UUID4 for one completed conversation turn.

    Returns
    -------
    str
        Lowercase, hyphen-free UUID4 hexadecimal identifier.

    """

    return uuid4().hex


def require_turn_uuid(value: object) -> str:
    """Require the canonical stored identity of a completed turn.

    Parameters
    ----------
    value: object
        Untrusted turn identifier from an API or conversation record.

    Returns
    -------
    str
        Validated canonical UUID4 hex string.

    Raises
    ------
    ValueError
        If the identifier is missing, malformed, or not a UUID4.

    """

    if (
        not isinstance(value, str)
        or _UUID_PATTERN.fullmatch(value) is None
        or UUID(hex=value).version != 4
    ):
        raise ValueError("A completed turn needs a valid UUID4.")
    return value


def require_feedback_role(value: object) -> FeedbackRole:
    """Validate a question or answer role for message feedback.

    Parameters
    ----------
    value: object
        Role supplied by an untrusted caller or saved JSON.

    Returns
    -------
    FeedbackRole
        Canonical ``user`` or ``assistant`` role.

    Raises
    ------
    ValueError
        If the role is not a feedback-capable message role.

    """

    if not isinstance(value, str) or value not in _ROLES:
        raise ValueError("Feedback role must be user or assistant.")
    return value


def require_saved_turn_ids(turns: object) -> tuple[str, ...]:
    """Validate distinct UUIDs in one current-format saved conversation.

    Parameters
    ----------
    turns: object
        Raw saved list of completed-turn dictionaries.

    Returns
    -------
    tuple[str, ...]
        Turn identities in the same order as the saved turns.

    Raises
    ------
    ValueError
        If the structure or any ID is unsupported or duplicated.

    Notes
    -----
    This is not a migration or a full answer/evidence parser. It ensures
    branches cannot duplicate an obsolete identity-less checkpoint.

    """

    if not isinstance(turns, list):
        raise ValueError("The saved conversation turns are invalid.")
    ids: list[str] = []
    for item in turns:
        if not isinstance(item, Mapping):
            raise ValueError("The saved conversation turns are invalid.")
        ids.append(require_turn_uuid(item.get("conversation_turn_uuid")))
    if len(ids) != len(set(ids)):
        raise ValueError("The saved conversation has duplicate turn UUIDs.")
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class MessageFeedback:
    """User-owned selection and rating for one question or answer.

    Parameters
    ----------
    saved: bool, optional
        Whether Save was selected independently of a Like. Defaults to False.
    reaction: FeedbackReaction, optional
        Neutral, liked, or disliked. Defaults to neutral.
    excluded: bool, optional
        Explicit exclusion from long-term retrieval. Defaults to False.

    Attributes
    ----------
    saved: bool
        Persistent explicit Save selection.
    reaction: FeedbackReaction
        Persistent rating, independent of Save.
    excluded: bool
        Researcher-controlled exclusion, independent of feedback.

    Notes
    -----
    A Like is an explicit selection even without Save. A Dislike or an
    explicit exclusion always removes the message from memory retrieval.
    Feedback does not certify
    the factual correctness of generated text.

    """

    saved: bool = False
    reaction: FeedbackReaction = "neutral"
    excluded: bool = False

    def __post_init__(self) -> None:
        """Reject coerced or unknown feedback values immediately.

        Raises
        ------
        ValueError
            If saved or reaction does not have its required type/value.

        """

        if type(self.excluded) is not bool or type(self.saved) is not bool or (
            not isinstance(self.reaction, str)
            or self.reaction not in _REACTIONS
        ):
            raise ValueError("Invalid conversation message feedback.")

    @property
    def selected(self) -> bool:
        """Return whether the message was explicitly selected for memory.

        Returns
        -------
        bool
            True for a saved or liked message, except when disliked.

        """

        return not self.excluded and self.reaction != "disliked" and (
            self.saved or self.reaction == "liked"
        )
