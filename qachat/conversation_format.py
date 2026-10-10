"""Define the persisted conversation-record structure used by QAChat.

This module owns the small JSON vocabulary shared by conversation storage,
RAG persistence, and structured tracing. It also provides the canonical
helper for locating resumable conversation state in both the current grouped
trace format and the supported legacy top-level format.

The module is intentionally dependency-light. It does not know about models,
projects, tracing behavior, or GUI state. Higher-level components use it to
interpret the same persisted representation without duplicating the format
contract.

Key Features
------------

- **Shared format vocabulary**

    Define the grouped conversation-session key, category, title, and resume
    state field in one place.

- **Legacy compatibility**

    Recognize the historical top-level ``conversation`` mapping before the
    grouped sessions representation.

- **Canonical state lookup**

    Locate resumable state consistently and optionally create the grouped
    container in place when persistence needs it.

Module Structure
----------------

- ``CONVERSATION_*`` constants

    Canonical identifiers used by the persisted conversation JSON format.

- ``conversation_state``

    Locate or create the mutable resumable-state mapping in a payload.

- ``new_conversation_session``

    Build a new grouped conversation-session record.

This module describes an internal storage contract. Its names are not exposed
through the package-level ``qachat`` API.

"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CONVERSATION_SESSION_KEY",
    "CONVERSATION_SESSION_TITLE",
    "CONVERSATION_SESSION_CATEGORY",
    "CONVERSATION_RESUME_STATE_KEY",
    "LEGACY_CONVERSATION_KEY",
    "new_conversation_session",
    "conversation_state",
]

CONVERSATION_SESSION_KEY: Final[str] = "05_conversations"
CONVERSATION_SESSION_TITLE: Final[str] = "Conversation turns"
CONVERSATION_SESSION_CATEGORY: Final[str] = "conversations"
CONVERSATION_RESUME_STATE_KEY: Final[str] = "resume_state"
LEGACY_CONVERSATION_KEY: Final[str] = "conversation"


def new_conversation_session(
    resume_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one grouped conversation-session record.

    Parameters
    ----------
    resume_state: dict[str, object] | None, optional
        Initial resumable conversation state. ``None`` creates an empty state.
        The supplied mapping is stored by reference.

    Returns
    -------
    dict[str, object]
        New conversation-session mapping with empty turn and operation groups.

    """

    state = {} if resume_state is None else resume_state
    return {
        "title": CONVERSATION_SESSION_TITLE,
        "category": CONVERSATION_SESSION_CATEGORY,
        "turns": {},
        "operations": {},
        CONVERSATION_RESUME_STATE_KEY: state,
    }


def conversation_state(
    payload: dict[str, object], *, create: bool = False
) -> dict[str, object]:
    """Locate resumable conversation state in a persisted payload.

    Parameters
    ----------
    payload: dict[str, object]
        Decoded conversation or trace JSON document.
    create: bool, optional
        Whether a missing grouped conversation session may be inserted into
        ``payload``. Defaults to ``False``.

    Returns
    -------
    dict[str, object]
        Mutable legacy conversation or grouped resume-state mapping contained
        in ``payload``.

    Raises
    ------
    ValueError
        If no valid state exists and creation is disabled, or if the grouped
        sessions container has an invalid structure.

    Notes
    -----
    The supported legacy top-level ``conversation`` mapping takes precedence
    over grouped sessions. With ``create=True``, the supplied payload may be
    modified in place.

    """

    legacy = payload.get(LEGACY_CONVERSATION_KEY)
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
            and session.get("category") == CONVERSATION_SESSION_CATEGORY
        ):
            state = (
                session.setdefault(CONVERSATION_RESUME_STATE_KEY, {})
                if create
                else session.get(
                    CONVERSATION_RESUME_STATE_KEY,
                    session.get(LEGACY_CONVERSATION_KEY),
                )
            )
            if isinstance(state, dict):
                return state

    if create:
        session = new_conversation_session()
        sessions[CONVERSATION_SESSION_KEY] = session
        state = session[CONVERSATION_RESUME_STATE_KEY]
        if isinstance(state, dict):
            return state

    raise ValueError("The saved conversation is invalid.")
