"""Define the canonical persisted conversation-record structure in QAChat.

Conversation storage, RAG persistence, and structured tracing share one JSON
format. Resumable state belongs exclusively to the ``resume_state`` mapping
under ``sessions["05_conversations"]``. Other trace sessions may coexist,
but cannot claim the conversation category.

The module is dependency-light. It does not manage models, projects, tracing
behavior, or GUI state. Callers use the shared contract to avoid interpreting
the same stored record differently.

Key Features
------------

- **One conversation session**

    Identify resumable state by its reserved grouped-session key.

- **Strict persisted structure**

    Reject obsolete fields, conflicting sessions, and malformed state without
    modifying the input document.

- **Explicit creation**

    Create the complete grouped session only when it is absent and creation
    has been requested.

Module Structure
----------------

- ``CONVERSATION_*`` constants

    Canonical identifiers shared by conversation JSON readers and writers.

- ``conversation_state``

    Locate or create the mutable resumable-state mapping in a document.

- ``new_conversation_session``

    Construct a complete grouped conversation-session record.

The names are internal to persistence and are not exported by ``qachat``.

Examples
--------
>>> from qachat.conversation_format import new_conversation_session
>>> record = new_conversation_session()
>>> record["category"]
'conversations'

"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CONVERSATION_SESSION_KEY",
    "CONVERSATION_SESSION_TITLE",
    "CONVERSATION_SESSION_CATEGORY",
    "CONVERSATION_RESUME_STATE_KEY",
    "new_conversation_session",
    "conversation_state",
]

CONVERSATION_SESSION_KEY: Final[str] = "05_conversations"
CONVERSATION_SESSION_TITLE: Final[str] = "Conversation turns"
CONVERSATION_SESSION_CATEGORY: Final[str] = "conversations"
CONVERSATION_RESUME_STATE_KEY: Final[str] = "resume_state"


def new_conversation_session(
    resume_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a complete grouped conversation-session record.

    Parameters
    ----------
    resume_state: dict[str, object] | None, optional
        Initial resumable state. ``None`` creates an empty state. A supplied
        mapping is stored by reference.

    Returns
    -------
    dict[str, object]
        New session with independent empty turn and operation groups.

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
    """Locate resumable state in the canonical grouped conversation session.

    Parameters
    ----------
    payload: dict[str, object]
        Decoded conversation or trace JSON document.
    create: bool, optional
        Create a complete canonical session if absent. Defaults to ``False``.

    Returns
    -------
    dict[str, object]
        The stored mutable ``resume_state`` mapping, without copying it.

    Raises
    ------
    ValueError
        If the payload is malformed, contains obsolete conversation fields,
        reserves the session key for another purpose, or has an ambiguous
        conversation category. Reading never modifies the supplied payload.

    Notes
    -----
    Creation does not migrate obsolete records. Both modes reject invalid
    existing data before making any change. Unrelated valid trace sessions
    and top-level metadata are retained.

    """

    invalid = "The saved conversation is invalid."
    if not isinstance(payload, dict) or "conversation" in payload:
        raise ValueError(invalid)
    trace = payload.get("trace")
    if isinstance(trace, dict) and "sessions" in trace:
        raise ValueError(invalid)

    sessions = payload.get("sessions")
    if sessions is None and "sessions" not in payload:
        if not create:
            raise ValueError(invalid)
        sessions = {}
    if not isinstance(sessions, dict):
        raise ValueError(invalid)

    # A category-only search is ambiguous. The reserved key is the sole
    # supported location even when another trace session claims this category.
    for key, session in sessions.items():
        if (
            key != CONVERSATION_SESSION_KEY
            and isinstance(session, dict)
            and session.get("category") == CONVERSATION_SESSION_CATEGORY
        ):
            raise ValueError(
                f"{invalid} Conflicting conversation session {key!r}."
            )

    if CONVERSATION_SESSION_KEY not in sessions:
        if not create:
            raise ValueError(invalid)
        session = new_conversation_session()
        state = session.get(CONVERSATION_RESUME_STATE_KEY)
        if not isinstance(state, dict):
            raise ValueError(invalid)
        # Do not mutate the payload until construction has succeeded.
        if "sessions" not in payload:
            payload["sessions"] = sessions
        sessions[CONVERSATION_SESSION_KEY] = session
        return state

    session = sessions[CONVERSATION_SESSION_KEY]
    if (
        not isinstance(session, dict)
        or session.get("category") != CONVERSATION_SESSION_CATEGORY
        or not isinstance(session.get("title"), str)
        or not isinstance(session.get("turns"), dict)
        or not isinstance(session.get("operations"), dict)
        or "conversation" in session
    ):
        raise ValueError(
            f"{invalid} Invalid reserved session "
            f"{CONVERSATION_SESSION_KEY!r}."
        )

    state = session.get(CONVERSATION_RESUME_STATE_KEY)
    if not isinstance(state, dict) or "chat_messages" in state:
        raise ValueError(invalid)
    return state
